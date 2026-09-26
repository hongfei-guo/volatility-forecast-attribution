from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import bnsv.forecast_generation as generation
from bnsv.forecast_features import (
    fit_feature_transform,
    prepare_estimation_data,
)
from bnsv.forecast_filter import FixedParameterGridFilter
from bnsv.forecast_paths import (
    compact_refit,
    fit_log_har_posterior,
    seed_from_context,
    simulate_log_har_paths,
)


class SyntheticFit:
    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self._variables = variables

    def stan_variable(self, name: str) -> np.ndarray:
        return self._variables[name]

    def stan_variables(self) -> dict[str, np.ndarray]:
        return self._variables


def neural_variables(*, observations: int, draws: int = 8) -> dict[str, np.ndarray]:
    draw_axis = np.linspace(-0.1, 0.1, draws)
    baseline = np.linspace(-0.3, 0.2, observations)
    return {
        "mu": draw_axis,
        "phi": np.linspace(0.85, 0.92, draws),
        "sigma_eta": np.linspace(0.10, 0.14, draws),
        "nu": np.linspace(7.0, 10.0, draws),
        "b": baseline[None, :] + draw_axis[:, None],
        "W1": np.zeros((draws, 1, 5)),
        "b1": np.zeros((draws, 5)),
        "output_weight": np.full((draws, 5), 0.05),
        "centering_mean": np.zeros(draws),
    }


def test_compact_refit_selects_reproducible_atoms(
    synthetic_frame: pd.DataFrame,
) -> None:
    end = synthetic_frame.loc[59, "date"]
    transform = fit_feature_transform(
        synthetic_frame, model_id="NN-SV", training_end=end
    )
    estimation = prepare_estimation_data(synthetic_frame, transform)
    fit = SyntheticFit(neural_variables(observations=estimation.z.size))

    first = compact_refit(
        fit,
        model_id="NN-SV",
        transform=transform,
        estimation_features=estimation.x,
        particle_count=4,
        rng=np.random.default_rng(41),
    )
    second = compact_refit(
        fit,
        model_id="NN-SV",
        transform=transform,
        estimation_features=estimation.x,
        particle_count=4,
        rng=np.random.default_rng(41),
    )

    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    selected = first["parameter__source_draw_index"]
    source = fit.stan_variables()
    np.testing.assert_array_equal(first["state"], source["b"][selected, -1])
    np.testing.assert_array_equal(first["ancestry"], np.arange(4))
    assert first["parameter__w1"].shape == (4, 1, 5)
    assert first["weights"].sum() == 1.0


def test_log_har_posterior_and_paths_have_declared_shapes() -> None:
    history = np.linspace(-10.0, -8.0, 60) + 0.05 * np.sin(np.arange(60))
    beta, sigma = fit_log_har_posterior(
        history,
        draws=7,
        rng=np.random.default_rng(52),
    )
    paths = simulate_log_har_paths(
        history,
        beta=beta,
        sigma=sigma,
        horizon=10,
        paths=9,
        rng=np.random.default_rng(53),
    )

    assert beta.shape == (7, 4)
    assert sigma.shape == (7,)
    assert paths.shape == (9, 10)
    assert np.isfinite(paths).all()
    assert np.all(sigma > 0)


def test_context_seed_is_fixed_and_context_specific() -> None:
    expected = 1742244529
    assert seed_from_context(
        20260730, "SP500", "2018-01-02", 10, "forecast_paths"
    ) == expected
    assert seed_from_context(
        20260730, "SP500", "2018-01-02", 10, "forecast_paths"
    ) == expected
    assert seed_from_context(
        20260730, "SP500", "2018-01-02", 5, "forecast_paths"
    ) != expected


def test_panel_forecasts_each_origin_before_filtering(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_position = 59
    second_position = 60
    first_origin = pd.Timestamp(synthetic_frame.loc[first_position, "date"])
    second_origin = pd.Timestamp(synthetic_frame.loc[second_position, "date"])
    events: list[tuple[str, int | None]] = []
    fit_calls: list[int] = []

    def fit_function(**kwargs: Any) -> tuple[SyntheticFit, list[dict[str, Any]]]:
        fit_calls.append(int(kwargs["seed"]))
        observations = int(kwargs["data"]["N"])
        return SyntheticFit(neural_variables(observations=observations)), []

    original_forecast = generation.simulate_forecast_paths

    def traced_forecast(**kwargs: Any) -> dict[str, np.ndarray]:
        events.append(("forecast", int(kwargs["horizon"])))
        return original_forecast(**kwargs)

    original_step = FixedParameterGridFilter.step

    def traced_step(
        self: FixedParameterGridFilter, **kwargs: Any
    ) -> dict[str, float | np.ndarray]:
        events.append(("filter", None))
        return original_step(self, **kwargs)

    monkeypatch.setattr(generation, "simulate_forecast_paths", traced_forecast)
    monkeypatch.setattr(FixedParameterGridFilter, "step", traced_step)

    arguments: dict[str, Any] = {
        "daily": synthetic_frame,
        "market": "SP500",
        "model_id": "NN-SV",
        "stan_file": "nn_sv.stan",
        "refit_dates": pd.DatetimeIndex([first_origin]),
        "base_seed": 20260730,
        "forecast_start": first_origin,
        "forecast_end": second_origin,
        "particle_count": 4,
        "grid_level": 0,
        "path_count": 8,
        "fit_function": fit_function,
    }
    panel = generation.generate_forecast_panel(
        **arguments,
        output_work_dir=tmp_path / "forecast-work",
    )

    expected_events = [
        ("forecast", 1),
        ("forecast", 5),
        ("forecast", 10),
        ("filter", None),
        ("forecast", 1),
        ("forecast", 5),
        ("forecast", 10),
    ]
    assert events == expected_events
    events.clear()
    repeated = generation.generate_forecast_panel(
        **arguments,
        output_work_dir=tmp_path / "forecast-work-repeat",
    )
    assert events == expected_events
    pd.testing.assert_frame_equal(panel, repeated)
    assert len(fit_calls) == 2 and fit_calls[0] == fit_calls[1]

    assert list(panel.columns) == [
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "mature_date",
        "forecast_kind",
        "variance_forecast",
        "cumulative_return_mean",
        "forecast_draw_count",
        "return_distribution",
        "variance_mean_method",
    ]
    assert panel.shape == (6, 11)
    assert panel["origin_date"].tolist() == [
        str(first_origin.date()),
        str(first_origin.date()),
        str(first_origin.date()),
        str(second_origin.date()),
        str(second_origin.date()),
        str(second_origin.date()),
    ]
    assert panel["horizon"].tolist() == [1, 5, 10, 1, 5, 10]
    assert set(panel["forecast_draw_count"]) == {8}
    assert set(panel["forecast_kind"]) == {"probabilistic"}
    assert set(panel["return_distribution"]) == {
        "variance_standardized_student_t"
    }
    assert set(panel["variance_mean_method"]) == {
        "mean_integrated_path_variance"
    }
    assert np.isfinite(
        panel[["variance_forecast", "cumulative_return_mean"]].to_numpy(float)
    ).all()
    assert np.all(panel["variance_forecast"].to_numpy(float) > 0)

    component_root = tmp_path / "forecast-panels" / "components"
    with_components = generation.generate_forecast_panel(
        **arguments,
        output_work_dir=tmp_path / "forecast-work-components",
        components_dir=component_root,
    )
    pd.testing.assert_frame_equal(
        panel,
        with_components.drop(columns="predictive_file"),
    )
    assert with_components["predictive_file"].str.endswith(".npz").all()
    for row in with_components.itertuples(index=False):
        component_path = component_root / row.predictive_file
        assert component_path.is_file()
        with np.load(component_path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
                "degrees_of_freedom",
            }
            expected_shape = (8, row.horizon)
            for name in (
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
            ):
                assert arrays[name].shape == expected_shape
            assert arrays["degrees_of_freedom"].shape == (8,)
            assert np.all(arrays["degrees_of_freedom"] > 2)
            assert float(np.mean(np.sum(arrays["raw_variances"], axis=1))) == (
                row.variance_forecast
            )


def test_horizon_one_model_keeps_the_last_valid_origin(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    origin = pd.Timestamp(synthetic_frame.iloc[-2]["date"])

    def linear_fit(**kwargs: Any) -> tuple[SyntheticFit, list[dict[str, Any]]]:
        observations = int(kwargs["data"]["N"])
        draws = 8
        axis = np.linspace(-0.1, 0.1, draws)
        variables = {
            "mu": axis,
            "phi": np.linspace(0.85, 0.92, draws),
            "sigma_eta": np.linspace(0.10, 0.14, draws),
            "nu": np.linspace(7.0, 10.0, draws),
            "b": np.linspace(-0.3, 0.2, observations)[None, :]
            + axis[:, None],
            "beta": np.zeros((draws, 2)),
        }
        return SyntheticFit(variables), []

    panel = generation.generate_forecast_panel(
        daily=synthetic_frame,
        market="SP500",
        model_id="RV-LIN-SV",
        stan_file="rv_lin_sv.stan",
        refit_dates=pd.DatetimeIndex([origin]),
        output_work_dir=tmp_path / "linear-work",
        base_seed=20260730,
        forecast_start=origin,
        forecast_end=origin,
        particle_count=4,
        grid_level=0,
        path_count=8,
        fit_function=linear_fit,
    )
    assert panel[["origin_date", "horizon", "mature_date"]].to_dict(
        "records"
    ) == [
        {
            "origin_date": str(origin.date()),
            "horizon": 1,
            "mature_date": str(
                pd.Timestamp(synthetic_frame.iloc[-1]["date"]).date()
            ),
        }
    ]
