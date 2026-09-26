from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bnsv.forecast_generation as generation
from bnsv.forecast_filter import FixedParameterGridFilter, filter_bank_is_adequate
from bnsv.forecast_fit import SamplingFailure
from bnsv.forecast_paths import seed_from_context


class SyntheticFit:
    def __init__(self, observations: int, draws: int = 8) -> None:
        axis = np.linspace(-0.1, 0.1, draws)
        self.variables = {
            "mu": axis,
            "phi": np.linspace(0.85, 0.92, draws),
            "sigma_eta": np.linspace(0.10, 0.14, draws),
            "nu": np.linspace(7.0, 10.0, draws),
            "b": np.linspace(-0.3, 0.2, observations)[None, :]
            + axis[:, None],
            "W1": np.zeros((draws, 2, 5)),
            "b1": np.zeros((draws, 5)),
            "output_weight": np.full((draws, 5), 0.05),
            "centering_mean": np.zeros(draws),
            "gamma_A": np.linspace(-0.05, 0.05, draws),
            "q_centering_mean": np.zeros(draws),
        }

    def stan_variable(self, name: str) -> np.ndarray:
        return self.variables[name]

    def stan_variables(self) -> dict[str, np.ndarray]:
        return self.variables


def _fit(**kwargs: Any) -> tuple[SyntheticFit, list[dict[str, Any]]]:
    return SyntheticFit(int(kwargs["data"]["N"])), []


def test_rv_ra_panel_and_five_scoring_components_are_deterministic(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch,
) -> None:
    origin = pd.Timestamp(synthetic_frame.loc[59, "date"])
    monkeypatch.setattr(generation, "filter_bank_is_adequate", lambda *a, **k: True)
    arguments = {
        "daily": synthetic_frame,
        "market": "SP500",
        "model_id": "RV-RA-NN-SV",
        "stan_file": "rv_ra_nn_sv.stan",
        "refit_dates": pd.DatetimeIndex([origin]),
        "base_seed": 20260730,
        "forecast_start": origin,
        "forecast_end": origin,
        "particle_count": 4,
        "path_count": 8,
        "fit_function": _fit,
        "certification_grid_levels": (0, 1, 2),
    }
    component_dir = tmp_path / "components"
    first = generation.generate_forecast_panel(
        **arguments,
        output_work_dir=tmp_path / "work-a",
        components_dir=component_dir,
    )
    second = generation.generate_forecast_panel(
        **arguments,
        output_work_dir=tmp_path / "work-b",
    )
    pd.testing.assert_frame_equal(first.drop(columns="predictive_file"), second)
    assert first["horizon"].tolist() == [1, 5, 10]
    for row in first.itertuples(index=False):
        with np.load(component_dir / row.predictive_file, allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
                "degrees_of_freedom",
            }
            assert arrays["raw_returns"].shape == (8, row.horizon)


def test_unsuccessful_rv_ra_refit_resumes_only_at_the_next_schedule_date(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch,
) -> None:
    origins = [pd.Timestamp(synthetic_frame.loc[index, "date"]) for index in (59, 61)]
    calls = 0

    def fit(**kwargs: Any) -> tuple[SyntheticFit, list[dict[str, Any]]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SamplingFailure([])
        return _fit(**kwargs)

    monkeypatch.setattr(generation, "filter_bank_is_adequate", lambda *a, **k: True)
    panel = generation.generate_forecast_panel(
        daily=synthetic_frame,
        market="SP500",
        model_id="RV-RA-NN-SV",
        stan_file="rv_ra_nn_sv.stan",
        refit_dates=pd.DatetimeIndex(origins),
        output_work_dir=tmp_path / "work",
        base_seed=20260730,
        forecast_start=origins[0],
        forecast_end=origins[1],
        particle_count=4,
        path_count=8,
        fit_function=fit,
        certification_grid_levels=(0, 1, 2),
    )
    assert calls == 2
    assert set(panel["origin_date"]) == {str(origins[1].date())}


def test_filter_certification_uses_three_adjacent_refinements() -> None:
    parameters = {
        "mu": np.linspace(-0.4, 0.4, 4),
        "phi": np.full(4, 0.9),
        "sigma_eta": np.full(4, 0.1),
        "nu": np.full(4, 8.0),
    }
    filters = {
        level: FixedParameterGridFilter(
            parameters={name: value.copy() for name, value in parameters.items()},
            initial_state=np.zeros(4),
            weights=np.full(4, 0.25),
            ancestry=np.arange(4),
            level=level,
        )
        for level in (0, 1, 2)
    }
    assert filter_bank_is_adequate(filters, initial_weights=np.full(4, 0.25))
    filters[2].step(
        x_t=np.zeros(1),
        observed_z=0.25,
        correction=lambda parameters, features: np.zeros(4),
    )
    assert not filter_bank_is_adequate(
        filters, initial_weights=np.full(4, 0.25)
    )


def test_rv_ra_refit_seed_matches_the_stated_context() -> None:
    assert seed_from_context(20260730, "DAX", "2020-05-04", "nuts") == 2137880760
