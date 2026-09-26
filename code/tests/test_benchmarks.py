from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bnsv.benchmarks as benchmarks
from bnsv.benchmarks import (
    BENCHMARK_IDS,
    FORECAST_PANEL_COLUMNS,
    GAUSSIAN_COMPONENT_NAMES,
    GaussianRealizedGarchFit,
    analytic_gaussian_variance_path,
    generate_benchmark_panel,
    har_design,
)


def _fixed_gaussian_fit(
    standardized_returns: np.ndarray,
    log_rv: np.ndarray,
    *,
    maxiter: int,
) -> tuple[GaussianRealizedGarchFit, np.ndarray]:
    del standardized_returns, maxiter
    fit = GaussianRealizedGarchFit(
        log_v_bar=-1.0,
        persistence=0.9,
        realized_share=0.2,
        xi=0.0,
        measurement_phi=1.0,
        tau1=0.0,
        tau2=0.0,
        sigma_u=0.1,
        log_v0=-1.0,
        omega=-0.1,
        beta=0.72,
        gamma=0.18,
        negative_log_likelihood=0.0,
        optimizer_start=0,
    )
    return fit, np.full(log_rv.size, -1.0)


def test_har_design_uses_daily_weekly_and_monthly_lags() -> None:
    values = np.arange(1.0, 31.0)
    design, target = har_design(values)
    np.testing.assert_allclose(
        design[0],
        [1.0, values[21], np.mean(values[17:22]), np.mean(values[:22])],
    )
    assert target[0] == values[22]


def test_gaussian_realized_garch_analytic_path() -> None:
    fit = GaussianRealizedGarchFit(
        log_v_bar=-0.5,
        persistence=0.8,
        realized_share=0.25,
        xi=0.0,
        measurement_phi=1.0,
        tau1=0.0,
        tau2=0.0,
        sigma_u=0.0,
        log_v0=-0.5,
        omega=-0.1,
        beta=0.6,
        gamma=0.2,
        negative_log_likelihood=0.0,
        optimizer_start=0,
    )
    path = analytic_gaussian_variance_path(
        fit, forecast_log_h=-0.4, horizon=2
    )
    np.testing.assert_allclose(path, np.exp([-0.4, -0.42]))


def test_generator_matches_public_panel_schema(
    synthetic_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_position = 105
    second_position = 106
    first_origin = pd.Timestamp(synthetic_frame.loc[first_position, "date"])
    second_origin = pd.Timestamp(synthetic_frame.loc[second_position, "date"])

    monkeypatch.setattr(
        benchmarks, "fit_gaussian_realized_garch", _fixed_gaussian_fit
    )
    panel = generate_benchmark_panel(
        frame=synthetic_frame,
        market="SP500",
        model_ids=BENCHMARK_IDS,
        refit_dates=pd.DatetimeIndex([first_origin]),
        forecast_start=first_origin,
        forecast_end=second_origin,
        forecast_draw_count=4096,
    )

    assert list(panel.columns) == list(FORECAST_PANEL_COLUMNS)
    assert panel.shape == (24, len(FORECAST_PANEL_COLUMNS))
    assert set(panel["model_id"]) == set(BENCHMARK_IDS)
    assert set(panel["horizon"]) == {1, 5, 10}
    assert set(panel["origin_date"]) == {
        str(first_origin.date()),
        str(second_origin.date()),
    }
    assert np.all(panel["variance_forecast"].to_numpy(float) > 0)
    gaussian = panel[
        panel["model_id"].eq("Realized-GARCH-Gaussian-QMLE")
    ]
    assert set(gaussian["forecast_kind"]) == {"probabilistic"}
    assert set(gaussian["forecast_draw_count"]) == {4096}
    assert set(gaussian["return_distribution"]) == {"gaussian"}
    classical = panel[~panel.index.isin(gaussian.index)]
    assert set(classical["forecast_kind"]) == {"variance_point_only"}
    assert classical["forecast_draw_count"].isna().all()


def test_gaussian_components_are_deterministic_and_scorer_compatible(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = pd.Timestamp(synthetic_frame.loc[105, "date"])
    monkeypatch.setattr(
        benchmarks, "fit_gaussian_realized_garch", _fixed_gaussian_fit
    )
    arguments = {
        "frame": synthetic_frame,
        "market": "SP500",
        "model_ids": ["Realized-GARCH-Gaussian-QMLE"],
        "refit_dates": pd.DatetimeIndex([origin]),
        "forecast_start": origin,
        "forecast_end": origin,
        "forecast_draw_count": 32,
        "base_seed": 20260730,
    }
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_benchmark_panel(**arguments, components_dir=first_root)
    second = generate_benchmark_panel(**arguments, components_dir=second_root)
    pd.testing.assert_frame_equal(first, second)
    assert first["predictive_file"].notna().all()
    for row in first.itertuples(index=False):
        with np.load(first_root / row.predictive_file, allow_pickle=False) as left:
            with np.load(second_root / row.predictive_file, allow_pickle=False) as right:
                assert set(left.files) == set(GAUSSIAN_COMPONENT_NAMES)
                assert set(right.files) == set(GAUSSIAN_COMPONENT_NAMES)
                for name in GAUSSIAN_COMPONENT_NAMES:
                    assert left[name].shape == (32, row.horizon)
                    np.testing.assert_array_equal(left[name], right[name])
                np.testing.assert_allclose(
                    left["raw_conditional_sds"] ** 2,
                    left["raw_variances"],
                )


def test_packaged_benchmarks_use_the_declared_schema_and_horizons() -> None:
    package = Path(__file__).resolve().parents[2]
    for name in ("DAX.parquet", "FTSE100.parquet", "SP500.parquet"):
        path = package / "forecasts" / name
        panel = pd.read_parquet(path)
        assert list(panel.columns) == list(FORECAST_PANEL_COLUMNS)
        selected = panel[panel["model_id"].isin(BENCHMARK_IDS)]
        assert set(selected["model_id"]) == set(BENCHMARK_IDS)
        for model_id in BENCHMARK_IDS:
            assert set(selected.loc[selected["model_id"].eq(model_id), "horizon"]) == {
                1,
                5,
                10,
            }
