from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bnsv.garch_models as garch_models
from bnsv.forecast_features import ReturnScaler
from bnsv.garch_models import (
    GARCH_T_ID,
    REALIZED_GARCH_T_ID,
    GarchTFit,
    GarchTFitError,
    StudentTRealizedGarchFit,
    forecast_realized_garch_log_variance,
    fit_garch_t,
    garch_variance_mean_path,
    generate_garch_panel,
)


def _garch_fit() -> GarchTFit:
    return GarchTFit(
        variance_bar=1.0,
        persistence=0.96,
        arch_share=0.10,
        omega=0.04,
        alpha=0.096,
        beta=0.864,
        nu=8.0,
        negative_log_likelihood=100.0,
        optimizer_start=0,
    )


def _realized_fit() -> StudentTRealizedGarchFit:
    return StudentTRealizedGarchFit(
        log_v_bar=-0.5,
        persistence=0.94,
        realized_share=0.25,
        xi=0.05,
        measurement_phi=1.0,
        tau1=-0.08,
        tau2=0.06,
        sigma_u=0.18,
        nu=8.0,
        log_v0=-0.5,
        omega=-0.0425,
        beta=0.705,
        gamma=0.235,
        negative_log_likelihood=100.0,
        optimizer_start=0,
    )


def _component_arrays(root: Path, relative: str) -> dict[str, np.ndarray]:
    with np.load(root / relative, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_garch_panel_writes_loss_builder_components(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    start_position = 100
    dates = pd.DatetimeIndex(synthetic_frame["date"])

    def fixed_fit(values: np.ndarray, *, maxiter: int):
        assert maxiter == 2000
        return _garch_fit(), np.ones(values.size)

    panel = generate_garch_panel(
        frame=synthetic_frame,
        market="SP500",
        model_id=GARCH_T_ID,
        refit_dates=pd.DatetimeIndex([dates[start_position]]),
        forecast_start=dates[start_position],
        forecast_end=dates[start_position + 2],
        horizons=(1, 5, 10),
        origin_alignment_horizon=10,
        optimizer_maxiter=2000,
        path_count=32,
        base_seed=20260730,
        components_dir=tmp_path,
        garch_fit_function=fixed_fit,
    )
    assert len(panel) == 9
    assert set(panel["model_id"]) == {GARCH_T_ID}
    assert panel["predictive_file"].notna().all()
    first = panel.iloc[0]
    scaler = ReturnScaler.fit(
        synthetic_frame.iloc[: start_position + 1]["return_cc"].to_numpy(float)
    )
    observed = float(
        scaler.standardize(synthetic_frame.loc[start_position, "return_cc"])
    )
    one_step = 0.04 + 0.096 * observed**2 + 0.864
    assert np.isclose(first["variance_forecast"], scaler.scale**2 * one_step)
    arrays = _component_arrays(tmp_path, str(first["predictive_file"]))
    assert set(arrays) == {
        "raw_returns",
        "raw_variances",
        "raw_return_locations",
        "raw_conditional_sds",
        "degrees_of_freedom",
    }
    assert arrays["raw_returns"].shape == (32, 1)
    assert arrays["raw_variances"].shape == (32, 1)
    np.testing.assert_allclose(arrays["raw_return_locations"], scaler.mean)
    np.testing.assert_allclose(arrays["degrees_of_freedom"], 8.0)


def test_failed_refit_has_no_stale_forecasts(
    synthetic_frame: pd.DataFrame,
) -> None:
    dates = pd.DatetimeIndex(synthetic_frame["date"])
    start_position = 100
    calls = 0

    def fail_then_fit(values: np.ndarray, *, maxiter: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GarchTFitError("synthetic failure")
        return _garch_fit(), np.ones(values.size)

    panel = generate_garch_panel(
        frame=synthetic_frame,
        market="SP500",
        model_id=GARCH_T_ID,
        refit_dates=pd.DatetimeIndex(
            [dates[start_position], dates[start_position + 3]]
        ),
        forecast_start=dates[start_position],
        forecast_end=dates[start_position + 4],
        horizons=(1, 5, 10),
        origin_alignment_horizon=10,
        optimizer_maxiter=2000,
        path_count=16,
        base_seed=20260730,
        garch_fit_function=fail_then_fit,
    )
    assert calls == 2
    assert set(panel["origin_date"]) == {
        str(dates[start_position + 3].date()),
        str(dates[start_position + 4].date()),
    }


def test_realized_garch_panel_uses_h1_transition_and_components(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    start_position = 100
    dates = pd.DatetimeIndex(synthetic_frame["date"])

    def fixed_fit(values: np.ndarray, log_rv: np.ndarray, *, maxiter: int):
        assert values.shape == log_rv.shape
        return _realized_fit(), np.full(values.size, -0.5)

    panel = generate_garch_panel(
        frame=synthetic_frame,
        market="SP500",
        model_id=REALIZED_GARCH_T_ID,
        refit_dates=pd.DatetimeIndex([dates[start_position]]),
        forecast_start=dates[start_position],
        forecast_end=dates[start_position + 2],
        horizons=(1,),
        origin_alignment_horizon=10,
        optimizer_maxiter=2000,
        path_count=32,
        base_seed=20260730,
        components_dir=tmp_path,
        realized_garch_fit_function=fixed_fit,
    )
    assert len(panel) == 3
    first = panel.iloc[0]
    scaler = ReturnScaler.fit(
        synthetic_frame.iloc[: start_position + 1]["return_cc"].to_numpy(float)
    )
    observed_log_rv = np.log(
        synthetic_frame.loc[start_position, "rv_oc"] / scaler.scale**2
    )
    forecast_log_h = forecast_realized_garch_log_variance(
        _realized_fit(), origin_log_h=-0.5, observed_log_rv=observed_log_rv
    )
    assert np.isclose(
        first["variance_forecast"], scaler.scale**2 * np.exp(forecast_log_h)
    )
    arrays = _component_arrays(tmp_path, str(first["predictive_file"]))
    np.testing.assert_allclose(
        arrays["raw_variances"], first["variance_forecast"]
    )
    np.testing.assert_allclose(arrays["raw_return_locations"], scaler.mean)
    np.testing.assert_allclose(arrays["degrees_of_freedom"], 8.0)


def test_garch_variance_mean_recursion() -> None:
    path = garch_variance_mean_path(
        _garch_fit(), forecast_variance=1.3, horizon=3
    )
    np.testing.assert_allclose(path, [1.3, 0.04 + 0.96 * 1.3, 0.04 + 0.96 * 1.288])


def test_garch_fit_requires_two_near_optimal_starts(monkeypatch) -> None:
    results = iter(
        [
            SimpleNamespace(
                success=True,
                fun=value,
                x=np.asarray([0.0, 3.0, -2.0, np.log(6.0)]),
            )
            for value in (100.0, 100.01, 100.5, 101.0)
        ]
    )
    monkeypatch.setattr(
        garch_models, "minimize", lambda *args, **kwargs: next(results)
    )
    with pytest.raises(GarchTFitError):
        fit_garch_t(np.random.default_rng(1).normal(size=200))
