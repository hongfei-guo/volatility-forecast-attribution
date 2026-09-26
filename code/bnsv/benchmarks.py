"""Expanding-window forecasts for the external variance benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import yaml

from .data_contract import read_daily_frame, validate_daily_frame
from .forecast_features import ReturnScaler
from .forecast_paths import seed_from_context


BENCHMARK_IDS = (
    "HAR-RV",
    "SHAR-RV",
    "log-HAR-RV",
    "Realized-GARCH-Gaussian-QMLE",
)
HORIZONS = (1, 5, 10)
FORECAST_PANEL_COLUMNS = (
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
    "candidate_availability",
    "candidate_model_ids",
    "available_candidate_model_ids",
    "combination_mode",
    "combination_weights",
)
GAUSSIAN_COMPONENT_NAMES = (
    "raw_returns",
    "raw_variances",
    "raw_return_locations",
    "raw_conditional_sds",
)


@dataclass(frozen=True)
class OLSFit:
    coefficients: np.ndarray
    residual_sd: float


@dataclass(frozen=True)
class GaussianRealizedGarchFit:
    log_v_bar: float
    persistence: float
    realized_share: float
    xi: float
    measurement_phi: float
    tau1: float
    tau2: float
    sigma_u: float
    log_v0: float
    omega: float
    beta: float
    gamma: float
    negative_log_likelihood: float
    optimizer_start: int


class RealizedGarchFitError(RuntimeError):
    """Raised when estimation cannot produce an admissible fitted model."""


def har_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < 24 or np.any(~np.isfinite(series)):
        raise ValueError("HAR requires at least 24 finite observations")
    rows = [
        [
            1.0,
            series[index - 1],
            np.mean(series[index - 5 : index]),
            np.mean(series[index - 22 : index]),
        ]
        for index in range(22, series.size)
    ]
    return np.asarray(rows), series[22:].copy()


def shar_design(
    rv: np.ndarray, rv_up: np.ndarray, rv_down: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(rv, dtype=float)
    up = np.asarray(rv_up, dtype=float)
    down = np.asarray(rv_down, dtype=float)
    if not (target.shape == up.shape == down.shape) or target.size < 24:
        raise ValueError("SHAR inputs must align and contain at least 24 observations")
    if np.any(~np.isfinite(np.column_stack((target, up, down)))):
        raise ValueError("SHAR inputs must be finite")
    rows = [
        [
            1.0,
            up[index - 1],
            down[index - 1],
            np.mean(target[index - 5 : index]),
            np.mean(target[index - 22 : index]),
        ]
        for index in range(22, target.size)
    ]
    return np.asarray(rows), target[22:].copy()


def fit_ols(x: np.ndarray, y: np.ndarray) -> OLSFit:
    design = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("benchmark design matrix is rank deficient")
    residual = target - design @ coefficients
    degrees_of_freedom = max(1, design.shape[0] - design.shape[1])
    residual_sd = float(np.sqrt(residual @ residual / degrees_of_freedom))
    return OLSFit(coefficients=coefficients, residual_sd=residual_sd)


def fit_har_levels(values: np.ndarray) -> OLSFit:
    return fit_ols(*har_design(values))


def fit_shar_levels(
    rv: np.ndarray, rv_up: np.ndarray, rv_down: np.ndarray
) -> OLSFit:
    return fit_ols(*shar_design(rv, rv_up, rv_down))


def forecast_har_levels(
    history: np.ndarray, fit: OLSFit, horizon: int, *, floor: float = 0.0
) -> np.ndarray:
    work = list(np.asarray(history, dtype=float))
    forecasts = np.empty(int(horizon), dtype=float)
    for step in range(int(horizon)):
        row = np.asarray(
            [1.0, work[-1], np.mean(work[-5:]), np.mean(work[-22:])]
        )
        forecasts[step] = max(float(floor), float(row @ fit.coefficients))
        work.append(forecasts[step])
    return forecasts


def forecast_shar_levels(
    rv_history: np.ndarray,
    up_history: np.ndarray,
    down_history: np.ndarray,
    fit: OLSFit,
    future_up: np.ndarray,
    future_down: np.ndarray,
    *,
    floor: float = 0.0,
) -> np.ndarray:
    rv_work = list(np.asarray(rv_history, dtype=float))
    up_work = list(np.asarray(up_history, dtype=float))
    down_work = list(np.asarray(down_history, dtype=float))
    up_forecast = np.asarray(future_up, dtype=float)
    down_forecast = np.asarray(future_down, dtype=float)
    if up_forecast.shape != down_forecast.shape:
        raise ValueError("future semivariance forecasts must align")
    forecasts = np.empty(up_forecast.size, dtype=float)
    for step in range(up_forecast.size):
        row = np.asarray(
            [
                1.0,
                up_work[-1],
                down_work[-1],
                np.mean(rv_work[-5:]),
                np.mean(rv_work[-22:]),
            ]
        )
        forecasts[step] = max(float(floor), float(row @ fit.coefficients))
        rv_work.append(forecasts[step])
        up_work.append(float(up_forecast[step]))
        down_work.append(float(down_forecast[step]))
    return forecasts


def forecast_log_har(history: np.ndarray, fit: OLSFit, horizon: int) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    log_work = list(np.log(np.maximum(values, np.finfo(float).tiny)))
    smearing = float(np.exp(0.5 * fit.residual_sd**2))
    forecasts = np.empty(int(horizon), dtype=float)
    for step in range(int(horizon)):
        row = np.asarray(
            [
                1.0,
                log_work[-1],
                np.mean(log_work[-5:]),
                np.mean(log_work[-22:]),
            ]
        )
        mean_log = float(row @ fit.coefficients)
        forecasts[step] = np.exp(mean_log) * smearing
        log_work.append(mean_log)
    return forecasts


def _unpack_realized_garch(raw: np.ndarray) -> dict[str, float]:
    (
        log_v_bar,
        persistence_raw,
        share_raw,
        xi,
        log_phi,
        tau1,
        tau2,
        log_sigma_u,
        log_v0,
    ) = np.asarray(raw, dtype=float)
    persistence = float(expit(persistence_raw))
    realized_share = float(expit(share_raw))
    measurement_phi = float(np.exp(log_phi))
    sigma_u = float(np.exp(log_sigma_u))
    beta = persistence * (1.0 - realized_share)
    gamma = persistence * realized_share / measurement_phi
    omega = (1.0 - persistence) * float(log_v_bar) - gamma * float(xi)
    return {
        "log_v_bar": float(log_v_bar),
        "persistence": persistence,
        "realized_share": realized_share,
        "xi": float(xi),
        "measurement_phi": measurement_phi,
        "tau1": float(tau1),
        "tau2": float(tau2),
        "sigma_u": sigma_u,
        "log_v0": float(log_v0),
        "omega": omega,
        "beta": beta,
        "gamma": gamma,
    }


def realized_garch_log_variance_path(
    log_rv: np.ndarray,
    *,
    log_v0: float,
    omega: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    measurements = np.asarray(log_rv, dtype=float)
    if (
        measurements.ndim != 1
        or measurements.size == 0
        or np.any(~np.isfinite(measurements))
    ):
        raise ValueError("log realised variance must be a nonempty finite vector")
    path = np.empty(measurements.size, dtype=float)
    path[0] = float(log_v0)
    for index in range(1, measurements.size):
        path[index] = omega + beta * path[index - 1] + gamma * measurements[index - 1]
    return path


def gaussian_realized_garch_negative_log_likelihood(
    raw: np.ndarray, standardized_returns: np.ndarray, log_rv: np.ndarray
) -> float:
    parameters = _unpack_realized_garch(raw)
    log_h = realized_garch_log_variance_path(
        log_rv,
        log_v0=parameters["log_v0"],
        omega=parameters["omega"],
        beta=parameters["beta"],
        gamma=parameters["gamma"],
    )
    if np.any(~np.isfinite(log_h)) or np.any(np.abs(log_h) > 100.0):
        return 1e100
    returns = np.asarray(standardized_returns, dtype=float)
    innovations = returns * np.exp(-0.5 * log_h)
    location = (
        parameters["xi"]
        + parameters["measurement_phi"] * log_h
        + parameters["tau1"] * innovations
        + parameters["tau2"] * (innovations * innovations - 1.0)
    )
    measurement_residual = (
        np.asarray(log_rv, dtype=float) - location
    ) / parameters["sigma_u"]
    return_log_density = -0.5 * (
        np.log(2.0 * np.pi) + log_h + returns * returns * np.exp(-log_h)
    )
    measurement_log_density = (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(parameters["sigma_u"])
        - 0.5 * measurement_residual * measurement_residual
    )
    objective = -float(np.sum(return_log_density + measurement_log_density))
    return objective if np.isfinite(objective) else 1e100


def moment_denominator(fit: GaussianRealizedGarchFit) -> float:
    denominator = 1.0 - 2.0 * fit.gamma * fit.tau2
    if not np.isfinite(denominator) or denominator <= 1e-6:
        raise ValueError("Gaussian Realized GARCH variance mean is undefined")
    return float(denominator)


def fit_gaussian_realized_garch(
    standardized_returns: np.ndarray,
    log_rv: np.ndarray,
    *,
    maxiter: int = 2000,
) -> tuple[GaussianRealizedGarchFit, np.ndarray]:
    returns = np.asarray(standardized_returns, dtype=float)
    measurements = np.asarray(log_rv, dtype=float)
    if (
        returns.ndim != 1
        or measurements.shape != returns.shape
        or returns.size < 100
        or np.any(~np.isfinite(np.column_stack((returns, measurements))))
    ):
        raise ValueError("Gaussian Realized GARCH requires 100 aligned finite rows")
    mean_log_rv = float(np.mean(measurements))
    initial_sigma_u = max(0.05, float(np.std(measurements, ddof=1)) * 0.25)
    starts = [
        np.asarray(
            [
                mean_log_rv,
                np.log(0.95 / 0.05),
                np.log(share / (1.0 - share)),
                0.0,
                0.0,
                0.0,
                0.0,
                np.log(initial_sigma_u),
                mean_log_rv,
            ]
        )
        for share in (0.10, 0.30, 0.50)
    ]
    bounds = [
        (-20.0, 20.0),
        (-8.0, 8.0),
        (-8.0, 8.0),
        (-10.0, 10.0),
        (-3.0, 3.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-8.0, 3.0),
        (-20.0, 20.0),
    ]
    results = [
        minimize(
            gaussian_realized_garch_negative_log_likelihood,
            start,
            args=(returns, measurements),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6},
        )
        for start in starts
    ]
    successful = [
        (index, result)
        for index, result in enumerate(results)
        if result.success and np.isfinite(result.fun)
    ]
    if not successful:
        raise RealizedGarchFitError("all Gaussian Realized GARCH optimizer starts failed")
    selected_index, selected = min(successful, key=lambda item: float(item[1].fun))
    fit = GaussianRealizedGarchFit(
        **_unpack_realized_garch(np.asarray(selected.x, dtype=float)),
        negative_log_likelihood=float(selected.fun),
        optimizer_start=int(selected_index),
    )
    try:
        moment_denominator(fit)
    except ValueError as error:
        raise RealizedGarchFitError(str(error)) from error
    path = realized_garch_log_variance_path(
        measurements,
        log_v0=fit.log_v0,
        omega=fit.omega,
        beta=fit.beta,
        gamma=fit.gamma,
    )
    return fit, path


def gaussian_log_shock_mgf(
    fit: GaussianRealizedGarchFit, coefficient: float
) -> float:
    value = float(coefficient)
    denominator = 1.0 - 2.0 * value * fit.gamma * fit.tau2
    if not np.isfinite(denominator) or denominator <= 1e-6:
        raise ValueError("Gaussian shock moment is undefined")
    return float(
        -value * fit.gamma * fit.tau2
        - 0.5 * np.log(denominator)
        + (value * fit.gamma * fit.tau1) ** 2 / (2.0 * denominator)
        + 0.5 * (value * fit.gamma * fit.sigma_u) ** 2
    )


def analytic_gaussian_variance_path(
    fit: GaussianRealizedGarchFit, *, forecast_log_h: float, horizon: int
) -> np.ndarray:
    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    moment_denominator(fit)
    values = np.empty(int(horizon), dtype=float)
    intercept = fit.omega + fit.gamma * fit.xi
    for step in range(1, int(horizon) + 1):
        deterministic = fit.persistence ** (step - 1) * float(forecast_log_h)
        if step > 1:
            deterministic += intercept * sum(
                fit.persistence**power for power in range(step - 1)
            )
        log_mean = deterministic + sum(
            gaussian_log_shock_mgf(fit, fit.persistence**power)
            for power in range(step - 1)
        )
        values[step - 1] = np.exp(log_mean)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Gaussian Realized GARCH variance forecasts are invalid")
    return values


def simulate_gaussian_realized_garch_paths(
    fit: GaussianRealizedGarchFit,
    *,
    forecast_log_h: float,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if int(horizon) < 1 or int(paths) < 2:
        raise ValueError("horizon must be positive and paths must be at least two")
    log_h = np.full(int(paths), float(forecast_log_h))
    standardized_returns = np.empty((paths, horizon), dtype=float)
    standardized_variances = np.empty((paths, horizon), dtype=float)
    for step in range(int(horizon)):
        return_innovation = rng.normal(size=paths)
        measurement_innovation = rng.normal(scale=fit.sigma_u, size=paths)
        standardized_variances[:, step] = np.exp(log_h)
        standardized_returns[:, step] = (
            np.exp(0.5 * log_h) * return_innovation
        )
        log_measurement = (
            fit.xi
            + fit.measurement_phi * log_h
            + fit.tau1 * return_innovation
            + fit.tau2 * (return_innovation * return_innovation - 1.0)
            + measurement_innovation
        )
        log_h = fit.omega + fit.beta * log_h + fit.gamma * log_measurement
    return {
        "standardized_returns": standardized_returns,
        "standardized_variances": standardized_variances,
    }


def load_refit_dates(path: str | Path, *, market: str) -> pd.DatetimeIndex:
    frame = pd.read_csv(path)
    missing = {"market", "refit_date"} - set(frame)
    if missing:
        raise ValueError(f"refit schedule is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["market"].astype(str) == market, "refit_date"]
    dates = pd.DatetimeIndex(pd.to_datetime(selected, errors="raise")).normalize()
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("refit dates must be nonempty, unique, and ordered")
    return dates


def benchmark_daily_frame(frame: pd.DataFrame, *, market: str) -> pd.DataFrame:
    validate_daily_frame(frame)
    required = ["date", "market", "return_cc", "rv_oc", "rv_cc", "rv_up", "rv_down"]
    selected = frame.loc[frame["market"].astype(str) == market, required].copy()
    if selected.empty:
        raise ValueError(f"daily data contain no rows for {market}")
    selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
    for column in required[2:]:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    selected = selected.sort_values("date").reset_index(drop=True)
    if selected["date"].duplicated().any():
        raise ValueError(f"daily data contain duplicate dates for {market}")
    return selected


def _panel_row(
    *,
    market: str,
    model_id: str,
    origin: pd.Timestamp,
    horizon: int,
    mature_date: pd.Timestamp,
    variance_forecast: float,
    cumulative_return_mean: float | None = None,
    forecast_draw_count: int | None = None,
    return_distribution: str | None = None,
    variance_mean_method: str | None = None,
) -> dict[str, Any]:
    row = {column: None for column in FORECAST_PANEL_COLUMNS}
    row.update(
        {
            "market": market,
            "model_id": model_id,
            "origin_date": str(origin.date()),
            "horizon": int(horizon),
            "mature_date": str(mature_date.date()),
            "forecast_kind": (
                "probabilistic"
                if model_id == "Realized-GARCH-Gaussian-QMLE"
                else "variance_point_only"
            ),
            "variance_forecast": float(variance_forecast),
            "cumulative_return_mean": cumulative_return_mean,
            "forecast_draw_count": forecast_draw_count,
            "return_distribution": return_distribution,
            "variance_mean_method": variance_mean_method,
        }
    )
    return row


def _classical_panel(
    daily: pd.DataFrame,
    *,
    market: str,
    model_id: str,
    refit_dates: pd.DatetimeIndex,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    horizons: Sequence[int],
) -> pd.DataFrame:
    max_horizon = max(horizons)
    positions = [
        index
        for index, value in enumerate(daily["date"])
        if forecast_start <= value <= forecast_end
        and index + max_horizon < len(daily)
    ]
    if not positions or daily.loc[positions[0], "date"] not in set(refit_dates):
        raise ValueError("the first forecast origin must be a documented refit date")
    fit: OLSFit | None = None
    component_fits: tuple[OLSFit, OLSFit] | None = None
    rows: list[dict[str, Any]] = []
    refit_set = set(refit_dates)
    for position in positions:
        origin = pd.Timestamp(daily.loc[position, "date"])
        history = daily.iloc[: position + 1]
        if fit is None or origin in refit_set:
            if model_id == "HAR-RV":
                fit = fit_har_levels(history["rv_cc"].to_numpy(float))
            elif model_id == "SHAR-RV":
                fit = fit_shar_levels(
                    history["rv_cc"].to_numpy(float),
                    history["rv_up"].to_numpy(float),
                    history["rv_down"].to_numpy(float),
                )
                component_fits = (
                    fit_har_levels(history["rv_up"].to_numpy(float)),
                    fit_har_levels(history["rv_down"].to_numpy(float)),
                )
            elif model_id == "log-HAR-RV":
                log_rv = np.log(
                    np.maximum(history["rv_cc"].to_numpy(float), np.finfo(float).tiny)
                )
                fit = fit_ols(*har_design(log_rv))
            else:
                raise ValueError(f"unsupported classical benchmark: {model_id}")
        assert fit is not None
        for horizon in horizons:
            if model_id == "HAR-RV":
                daily_forecast = forecast_har_levels(
                    history["rv_cc"].to_numpy(float), fit, int(horizon)
                )
            elif model_id == "SHAR-RV":
                assert component_fits is not None
                future_up = forecast_har_levels(
                    history["rv_up"].to_numpy(float), component_fits[0], int(horizon)
                )
                future_down = forecast_har_levels(
                    history["rv_down"].to_numpy(float), component_fits[1], int(horizon)
                )
                daily_forecast = forecast_shar_levels(
                    history["rv_cc"].to_numpy(float),
                    history["rv_up"].to_numpy(float),
                    history["rv_down"].to_numpy(float),
                    fit,
                    future_up,
                    future_down,
                )
            else:
                daily_forecast = forecast_log_har(
                    history["rv_cc"].to_numpy(float), fit, int(horizon)
                )
            rows.append(
                _panel_row(
                    market=market,
                    model_id=model_id,
                    origin=origin,
                    horizon=int(horizon),
                    mature_date=pd.Timestamp(daily.loc[position + int(horizon), "date"]),
                    variance_forecast=float(np.sum(daily_forecast)),
                )
            )
    return pd.DataFrame(rows, columns=FORECAST_PANEL_COLUMNS)


def _realized_garch_panel(
    daily: pd.DataFrame,
    *,
    market: str,
    refit_dates: pd.DatetimeIndex,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    horizons: Sequence[int],
    optimizer_maxiter: int,
    forecast_draw_count: int,
    base_seed: int,
    components_dir: Path | None,
) -> pd.DataFrame:
    max_horizon = max(horizons)
    positions = [
        index
        for index, value in enumerate(daily["date"])
        if forecast_start <= value <= forecast_end
        and index + max_horizon < len(daily)
    ]
    if not positions or daily.loc[positions[0], "date"] not in set(refit_dates):
        raise ValueError("the first forecast origin must be a documented refit date")
    refit_set = set(refit_dates)
    fit: GaussianRealizedGarchFit | None = None
    scaler: ReturnScaler | None = None
    current_log_h: float | None = None
    rows: list[dict[str, Any]] = []
    for position in positions:
        origin = pd.Timestamp(daily.loc[position, "date"])
        history = daily.iloc[: position + 1]
        if fit is None or origin in refit_set:
            candidate_scaler = ReturnScaler.fit(history["return_cc"].to_numpy(float))
            standardized = candidate_scaler.standardize(
                history["return_cc"].to_numpy(float)
            )
            log_rv = np.log(
                np.maximum(
                    history["rv_oc"].to_numpy(float) / candidate_scaler.scale**2,
                    np.finfo(float).tiny,
                )
            )
            candidate_fit, fitted_path = fit_gaussian_realized_garch(
                standardized, log_rv, maxiter=optimizer_maxiter
            )
            fit = candidate_fit
            scaler = candidate_scaler
            current_log_h = float(fitted_path[-1])
        if fit is None or scaler is None or current_log_h is None:
            continue
        observed_log_rv = float(
            np.log(
                max(
                    daily.loc[position, "rv_oc"] / scaler.scale**2,
                    np.finfo(float).tiny,
                )
            )
        )
        forecast_log_h = float(
            fit.omega + fit.beta * current_log_h + fit.gamma * observed_log_rv
        )
        analytic_path = analytic_gaussian_variance_path(
            fit, forecast_log_h=forecast_log_h, horizon=max_horizon
        )
        for horizon in horizons:
            row = _panel_row(
                market=market,
                model_id="Realized-GARCH-Gaussian-QMLE",
                origin=origin,
                horizon=int(horizon),
                mature_date=pd.Timestamp(daily.loc[position + int(horizon), "date"]),
                variance_forecast=float(
                    scaler.scale**2 * np.sum(analytic_path[: int(horizon)])
                ),
                cumulative_return_mean=float(int(horizon) * scaler.mean),
                forecast_draw_count=int(forecast_draw_count),
                return_distribution="gaussian",
                variance_mean_method="analytic_gaussian_realized_garch",
            )
            if components_dir is not None:
                seed = seed_from_context(
                    base_seed,
                    market,
                    str(origin.date()),
                    int(horizon),
                    "Realized-GARCH-Gaussian-QMLE",
                )
                paths = simulate_gaussian_realized_garch_paths(
                    fit,
                    forecast_log_h=forecast_log_h,
                    horizon=int(horizon),
                    paths=int(forecast_draw_count),
                    rng=np.random.default_rng(seed),
                )
                raw_variances = scaler.scale**2 * paths["standardized_variances"]
                raw_returns = (
                    scaler.mean + scaler.scale * paths["standardized_returns"]
                )
                component_arrays = {
                    "raw_returns": raw_returns,
                    "raw_variances": raw_variances,
                    "raw_return_locations": np.full_like(raw_returns, scaler.mean),
                    "raw_conditional_sds": scaler.scale
                    * np.sqrt(paths["standardized_variances"]),
                }
                expected_shape = (int(forecast_draw_count), int(horizon))
                if any(
                    component_arrays[name].shape != expected_shape
                    or np.any(~np.isfinite(component_arrays[name]))
                    for name in GAUSSIAN_COMPONENT_NAMES
                ):
                    raise FloatingPointError(
                        "Gaussian predictive components are invalid"
                    )
                if (
                    np.any(component_arrays["raw_variances"] <= 0)
                    or np.any(component_arrays["raw_conditional_sds"] <= 0)
                ):
                    raise FloatingPointError(
                        "Gaussian predictive variances and scales must be positive"
                    )
                component_name = (
                    f"{market}_Realized-GARCH-Gaussian-QMLE_"
                    f"{origin:%Y-%m-%d}_h{int(horizon):02d}.npz"
                )
                np.savez_compressed(
                    components_dir / component_name, **component_arrays
                )
                row["predictive_file"] = component_name
            rows.append(row)
        current_log_h = forecast_log_h
    columns = list(FORECAST_PANEL_COLUMNS)
    if components_dir is not None:
        columns.append("predictive_file")
    return pd.DataFrame(rows, columns=columns)


def generate_benchmark_panel(
    *,
    frame: pd.DataFrame,
    market: str,
    model_ids: Iterable[str],
    refit_dates: pd.DatetimeIndex,
    forecast_start: str | pd.Timestamp,
    forecast_end: str | pd.Timestamp,
    horizons: Sequence[int] = HORIZONS,
    optimizer_maxiter: int = 2000,
    forecast_draw_count: int = 4096,
    base_seed: int = 20260730,
    components_dir: str | Path | None = None,
) -> pd.DataFrame:
    selected_models = tuple(model_ids)
    unknown = set(selected_models) - set(BENCHMARK_IDS)
    if not selected_models or unknown:
        raise ValueError(f"unsupported benchmark models: {sorted(unknown)}")
    selected_horizons = tuple(int(value) for value in horizons)
    if selected_horizons != HORIZONS:
        raise ValueError(f"benchmark horizons must be {HORIZONS}")
    daily = benchmark_daily_frame(frame, market=market)
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    if start > end:
        raise ValueError("forecast_start must not be later than forecast_end")
    component_root = None if components_dir is None else Path(components_dir)
    if component_root is not None:
        component_root.mkdir(parents=True, exist_ok=True)
    panels = []
    for model_id in selected_models:
        if model_id == "Realized-GARCH-Gaussian-QMLE":
            panel = _realized_garch_panel(
                daily,
                market=market,
                refit_dates=refit_dates,
                forecast_start=start,
                forecast_end=end,
                horizons=selected_horizons,
                optimizer_maxiter=optimizer_maxiter,
                forecast_draw_count=forecast_draw_count,
                base_seed=base_seed,
                components_dir=component_root,
            )
        else:
            panel = _classical_panel(
                daily,
                market=market,
                model_id=model_id,
                refit_dates=refit_dates,
                forecast_start=start,
                forecast_end=end,
                horizons=selected_horizons,
            )
        panels.append(panel)
    rows = [record for panel in panels for record in panel.to_dict("records")]
    columns = list(FORECAST_PANEL_COLUMNS)
    if component_root is not None:
        columns.append("predictive_file")
    return pd.DataFrame(rows, columns=columns)


def generate_from_design(
    *,
    data_path: str | Path,
    design_path: str | Path,
    market: str,
    output_path: str | Path,
    model_ids: Iterable[str] | None = None,
    origin_end: str | None = None,
    components_dir: str | Path | None = None,
) -> pd.DataFrame:
    design_file = Path(design_path)
    with design_file.open("r", encoding="utf-8") as handle:
        design = yaml.safe_load(handle)
    if not isinstance(design, dict):
        raise ValueError("benchmark design must be a mapping")
    if market not in design["markets"]:
        raise ValueError(f"market is not in the benchmark design: {market}")
    schedule_path = design_file.parent / design["refits"]["analysis_dates"]
    frame = read_daily_frame(data_path)
    sample = design["sample"]
    end = sample["forecast_end"] if origin_end is None else origin_end
    realized_garch = design["models"]["Realized-GARCH-Gaussian-QMLE"]
    if tuple(design["forecast"]["gaussian_predictive_components"]) != (
        GAUSSIAN_COMPONENT_NAMES
    ):
        raise ValueError("Gaussian predictive-component names differ from the design")
    panel = generate_benchmark_panel(
        frame=frame,
        market=market,
        model_ids=BENCHMARK_IDS if model_ids is None else model_ids,
        refit_dates=load_refit_dates(schedule_path, market=market),
        forecast_start=sample["forecast_start"],
        forecast_end=end,
        horizons=tuple(design["horizons"]),
        optimizer_maxiter=int(realized_garch["optimizer_maxiter"]),
        forecast_draw_count=int(design["forecast"]["paths"]),
        base_seed=int(design["seed"]),
        components_dir=components_dir,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)
    return panel
