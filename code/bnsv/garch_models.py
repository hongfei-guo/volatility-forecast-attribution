"""Expanding-window Student-t GARCH forecast generators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
import yaml

from .benchmarks import (
    FORECAST_PANEL_COLUMNS,
    benchmark_daily_frame,
    load_refit_dates,
    realized_garch_log_variance_path,
)
from .data_contract import read_daily_frame
from .forecast_features import ReturnScaler
from .forecast_paths import seed_from_context


GARCH_T_ID = "GARCH-t"
REALIZED_GARCH_T_ID = "Realized-GARCH-t-MLE"
MODEL_IDS = (GARCH_T_ID, REALIZED_GARCH_T_ID)
RETURN_DISTRIBUTION = "variance_standardized_student_t"
GARCH_VARIANCE_MEAN_METHOD = "analytic_garch_11_t"
REALIZED_GARCH_VARIANCE_MEAN_METHOD = "analytic_one_day_conditional_variance"
MIN_NEAR_BEST_STARTS = 2
NEAR_BEST_ABSOLUTE_TOLERANCE = 1e-4
NEAR_BEST_RELATIVE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class GarchTFit:
    variance_bar: float
    persistence: float
    arch_share: float
    omega: float
    alpha: float
    beta: float
    nu: float
    negative_log_likelihood: float
    optimizer_start: int


@dataclass(frozen=True)
class StudentTRealizedGarchFit:
    log_v_bar: float
    persistence: float
    realized_share: float
    xi: float
    measurement_phi: float
    tau1: float
    tau2: float
    sigma_u: float
    nu: float
    log_v0: float
    omega: float
    beta: float
    gamma: float
    negative_log_likelihood: float
    optimizer_start: int


class GarchTFitError(RuntimeError):
    """Raised when the fixed-start GARCH-t fit is not accepted."""


class StudentTRealizedGarchFitError(RuntimeError):
    """Raised when all fixed Realized GARCH-t starts fail."""


def _unpack_garch_t(raw: np.ndarray) -> dict[str, float]:
    log_variance_bar, persistence_raw, share_raw, log_nu_minus_two = np.asarray(
        raw, dtype=float
    )
    variance_bar = float(np.exp(log_variance_bar))
    persistence = float(expit(persistence_raw))
    arch_share = float(expit(share_raw))
    alpha = persistence * arch_share
    beta = persistence * (1.0 - arch_share)
    return {
        "variance_bar": variance_bar,
        "persistence": persistence,
        "arch_share": arch_share,
        "omega": (1.0 - persistence) * variance_bar,
        "alpha": alpha,
        "beta": beta,
        "nu": float(2.0 + np.exp(log_nu_minus_two)),
    }


def garch_variance_path(
    standardized_returns: np.ndarray,
    *,
    variance_bar: float,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    values = np.asarray(standardized_returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("finite one-dimensional returns are required")
    path = np.empty(values.size, dtype=float)
    path[0] = float(variance_bar)
    for index in range(1, values.size):
        path[index] = omega + alpha * values[index - 1] ** 2 + beta * path[index - 1]
    if np.any(~np.isfinite(path)) or np.any(path <= 0):
        raise ValueError("GARCH variance path must be finite and positive")
    return path


def standardized_t_log_density(
    values: np.ndarray,
    *,
    conditional_variances: np.ndarray,
    nu: float,
) -> np.ndarray:
    observations = np.asarray(values, dtype=float)
    variances = np.asarray(conditional_variances, dtype=float)
    if observations.shape != variances.shape:
        raise ValueError("returns and variances must align")
    if nu <= 2 or np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        raise ValueError("Student-t inputs are invalid")
    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(np.pi * (nu - 2.0))
        - 0.5 * np.log(variances)
        - 0.5
        * (nu + 1.0)
        * np.log1p(observations**2 / ((nu - 2.0) * variances))
    )


def garch_t_negative_log_likelihood(
    raw: np.ndarray, standardized_returns: np.ndarray
) -> float:
    parameters = _unpack_garch_t(raw)
    try:
        variances = garch_variance_path(
            standardized_returns,
            variance_bar=parameters["variance_bar"],
            omega=parameters["omega"],
            alpha=parameters["alpha"],
            beta=parameters["beta"],
        )
        log_density = standardized_t_log_density(
            standardized_returns,
            conditional_variances=variances,
            nu=parameters["nu"],
        )
    except ValueError:
        return 1e100
    objective = -float(np.sum(log_density))
    return objective if np.isfinite(objective) else 1e100


def fit_garch_t(
    standardized_returns: np.ndarray, *, maxiter: int = 2000
) -> tuple[GarchTFit, np.ndarray]:
    values = np.asarray(standardized_returns, dtype=float)
    if values.ndim != 1 or values.size < 100 or np.any(~np.isfinite(values)):
        raise ValueError("GARCH-t requires at least 100 finite standardized returns")
    variance_anchor = max(float(np.mean(values**2)), 1e-6)
    starts = [
        np.asarray(
            [
                np.log(variance_anchor),
                np.log(persistence / (1.0 - persistence)),
                np.log(share / (1.0 - share)),
                np.log(6.0),
            ]
        )
        for persistence, share in (
            (0.90, 0.05),
            (0.90, 0.15),
            (0.98, 0.05),
            (0.98, 0.15),
        )
    ]
    bounds = [
        (np.log(1e-4), np.log(100.0)),
        (-8.0, 8.0),
        (-8.0, 8.0),
        (-6.0, 8.0),
    ]
    results = [
        minimize(
            garch_t_negative_log_likelihood,
            start,
            args=(values,),
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
        raise GarchTFitError("all fixed GARCH-t starts failed")
    best_objective = min(float(result.fun) for _, result in successful)
    tolerance = max(
        NEAR_BEST_ABSOLUTE_TOLERANCE,
        abs(best_objective) * NEAR_BEST_RELATIVE_TOLERANCE,
    )
    near_best = sum(
        abs(float(result.fun) - best_objective) <= tolerance
        for _, result in successful
    )
    if near_best < MIN_NEAR_BEST_STARTS:
        raise GarchTFitError("fewer than two fixed starts reached the same optimum")
    selected_start, selected = min(successful, key=lambda item: float(item[1].fun))
    fit = GarchTFit(
        **_unpack_garch_t(np.asarray(selected.x, dtype=float)),
        negative_log_likelihood=float(selected.fun),
        optimizer_start=int(selected_start),
    )
    path = garch_variance_path(
        values,
        variance_bar=fit.variance_bar,
        omega=fit.omega,
        alpha=fit.alpha,
        beta=fit.beta,
    )
    return fit, path


def garch_variance_mean_path(
    fit: GarchTFit, *, forecast_variance: float, horizon: int
) -> np.ndarray:
    if horizon < 1 or not np.isfinite(forecast_variance) or forecast_variance <= 0:
        raise ValueError("a positive finite one-step variance is required")
    values = np.empty(int(horizon), dtype=float)
    values[0] = float(forecast_variance)
    for step in range(1, int(horizon)):
        values[step] = fit.omega + fit.persistence * values[step - 1]
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("GARCH-t variance means must be finite and positive")
    return values


def simulate_garch_t_paths(
    fit: GarchTFit,
    *,
    forecast_variance: float,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1 or paths < 2:
        raise ValueError("horizon must be positive and paths must be at least two")
    variance = np.full(int(paths), float(forecast_variance), dtype=float)
    returns = np.empty((int(paths), int(horizon)), dtype=float)
    variances = np.empty_like(returns)
    scale = np.sqrt((fit.nu - 2.0) / fit.nu)
    for step in range(int(horizon)):
        innovation = rng.standard_t(fit.nu, size=int(paths)) * scale
        variances[:, step] = variance
        returns[:, step] = np.sqrt(variance) * innovation
        variance = fit.omega + fit.alpha * returns[:, step] ** 2 + fit.beta * variance
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("GARCH-t predictive variance became invalid")
    return returns, variances


def _unpack_realized_garch_t(raw: np.ndarray) -> dict[str, float]:
    (
        log_v_bar,
        persistence_raw,
        share_raw,
        xi,
        log_phi,
        tau1,
        tau2,
        log_sigma_u,
        log_nu_minus_two,
        log_v0,
    ) = np.asarray(raw, dtype=float)
    persistence = float(expit(persistence_raw))
    realized_share = float(expit(share_raw))
    measurement_phi = float(np.exp(log_phi))
    gamma = persistence * realized_share / measurement_phi
    beta = persistence * (1.0 - realized_share)
    return {
        "log_v_bar": float(log_v_bar),
        "persistence": persistence,
        "realized_share": realized_share,
        "xi": float(xi),
        "measurement_phi": measurement_phi,
        "tau1": float(tau1),
        "tau2": float(tau2),
        "sigma_u": float(np.exp(log_sigma_u)),
        "nu": float(2.0 + np.exp(log_nu_minus_two)),
        "log_v0": float(log_v0),
        "omega": (1.0 - persistence) * float(log_v_bar) - gamma * float(xi),
        "beta": beta,
        "gamma": gamma,
    }


def _realized_garch_t_log_density(
    values: np.ndarray,
    *,
    nu: float,
    log_variance: np.ndarray,
) -> np.ndarray:
    observations = np.asarray(values, dtype=float)
    log_scale = (
        0.5 * np.asarray(log_variance, dtype=float)
        + 0.5 * (np.log(nu - 2.0) - np.log(nu))
    )
    scaled = observations * np.exp(-log_scale)
    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - log_scale
        - 0.5 * (nu + 1.0) * np.log1p((scaled * scaled) / nu)
    )


def student_t_realized_garch_negative_log_likelihood(
    raw: np.ndarray,
    standardized_returns: np.ndarray,
    log_rv: np.ndarray,
) -> float:
    parameters = _unpack_realized_garch_t(raw)
    returns = np.asarray(standardized_returns, dtype=float)
    measure = np.asarray(log_rv, dtype=float)
    log_h = realized_garch_log_variance_path(
        measure,
        log_v0=parameters["log_v0"],
        omega=parameters["omega"],
        beta=parameters["beta"],
        gamma=parameters["gamma"],
    )
    if np.any(~np.isfinite(log_h)) or np.any(np.abs(log_h) > 100.0):
        return 1e100
    innovations = returns * np.exp(-0.5 * log_h)
    location = (
        parameters["xi"]
        + parameters["measurement_phi"] * log_h
        + parameters["tau1"] * innovations
        + parameters["tau2"] * (innovations**2 - 1.0)
    )
    residual = (measure - location) / parameters["sigma_u"]
    return_density = _realized_garch_t_log_density(
        returns,
        nu=parameters["nu"],
        log_variance=log_h,
    )
    measurement_density = (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(parameters["sigma_u"])
        - 0.5 * residual**2
    )
    objective = -float(np.sum(return_density + measurement_density))
    return objective if np.isfinite(objective) else 1e100


def fit_student_t_realized_garch(
    standardized_returns: np.ndarray,
    log_rv: np.ndarray,
    *,
    maxiter: int = 2000,
) -> tuple[StudentTRealizedGarchFit, np.ndarray]:
    returns = np.asarray(standardized_returns, dtype=float)
    measure = np.asarray(log_rv, dtype=float)
    if returns.ndim != 1 or measure.shape != returns.shape or returns.size < 100:
        raise ValueError("Realized GARCH-t inputs must align and contain 100 rows")
    if np.any(~np.isfinite(returns)) or np.any(~np.isfinite(measure)):
        raise ValueError("Realized GARCH-t inputs must be finite")
    mean_log_rv = float(np.mean(measure))
    sigma_u = max(0.05, float(np.std(measure, ddof=1)) * 0.25)
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
                np.log(sigma_u),
                np.log(6.0),
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
        (-3.0, 5.0),
        (-20.0, 20.0),
    ]
    results = [
        minimize(
            student_t_realized_garch_negative_log_likelihood,
            start,
            args=(returns, measure),
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
        raise StudentTRealizedGarchFitError("all fixed Realized GARCH-t starts failed")
    selected_start, selected = min(successful, key=lambda item: float(item[1].fun))
    fit = StudentTRealizedGarchFit(
        **_unpack_realized_garch_t(np.asarray(selected.x, dtype=float)),
        negative_log_likelihood=float(selected.fun),
        optimizer_start=int(selected_start),
    )
    path = realized_garch_log_variance_path(
        measure,
        log_v0=fit.log_v0,
        omega=fit.omega,
        beta=fit.beta,
        gamma=fit.gamma,
    )
    if np.any(~np.isfinite(path)):
        raise StudentTRealizedGarchFitError("fitted log-variance path is non-finite")
    return fit, path


def forecast_realized_garch_log_variance(
    fit: StudentTRealizedGarchFit,
    *,
    origin_log_h: float,
    observed_log_rv: float,
) -> float:
    value = fit.omega + fit.beta * origin_log_h + fit.gamma * observed_log_rv
    if not np.isfinite(value) or abs(value) > 100.0:
        raise ValueError("one-day forecast log variance is invalid")
    return float(value)


def _write_components(
    *,
    output_root: Path,
    relative_path: Path,
    raw_returns: np.ndarray,
    raw_variances: np.ndarray,
    raw_locations: np.ndarray,
    raw_sds: np.ndarray,
    nu: float,
) -> str:
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        raw_returns=raw_returns,
        raw_variances=raw_variances,
        raw_return_locations=raw_locations,
        raw_conditional_sds=raw_sds,
        degrees_of_freedom=np.full(raw_returns.shape[0], float(nu)),
    )
    return relative_path.as_posix()


def _panel_row(
    *,
    market: str,
    model_id: str,
    origin: pd.Timestamp,
    horizon: int,
    mature_date: pd.Timestamp,
    variance_forecast: float,
    cumulative_return_mean: float,
    path_count: int,
    variance_mean_method: str,
    predictive_file: str | None,
) -> dict[str, Any]:
    row = {column: None for column in FORECAST_PANEL_COLUMNS}
    row.update(
        {
            "market": market,
            "model_id": model_id,
            "origin_date": str(origin.date()),
            "horizon": int(horizon),
            "mature_date": str(mature_date.date()),
            "forecast_kind": "probabilistic",
            "variance_forecast": float(variance_forecast),
            "cumulative_return_mean": float(cumulative_return_mean),
            "forecast_draw_count": int(path_count),
            "return_distribution": RETURN_DISTRIBUTION,
            "variance_mean_method": variance_mean_method,
        }
    )
    if predictive_file is not None:
        row["predictive_file"] = predictive_file
    return row


GarchFitFunction = Callable[..., tuple[GarchTFit, np.ndarray]]
RealizedGarchFitFunction = Callable[
    ..., tuple[StudentTRealizedGarchFit, np.ndarray]
]


def generate_garch_panel(
    *,
    frame: pd.DataFrame,
    market: str,
    model_id: str,
    refit_dates: pd.DatetimeIndex,
    forecast_start: str | pd.Timestamp,
    forecast_end: str | pd.Timestamp,
    horizons: Sequence[int],
    origin_alignment_horizon: int,
    optimizer_maxiter: int,
    path_count: int,
    base_seed: int,
    components_dir: str | Path | None = None,
    garch_fit_function: GarchFitFunction = fit_garch_t,
    realized_garch_fit_function: RealizedGarchFitFunction = fit_student_t_realized_garch,
) -> pd.DataFrame:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model: {model_id}")
    selected_horizons = tuple(int(value) for value in horizons)
    expected = (1, 5, 10) if model_id == GARCH_T_ID else (1,)
    if selected_horizons != expected:
        raise ValueError(f"{model_id} horizons must be {expected}")
    if origin_alignment_horizon < max(selected_horizons):
        raise ValueError("origin_alignment_horizon is shorter than an output horizon")
    if optimizer_maxiter <= 0 or path_count < 2 or base_seed <= 0:
        raise ValueError("optimizer iterations, paths, and seed must be positive")
    daily = benchmark_daily_frame(frame, market=market)
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    if start > end:
        raise ValueError("forecast_start must not be later than forecast_end")
    positions = [
        index
        for index, value in enumerate(daily["date"])
        if start <= value <= end and index + int(origin_alignment_horizon) < len(daily)
    ]
    refits = pd.DatetimeIndex(refit_dates).normalize()
    if (
        not positions
        or refits.empty
        or refits.has_duplicates
        or not refits.is_monotonic_increasing
        or daily.loc[positions[0], "date"] != refits[0]
    ):
        raise ValueError("the documented refit schedule does not align with origins")
    origin_dates = set(pd.DatetimeIndex(daily.loc[positions, "date"]).normalize())
    last_origin = pd.Timestamp(daily.loc[positions[-1], "date"])
    relevant_refits = {value for value in refits if value <= last_origin}
    if not relevant_refits.issubset(origin_dates):
        raise ValueError("refit dates must be forecast origins")
    refit_set = set(refits)
    component_root = None if components_dir is None else Path(components_dir)
    if component_root is not None:
        component_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scaler: ReturnScaler | None = None
    current_state: float | None = None
    garch_fit: GarchTFit | None = None
    realized_fit: StudentTRealizedGarchFit | None = None

    for position in positions:
        origin = pd.Timestamp(daily.loc[position, "date"])
        history = daily.iloc[: position + 1]
        if origin in refit_set:
            candidate_scaler = ReturnScaler.fit(history["return_cc"].to_numpy(float))
            standardized = candidate_scaler.standardize(
                history["return_cc"].to_numpy(float)
            )
            try:
                if model_id == GARCH_T_ID:
                    candidate_fit, fitted_path = garch_fit_function(
                        standardized, maxiter=int(optimizer_maxiter)
                    )
                    garch_fit = candidate_fit
                    realized_fit = None
                else:
                    log_rv = np.log(
                        np.maximum(
                            history["rv_oc"].to_numpy(float)
                            / candidate_scaler.scale**2,
                            np.finfo(float).tiny,
                        )
                    )
                    candidate_fit, fitted_path = realized_garch_fit_function(
                        standardized, log_rv, maxiter=int(optimizer_maxiter)
                    )
                    realized_fit = candidate_fit
                    garch_fit = None
            except (GarchTFitError, StudentTRealizedGarchFitError):
                scaler = None
                current_state = None
                garch_fit = None
                realized_fit = None
                continue
            scaler = candidate_scaler
            current_state = float(fitted_path[-1])

        if scaler is None or current_state is None:
            continue

        if model_id == GARCH_T_ID:
            assert garch_fit is not None
            observed = float(scaler.standardize(daily.loc[position, "return_cc"]))
            forecast_variance = (
                garch_fit.omega
                + garch_fit.alpha * observed**2
                + garch_fit.beta * current_state
            )
            analytic_path = garch_variance_mean_path(
                garch_fit,
                forecast_variance=forecast_variance,
                horizon=max(selected_horizons),
            )
            for horizon in selected_horizons:
                predictive_file = None
                if component_root is not None:
                    seed = seed_from_context(
                        base_seed, market, str(origin.date()), horizon, model_id
                    )
                    simulated_returns, simulated_variances = simulate_garch_t_paths(
                        garch_fit,
                        forecast_variance=forecast_variance,
                        horizon=horizon,
                        paths=path_count,
                        rng=np.random.default_rng(seed),
                    )
                    raw_returns = scaler.mean + scaler.scale * simulated_returns
                    raw_variances = scaler.scale**2 * simulated_variances
                    raw_locations = np.full_like(raw_returns, scaler.mean)
                    raw_sds = scaler.scale * np.sqrt(simulated_variances)
                    predictive_file = _write_components(
                        output_root=component_root,
                        relative_path=Path(market)
                        / GARCH_T_ID
                        / f"{origin:%Y%m%d}_h{horizon}.npz",
                        raw_returns=raw_returns,
                        raw_variances=raw_variances,
                        raw_locations=raw_locations,
                        raw_sds=raw_sds,
                        nu=garch_fit.nu,
                    )
                rows.append(
                    _panel_row(
                        market=market,
                        model_id=model_id,
                        origin=origin,
                        horizon=horizon,
                        mature_date=pd.Timestamp(daily.loc[position + horizon, "date"]),
                        variance_forecast=float(
                            scaler.scale**2 * np.sum(analytic_path[:horizon])
                        ),
                        cumulative_return_mean=float(horizon * scaler.mean),
                        path_count=path_count,
                        variance_mean_method=GARCH_VARIANCE_MEAN_METHOD,
                        predictive_file=predictive_file,
                    )
                )
            current_state = float(forecast_variance)
            continue

        assert realized_fit is not None
        observed_log_rv = float(
            np.log(
                max(
                    daily.loc[position, "rv_oc"] / scaler.scale**2,
                    np.finfo(float).tiny,
                )
            )
        )
        forecast_log_h = forecast_realized_garch_log_variance(
            realized_fit,
            origin_log_h=current_state,
            observed_log_rv=observed_log_rv,
        )
        variance = float(np.exp(forecast_log_h))
        predictive_file = None
        if component_root is not None:
            seed = seed_from_context(
                base_seed, market, str(origin.date()), 1, model_id
            )
            t_scale = np.sqrt((realized_fit.nu - 2.0) / realized_fit.nu)
            innovations = (
                np.random.default_rng(seed).standard_t(
                    realized_fit.nu, size=(path_count, 1)
                )
                * t_scale
            )
            simulated_variances = np.full((path_count, 1), variance)
            simulated_returns = np.sqrt(simulated_variances) * innovations
            raw_returns = scaler.mean + scaler.scale * simulated_returns
            raw_variances = scaler.scale**2 * simulated_variances
            raw_locations = np.full_like(raw_returns, scaler.mean)
            raw_sds = scaler.scale * np.sqrt(simulated_variances)
            predictive_file = _write_components(
                output_root=component_root,
                relative_path=Path(market)
                / REALIZED_GARCH_T_ID
                / f"{origin:%Y%m%d}_h1.npz",
                raw_returns=raw_returns,
                raw_variances=raw_variances,
                raw_locations=raw_locations,
                raw_sds=raw_sds,
                nu=realized_fit.nu,
            )
        rows.append(
            _panel_row(
                market=market,
                model_id=model_id,
                origin=origin,
                horizon=1,
                mature_date=pd.Timestamp(daily.loc[position + 1, "date"]),
                variance_forecast=float(scaler.scale**2 * variance),
                cumulative_return_mean=float(scaler.mean),
                path_count=path_count,
                variance_mean_method=REALIZED_GARCH_VARIANCE_MEAN_METHOD,
                predictive_file=predictive_file,
            )
        )
        current_state = forecast_log_h

    columns = list(FORECAST_PANEL_COLUMNS)
    if component_root is not None:
        columns.append("predictive_file")
    return pd.DataFrame(rows, columns=columns)


def generate_garch_from_design(
    *,
    data_path: str | Path,
    design_path: str | Path,
    market: str,
    output_path: str | Path,
    components_dir: str | Path | None = None,
    origin_end: str | None = None,
    expected_model_id: str | None = None,
) -> pd.DataFrame:
    design_file = Path(design_path)
    design = yaml.safe_load(design_file.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("GARCH design must be a mapping")
    model = design.get("model", design)
    model_id = str(model.get("model_id"))
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported design model: {model_id}")
    if expected_model_id is not None and model_id != expected_model_id:
        raise ValueError("selected model and design model differ")
    if model_id == GARCH_T_ID:
        expected_fields = {
            "estimator": "fixed_four_start_mle",
            "return_distribution": RETURN_DISTRIBUTION,
            "parameter_uncertainty_propagated": False,
            "optimizer": "L-BFGS-B",
            "optimizer_maxiter": 2000,
            "variance_mean_method": GARCH_VARIANCE_MEAN_METHOD,
            "refit_failure_rule": "forecasts_unavailable_until_next_scheduled_refit",
        }
        acceptance = model.get("optimizer_acceptance")
        expected_acceptance = {
            "minimum_near_optimal_starts": MIN_NEAR_BEST_STARTS,
            "absolute_objective_tolerance": NEAR_BEST_ABSOLUTE_TOLERANCE,
            "relative_objective_tolerance": NEAR_BEST_RELATIVE_TOLERANCE,
        }
        if acceptance != expected_acceptance:
            raise ValueError("GARCH-t optimizer acceptance differs from the design")
    else:
        expected_fields = {
            "estimator": "joint_mle_fixed_three_starts",
            "measurement_variable": "rv_oc",
            "return_innovation": RETURN_DISTRIBUTION,
            "measurement_innovation": "gaussian",
            "parameter_uncertainty_propagated": False,
            "optimizer": "L-BFGS-B",
            "optimizer_maxiter": 2000,
            "optimizer_selection_rule": (
                "lowest_finite_objective_among_successful_fixed_starts"
            ),
            "refit_failure_rule": "forecasts_unavailable_until_next_scheduled_refit",
        }
    if any(model.get(key) != value for key, value in expected_fields.items()):
        raise ValueError(f"{model_id} design differs from the implemented estimator")
    markets = [str(value) for value in design.get("markets", [])]
    if market not in markets:
        raise ValueError(f"market is not in the design: {market}")
    horizons = tuple(int(value) for value in design["horizons"])
    sample = design["sample"]
    schedule_name = model.get("refit_schedule", design.get("refit_schedule"))
    schedule_path = design_file.parent / str(schedule_name)
    path_count = int(design.get("forecast_draws", design.get("forward_paths", 0)))
    if path_count != 4096 or int(design.get("seed", 0)) != 20260730:
        raise ValueError("forecast paths or seed differ from the published design")
    end = sample["forecast_end"] if origin_end is None else origin_end
    panel = generate_garch_panel(
        frame=read_daily_frame(data_path),
        market=market,
        model_id=model_id,
        refit_dates=load_refit_dates(schedule_path, market=market),
        forecast_start=sample["forecast_start"],
        forecast_end=end,
        horizons=horizons,
        origin_alignment_horizon=int(sample["origin_alignment_horizon"]),
        optimizer_maxiter=int(model["optimizer_maxiter"]),
        path_count=path_count,
        base_seed=int(design["seed"]),
        components_dir=components_dir,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)
    return panel
