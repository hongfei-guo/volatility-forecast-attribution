"""Expanding estimation, daily filtering, and forecast-panel construction."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data_contract import read_daily_frame
from .forecast_features import (
    fit_feature_transform,
    model_spec,
    prepare_estimation_data,
    transform_features,
)
from .forecast_fit import SamplingFailure, build_stan_data, fit_model
from .forecast_filter import FixedParameterGridFilter, filter_bank_is_adequate
from .forecast_paths import (
    compact_refit,
    filter_bank_from_refit,
    filter_from_refit,
    fit_log_har_posterior,
    fit_rv_ra_auxiliary_posterior,
    observable_correction,
    seed_from_context,
    simulate_forecast_paths,
    simulate_log_har_paths,
    simulate_rv_ra_auxiliary_paths,
)


FitFunction = Callable[..., tuple[Any, list[dict[str, Any]]]]


def load_forecast_design(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        design = yaml.safe_load(handle)
    if not isinstance(design, dict):
        raise ValueError("forecast design must be a mapping")
    return design


def load_refit_dates(
    path: str | Path, *, market: str, model_id: str
) -> pd.DatetimeIndex:
    model_spec(model_id)
    frame = pd.read_csv(path)
    required = {"market", "refit_date"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"refit calendar is missing columns: {sorted(missing)}")
    selected = frame[frame["market"].astype(str) == market]
    dates = pd.DatetimeIndex(pd.to_datetime(selected["refit_date"], errors="raise"))
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("refit dates must be nonempty, unique, and ordered")
    return dates.normalize()


def market_daily_frame(
    path: str | Path, *, market: str, model_id: str | None = None
) -> pd.DataFrame:
    frame = read_daily_frame(path)
    required = {"date", "market", "return_cc", "rv_oc"}
    if model_id == "RV-RA-NN-SV":
        required.add("asymmetry")
    missing = required - set(frame)
    if missing:
        raise ValueError(f"daily data are missing columns: {sorted(missing)}")
    selected = frame[frame["market"].astype(str) == market].copy()
    if selected.empty:
        raise ValueError(f"daily data contain no rows for {market}")
    selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
    selected = selected.sort_values("date").reset_index(drop=True)
    if selected["date"].duplicated().any():
        raise ValueError(f"daily data contain duplicate dates for {market}")
    columns = ["date", "return_cc", "rv_oc"]
    if model_id == "RV-RA-NN-SV":
        columns.append("asymmetry")
    return selected.loc[:, columns]


def _fit_auxiliary_log_har(
    daily: pd.DataFrame,
    *,
    origin_position: int,
    paths: int,
    rng: np.random.Generator,
    prior_variance: float,
    inverse_gamma_shape: float,
    inverse_gamma_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rv = daily.iloc[: origin_position + 1]["rv_oc"].to_numpy(float)
    log_rv = np.log(np.maximum(rv, np.finfo(float).tiny))
    return fit_log_har_posterior(
        log_rv,
        draws=paths,
        rng=rng,
        prior_variance=prior_variance,
        inverse_gamma_shape=inverse_gamma_shape,
        inverse_gamma_scale=inverse_gamma_scale,
    )



def forecast_record(
    *, paths: Mapping[str, np.ndarray], market: str, model_id: str,
    origin: pd.Timestamp, mature_date: pd.Timestamp, horizon: int,
    path_count: int, component_root: Path | None,
) -> dict[str, object]:
    """Summarize paths and optionally save the shared predictive components."""
    integrated_variance = np.sum(paths["raw_variances"], axis=1)
    cumulative_return = np.sum(paths["raw_returns"], axis=1)
    if np.any(~np.isfinite(integrated_variance)) or np.any(
        integrated_variance <= 0
    ):
        raise FloatingPointError("forecast paths produced invalid variance")
    row: dict[str, object] = {
        "market": market,
        "model_id": model_id,
        "origin_date": str(origin.date()),
        "horizon": int(horizon),
        "mature_date": str(mature_date.date()),
        "forecast_kind": "probabilistic",
        "variance_forecast": float(np.mean(integrated_variance)),
        "cumulative_return_mean": float(np.mean(cumulative_return)),
        "forecast_draw_count": int(path_count),
        "return_distribution": "variance_standardized_student_t",
        "variance_mean_method": "mean_integrated_path_variance",
    }
    if component_root is not None:
        component_name = (
            f"{market}_{model_id}_{origin:%Y-%m-%d}_h{horizon:02d}.npz"
        )
        component_arrays = {
            name: np.asarray(paths[name])
            for name in (
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
                "degrees_of_freedom",
            )
        }
        path_shape = (path_count, horizon)
        if any(
            component_arrays[name].shape != path_shape
            for name in (
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
            )
        ) or component_arrays["degrees_of_freedom"].shape != (
            path_count,
        ):
            raise ValueError("predictive components do not align with the panel")
        if any(
            np.any(~np.isfinite(value))
            for value in component_arrays.values()
        ):
            raise FloatingPointError(
                "predictive components contain non-finite values"
            )
        if (
            np.any(component_arrays["raw_variances"] <= 0)
            or np.any(component_arrays["raw_conditional_sds"] <= 0)
            or np.any(component_arrays["degrees_of_freedom"] <= 2)
        ):
            raise FloatingPointError(
                "predictive variances, scales, and degrees of freedom are invalid"
            )
        np.savez_compressed(
            component_root / component_name, **component_arrays
        )
        row["predictive_file"] = component_name
    return row


def generate_forecast_panel(
    *,
    daily: pd.DataFrame,
    market: str,
    model_id: str,
    stan_file: str | Path,
    refit_dates: pd.DatetimeIndex,
    output_work_dir: str | Path,
    base_seed: int,
    forecast_start: str | pd.Timestamp,
    forecast_end: str | pd.Timestamp,
    particle_count: int = 4000,
    grid_level: int = 4,
    path_count: int = 4096,
    prior_variance: float = 100.0,
    inverse_gamma_shape: float = 2.0,
    inverse_gamma_scale: float = 0.1,
    fit_function: FitFunction = fit_model,
    show_progress: bool = False,
    components_dir: str | Path | None = None,
    certification_grid_levels: tuple[int, int, int] = (2, 3, 4),
    warning_ess_fraction: float = 0.20,
) -> pd.DataFrame:
    """Generate one market-model panel in forecast-before-filter order."""
    spec = model_spec(model_id)
    columns = ["date", "return_cc", "rv_oc"]
    if model_id == "RV-RA-NN-SV":
        columns.append("asymmetry")
    missing = set(columns) - set(daily)
    if missing:
        raise ValueError(f"daily data are missing columns: {sorted(missing)}")
    frame = daily.loc[:, columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("single-market daily data must have unique ordered dates")
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    if start > end:
        raise ValueError("forecast_start must not be later than forecast_end")
    refit_set = set(pd.DatetimeIndex(refit_dates).normalize())
    origin_positions = [
        position
        for position, date in enumerate(frame["date"])
        if start <= pd.Timestamp(date) <= end
        and position + max(spec.horizons) < len(frame)
    ]
    if not origin_positions:
        raise ValueError("daily data contain no valid forecast origins")
    first_origin = pd.Timestamp(frame.loc[origin_positions[0], "date"])
    if first_origin not in refit_set:
        raise ValueError("the first forecast origin must be a refit date")

    work = Path(output_work_dir)
    work.mkdir(parents=True, exist_ok=True)
    component_root = Path(components_dir) if components_dir is not None else None
    if component_root is not None:
        component_root.mkdir(parents=True, exist_ok=True)
    state_filter: FixedParameterGridFilter | None = None
    filter_bank: dict[int, FixedParameterGridFilter] = {}
    initial_filter_weights: np.ndarray | None = None
    forecasts_withheld = False
    refit: dict[str, np.ndarray] | None = None
    transform = None
    transformed = None
    auxiliary: tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray] | None = None
    rows: list[dict[str, object]] = []

    for origin_position in origin_positions:
        origin = pd.Timestamp(frame.loc[origin_position, "date"])
        if forecasts_withheld and origin not in refit_set:
            continue
        if origin in refit_set:
            transform = fit_feature_transform(
                frame, model_id=model_id, training_end=origin
            )
            estimation = prepare_estimation_data(frame, transform)
            stan_data = build_stan_data(
                model_id=model_id,
                market=market,
                z=estimation.z,
                x=estimation.x,
            )
            fit_seed = seed_from_context(
                base_seed, market, str(origin.date()), "nuts"
            )
            fit_directory = work / f"{market}_{model_id}_{origin:%Y%m%d}"
            try:
                fit, _ = fit_function(
                    stan_file=stan_file,
                    model_id=model_id,
                    data=stan_data,
                    seed=fit_seed,
                    output_dir=fit_directory,
                    show_progress=show_progress,
                )
            except SamplingFailure:
                if model_id not in {"RV-LIN-SV", "RV-RA-NN-SV"}:
                    raise
                state_filter = None
                filter_bank = {}
                initial_filter_weights = None
                forecasts_withheld = model_id == "RV-RA-NN-SV"
                refit = None
                transform = None
                transformed = None
                auxiliary = None
                continue
            refit = compact_refit(
                fit,
                model_id=model_id,
                transform=transform,
                estimation_features=estimation.x,
                particle_count=particle_count,
                rng=np.random.default_rng(
                    seed_from_context(
                        base_seed,
                        market,
                        str(origin.date()),
                        "posterior_particles",
                    )
                ),
            )
            if model_id == "RV-RA-NN-SV":
                filter_bank = filter_bank_from_refit(
                    refit, levels=certification_grid_levels
                )
                state_filter = filter_bank[max(filter_bank)]
                initial_filter_weights = np.asarray(
                    refit["weights"], dtype=float
                ).copy()
                forecasts_withheld = False
            else:
                filter_bank = {}
                initial_filter_weights = None
                state_filter = filter_from_refit(refit, level=grid_level)
            transformed = transform_features(frame, transform)
            if model_id == "RV-NN-SV":
                auxiliary = _fit_auxiliary_log_har(
                    frame,
                    origin_position=origin_position,
                    paths=path_count,
                    rng=np.random.default_rng(
                        seed_from_context(
                            base_seed,
                            market,
                            str(origin.date()),
                            "rv",
                            "auxiliary_posterior",
                        )
                    ),
                    prior_variance=prior_variance,
                    inverse_gamma_shape=inverse_gamma_shape,
                    inverse_gamma_scale=inverse_gamma_scale,
                )
            elif model_id == "RV-RA-NN-SV":
                history = frame.iloc[: origin_position + 1]
                auxiliary = fit_rv_ra_auxiliary_posterior(
                    np.log(
                        np.maximum(
                            history["rv_oc"].to_numpy(float),
                            np.finfo(float).tiny,
                        )
                    ),
                    history["asymmetry"].to_numpy(float),
                    draws=path_count,
                    rng=np.random.default_rng(
                        seed_from_context(
                            base_seed,
                            market,
                            str(origin.date()),
                            "rv",
                            "auxiliary_posterior",
                        )
                    ),
                    prior_variance=prior_variance,
                    inverse_gamma_shape=inverse_gamma_shape,
                    inverse_gamma_scale=inverse_gamma_scale,
                )
            else:
                auxiliary = None

        if state_filter is None or refit is None or transform is None:
            continue
        assert transformed is not None
        history = frame.iloc[: origin_position + 1]
        z_history = transform.return_scaler.standardize(
            history["return_cc"].to_numpy(float)
        )
        rv_floor = transform.rv_floor or np.finfo(float).tiny
        log_rv_history = np.log(
            np.maximum(history["rv_oc"].to_numpy(float), rv_floor)
        )
        for horizon in spec.horizons:
            future_log_rv = None
            future_asymmetry = None
            if model_id == "RV-NN-SV":
                if auxiliary is None:
                    raise RuntimeError("RV-NN-SV auxiliary posterior is unavailable")
                auxiliary_rng = np.random.default_rng(
                    seed_from_context(
                        base_seed,
                        market,
                        str(origin.date()),
                        horizon,
                        "auxiliary_paths",
                    )
                )
                raw_log_rv = np.log(
                    np.maximum(
                        history["rv_oc"].to_numpy(float),
                        np.finfo(float).tiny,
                    )
                )
                future_log_rv = simulate_log_har_paths(
                    raw_log_rv,
                    beta=auxiliary[0],
                    sigma=auxiliary[1],
                    horizon=horizon,
                    paths=path_count,
                    rng=auxiliary_rng,
                )
            elif model_id == "RV-RA-NN-SV":
                if not isinstance(auxiliary, dict):
                    raise RuntimeError(
                        "RV-RA-NN-SV auxiliary posterior is unavailable"
                    )
                auxiliary_rng = np.random.default_rng(
                    seed_from_context(
                        base_seed,
                        market,
                        str(origin.date()),
                        horizon,
                        "auxiliary_paths",
                    )
                )
                future_log_rv, future_asymmetry = (
                    simulate_rv_ra_auxiliary_paths(
                        np.log(
                            np.maximum(
                                history["rv_oc"].to_numpy(float),
                                np.finfo(float).tiny,
                            )
                        ),
                        history["asymmetry"].to_numpy(float),
                        posterior=auxiliary,
                        horizon=horizon,
                        paths=path_count,
                        rng=auxiliary_rng,
                    )
                )
            forecast_rng = np.random.default_rng(
                seed_from_context(
                    base_seed,
                    market,
                    str(origin.date()),
                    horizon,
                    "forecast_paths",
                )
            )
            paths = simulate_forecast_paths(
                model_id=model_id,
                state_filter=state_filter,
                refit=refit,
                standardized_return_history=z_history,
                log_rv_history=log_rv_history,
                horizon=horizon,
                paths=path_count,
                rng=forecast_rng,
                future_log_rv=future_log_rv,
                asymmetry_history=(
                    history["asymmetry"].to_numpy(float)
                    if model_id == "RV-RA-NN-SV"
                    else None
                ),
                future_asymmetry=future_asymmetry,
            )
            rows.append(forecast_record(
                paths=paths, market=market, model_id=model_id, origin=origin,
                mature_date=pd.Timestamp(frame.loc[origin_position + horizon, "date"]),
                horizon=horizon, path_count=path_count, component_root=component_root,
            ))

        next_position = origin_position + 1
        if next_position < len(frame) and pd.Timestamp(
            frame.loc[next_position, "date"]
        ) <= end:
            feature_row = transformed.loc[
                transformed["date"] == frame.loc[next_position, "date"],
                list(transform.feature_names),
            ]
            if feature_row.empty or feature_row.isna().any(axis=None):
                raise ValueError("the next-day feature row is unavailable")
            next_z = float(
                transform.return_scaler.standardize(
                    frame.loc[next_position, "return_cc"]
                )
            )
            correction = lambda parameters, features: observable_correction(
                model_id, parameters, features
            )
            if filter_bank:
                for filtered in filter_bank.values():
                    filtered.step(
                        x_t=feature_row.to_numpy(float)[0],
                        observed_z=next_z,
                        correction=correction,
                    )
                assert initial_filter_weights is not None
                if not filter_bank_is_adequate(
                    filter_bank,
                    initial_weights=initial_filter_weights,
                    warning_fraction=warning_ess_fraction,
                ):
                    forecasts_withheld = True
            else:
                state_filter.step(
                    x_t=feature_row.to_numpy(float)[0],
                    observed_z=next_z,
                    correction=correction,
                )

    columns = [
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
    if component_root is not None:
        columns.append("predictive_file")
    panel = pd.DataFrame(rows, columns=columns)
    if panel.empty:
        raise RuntimeError("no forecasts were generated")
    return panel.sort_values(
        ["origin_date", "horizon"], kind="stable"
    ).reset_index(drop=True)


def generate_from_design(
    *,
    data_path: str | Path,
    design_path: str | Path,
    market: str,
    model_id: str,
    output_path: str | Path,
    work_dir: str | Path,
    origin_start: str | None = None,
    origin_end: str | None = None,
    refit_schedule: str | Path | None = None,
    show_progress: bool = False,
    components_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the documented design and write one forecast panel."""
    design_file = Path(design_path)
    design = load_forecast_design(design_file)
    if market not in design["markets"]:
        raise ValueError(f"unsupported market: {market}")
    if model_id not in design["models"]:
        raise ValueError(f"unsupported model: {model_id}")
    package_root = design_file.resolve().parent.parent
    model_design: Mapping[str, Any] = design["models"][model_id]
    stan_file = package_root / str(model_design["stan_file"])
    schedule_file = (
        design_file.parent / str(design["refits"]["analysis_dates"])
        if refit_schedule is None
        else Path(refit_schedule)
    )
    refit_dates = load_refit_dates(
        schedule_file, market=market, model_id=model_id
    )
    daily = market_daily_frame(data_path, market=market, model_id=model_id)
    forecast_config: Mapping[str, Any] = design["forecast"]
    auxiliary: Mapping[str, Any] = forecast_config["future_realised_variance"]
    panel = generate_forecast_panel(
        daily=daily,
        market=market,
        model_id=model_id,
        stan_file=stan_file,
        refit_dates=refit_dates,
        output_work_dir=work_dir,
        base_seed=int(design["seed"]),
        forecast_start=(
            design["sample"]["forecast_start"]
            if origin_start is None
            else origin_start
        ),
        forecast_end=(
            design["sample"]["forecast_end"] if origin_end is None else origin_end
        ),
        particle_count=int(design["state_filter"]["parameter_atoms"]),
        grid_level=int(design["state_filter"]["deterministic_grid_level"]),
        certification_grid_levels=tuple(
            int(value)
            for value in design["state_filter"].get(
                "certification_grid_levels", [2, 3, 4]
            )
        ),
        warning_ess_fraction=float(
            design["state_filter"].get("warning_ess_fraction", 0.20)
        ),
        path_count=int(forecast_config["paths"]),
        prior_variance=float(auxiliary["prior_variance"]),
        inverse_gamma_shape=float(auxiliary["inverse_gamma_shape"]),
        inverse_gamma_scale=float(auxiliary["inverse_gamma_scale"]),
        show_progress=show_progress,
        components_dir=components_dir,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)
    return panel
