"""Prequential combinations of the NN-SV and RV-NN-SV forecasts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import optimize
from scipy.special import logsumexp

from .evaluation import qlike


WEIGHT_COLUMNS = (
    "market",
    "horizon",
    "mode",
    "update_date",
    "model_id",
    "available",
    "discount_age_unit",
    "evaluation_sample_policy",
    "matured_observations",
    "prequential_weight",
    "availability_adjusted_weight",
)

FORECAST_COLUMNS = (
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

PREDICTIVE_COMPONENT_NAMES = (
    "raw_returns",
    "raw_variances",
    "raw_return_locations",
    "raw_conditional_sds",
    "degrees_of_freedom",
)


def load_combination_design(path: str | Path) -> dict[str, Any]:
    design = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("combination design must be a mapping")
    required = {
        "markets",
        "horizons",
        "candidate_models",
        "sample",
        "weighting",
    }
    missing = required - set(design)
    if missing:
        raise ValueError(f"combination design lacks fields: {sorted(missing)}")
    candidates = [str(value) for value in design["candidate_models"]]
    if candidates != ["NN-SV", "RV-NN-SV"]:
        raise ValueError("candidate_models must be NN-SV followed by RV-NN-SV")
    markets = [str(value) for value in design["markets"]]
    horizons = [int(value) for value in design["horizons"]]
    if not markets or len(markets) != len(set(markets)):
        raise ValueError("markets must be nonempty and unique")
    if horizons != [1, 5, 10]:
        raise ValueError("horizons must be [1, 5, 10]")
    if "seed" in design:
        seed = design["seed"]
        if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
    sample = design["sample"]
    weighting = design["weighting"]
    if not isinstance(sample, dict) or not isinstance(weighting, dict):
        raise ValueError("sample and weighting must be mappings")
    for field in ("objective_start", "forecast_start", "forecast_end"):
        if field not in sample:
            raise ValueError(f"sample lacks {field}")
    objective_start = pd.Timestamp(sample["objective_start"]).normalize()
    forecast_start = pd.Timestamp(sample["forecast_start"]).normalize()
    forecast_end = pd.Timestamp(sample["forecast_end"]).normalize()
    if not objective_start < forecast_start <= forecast_end:
        raise ValueError("combination sample dates are not ordered")
    expected_weighting = {
        "half_life_sessions",
        "update_frequency",
        "evaluation_sample_policy",
        "eligibility_column",
        "availability_column",
        "modes",
    }
    if expected_weighting - set(weighting):
        raise ValueError(
            "weighting lacks fields: " f"{sorted(expected_weighting - set(weighting))}"
        )
    if float(weighting["half_life_sessions"]) <= 0:
        raise ValueError("half_life_sessions must be positive")
    if weighting["update_frequency"] != "calendar_month":
        raise ValueError("update_frequency must be calendar_month")
    if weighting["evaluation_sample_policy"] != "main_gap_excluded":
        raise ValueError("evaluation_sample_policy must be main_gap_excluded")
    modes = weighting["modes"]
    if not isinstance(modes, dict) or set(modes) != {"equal", "qlike", "log_score"}:
        raise ValueError("modes must define equal, qlike, and log_score")
    for mode, allowed in modes.items():
        values = [int(value) for value in allowed]
        if not values or not set(values) <= set(horizons):
            raise ValueError(f"invalid horizons for {mode}")
    if [int(value) for value in modes["log_score"]] != [1]:
        raise ValueError("log_score combination must be restricted to horizon 1")
    return design


def exponential_discount_weights(ages: np.ndarray, *, half_life: float) -> np.ndarray:
    values = np.asarray(ages, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("ages must be a nonempty finite vector")
    if np.any(values < 0) or not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("ages must be nonnegative and half_life must be positive")
    weights = np.exp(-np.log(2.0) * values / half_life)
    return weights / weights.sum()


def _observation_weights(values: np.ndarray | None, n: int) -> np.ndarray:
    weights = (
        np.ones(n, dtype=float) if values is None else np.asarray(values, dtype=float)
    )
    if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("observation weights must be a finite nonnegative vector")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("observation weights must have positive sum")
    return weights / total


def optimize_qlike_weights(
    realized: np.ndarray,
    forecasts: np.ndarray,
    observation_weights: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(realized, dtype=float)
    f = np.asarray(forecasts, dtype=float)
    if y.ndim != 1 or f.ndim != 2 or f.shape[0] != y.size or f.shape[1] < 2:
        raise ValueError(
            "QLIKE inputs must have shapes (observation,) and (observation, model)"
        )
    if (
        np.any(~np.isfinite(y))
        or np.any(y <= 0)
        or np.any(~np.isfinite(f))
        or np.any(f <= 0)
    ):
        raise ValueError("QLIKE inputs must be finite and strictly positive")
    obs_weights = _observation_weights(observation_weights, y.size)
    model_count = f.shape[1]

    def objective(weights: np.ndarray) -> float:
        return float(np.sum(obs_weights * qlike(y, f @ weights)))

    result = optimize.minimize(
        objective,
        np.full(model_count, 1.0 / model_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"QLIKE weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def optimize_log_score_weights(
    log_densities: np.ndarray,
    observation_weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(log_densities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("log densities must have shape (observation, model)")
    if np.any(~np.isfinite(values)):
        raise ValueError("log densities must be finite")
    obs_weights = _observation_weights(observation_weights, values.shape[0])
    model_count = values.shape[1]

    def objective(weights: np.ndarray) -> float:
        mixture = logsumexp(values + np.log(np.maximum(weights, 1e-15)), axis=1)
        return float(-np.sum(obs_weights * mixture))

    result = optimize.minimize(
        objective,
        np.full(model_count, 1.0 / model_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"log-score weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def _normalise_dates(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result


def _prepare_forecasts(
    forecasts: pd.DataFrame,
    *,
    candidates: Sequence[str],
    markets: Sequence[str],
    horizons: Sequence[int],
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    required = {
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "mature_date",
        "variance_forecast",
    }
    missing = required - set(forecasts)
    if missing:
        raise ValueError(f"forecast panels lack columns: {sorted(missing)}")
    selected = forecasts[forecasts["model_id"].astype(str).isin(candidates)].copy()
    selected = _normalise_dates(selected, ("origin_date", "mature_date"))
    selected["market"] = selected["market"].astype(str)
    selected["model_id"] = selected["model_id"].astype(str)
    selected["horizon"] = pd.to_numeric(selected["horizon"], errors="raise").astype(int)
    selected = selected[
        selected["market"].isin(markets)
        & selected["horizon"].isin(horizons)
        & selected["origin_date"].between(forecast_start, forecast_end)
    ].copy()
    keys = ["market", "model_id", "origin_date", "horizon"]
    if selected.empty or selected.duplicated(keys).any():
        raise ValueError("candidate forecasts are empty or contain duplicate rows")
    variance = pd.to_numeric(selected["variance_forecast"], errors="raise")
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("candidate variance forecasts must be finite and positive")
    selected["variance_forecast"] = variance
    if "cumulative_return_mean" not in selected:
        selected["cumulative_return_mean"] = np.nan
    return selected.sort_values(keys, kind="stable").reset_index(drop=True)


def _prepare_objectives(
    losses: pd.DataFrame,
    warmup_losses: pd.DataFrame,
    *,
    candidates: Sequence[str],
    markets: Sequence[str],
    horizons: Sequence[int],
    objective_start: pd.Timestamp,
    forecast_start: pd.Timestamp,
    eligibility_column: str,
    availability_column: str,
) -> pd.DataFrame:
    required = {
        "market",
        "model_id",
        "origin_date",
        "mature_date",
        "horizon",
        "realized_variance",
        "variance_forecast",
        eligibility_column,
    }
    prepared_inputs = []
    for label, source in (("losses", losses), ("warmup losses", warmup_losses)):
        frame = source.copy()
        if availability_column not in frame:
            frame[availability_column] = True
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} lack columns: {sorted(missing)}")
        available = frame[availability_column].map({True: True, False: False})
        if available.isna().any():
            raise ValueError(f"{label} {availability_column} must contain booleans")
        prepared_inputs.append(frame.loc[available].copy())
    warmup = _normalise_dates(prepared_inputs[1], ("origin_date", "mature_date"))
    if warmup.empty or warmup["origin_date"].min() < objective_start:
        raise ValueError("warmup losses are empty or precede objective_start")
    if np.any(warmup["origin_date"] >= forecast_start) or np.any(
        warmup["mature_date"] >= forecast_start
    ):
        raise ValueError(
            "warmup losses must originate and mature before forecast_start"
        )
    main = _normalise_dates(prepared_inputs[0], ("origin_date", "mature_date"))
    if np.any(main["origin_date"] < forecast_start):
        raise ValueError("main losses overlap the warmup period")
    objectives = pd.concat([warmup, main], ignore_index=True, sort=False)
    objectives["market"] = objectives["market"].astype(str)
    objectives["model_id"] = objectives["model_id"].astype(str)
    objectives["horizon"] = pd.to_numeric(objectives["horizon"], errors="raise").astype(
        int
    )
    objectives = objectives[
        objectives["model_id"].isin(candidates)
        & objectives["market"].isin(markets)
        & objectives["horizon"].isin(horizons)
    ].copy()
    keys = ["market", "model_id", "origin_date", "horizon"]
    if objectives.empty or objectives.duplicated(keys).any():
        raise ValueError("objective losses are empty or contain duplicate rows")
    objectives[eligibility_column] = objectives[eligibility_column].map(
        {True: True, False: False}
    )
    if objectives[eligibility_column].isna().any():
        raise ValueError(f"{eligibility_column} must contain booleans")
    for column in ("realized_variance", "variance_forecast"):
        values = pd.to_numeric(objectives[column], errors="raise")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{column} must be finite and positive")
        objectives[column] = values
    return objectives.sort_values(keys, kind="stable").reset_index(drop=True)


def warmup_losses_from_forecasts(
    warmup_forecasts: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    eligibility_column: str,
    availability_column: str,
) -> pd.DataFrame:
    """Attach legally obtained realised variance to prediction-only warmup rows."""
    required = {
        "market",
        "model_id",
        "origin_date",
        "mature_date",
        "horizon",
        "variance_forecast",
        availability_column,
    }
    missing = required - set(warmup_forecasts)
    if missing:
        raise ValueError(f"warmup forecasts lack columns: {sorted(missing)}")
    daily_required = {"market", "date", "rv_cc"}
    missing_daily = daily_required - set(daily)
    if missing_daily:
        raise ValueError(f"daily data lack columns: {sorted(missing_daily)}")
    source = warmup_forecasts.copy()
    if eligibility_column not in source:
        source[eligibility_column] = True
    available = source[availability_column].map({True: True, False: False})
    if available.isna().any():
        raise ValueError(f"{availability_column} must contain booleans")
    forecasts = _normalise_dates(
        source.loc[available].copy(), ("origin_date", "mature_date")
    )
    if forecasts.empty:
        raise ValueError("warmup forecasts contain no available predictions")
    daily_frame = _normalise_dates(daily, ("date",))
    daily_frame["market"] = daily_frame["market"].astype(str)
    forecasts["market"] = forecasts["market"].astype(str)
    markets = {
        market: group.sort_values("date").reset_index(drop=True)
        for market, group in daily_frame.groupby("market", sort=False)
    }
    positions = {
        market: {date: index for index, date in enumerate(group["date"])}
        for market, group in markets.items()
    }
    realized: dict[tuple[str, pd.Timestamp, int], float] = {}
    for record in forecasts.to_dict("records"):
        market = str(record["market"])
        origin = pd.Timestamp(record["origin_date"])
        horizon = int(record["horizon"])
        key = (market, origin, horizon)
        if market not in positions or origin not in positions[market]:
            raise ValueError(f"warmup origin is absent from daily data: {key}")
        if key not in realized:
            position = positions[market][origin]
            target = markets[market].iloc[position + 1 : position + horizon + 1]
            if len(target) != horizon:
                raise ValueError(f"warmup target is incomplete: {key}")
            mature = pd.Timestamp(target["date"].iloc[-1])
            if mature != pd.Timestamp(record["mature_date"]):
                raise ValueError(f"warmup mature date differs from daily data: {key}")
            values = pd.to_numeric(target["rv_cc"], errors="raise").to_numpy(float)
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"warmup realised variance is invalid: {key}")
            total = float(values.sum())
            if total <= 0:
                raise ValueError(f"warmup integrated variance is not positive: {key}")
            realized[key] = total
        record_mature = pd.Timestamp(record["mature_date"])
        expected_mature = pd.Timestamp(
            markets[market].iloc[positions[market][origin] + horizon]["date"]
        )
        if record_mature != expected_mature:
            raise ValueError(f"candidate warmup mature dates differ: {key}")
    result = forecasts.copy()
    result["realized_variance"] = [
        realized[(str(row.market), pd.Timestamp(row.origin_date), int(row.horizon))]
        for row in result.itertuples(index=False)
    ]
    if "return_log_density" not in result:
        result["return_log_density"] = np.nan
    return result


def _common_objectives(
    objectives: pd.DataFrame,
    *,
    candidates: Sequence[str],
    eligibility_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_keys = ["market", "horizon", "origin_date"]
    for (market, horizon, origin), group in objectives.groupby(group_keys, sort=True):
        indexed = group.set_index("model_id").reindex(candidates)
        available = indexed["mature_date"].notna().to_numpy()
        if not np.all(available):
            continue
        mature = pd.DatetimeIndex(indexed["mature_date"])
        if len(mature.unique()) != 1:
            raise ValueError(
                f"candidate mature dates differ at {market} {origin} h={horizon}"
            )
        realized = indexed["realized_variance"].to_numpy(float)
        if not np.allclose(realized, realized[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"candidate realized variances differ at {market} {origin} h={horizon}"
            )
        eligible = indexed[eligibility_column].to_numpy(bool)
        if not np.all(eligible == eligible[0]):
            raise ValueError(
                f"candidate eligibility differs at {market} {origin} h={horizon}"
            )
        log_density = np.full(len(candidates), np.nan)
        if "return_log_density" in indexed:
            log_density = pd.to_numeric(
                indexed["return_log_density"], errors="coerce"
            ).to_numpy(float)
        rows.append(
            {
                "market": market,
                "horizon": int(horizon),
                "origin_date": pd.Timestamp(origin),
                "mature_date": mature[0],
                "eligible": bool(eligible[0]),
                "realized_variance": float(realized[0]),
                "variance_forecasts": indexed["variance_forecast"].to_numpy(float),
                "log_densities": log_density,
            }
        )
    if not rows:
        raise ValueError("no objective dates contain both candidate models")
    return pd.DataFrame(rows)


def _session_indices(
    daily: pd.DataFrame,
    markets: Sequence[str],
    calendar_report: Mapping[str, Any],
) -> dict[str, dict[pd.Timestamp, int]]:
    required = {"market", "date"}
    missing = required - set(daily)
    if missing:
        raise ValueError(f"daily data lack columns: {sorted(missing)}")
    frame = _normalise_dates(daily, ("date",))
    frame["market"] = frame["market"].astype(str)
    if calendar_report.get("valid") is not True:
        raise ValueError("calendar report must be valid")
    market_reports = calendar_report.get("markets")
    if not isinstance(market_reports, list):
        raise ValueError("calendar report lacks market records")
    reports = {
        str(record.get("market")): record
        for record in market_reports
        if isinstance(record, dict)
    }
    result: dict[str, dict[pd.Timestamp, int]] = {}
    for market in markets:
        observed = pd.DatetimeIndex(
            frame.loc[frame["market"] == market, "date"]
        ).sort_values()
        if observed.empty or observed.has_duplicates:
            raise ValueError(f"daily dates for {market} are empty or duplicated")
        if market not in reports:
            raise ValueError(f"calendar report lacks {market}")
        report = reports[market]
        if report.get("observed_non_sessions") or report.get("source_label_mismatches"):
            raise ValueError(f"calendar report contains unresolved labels for {market}")
        if (
            pd.Timestamp(report.get("first_date")).normalize() != observed.min()
            or pd.Timestamp(report.get("last_date")).normalize() != observed.max()
        ):
            raise ValueError(
                f"calendar report date range differs from daily data for {market}"
            )
        missing_dates = pd.to_datetime(
            report.get("missing_expected_sessions", []), errors="raise"
        )
        missing = pd.DatetimeIndex(missing_dates).normalize()
        dates = observed.union(missing).sort_values()
        if int(report.get("expected_sessions", -1)) != len(dates):
            raise ValueError(f"calendar report session count differs for {market}")
        result[market] = {date: index for index, date in enumerate(dates)}
    return result


def _adjust_weights(base: np.ndarray, availability: np.ndarray) -> np.ndarray:
    weights = np.asarray(base, dtype=float)
    available = np.asarray(availability, dtype=bool)
    if (
        weights.shape != available.shape
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("invalid base weights or availability")
    adjusted = np.where(available, weights, 0.0)
    total = float(adjusted.sum())
    if total <= 0:
        raise ValueError("no candidate with positive weight is available")
    return adjusted / total


def _weights_from_objectives(
    objectives: pd.DataFrame,
    *,
    market: str,
    horizon: int,
    mode: str,
    as_of_date: pd.Timestamp,
    session_index: Mapping[pd.Timestamp, int],
    half_life: float,
) -> tuple[np.ndarray, int]:
    cell = objectives[
        (objectives["market"] == market)
        & (objectives["horizon"].astype(int) == horizon)
        & objectives["eligible"]
        & (objectives["mature_date"] <= as_of_date)
    ].copy()
    if cell.empty:
        raise ValueError(
            f"no eligible warmup objective has matured by {as_of_date.date()}"
        )
    if as_of_date not in session_index:
        raise ValueError(f"forecast origin is absent from {market} exchange sessions")
    missing_dates = [date for date in cell["mature_date"] if date not in session_index]
    if missing_dates:
        raise ValueError("objective mature dates are absent from exchange sessions")
    ages = np.asarray(
        [
            session_index[as_of_date] - session_index[date]
            for date in cell["mature_date"]
        ],
        dtype=int,
    )
    observation_weights = exponential_discount_weights(ages, half_life=half_life)
    if mode == "equal":
        base = np.full(2, 0.5)
    elif mode == "qlike":
        base = optimize_qlike_weights(
            cell["realized_variance"].to_numpy(float),
            np.stack(cell["variance_forecasts"].to_numpy()),
            observation_weights,
        )
    elif mode == "log_score":
        densities = np.stack(cell["log_densities"].to_numpy())
        if np.any(~np.isfinite(densities)):
            raise ValueError(
                "log-score objectives require finite one-day log densities"
            )
        base = optimize_log_score_weights(densities, observation_weights)
    else:
        raise ValueError(f"unsupported combination mode: {mode}")
    return base, int(len(cell))


def _events_from_objectives(
    forecasts: pd.DataFrame,
    objectives: pd.DataFrame,
    *,
    candidates: Sequence[str],
    modes: Mapping[str, Sequence[int]],
    session_indices: Mapping[str, Mapping[pd.Timestamp, int]],
    half_life: float,
    sample_policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market in sorted(forecasts["market"].unique()):
        for horizon in sorted(
            forecasts.loc[forecasts["market"] == market, "horizon"].unique()
        ):
            cell = forecasts[
                (forecasts["market"] == market)
                & (forecasts["horizon"].astype(int) == int(horizon))
            ]
            origins = pd.DatetimeIndex(cell["origin_date"].unique()).sort_values()
            for mode, allowed_horizons in modes.items():
                if int(horizon) not in {int(value) for value in allowed_horizons}:
                    continue
                current_month: tuple[int, int] | None = None
                base: np.ndarray | None = None
                matured = 0
                previous_availability: tuple[bool, ...] | None = None
                for origin in origins:
                    indexed = cell[cell["origin_date"] == origin].set_index("model_id")
                    availability = np.asarray(
                        [candidate in indexed.index for candidate in candidates]
                    )
                    if not availability.any():
                        raise ValueError(
                            "no candidate forecast is available at an origin"
                        )
                    month = (origin.year, origin.month)
                    monthly_update = base is None or month != current_month
                    if monthly_update:
                        base, matured = _weights_from_objectives(
                            objectives,
                            market=str(market),
                            horizon=int(horizon),
                            mode=str(mode),
                            as_of_date=origin,
                            session_index=session_indices[str(market)],
                            half_life=half_life,
                        )
                        current_month = month
                    assert base is not None
                    adjusted = _adjust_weights(base, availability)
                    availability_key = tuple(bool(value) for value in availability)
                    if monthly_update or availability_key != previous_availability:
                        for index, candidate in enumerate(candidates):
                            rows.append(
                                {
                                    "market": market,
                                    "horizon": int(horizon),
                                    "mode": mode,
                                    "update_date": origin,
                                    "model_id": candidate,
                                    "available": bool(availability[index]),
                                    "discount_age_unit": "exchange_session",
                                    "evaluation_sample_policy": sample_policy,
                                    "matured_observations": matured,
                                    "prequential_weight": float(base[index]),
                                    "availability_adjusted_weight": float(
                                        adjusted[index]
                                    ),
                                }
                            )
                    previous_availability = availability_key
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


def _validate_packaged_weights(
    weights: pd.DataFrame,
    *,
    forecasts: pd.DataFrame,
    candidates: Sequence[str],
    modes: Mapping[str, Sequence[int]],
    sample_policy: str,
) -> pd.DataFrame:
    missing = set(WEIGHT_COLUMNS) - set(weights)
    if missing:
        raise ValueError(f"packaged weights lack columns: {sorted(missing)}")
    result = _normalise_dates(weights.loc[:, WEIGHT_COLUMNS], ("update_date",))
    result["model_id"] = result["model_id"].astype(str)
    result["mode"] = result["mode"].astype(str)
    result["market"] = result["market"].astype(str)
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise").astype(int)
    result = result[result["mode"].isin(modes)].copy()
    keys = ["market", "horizon", "mode", "update_date", "model_id"]
    if result.empty or result.duplicated(keys).any():
        raise ValueError("packaged weights are empty or duplicated")
    if set(result["model_id"]) != set(candidates):
        raise ValueError("packaged weights contain a different candidate set")
    if set(result["discount_age_unit"]) != {"exchange_session"}:
        raise ValueError("packaged weights must use exchange-session ages")
    if set(result["evaluation_sample_policy"]) != {sample_policy}:
        raise ValueError("packaged weights use a different evaluation sample")
    for column in ("prequential_weight", "availability_adjusted_weight"):
        values = pd.to_numeric(result[column], errors="raise")
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"invalid {column}")
        result[column] = values
    for key, group in result.groupby(["market", "horizon", "mode", "update_date"]):
        ordered = group.set_index("model_id").reindex(candidates)
        if ordered.isna().all(axis=1).any():
            raise ValueError(f"packaged weight event lacks a candidate: {key}")
        if not np.isclose(ordered["prequential_weight"].sum(), 1.0):
            raise ValueError(f"prequential weights do not sum to one: {key}")
        if not np.isclose(ordered["availability_adjusted_weight"].sum(), 1.0):
            raise ValueError(f"adjusted weights do not sum to one: {key}")
    expected = []
    for (market, horizon), group in forecasts.groupby(["market", "horizon"]):
        first_by_month = (
            group[["origin_date"]]
            .drop_duplicates()
            .sort_values("origin_date")
            .assign(month=lambda frame: frame["origin_date"].dt.to_period("M"))
            .groupby("month", sort=True)["origin_date"]
            .first()
        )
        for mode, allowed in modes.items():
            if int(horizon) in {int(value) for value in allowed}:
                expected.extend(
                    (str(market), int(horizon), mode, date) for date in first_by_month
                )
    observed = set(
        tuple(row)
        for row in result[["market", "horizon", "mode", "update_date"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing_events = [key for key in expected if key not in observed]
    if missing_events:
        raise ValueError(
            "packaged weights lack required monthly updates; "
            f"first missing event: {missing_events[0]}"
        )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _weight_event_lookup(
    weights: pd.DataFrame,
    *,
    candidates: Sequence[str],
) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, pd.DataFrame]]:
    events: dict[pd.Timestamp, pd.DataFrame] = {}
    for update, group in weights.groupby("update_date", sort=True):
        date = pd.Timestamp(update).normalize()
        events[date] = group.set_index("model_id").reindex(candidates)
    dates = pd.DatetimeIndex(sorted(events))
    if dates.empty:
        raise ValueError("no weight events are available")
    return dates, events


def _event_from_lookup(
    dates: pd.DatetimeIndex,
    events: Mapping[pd.Timestamp, pd.DataFrame],
    *,
    origin: pd.Timestamp,
    availability: np.ndarray,
) -> np.ndarray:
    position = int(dates.searchsorted(origin, side="right")) - 1
    if position < 0:
        raise ValueError(f"no weight event is available for {origin.date()}")
    event = events[pd.Timestamp(dates[position])]
    recorded = event["available"].map({True: True, False: False})
    if recorded.isna().any() or not np.array_equal(
        recorded.to_numpy(bool), availability
    ):
        raise ValueError("weight event availability differs from candidate forecasts")
    base = event["prequential_weight"].to_numpy(float)
    adjusted = event["availability_adjusted_weight"].to_numpy(float)
    expected = _adjust_weights(base, availability)
    if not np.allclose(adjusted, expected, rtol=0.0, atol=1e-12):
        raise ValueError("recorded availability adjustment is inconsistent")
    return adjusted


def _candidate_predictive_components(
    rows: pd.DataFrame,
    *,
    predictive_root: Path,
    horizon: int,
) -> list[dict[str, np.ndarray]]:
    if "predictive_file" not in rows:
        raise ValueError(
            "whole-path combinations require predictive_file for each candidate"
        )
    components: list[dict[str, np.ndarray]] = []
    expected_draws: int | None = None
    for row in rows.itertuples(index=False):
        relative = Path(str(row.predictive_file))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("candidate predictive_file paths must be relative")
        path = predictive_root / relative
        if not path.is_file():
            raise ValueError(f"candidate predictive file is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(PREDICTIVE_COMPONENT_NAMES) - set(archive.files)
            if missing:
                raise ValueError(
                    "candidate predictive file lacks required arrays: "
                    f"{sorted(missing)}"
                )
            item = {
                name: np.asarray(archive[name]).copy()
                for name in PREDICTIVE_COMPONENT_NAMES
            }
        path_shape = item["raw_returns"].shape
        if len(path_shape) != 2 or path_shape[1] != horizon:
            raise ValueError(
                "candidate predictive paths do not match the forecast horizon"
            )
        draws = int(path_shape[0])
        if draws == 0 or any(
            item[name].shape != path_shape
            for name in PREDICTIVE_COMPONENT_NAMES[:-1]
        ):
            raise ValueError("candidate predictive path arrays must align")
        if item["degrees_of_freedom"].shape != (draws,):
            raise ValueError(
                "degrees_of_freedom must contain one value per predictive path"
            )
        reported_draws = getattr(row, "forecast_draw_count", np.nan)
        if pd.notna(reported_draws) and int(reported_draws) != draws:
            raise ValueError(
                "candidate forecast_draw_count differs from its predictive file"
            )
        if expected_draws is None:
            expected_draws = draws
        elif draws != expected_draws:
            raise ValueError("candidate models have different predictive path counts")
        if any(np.any(~np.isfinite(item[name])) for name in PREDICTIVE_COMPONENT_NAMES):
            raise ValueError("candidate predictive components must be finite")
        if np.any(item["raw_variances"] <= 0):
            raise ValueError("candidate raw variances must be strictly positive")
        if np.any(item["raw_conditional_sds"] <= 0):
            raise ValueError("candidate conditional standard deviations must be positive")
        if np.any(item["degrees_of_freedom"] <= 2):
            raise ValueError("candidate degrees of freedom must exceed two")
        components.append(item)
    return components


def _write_whole_path_mixture(
    rows: pd.DataFrame,
    *,
    weights: np.ndarray,
    candidate_indices: np.ndarray,
    predictive_root: Path,
    output_path: Path,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    components = _candidate_predictive_components(
        rows,
        predictive_root=predictive_root,
        horizon=horizon,
    )
    draw_count = int(components[0]["raw_returns"].shape[0])
    local_members = rng.choice(len(components), size=draw_count, p=weights)
    source_draws = rng.integers(0, draw_count, size=draw_count)
    source_models = candidate_indices[local_members]
    selected: dict[str, np.ndarray] = {}
    for name in PREDICTIVE_COMPONENT_NAMES:
        stacked = np.stack([component[name] for component in components], axis=0)
        selected[name] = stacked[local_members, source_draws]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **selected,
        source_model_indices=source_models,
        source_draw_indices=source_draws,
    )
    cumulative_return_mean = float(np.mean(np.sum(selected["raw_returns"], axis=1)))
    return draw_count, cumulative_return_mean


def _ensemble_rows(
    forecasts: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    candidates: Sequence[str],
    modes: Mapping[str, Sequence[int]],
    predictive_root: Path | None = None,
    predictive_output_dir: Path | None = None,
    mixture_seed: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = {
        "equal": "NN-ENSEMBLE-EQUAL",
        "qlike": "NN-ENSEMBLE-QLIKE",
        "log_score": "NN-ENSEMBLE-LOG_SCORE",
    }
    for (market, horizon), cell in forecasts.groupby(["market", "horizon"], sort=True):
        origin_groups = {
            pd.Timestamp(origin): group.set_index("model_id")
            for origin, group in cell.groupby("origin_date", sort=True)
        }
        origins = pd.DatetimeIndex(sorted(origin_groups))
        for mode, allowed_horizons in modes.items():
            if int(horizon) not in {int(value) for value in allowed_horizons}:
                continue
            mode_weights = weights[
                (weights["market"] == market)
                & (weights["horizon"].astype(int) == int(horizon))
                & (weights["mode"] == mode)
            ]
            event_dates, event_lookup = _weight_event_lookup(
                mode_weights, candidates=candidates
            )
            mixture_rng = (
                np.random.default_rng(mixture_seed)
                if predictive_root is not None and mode in {"equal", "log_score"}
                else None
            )
            for origin in origins:
                indexed = origin_groups[pd.Timestamp(origin)]
                availability = np.asarray(
                    [candidate in indexed.index for candidate in candidates]
                )
                adjusted = _event_from_lookup(
                    event_dates,
                    event_lookup,
                    origin=origin,
                    availability=availability,
                )
                available_models = [
                    candidate
                    for candidate, present in zip(candidates, availability, strict=True)
                    if present
                ]
                active_rows = indexed.loc[available_models]
                mature = pd.DatetimeIndex(active_rows["mature_date"])
                if len(mature.unique()) != 1:
                    raise ValueError("candidate forecasts disagree on mature_date")
                exact_variance = float(
                    np.sum(
                        active_rows["variance_forecast"].to_numpy(float)
                        * adjusted[availability]
                    )
                )
                if not np.isfinite(exact_variance) or exact_variance <= 0:
                    raise ValueError(
                        "combined variance mean must be finite and positive"
                    )
                common: dict[str, object] = {
                    "market": market,
                    "model_id": model_names[mode],
                    "origin_date": origin,
                    "horizon": int(horizon),
                    "mature_date": mature[0],
                    "variance_forecast": exact_variance,
                    "return_distribution": None,
                    "variance_mean_method": "exact_weighted_candidate_means",
                    "candidate_availability": availability.tolist(),
                    "candidate_model_ids": list(candidates),
                    "available_candidate_model_ids": available_models,
                    "combination_mode": mode,
                    "combination_weights": adjusted.tolist(),
                }
                return_means = pd.to_numeric(
                    active_rows["cumulative_return_mean"], errors="coerce"
                ).to_numpy(float)
                cumulative_mean = (
                    float(np.sum(return_means * adjusted[availability]))
                    if np.all(np.isfinite(return_means))
                    else np.nan
                )
                if mixture_rng is not None:
                    assert predictive_root is not None
                    assert predictive_output_dir is not None
                    assert mixture_seed is not None
                    if "return_distribution" not in active_rows:
                        raise ValueError(
                            "whole-path combinations require return_distribution"
                        )
                    distributions = set(active_rows["return_distribution"].astype(str))
                    if distributions != {"variance_standardized_student_t"}:
                        raise ValueError(
                            "candidate predictive components must use the documented "
                            "variance-standardized Student-t distribution"
                        )
                    predictive_file = (
                        f"{market}_{model_names[mode]}_"
                        f"{origin.strftime('%Y%m%d')}_h{int(horizon)}.npz"
                    )
                    draw_count, cumulative_mean = _write_whole_path_mixture(
                        active_rows,
                        weights=adjusted[availability],
                        candidate_indices=np.flatnonzero(availability),
                        predictive_root=predictive_root,
                        output_path=predictive_output_dir / predictive_file,
                        horizon=int(horizon),
                        rng=mixture_rng,
                    )
                    forecast_fields: dict[str, object] = {
                        "forecast_kind": "probabilistic",
                        "forecast_draw_count": draw_count,
                        "return_distribution": (
                            "variance_standardized_student_t"
                        ),
                        "predictive_file": predictive_file,
                    }
                else:
                    forecast_fields = {
                        "forecast_kind": "variance_point_only",
                        "forecast_draw_count": np.nan,
                    }
                rows.append(
                    {
                        **common,
                        **forecast_fields,
                        "cumulative_return_mean": cumulative_mean,
                    }
                )
    columns = list(FORECAST_COLUMNS)
    if predictive_root is not None:
        columns.append("predictive_file")
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["market", "model_id", "origin_date", "horizon"], kind="stable")
        .reset_index(drop=True)
    )


def generate_combination_panels(
    *,
    daily: pd.DataFrame,
    forecasts: pd.DataFrame,
    design: Mapping[str, Any],
    losses: pd.DataFrame | None = None,
    warmup_losses: pd.DataFrame | None = None,
    warmup_forecasts: pd.DataFrame | None = None,
    packaged_weights: pd.DataFrame | None = None,
    calendar_report: Mapping[str, Any] | None = None,
    selected_modes: Sequence[str] | None = None,
    predictive_root: str | Path | None = None,
    predictive_output_dir: str | Path | None = None,
    mixture_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [str(value) for value in design["candidate_models"]]
    markets = [str(value) for value in design["markets"]]
    horizons = [int(value) for value in design["horizons"]]
    sample = design["sample"]
    weighting = design["weighting"]
    prepared_forecasts = _prepare_forecasts(
        forecasts,
        candidates=candidates,
        markets=markets,
        horizons=horizons,
        forecast_start=pd.Timestamp(sample["forecast_start"]).normalize(),
        forecast_end=pd.Timestamp(sample["forecast_end"]).normalize(),
    )
    all_modes = {
        key: [int(value) for value in values]
        for key, values in weighting["modes"].items()
    }
    requested_modes = (
        list(all_modes)
        if selected_modes is None
        else [str(value) for value in selected_modes]
    )
    if not requested_modes or len(requested_modes) != len(set(requested_modes)):
        raise ValueError("selected_modes must be nonempty and unique")
    unknown_modes = set(requested_modes) - set(all_modes)
    if unknown_modes:
        raise ValueError(f"unsupported selected_modes: {sorted(unknown_modes)}")
    modes = {mode: all_modes[mode] for mode in requested_modes}
    if (predictive_root is None) != (predictive_output_dir is None):
        raise ValueError(
            "predictive_root and predictive_output_dir must be supplied together"
        )
    predictive_root_path = (
        None if predictive_root is None else Path(predictive_root)
    )
    predictive_output_path = (
        None if predictive_output_dir is None else Path(predictive_output_dir)
    )
    if predictive_root_path is not None:
        if mixture_seed is None:
            if "seed" not in design:
                raise ValueError("whole-path combinations require a documented seed")
            mixture_seed = int(design["seed"])
        if mixture_seed < 0:
            raise ValueError("mixture_seed must be nonnegative")
    sample_policy = str(weighting["evaluation_sample_policy"])
    if packaged_weights is not None:
        if (
            losses is not None
            or warmup_losses is not None
            or warmup_forecasts is not None
        ):
            raise ValueError("choose regenerated weights or packaged weights, not both")
        weights = _validate_packaged_weights(
            packaged_weights,
            forecasts=prepared_forecasts,
            candidates=candidates,
            modes=modes,
            sample_policy=sample_policy,
        )
    else:
        if warmup_losses is not None and warmup_forecasts is not None:
            raise ValueError("choose warmup_losses or warmup_forecasts, not both")
        if losses is None or (warmup_losses is None and warmup_forecasts is None):
            raise ValueError(
                "regenerating prequential weights requires main losses and one warmup input; "
                "otherwise supply the complete packaged monthly weight path"
            )
        if warmup_forecasts is not None:
            if "log_score" in modes:
                raise ValueError(
                    "log_score weight regeneration requires warmup_losses with evaluated "
                    "log densities; prediction-only warmup forecasts support equal and QLIKE"
                )
            warmup_losses = warmup_losses_from_forecasts(
                warmup_forecasts,
                daily,
                eligibility_column=str(weighting["eligibility_column"]),
                availability_column=str(weighting["availability_column"]),
            )
        assert warmup_losses is not None
        if calendar_report is None:
            raise ValueError("weight regeneration requires a valid calendar_report")
        prepared_objectives = _prepare_objectives(
            losses,
            warmup_losses,
            candidates=candidates,
            markets=markets,
            horizons=horizons,
            objective_start=pd.Timestamp(sample["objective_start"]).normalize(),
            forecast_start=pd.Timestamp(sample["forecast_start"]).normalize(),
            eligibility_column=str(weighting["eligibility_column"]),
            availability_column=str(weighting["availability_column"]),
        )
        common_objectives = _common_objectives(
            prepared_objectives,
            candidates=candidates,
            eligibility_column=str(weighting["eligibility_column"]),
        )
        weights = _events_from_objectives(
            prepared_forecasts,
            common_objectives,
            candidates=candidates,
            modes=modes,
            session_indices=_session_indices(daily, markets, calendar_report),
            half_life=float(weighting["half_life_sessions"]),
            sample_policy=sample_policy,
        )
    ensemble = _ensemble_rows(
        prepared_forecasts,
        weights,
        candidates=candidates,
        modes=modes,
        predictive_root=predictive_root_path,
        predictive_output_dir=predictive_output_path,
        mixture_seed=mixture_seed,
    )
    weights = weights.loc[:, WEIGHT_COLUMNS].copy()
    for column in ("update_date",):
        weights[column] = pd.to_datetime(weights[column]).dt.strftime("%Y-%m-%d")
    for column in ("origin_date", "mature_date"):
        ensemble[column] = pd.to_datetime(ensemble[column]).dt.strftime("%Y-%m-%d")
    return ensemble, weights
