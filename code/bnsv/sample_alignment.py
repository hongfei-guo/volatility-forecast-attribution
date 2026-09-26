"""Exchange-session gaps and the evaluation sample.

The Oxford-Man archive is the value authority and is never imputed.  The
independent exchange-calendar audit identifies expected sessions absent from
that archive.  Loss rows retain all observed targets, but main evaluation
excludes horizons that span at least one such missing session.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


GAP_POLICY = "exchange_calendar_gap_exclusion"
EVALUATION_SAMPLE_POLICY = "main_gap_excluded"


def read_loss_archive(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError("loss archive must be CSV or Parquet")


def write_loss_archive(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(
            destination, index=False, engine="pyarrow", compression="zstd"
        )
    elif destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError("loss archive output must be Parquet or CSV")


def load_calendar_audit(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"exchange-calendar audit is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        raise ValueError("exchange-calendar audit must be a valid audit object")
    if payload.get("observed_non_sessions"):
        raise ValueError("calendar audit contains observed non-sessions")
    if payload.get("source_label_mismatches"):
        raise ValueError("calendar audit contains source-label mismatches")
    markets = payload.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("calendar audit contains no market reports")
    names = [str(item.get("market", "")) for item in markets]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("calendar audit market reports must have unique names")
    for item in markets:
        missing = pd.DatetimeIndex(
            pd.to_datetime(item.get("missing_expected_sessions", []), errors="raise")
        ).normalize()
        if missing.has_duplicates or not missing.is_monotonic_increasing:
            raise ValueError(
                f"missing sessions are not unique and sorted for {item['market']}"
            )
    return payload


def missing_sessions_by_market(
    calendar_audit: Mapping[str, Any],
) -> dict[str, pd.DatetimeIndex]:
    return {
        str(item["market"]): pd.DatetimeIndex(
            pd.to_datetime(item.get("missing_expected_sessions", []), errors="raise")
        ).normalize()
        for item in calendar_audit["markets"]
    }


def _normal_date(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def strict_bool_series(values: pd.Series, *, name: str) -> pd.Series:
    """Parse a persisted boolean column without Python truthiness surprises."""
    series = pd.Series(values, index=values.index, name=values.name)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    if series.isna().any():
        raise ValueError(f"boolean column {name!r} contains missing values")
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"boolean column {name!r} contains values outside 0/1")
        return numeric.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(
            f"boolean column {name!r} contains unrecognized values: {unknown}"
        )
    return normalized.map(mapping).astype(bool)


def gap_metadata(
    *,
    market: str,
    origin_date: object,
    target_dates: Sequence[object],
    previous_observed_date: object | None,
    missing_by_market: Mapping[str, pd.DatetimeIndex],
) -> dict[str, object]:
    """Return gap flags for one origin and forecast target."""
    if market not in missing_by_market:
        raise ValueError(f"market {market!r} is absent from the calendar audit")
    origin = _normal_date(origin_date)
    targets = pd.DatetimeIndex([_normal_date(value) for value in target_dates])
    if len(targets) == 0:
        raise ValueError("target dates cannot be empty")
    if targets.has_duplicates or not targets.is_monotonic_increasing:
        raise ValueError("target dates must be unique and strictly increasing")
    if targets[0] <= origin:
        raise ValueError("target dates must follow the forecast origin")
    missing = missing_by_market[market]
    target_missing = missing[(missing > origin) & (missing <= targets[-1])]

    if previous_observed_date is None:
        prior_missing = pd.DatetimeIndex([])
    else:
        previous = _normal_date(previous_observed_date)
        if previous >= origin:
            raise ValueError("previous observed date must precede the origin")
        prior_missing = missing[(missing > previous) & (missing <= origin)]

    target_missing_strings = [str(value.date()) for value in target_missing]
    prior_missing_strings = [str(value.date()) for value in prior_missing]
    spans_target_gap = bool(target_missing_strings)
    follows_origin_gap = bool(prior_missing_strings)
    return {
        "gap_policy": GAP_POLICY,
        "target_spans_archive_gap": spans_target_gap,
        "target_missing_expected_session_count": len(target_missing_strings),
        "target_missing_expected_sessions_json": json.dumps(
            target_missing_strings, separators=(",", ":")
        ),
        "target_expected_session_span": int(len(targets) + len(target_missing_strings)),
        "origin_follows_archive_gap": follows_origin_gap,
        "origin_prior_missing_expected_session_count": len(prior_missing_strings),
        "origin_prior_missing_expected_sessions_json": json.dumps(
            prior_missing_strings, separators=(",", ":")
        ),
        "main_evaluation_eligible": not spans_target_gap,
        "evaluation_sample_policy": "exclude_targets_spanning_missing_sessions",
    }


def annotate_loss_table(
    losses: pd.DataFrame,
    *,
    calendar_audit: Mapping[str, Any],
    observed_dates_by_market: Mapping[str, Sequence[object]],
) -> pd.DataFrame:
    """Add exchange-session gap fields to a loss table."""
    required = {
        "market",
        "origin_date",
        "horizon",
        "target_start",
        "target_end",
        "mature_date",
        "target_dates_json",
    }
    missing_columns = required - set(losses.columns)
    if missing_columns:
        raise ValueError(
            f"loss archive lacks gap-annotation fields: {sorted(missing_columns)}"
        )
    missing_map = missing_sessions_by_market(calendar_audit)
    observed_maps: dict[str, tuple[pd.DatetimeIndex, dict[pd.Timestamp, int]]] = {}
    for market, values in observed_dates_by_market.items():
        dates = pd.DatetimeIndex([_normal_date(value) for value in values])
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"observed dates are not unique and sorted for {market}")
        observed_maps[str(market)] = (dates, {date: i for i, date in enumerate(dates)})

    annotations: list[dict[str, object]] = []
    for row in losses.itertuples(index=False):
        market = str(row.market)
        if market not in observed_maps:
            raise ValueError(f"market {market!r} is absent from evaluation data")
        origin = _normal_date(row.origin_date)
        observed, positions = observed_maps[market]
        if origin not in positions:
            raise ValueError(f"origin {origin.date()} is absent from evaluation data")
        position = positions[origin]
        previous = observed[position - 1] if position > 0 else None
        horizon = int(row.horizon)
        if horizon <= 0:
            raise ValueError("loss archive horizon must be positive")
        target_dates = json.loads(str(row.target_dates_json))
        if not isinstance(target_dates, list):
            raise ValueError("target_dates_json must encode a list")
        expected_targets = observed[position + 1 : position + horizon + 1]
        if len(expected_targets) != horizon:
            raise ValueError(
                f"origin {origin.date()} lacks a complete horizon-{horizon} target"
            )
        stored_targets = pd.DatetimeIndex(
            [_normal_date(value) for value in target_dates]
        )
        if len(stored_targets) != horizon or not np.array_equal(
            stored_targets.to_numpy(), expected_targets.to_numpy()
        ):
            raise ValueError(
                "stored target dates do not equal the next horizon observed "
                f"dates for {market} at {origin.date()}"
            )
        expected_start = expected_targets[0]
        expected_end = expected_targets[-1]
        for column, value, expected in (
            ("target_start", row.target_start, expected_start),
            ("target_end", row.target_end, expected_end),
            ("mature_date", row.mature_date, expected_end),
        ):
            if _normal_date(value) != expected:
                raise ValueError(
                    f"stored {column} does not match the observed target sequence"
                )
        annotation = gap_metadata(
            market=market,
            origin_date=origin,
            target_dates=stored_targets,
            previous_observed_date=previous,
            missing_by_market=missing_map,
        )
        annotations.append(annotation)
    annotation_frame = pd.DataFrame(annotations, index=losses.index)
    overlap = set(annotation_frame.columns) & set(losses.columns)
    result = losses.drop(columns=sorted(overlap)).copy()
    for column in annotation_frame.columns:
        result[column] = annotation_frame[column]
    return result


def select_evaluation_sample(
    losses: pd.DataFrame, policy: str = "main_gap_excluded"
) -> pd.DataFrame:
    """Exclude targets that span a missing expected exchange session."""
    if policy != EVALUATION_SAMPLE_POLICY:
        raise ValueError(f"unsupported evaluation sample {policy!r}")
    required = {
        "target_spans_archive_gap",
        "main_evaluation_eligible",
    }
    missing = required - set(losses.columns)
    if missing:
        raise ValueError(
            "loss archive has not been exchange-gap annotated "
            f"(missing {sorted(missing)})"
        )
    if "gap_policy" not in losses:
        raise ValueError("loss archive lacks a gap-policy identity")
    if not (losses["gap_policy"].astype(str) == GAP_POLICY).all():
        raise ValueError("loss archive uses an unsupported gap policy")
    target_gap = strict_bool_series(
        losses["target_spans_archive_gap"], name="target_spans_archive_gap"
    )
    main_eligible = strict_bool_series(
        losses["main_evaluation_eligible"], name="main_evaluation_eligible"
    )
    if not (main_eligible == ~target_gap).all():
        raise ValueError("main_evaluation_eligible contradicts target gap flags")
    selected = losses[main_eligible]
    if selected.empty:
        raise ValueError(f"evaluation sample {policy!r} is empty")
    return selected.copy()
