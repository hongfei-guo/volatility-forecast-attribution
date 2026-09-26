"""Build the daily empirical input from the Oxford-Man Realized Library."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .data_contract import (
    FULL_DAY_PROXY_PAIRS,
    OMI_SOURCE_PROXY_COLUMNS,
    ValidationReport,
    validate_daily_frame,
)
from .file_identity import sha256_file


SOURCE_DATE_COLUMN = "Unnamed: 0"
MARKET_SYMBOL_MAP = {
    ".SPX": "SP500",
    ".GDAXI": "DAX",
    ".FTSE": "FTSE100",
}

SOURCE_REQUIRED_COLUMNS = (
    SOURCE_DATE_COLUMN,
    "Symbol",
    "rv5_ss",
    "close_time",
    "rsv_ss",
    "open_time",
    "rk_parzen",
    "rv5",
    "rv10_ss",
    "bv",
    "close_price",
    "rv10",
    "open_price",
    "rk_twoscale",
    "medrv",
    "bv_ss",
    "open_to_close",
    "nobs",
    "rsv",
    "rk_th2",
)

CANONICAL_ALIAS_MAP = {
    "rv5": "rv_oc",
    "rv5_ss": "rv5_ss_oc",
    "rv10": "rv10_oc",
    "rv10_ss": "rv10_ss_oc",
    "bv": "bv_oc",
    "bv_ss": "bv_ss_oc",
    "medrv": "medrv_oc",
    "rk_parzen": "rk_parzen_oc",
    "rk_twoscale": "rk_twoscale_oc",
    "rk_th2": "rk_th2_oc",
    "rsv": "rv_down",
    "rsv_ss": "rv_down_ss",
}


class OMIBuilderError(ValueError):
    """Raised when the source release cannot be transformed without repair."""


@dataclass(frozen=True)
class MarketAudit:
    market: str
    source_symbol: str
    source_rows: int
    output_rows: int
    first_source_date: str
    first_output_date: str
    last_output_date: str
    dropped_initial_rows: int
    duplicate_source_dates: int
    weekend_source_dates: int
    utc_conversion_date_shift_count: int
    maximum_calendar_gap_days: int


@dataclass(frozen=True)
class OMIBuildAudit:
    source_sha256: str
    source_rows_all_markets: int
    source_rows_selected_markets: int
    output_rows: int
    date_policy: str
    max_open_to_close_identity_error: float
    max_close_to_close_identity_error: float
    validation: dict[str, Any]
    markets: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_date_labels(values: pd.Series) -> pd.Series:
    """Use the source's calendar label, not its UTC-converted calendar date."""
    raw = values.astype(str)
    labels = raw.str.slice(0, 10)
    if (~labels.str.fullmatch(r"\d{4}-\d{2}-\d{2}")).any():
        raise OMIBuilderError("source timestamps do not start with YYYY-MM-DD")
    return pd.to_datetime(labels, format="%Y-%m-%d", errors="raise")


def _utc_conversion_shift_count(values: pd.Series, labels: pd.Series) -> int:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    utc_labels = parsed.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    return int(np.sum(utc_labels.to_numpy() != labels.dt.normalize().to_numpy()))


def build_omi_daily_frame(
    source: pd.DataFrame,
    *,
    source_sha256: str,
    market_symbol_map: dict[str, str] | None = None,
    atol: float = 1e-10,
) -> tuple[pd.DataFrame, OMIBuildAudit]:
    """Build the three-market daily contract without imputing or repairing rows."""
    mapping = MARKET_SYMBOL_MAP if market_symbol_map is None else market_symbol_map
    missing = set(SOURCE_REQUIRED_COLUMNS) - set(source.columns)
    if missing:
        raise OMIBuilderError(f"Oxford-Man source is missing columns: {sorted(missing)}")
    if not mapping:
        raise OMIBuilderError("market-symbol map cannot be empty")

    selected = source[source["Symbol"].isin(mapping)].copy()
    observed_symbols = set(selected["Symbol"].astype(str))
    missing_symbols = set(mapping) - observed_symbols
    if missing_symbols:
        raise OMIBuilderError(
            f"Oxford-Man source is missing requested symbols: {sorted(missing_symbols)}"
        )
    selected["date"] = _source_date_labels(selected[SOURCE_DATE_COLUMN])
    selected["market"] = selected["Symbol"].map(mapping)

    numeric_source = [
        column
        for column in SOURCE_REQUIRED_COLUMNS
        if column not in {SOURCE_DATE_COLUMN, "Symbol"}
    ]
    for column in numeric_source:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        values = selected[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            raise OMIBuilderError(f"source column {column} contains non-finite values")
    if (selected[["open_price", "close_price"]] <= 0).any().any():
        raise OMIBuilderError("open and close prices must be strictly positive")
    if (selected[list(OMI_SOURCE_PROXY_COLUMNS)] < 0).any().any():
        raise OMIBuilderError("Oxford-Man realised proxy fields must be non-negative")

    selected = selected.sort_values(["market", "date"]).reset_index(drop=True)
    market_audits: list[MarketAudit] = []
    output_groups: list[pd.DataFrame] = []
    max_open_close_error = 0.0
    max_close_close_error = 0.0

    for source_symbol, market in mapping.items():
        group = selected[selected["market"] == market].copy()
        duplicate_count = int(group["date"].duplicated().sum())
        if duplicate_count:
            raise OMIBuilderError(f"duplicate source trading date for {market}")
        if not group["date"].is_monotonic_increasing:
            raise OMIBuilderError(f"source dates are not increasing for {market}")
        weekend_count = int(np.sum(group["date"].dt.dayofweek >= 5))
        if weekend_count:
            raise OMIBuilderError(f"weekend trading labels found for {market}")

        source_timestamp = group[SOURCE_DATE_COLUMN].astype(str).copy()
        previous_close = group["close_price"].shift(1)
        group["return_overnight"] = np.log(group["open_price"] / previous_close)
        calculated_oc = np.log(group["close_price"] / group["open_price"])
        open_close_error = np.abs(calculated_oc - group["open_to_close"])
        max_open_close_error = max(max_open_close_error, float(open_close_error.max()))
        if float(open_close_error.max()) > atol:
            raise OMIBuilderError(f"open-to-close return identity failed for {market}")
        group["return_oc"] = group["open_to_close"]
        group["return_cc"] = np.log(group["close_price"] / previous_close)
        close_close_error = np.abs(
            group["return_cc"] - (group["return_overnight"] + group["return_oc"])
        )
        max_close_close_error = max(
            max_close_close_error, float(close_close_error.dropna().max())
        )
        if float(close_close_error.dropna().max()) > atol:
            raise OMIBuilderError(f"close-to-close return identity failed for {market}")

        for source_name, canonical_name in CANONICAL_ALIAS_MAP.items():
            group[canonical_name] = group[source_name]
        group["rv_up"] = group["rv_oc"] - group["rv_down"]
        if (group["rv_up"] < -atol).any():
            raise OMIBuilderError(f"downside semivariance exceeds RV for {market}")
        group["rv_up"] = group["rv_up"].clip(lower=0.0)
        group["asymmetry"] = (group["rv_down"] - group["rv_up"]) / (
            group["rv_down"] + group["rv_up"] + 1e-12
        )
        overnight_square = group["return_overnight"] ** 2
        for full_day, intraday in FULL_DAY_PROXY_PAIRS.items():
            group[full_day] = group[intraday] + overnight_square

        group["source_symbol"] = source_symbol
        group["source_timestamp"] = source_timestamp
        valid = group["return_overnight"].notna()
        if int((~valid).sum()) != 1 or not bool((~valid).iloc[0]):
            raise OMIBuilderError(
                f"expected exactly one initial missing previous close for {market}"
            )
        output_group = group.loc[valid].copy()
        output_groups.append(output_group)
        gaps = group["date"].diff().dt.days.dropna()
        market_audits.append(
            MarketAudit(
                market=market,
                source_symbol=source_symbol,
                source_rows=int(len(group)),
                output_rows=int(len(output_group)),
                first_source_date=str(group["date"].iloc[0].date()),
                first_output_date=str(output_group["date"].iloc[0].date()),
                last_output_date=str(output_group["date"].iloc[-1].date()),
                dropped_initial_rows=1,
                duplicate_source_dates=duplicate_count,
                weekend_source_dates=weekend_count,
                utc_conversion_date_shift_count=_utc_conversion_shift_count(
                    group[SOURCE_DATE_COLUMN], group["date"]
                ),
                maximum_calendar_gap_days=int(gaps.max()),
            )
        )

    output = pd.concat(output_groups, ignore_index=True)
    ordered_columns = [
        "date",
        "market",
        "return_cc",
        "return_overnight",
        "return_oc",
        "rv_oc",
        "rv_cc",
        "rv_down",
        "rv_up",
        "asymmetry",
        *OMI_SOURCE_PROXY_COLUMNS,
        *[
            value
            for value in CANONICAL_ALIAS_MAP.values()
            if value not in {"rv_oc", "rv_down"}
        ],
        *[
            value
            for value in FULL_DAY_PROXY_PAIRS
            if value != "rv_cc"
        ],
        "source_symbol",
        "source_timestamp",
        "open_price",
        "close_price",
        "open_time",
        "close_time",
        "nobs",
    ]
    if len(ordered_columns) != len(set(ordered_columns)):
        raise OMIBuilderError("output column order contains duplicates")
    output = output[ordered_columns].sort_values(["market", "date"]).reset_index(drop=True)
    validation: ValidationReport = validate_daily_frame(output, atol=atol)
    audit = OMIBuildAudit(
        source_sha256=source_sha256,
        source_rows_all_markets=int(len(source)),
        source_rows_selected_markets=int(len(selected)),
        output_rows=int(len(output)),
        date_policy=(
            "source_timestamp_first_10_characters; do not convert the session "
            "label to UTC before extracting the trading date"
        ),
        max_open_to_close_identity_error=max_open_close_error,
        max_close_to_close_identity_error=max_close_close_error,
        validation=validation.to_dict(),
        markets=[asdict(item) for item in market_audits],
    )
    return output, audit


def read_omi_source(path: str | Path) -> pd.DataFrame:
    """Read only the columns required by the daily-data builder."""
    source = Path(path)
    if not source.is_file():
        raise OMIBuilderError(f"Oxford-Man source file does not exist: {source}")
    return pd.read_csv(source, usecols=list(SOURCE_REQUIRED_COLUMNS))


def _atomic_parquet_write(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp.parquet",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, destination)
    except ImportError as exc:
        raise OMIBuilderError(
            "writing the Parquet archive requires pyarrow"
        ) from exc
    finally:
        if "temporary" in locals() and temporary.exists():
            temporary.unlink()


def build_omi_archive(
    *,
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Build the daily Parquet archive and one validation report."""
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    report = Path(report_path).resolve()
    if len({source, output, report}) != 3:
        raise OMIBuilderError("source, output, and report paths must differ")
    source_hash = sha256_file(source)
    daily, audit = build_omi_daily_frame(
        read_omi_source(source), source_sha256=source_hash
    )
    _atomic_parquet_write(daily, output)
    reloaded = pd.read_parquet(output)
    readback_validation = validate_daily_frame(reloaded).to_dict()
    if len(reloaded) != len(daily) or list(reloaded.columns) != list(daily.columns):
        raise OMIBuilderError("Parquet read-back changed rows or columns")

    payload = {
        "source": "Oxford-Man Institute Realized Library",
        "source_sha256": source_hash,
        "market_symbol_map": MARKET_SYMBOL_MAP,
        "date_policy": audit.date_policy,
        "column_aliases": CANONICAL_ALIAS_MAP,
        "full_day_proxy_pairs": FULL_DAY_PROXY_PAIRS,
        "output": {
            "path": os.path.relpath(output, report.parent),
            "size": output.stat().st_size,
        },
        "rows": int(len(daily)),
        "columns": list(daily.columns),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "validation": readback_validation,
        "markets": audit.markets,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
