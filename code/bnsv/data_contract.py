"""Definitions and checks for the daily empirical data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE_COLUMNS = (
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
)

# Preserve both the Oxford-Man release names and canonical intraday/full-day
# aliases.  The full-day variants add the same overnight squared return used by
# the main close-to-close target, so changing evaluation proxy never requires
# refitting or regenerating forecast paths.
OMI_SOURCE_PROXY_COLUMNS = (
    "rv5",
    "rv5_ss",
    "rv10",
    "rv10_ss",
    "rsv",
    "rsv_ss",
    "bv",
    "bv_ss",
    "medrv",
    "rk_parzen",
    "rk_twoscale",
    "rk_th2",
)

CANONICAL_INTRADAY_PROXY_COLUMNS = (
    "rv5_ss_oc",
    "rv10_oc",
    "rv10_ss_oc",
    "bv_oc",
    "bv_ss_oc",
    "medrv_oc",
    "rk_parzen_oc",
    "rk_twoscale_oc",
    "rk_th2_oc",
    "rv_down_ss",
)

FULL_DAY_PROXY_PAIRS = {
    "rv_cc": "rv_oc",
    "rv5_ss_cc": "rv5_ss_oc",
    "rv10_cc": "rv10_oc",
    "rv10_ss_cc": "rv10_ss_oc",
    "bv_cc": "bv_oc",
    "bv_ss_cc": "bv_ss_oc",
    "medrv_cc": "medrv_oc",
    "rk_parzen_cc": "rk_parzen_oc",
    "rk_twoscale_cc": "rk_twoscale_oc",
    "rk_th2_cc": "rk_th2_oc",
}

EVALUATION_PROXY_COLUMNS = tuple(FULL_DAY_PROXY_PAIRS)

SOURCE_METADATA_COLUMNS = (
    "source_symbol",
    "source_timestamp",
    "open_price",
    "close_price",
    "open_time",
    "close_time",
    "nobs",
)

REQUIRED_COLUMNS = (
    *CORE_COLUMNS,
    *OMI_SOURCE_PROXY_COLUMNS,
    *CANONICAL_INTRADAY_PROXY_COLUMNS,
    *tuple(
        column for column in EVALUATION_PROXY_COLUMNS if column not in CORE_COLUMNS
    ),
    *SOURCE_METADATA_COLUMNS,
)

NUMERIC_COLUMNS = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column not in {"date", "market", "source_symbol", "source_timestamp"}
)

NONNEGATIVE_COLUMNS = (
    "rv_oc",
    "rv_cc",
    "rv_down",
    "rv_up",
    *OMI_SOURCE_PROXY_COLUMNS,
    *CANONICAL_INTRADAY_PROXY_COLUMNS,
    *tuple(
        column for column in EVALUATION_PROXY_COLUMNS if column not in CORE_COLUMNS
    ),
)

ALLOWED_MARKETS = {"SP500", "DAX", "FTSE100"}


class DataContractError(ValueError):
    """Raised when the daily data fail the stated definitions."""


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    markets: list[str]
    first_date: str
    last_date: str
    max_rv_identity_error: float
    max_semivariance_identity_error: float
    max_asymmetry_identity_error: float
    max_alternative_proxy_identity_error: float
    evaluation_proxy_columns: list[str]
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_daily_frame(path: str | Path) -> pd.DataFrame:
    """Read a daily input archive without changing its values."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(source)
        except ImportError as exc:
            raise DataContractError(
                "reading the Parquet archive requires pyarrow"
            ) from exc
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(source)
    else:
        raise DataContractError(
            f"unsupported daily-data format {source.suffix!r}; use Parquet or CSV"
        )
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def validate_daily_frame(
    frame: pd.DataFrame,
    *,
    atol: float = 1e-10,
    allowed_markets: set[str] | None = None,
) -> ValidationReport:
    """Validate values, dates, and algebraic identities without repairing data."""
    allowed = ALLOWED_MARKETS if allowed_markets is None else allowed_markets
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataContractError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        raise DataContractError("data frame is empty")

    df = frame.copy()
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:
        raise DataContractError("date column is not parseable") from exc
    if df[list(NUMERIC_COLUMNS)].isna().any().any():
        raise DataContractError("numeric contract columns contain missing values")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise DataContractError(f"{col} contains non-finite values")
    for col in ("market", "source_symbol", "source_timestamp"):
        if df[col].isna().any() or (df[col].astype(str).str.len() == 0).any():
            raise DataContractError(f"{col} contains missing or empty values")

    observed_markets = set(df["market"].astype(str))
    unknown = observed_markets - allowed
    if unknown:
        raise DataContractError(f"unknown market IDs: {sorted(unknown)}")

    if (df[list(NONNEGATIVE_COLUMNS)] < 0).any().any():
        raise DataContractError("variance, proxy, and semivariance fields must be non-negative")
    if ((df["asymmetry"] < -1 - atol) | (df["asymmetry"] > 1 + atol)).any():
        raise DataContractError("asymmetry lies outside [-1,1]")

    for market, group in df.groupby("market", sort=True):
        dates = group["date"]
        if dates.duplicated().any():
            raise DataContractError(f"duplicate date within {market}")
        if not dates.is_monotonic_increasing:
            raise DataContractError(f"dates are not strictly increasing within {market}")

    rv_expected = df["return_overnight"] ** 2 + df["rv_oc"]
    rv_error = np.abs(df["rv_cc"] - rv_expected)
    semi_error = np.abs(df["rv_oc"] - (df["rv_down"] + df["rv_up"]))
    asym_expected = (df["rv_down"] - df["rv_up"]) / (
        df["rv_down"] + df["rv_up"] + 1e-12
    )
    asym_error = np.abs(df["asymmetry"] - asym_expected)
    if float(rv_error.max()) > atol:
        raise DataContractError("rv_cc identity failed")
    if float(semi_error.max()) > atol:
        raise DataContractError("semivariance identity failed")
    if float(asym_error.max()) > max(atol, 1e-9):
        raise DataContractError("asymmetry identity failed")

    overnight_square = df["return_overnight"] ** 2
    alternative_errors: list[np.ndarray] = []
    for full_day, intraday in FULL_DAY_PROXY_PAIRS.items():
        error = np.abs(df[full_day] - (df[intraday] + overnight_square))
        alternative_errors.append(error.to_numpy(dtype=float))
        if float(error.max()) > atol:
            raise DataContractError(
                f"full-day alternative proxy identity failed: {full_day}"
            )
    max_alternative_error = float(
        max(np.max(value) for value in alternative_errors)
    )

    return ValidationReport(
        rows=int(len(df)),
        markets=sorted(observed_markets),
        first_date=str(df["date"].min().date()),
        last_date=str(df["date"].max().date()),
        max_rv_identity_error=float(rv_error.max()),
        max_semivariance_identity_error=float(semi_error.max()),
        max_asymmetry_identity_error=float(asym_error.max()),
        max_alternative_proxy_identity_error=max_alternative_error,
        evaluation_proxy_columns=list(EVALUATION_PROXY_COLUMNS),
        valid=True,
    )


def horizon_targets(dates: pd.Series | pd.DatetimeIndex, horizons: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise DataContractError("dates must be sorted and unique")
    records: list[dict[str, Any]] = []
    for i, origin in enumerate(idx):
        for h in horizons:
            if h <= 0:
                raise DataContractError("horizons must be positive")
            if i + h < len(idx):
                records.append({
                    "origin_date": origin,
                    "horizon": h,
                    "target_start": idx[i + 1],
                    "target_end": idx[i + h],
                    "mature_date": idx[i + h],
                })
    return pd.DataFrame.from_records(records)
