"""Independent exchange-calendar audit for the Oxford-Man daily archive."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_contract import read_daily_frame


MARKET_CALENDAR_MAP = {
    "SP500": "XNYS",
    "DAX": "XETR",
    "FTSE100": "XLON",
}


class CalendarAuditError(ValueError):
    """Raised when observed labels cannot be reconciled with exchange sessions."""


@dataclass(frozen=True)
class MarketCalendarAudit:
    market: str
    calendar: str
    first_date: str
    last_date: str
    observed_sessions: int
    expected_sessions: int
    coverage_fraction: float
    missing_expected_sessions: list[str]
    observed_non_sessions: list[str]
    source_label_mismatches: list[str]
    source_utc_offsets: dict[str, int]
    utc_conversion_date_shift_count: int
    missing_by_sample_segment: dict[str, int]


def _calendar_sessions(name: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise CalendarAuditError(
            "exchange-calendar audit requires the optional 'audit' dependencies"
        ) from exc
    calendar = xcals.get_calendar(
        name,
        start=first - pd.Timedelta(days=366),
        end=last + pd.Timedelta(days=366),
    )
    sessions = calendar.sessions_in_range(first, last)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def audit_exchange_calendars(
    data: str | Path | pd.DataFrame,
    *,
    market_calendar_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare every observed label with an independent exchange calendar.

    Missing expected sessions are documented but are not silently synthesized.
    Observed labels that are not exchange sessions, or source-label/date
    mismatches, make the audit fail.
    """
    frame = read_daily_frame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    mapping = MARKET_CALENDAR_MAP if market_calendar_map is None else market_calendar_map
    required = {"market", "date", "source_timestamp"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise CalendarAuditError(
            f"calendar audit requires columns: {sorted(missing_columns)}"
        )
    observed_markets = set(frame["market"].astype(str))
    if observed_markets != set(mapping):
        raise CalendarAuditError(
            f"market/calendar map mismatch: observed={sorted(observed_markets)} "
            f"configured={sorted(mapping)}"
        )

    market_reports: list[MarketCalendarAudit] = []
    all_observed_non_sessions: list[str] = []
    all_label_mismatches: list[str] = []
    for market, calendar_name in mapping.items():
        group = frame.loc[frame["market"] == market].copy()
        group["date"] = pd.to_datetime(group["date"], errors="raise").dt.normalize()
        group = group.sort_values("date")
        observed = pd.DatetimeIndex(group["date"])
        if observed.has_duplicates:
            raise CalendarAuditError(f"duplicate observed dates for {market}")
        expected = _calendar_sessions(calendar_name, observed.min(), observed.max())
        missing_expected = expected.difference(observed)
        observed_non_sessions = observed.difference(expected)

        source_text = group["source_timestamp"].astype(str)
        source_labels = pd.to_datetime(
            source_text.str.slice(0, 10),
            format="%Y-%m-%d",
            errors="raise",
        )
        mismatch_mask = source_labels.to_numpy() != group["date"].to_numpy()
        label_mismatches = [
            f"{market}:{source_text.iloc[i]}->{group['date'].iloc[i].date()}"
            for i in range(len(group))
            if bool(mismatch_mask[i])
        ]
        parsed_utc = pd.to_datetime(source_text, utc=True, errors="raise")
        utc_labels = (
            parsed_utc.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        )
        utc_shift_count = int(
            (utc_labels.to_numpy() != source_labels.to_numpy()).sum()
        )
        offset_counts = {
            str(offset): int(count)
            for offset, count in source_text.str.slice(-6).value_counts().sort_index().items()
        }
        missing_strings = [str(item.date()) for item in missing_expected]
        missing_by_segment = {
            "estimation_through_2016": int((missing_expected <= "2016-12-31").sum()),
            "calibration_2017": int(
                ((missing_expected >= "2017-01-01") & (missing_expected <= "2017-12-31")).sum()
            ),
            "evaluation_2018_to_2022_02_25": int(
                ((missing_expected >= "2018-01-01") & (missing_expected <= "2022-02-25")).sum()
            ),
        }
        report = MarketCalendarAudit(
            market=market,
            calendar=calendar_name,
            first_date=str(observed.min().date()),
            last_date=str(observed.max().date()),
            observed_sessions=int(len(observed)),
            expected_sessions=int(len(expected)),
            coverage_fraction=float(len(observed) / len(expected)),
            missing_expected_sessions=missing_strings,
            observed_non_sessions=[str(item.date()) for item in observed_non_sessions],
            source_label_mismatches=label_mismatches,
            source_utc_offsets=offset_counts,
            utc_conversion_date_shift_count=utc_shift_count,
            missing_by_sample_segment=missing_by_segment,
        )
        market_reports.append(report)
        all_observed_non_sessions.extend(
            f"{market}:{item.date()}" for item in observed_non_sessions
        )
        all_label_mismatches.extend(label_mismatches)

    return {
        "calendar_library": "exchange_calendars",
        "date_policy": (
            "Oxford-Man source timestamp first 10 characters define the "
            "session label; UTC conversion is audited but never used to relabel sessions"
        ),
        "valid": not all_observed_non_sessions and not all_label_mismatches,
        "observed_non_sessions": all_observed_non_sessions,
        "source_label_mismatches": all_label_mismatches,
        "markets": [asdict(item) for item in market_reports],
    }
