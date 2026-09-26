#!/usr/bin/env python3
"""Evaluate packaged variance forecasts against lawfully obtained daily data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bnsv.data_contract import (
    EVALUATION_PROXY_COLUMNS,
    read_daily_frame,
    validate_daily_frame,
)
from bnsv.evaluation import qlike, variance_mse
from bnsv.sample_alignment import (
    gap_metadata,
    load_calendar_audit,
    missing_sessions_by_market,
    write_loss_archive,
)


FORECAST_COLUMNS = {
    "market",
    "model_id",
    "origin_date",
    "horizon",
    "mature_date",
    "forecast_kind",
    "variance_forecast",
    "variance_mean_method",
    "return_distribution",
}


def _read_forecasts(paths: list[str]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        path = Path(raw)
        frame = pd.read_parquet(path)
        missing = FORECAST_COLUMNS - set(frame)
        if missing:
            raise ValueError(f"forecast panel {path} lacks columns: {sorted(missing)}")
        frame = frame[list(sorted(FORECAST_COLUMNS))].copy()
        frames.append(frame)
    forecasts = pd.concat(frames, ignore_index=True)
    keys = ["market", "model_id", "origin_date", "horizon"]
    if forecasts.empty or forecasts.duplicated(keys).any():
        raise ValueError("forecast panels are empty or contain duplicate model origins")
    forecasts["origin_date"] = pd.to_datetime(
        forecasts["origin_date"], errors="raise"
    ).dt.normalize()
    forecasts["mature_date"] = pd.to_datetime(
        forecasts["mature_date"], errors="raise"
    ).dt.normalize()
    forecasts["horizon"] = pd.to_numeric(
        forecasts["horizon"], errors="raise"
    ).astype(int)
    if not set(forecasts["horizon"]).issubset({1, 5, 10}):
        raise ValueError("forecast horizons must be 1, 5, or 10 sessions")
    variance = pd.to_numeric(forecasts["variance_forecast"], errors="raise")
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance forecasts must be finite and strictly positive")
    forecasts["variance_forecast"] = variance
    return forecasts.sort_values(keys).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--forecasts", nargs="+", required=True)
    parser.add_argument("--calendar-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = read_daily_frame(args.data)
    validate_daily_frame(data)
    calendar = load_calendar_audit(args.calendar_report)
    missing_sessions = missing_sessions_by_market(calendar)
    forecasts = _read_forecasts(args.forecasts)
    if set(forecasts["market"].astype(str)) != set(data["market"].astype(str)):
        raise ValueError("forecast and daily-data market sets differ")

    daily = {
        str(market): group.sort_values("date").reset_index(drop=True)
        for market, group in data.groupby("market", sort=False)
    }
    positions = {
        market: {pd.Timestamp(date).normalize(): i for i, date in enumerate(group["date"])}
        for market, group in daily.items()
    }
    target_cache: dict[tuple[str, pd.Timestamp, int], dict[str, object]] = {}
    rows = []
    for record in forecasts.to_dict("records"):
        market = str(record["market"])
        origin = pd.Timestamp(record["origin_date"]).normalize()
        horizon = int(record["horizon"])
        key = (market, origin, horizon)
        if key not in target_cache:
            if origin not in positions[market]:
                raise ValueError(f"forecast origin is absent from daily data: {key}")
            position = positions[market][origin]
            target = daily[market].iloc[position + 1 : position + horizon + 1]
            if len(target) != horizon:
                raise ValueError(f"incomplete target window: {key}")
            mature = pd.Timestamp(target["date"].iloc[-1]).normalize()
            if mature != pd.Timestamp(record["mature_date"]).normalize():
                raise ValueError(f"forecast mature date and daily data differ: {key}")
            target_dates = [str(pd.Timestamp(value).date()) for value in target["date"]]
            fields: dict[str, object] = {
                **gap_metadata(
                    market=market,
                    origin_date=origin,
                    target_dates=target_dates,
                    previous_observed_date=(
                        daily[market].iloc[position - 1]["date"]
                        if position > 0
                        else None
                    ),
                    missing_by_market=missing_sessions,
                ),
                "target_start": target_dates[0],
                "target_end": target_dates[-1],
                "mature_date": target_dates[-1],
                "target_dates_json": json.dumps(target_dates, separators=(",", ":")),
            }
            for proxy in EVALUATION_PROXY_COLUMNS:
                values = target[proxy].to_numpy(float)
                if np.any(~np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"invalid realised variance proxy: {key} {proxy}")
                fields[f"realized_variance__{proxy}"] = float(values.sum())
            target_cache[key] = fields
        fields = target_cache[key]
        forecast = float(record["variance_forecast"])
        row = {
            **record,
            **fields,
            "origin_date": str(origin.date()),
        }
        for proxy in EVALUATION_PROXY_COLUMNS:
            realized = float(fields[f"realized_variance__{proxy}"])
            row[f"qlike__{proxy}"] = float(qlike(realized, forecast))
            row[f"variance_mse__{proxy}"] = float(variance_mse(realized, forecast))
        row["realized_variance"] = row["realized_variance__rv_cc"]
        row["qlike"] = row["qlike__rv_cc"]
        row["variance_mse"] = row["variance_mse__rv_cc"]
        rows.append(row)

    output = pd.DataFrame(rows)
    write_loss_archive(output, args.output)
    print(f"Wrote {len(output)} variance-loss rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
