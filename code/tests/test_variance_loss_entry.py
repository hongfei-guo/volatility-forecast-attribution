from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from bnsv.evaluation import qlike


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_forecast_to_variance_loss_panel(
    tmp_path: Path, synthetic_frame: pd.DataFrame, monkeypatch
) -> None:
    markets = {"SP500": ".SPX", "FTSE100": ".FTSE", "DAX": ".GDAXI"}
    daily_frames = []
    for market, symbol in markets.items():
        frame = synthetic_frame.copy()
        frame["market"] = market
        frame["source_symbol"] = symbol
        daily_frames.append(frame)
    daily = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["market", "date"]
    )
    data_path = tmp_path / "daily.parquet"
    daily.to_parquet(data_path, index=False)
    calendar_path = tmp_path / "calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "valid": True,
                "observed_non_sessions": [],
                "source_label_mismatches": [],
                "markets": [
                    {"market": market, "missing_expected_sessions": []}
                    for market in markets
                ],
            }
        ),
        encoding="utf-8",
    )
    origin = synthetic_frame["date"].iloc[20]
    mature = synthetic_frame["date"].iloc[21]
    forecast = 0.0002
    forecasts = pd.DataFrame(
        [
            {
                "market": market,
                "model_id": "NN-SV",
                "origin_date": str(origin.date()),
                "horizon": 1,
                "mature_date": str(mature.date()),
                "forecast_kind": "probabilistic",
                "variance_forecast": forecast,
                "variance_mean_method": "predictive_mean",
                "return_distribution": "variance_standardized_student_t",
            }
            for market in markets
        ]
    )
    forecast_path = tmp_path / "forecasts.parquet"
    forecasts.to_parquet(forecast_path, index=False)
    output_path = tmp_path / "losses.parquet"

    script = ROOT / "scripts" / "build_variance_losses.py"
    spec = importlib.util.spec_from_file_location("variance_losses", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--data",
            str(data_path),
            "--forecasts",
            str(forecast_path),
            "--calendar-report",
            str(calendar_path),
            "--output",
            str(output_path),
        ],
    )
    assert module.main() == 0
    losses = pd.read_parquet(output_path)
    assert len(losses) == 3
    assert set(losses["market"]) == set(markets)
    realized = float(synthetic_frame["rv_cc"].iloc[21])
    assert (losses["realized_variance"] == realized).all()
    assert (losses["qlike"] == qlike(realized, forecast)).all()
    assert losses["main_evaluation_eligible"].all()
