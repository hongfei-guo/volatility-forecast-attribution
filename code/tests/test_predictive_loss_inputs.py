from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import build_predictive_losses


def _calendar_audit(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "valid": True,
                "observed_non_sessions": [],
                "source_label_mismatches": [],
                "markets": [
                    {"market": "SP500", "missing_expected_sessions": []}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_forecast_record_loader_accepts_parquet_and_jsonl(tmp_path: Path) -> None:
    parquet_path = tmp_path / "panel.parquet"
    pd.DataFrame([{"model_id": "NN-SV", "variance_forecast": 0.2}]).to_parquet(
        parquet_path, index=False
    )
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        json.dumps({"model_id": "SV", "variance_mean": 0.3}) + "\n",
        encoding="utf-8",
    )
    assert build_predictive_losses._load_forecast_records(
        [parquet_path, jsonl_path]
    ) == [
        {"model_id": "NN-SV", "variance_forecast": 0.2},
        {"model_id": "SV", "variance_mean": 0.3},
    ]


@pytest.mark.parametrize("path", ["/absolute/component.npz", "../component.npz"])
def test_predictive_file_must_stay_within_forecast_root(path: str) -> None:
    with pytest.raises(ValueError, match="without parent traversal"):
        build_predictive_losses._predictive_file({"predictive_file": path})


def test_saved_draw_mean_method_must_match_components() -> None:
    with pytest.raises(ValueError, match="does not match predictive components"):
        build_predictive_losses._variance_mean_for_evaluation(
            {
                "variance_forecast": 0.0004,
                "variance_mean_method": "mean_integrated_path_variance",
            },
            np.full(4, 0.0002),
        )


def test_generated_parquet_panels_feed_predictive_loss_builder(
    synthetic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_path = tmp_path / "daily.parquet"
    synthetic_frame.to_parquet(data_path, index=False)
    calendar_path = tmp_path / "calendar.json"
    _calendar_audit(calendar_path)

    origin = synthetic_frame.loc[20, "date"]
    mature = synthetic_frame.loc[21, "date"]
    component_dir = tmp_path / "components"
    component_dir.mkdir()
    np.savez_compressed(
        component_dir / "probabilistic.npz",
        raw_returns=np.array([[-0.02], [0.0], [0.015], [0.025]]),
        raw_variances=np.full((4, 1), 0.0002),
        raw_return_locations=np.zeros((4, 1)),
        raw_conditional_sds=np.full((4, 1), 0.02),
        degrees_of_freedom=np.full(4, 8.0),
    )
    common = {
        "market": "SP500",
        "origin_date": str(origin.date()),
        "horizon": 1,
        "mature_date": str(mature.date()),
    }
    probabilistic_panel = tmp_path / "probabilistic.parquet"
    pd.DataFrame(
        [
            {
                **common,
                "model_id": "NN-SV",
                "forecast_kind": "probabilistic",
                "variance_forecast": 0.0004,
                "variance_mean_method": "exact_weighted_candidate_means",
                "return_distribution": "variance_standardized_student_t",
                "predictive_file": "probabilistic.npz",
            }
        ]
    ).to_parquet(probabilistic_panel, index=False)
    point_panel = tmp_path / "point.parquet"
    pd.DataFrame(
        [
            {
                **common,
                "model_id": "HAR-RV",
                "forecast_kind": "variance_point_only",
                "variance_forecast": 0.0003,
                "predictive_file": None,
            }
        ]
    ).to_parquet(point_panel, index=False)
    output = tmp_path / "losses.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_predictive_losses.py",
            "--data",
            str(data_path),
            "--forecasts",
            str(probabilistic_panel),
            str(point_panel),
            "--forecast-root",
            str(component_dir),
            "--calendar-audit",
            str(calendar_path),
            "--output",
            str(output),
        ],
    )
    assert build_predictive_losses.main() == 0
    losses = pd.read_parquet(output).set_index("model_id")
    assert set(losses.index) == {"NN-SV", "HAR-RV"}
    assert losses.loc["NN-SV", "variance_forecast"] == 0.0004
    assert losses.loc["NN-SV", "variance_forecast_source"] == (
        "exact_weighted_candidate_means"
    )
    assert np.isfinite(losses.loc["NN-SV", "return_log_score"])
    assert 0 < losses.loc["NN-SV", "return_pit"] < 1
    assert np.isfinite(losses.loc["NN-SV", "var_es_01_fz0"])
    assert losses.loc["HAR-RV", "forecast_kind"] == "variance_point_only"
    assert losses.loc["HAR-RV", "variance_forecast"] == 0.0003
