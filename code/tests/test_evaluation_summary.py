from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    script = ROOT / "scripts" / "summarize_evaluation.py"
    spec = importlib.util.spec_from_file_location("evaluation_summary", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _losses() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2018-01-02", periods=30)
    for model_index, model in enumerate(("MODEL-A", "MODEL-B", "SPARSE")):
        for date_index, date in enumerate(dates):
            value = 0.1 * model_index + date_index / 100
            if model == "SPARSE" and date_index >= 7:
                value = np.nan
            rows.append(
                {
                    "market": "SP500",
                    "model_id": model,
                    "origin_date": str(date.date()),
                    "horizon": 1,
                    "forecast_kind": "variance_point_only",
                    "qlike": value,
                    "gap_policy": "exchange_calendar_gap_exclusion",
                    "target_spans_archive_gap": False,
                    "origin_follows_archive_gap": False,
                    "main_evaluation_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def _design() -> dict:
    return {
        "seed": 19,
        "markets": ["SP500"],
        "horizons": [1],
        "evaluation": {
            "main_sample_policy": "main_gap_excluded",
            "mcs": {
                "models": ["MODEL-A", "MODEL-B"],
                "replications": 30,
                "alpha": 0.10,
                "mean_block_length": 5,
            },
            "dm": {
                "multiplicity_adjustment": "none",
                "pairs": [
                    {
                        "loss_column": "qlike",
                        "model_a": "MODEL-A",
                        "model_b": "MODEL-B",
                        "horizons": [1],
                    }
                ],
            },
        },
    }


def test_variance_summary_uses_declared_models_and_pairwise_dates() -> None:
    module = _module()
    losses = _losses()
    selected = losses[losses["main_evaluation_eligible"]].copy()
    full = module._evaluation_rows(selected, design=_design(), scope="variance")
    reduced = module._evaluation_rows(
        selected[selected["model_id"] != "SPARSE"].copy(),
        design=_design(),
        scope="variance",
    )
    assert full == reduced
    mcs_rows, mean_rows, dm_rows = full
    assert mcs_rows[0]["n_common"] == 30
    assert [row["model_id"] for row in mean_rows] == ["MODEL-A", "MODEL-B"]
    assert dm_rows[0]["n_common"] == 30


def test_full_scope_rejects_variance_only_losses() -> None:
    with pytest.raises(ValueError, match="predictive-distribution columns"):
        _module()._validate_full_losses(_losses())
