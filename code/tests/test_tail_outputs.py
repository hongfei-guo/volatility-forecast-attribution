from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_realized_garch_results_are_distribution_only() -> None:
    metrics = pd.read_csv(ROOT / "results/realized_garch_t/model_metrics.csv")
    pairwise = pd.read_csv(ROOT / "results/realized_garch_t/pairwise_dm.csv")
    assert metrics.shape == (9, 9)
    assert metrics.columns.tolist() == [
        "market",
        "model_id",
        "observations",
        "mean_qlike",
        "mean_return_log_score",
        "mean_return_crps",
        "pit_mean",
        "pit_variance",
        "pit_ks_p_value",
    ]
    assert pairwise.shape == (18, 12)
    assert set(pairwise["loss"]) == {"qlike", "return_log_score", "return_crps"}


def test_analytic_tail_table_contains_reported_realized_garch_cells() -> None:
    pairwise = pd.read_csv(ROOT / "results/tail/pairwise_dm.csv")
    cells = [
        ("FTSE100", 0.05, -0.0093638970588298, 0.0001901037271856),
        ("DAX", 0.05, -0.0057014221224391, 0.1810014742803556),
        ("SP500", 0.01, -0.2111629820078865, 0.0015348098381412),
    ]
    for market, alpha, difference, p_value in cells:
        row = pairwise[
            (pairwise["market"] == market)
            & (pairwise["alpha"] == alpha)
            & (pairwise["model_a"] == "Realized-GARCH-t-MLE")
            & (pairwise["model_b"] == "Realized-GARCH-Gaussian-QMLE")
        ]
        assert len(row) == 1
        assert row.iloc[0]["mean_loss_difference"] == pytest.approx(
            difference, abs=1e-14
        )
        assert row.iloc[0]["p_value"] == pytest.approx(p_value, abs=1e-14)
