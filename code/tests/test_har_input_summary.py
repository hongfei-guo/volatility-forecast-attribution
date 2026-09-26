from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bnsv.forecast_features import HAR_INPUT_MODELS
from bnsv.har_input_summary import summarize_har_input_losses
from bnsv.mcs_dm import diebold_mariano
from bnsv.sample_alignment import GAP_POLICY


def losses():
    rows = []
    dates = pd.bdate_range("2018-01-02", periods=13)
    for name in HAR_INPUT_MODELS:
        for i in range(12):
            rows.append({
                "market": "SP500", "model_id": name, "horizon": 1,
                "origin_date": dates[i], "mature_date": dates[i + 1],
                "qlike__rv_cc": (0.2 + 0.01 * i if name == HAR_INPUT_MODELS[0] else 0.25 + 0.03 * np.sin(i)),
                "qlike": 99999,  # The explicit CC loss, not another proxy/alias, is used.
                "gap_policy": GAP_POLICY, "target_spans_archive_gap": i == 3,
                "main_evaluation_eligible": i != 3,
            })
    return pd.DataFrame(rows)


def test_gap_and_availability_filtering_uses_one_sample_for_both_means_and_dm():
    frame = losses()
    frame = frame.drop(index=13)  # Linear member unavailable at the second origin.
    frame.loc[2, "qlike__rv_cc"] = np.nan
    summary, common = summarize_har_input_losses(frame.sample(frac=1, random_state=12))
    row = summary.iloc[0]
    assert row["n_common"] == 9
    assert row["n_eligible_a"] == 11 and row["n_eligible_b"] == 10
    assert common["origin_date"].is_monotonic_increasing
    assert row["mean_loss_a"] == common["loss_a"].mean()
    assert row["mean_loss_b"] == common["loss_b"].mean()
    reference = diebold_mariano(common["loss_a"].to_numpy(), common["loss_b"].to_numpy(), horizon=1)
    assert row["mean_loss_difference"] == reference["mean_loss_difference"]
    assert row["p_value_two_sided"] == reference["p_value"]


@pytest.mark.parametrize("problem", ["duplicate", "maturity", "flags", "different_gap", "infinity", "horizon"])
def test_ambiguous_or_invalid_inputs_fail(problem):
    frame = losses()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "maturity":
        frame.loc[12, "mature_date"] += pd.Timedelta(days=1)
    elif problem == "flags":
        frame.loc[0, "main_evaluation_eligible"] = False
    elif problem == "different_gap":
        frame.loc[0, "target_spans_archive_gap"] = True
        frame.loc[0, "main_evaluation_eligible"] = False
    elif problem == "infinity":
        frame.loc[0, "qlike__rv_cc"] = np.inf
    else:
        frame.loc[0, "horizon"] = 5
    with pytest.raises(ValueError):
        summarize_har_input_losses(frame)


def test_short_pilot_is_not_presented_as_a_dm_result():
    frame = losses()
    frame = frame.groupby("model_id").head(3)
    with pytest.raises(ValueError, match="five common"):
        summarize_har_input_losses(frame)
