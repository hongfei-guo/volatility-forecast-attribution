from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bnsv.sample_alignment import (
    GAP_POLICY,
    annotate_loss_table,
    gap_metadata,
    load_calendar_audit,
    select_evaluation_sample,
)


def _audit() -> dict:
    return {
        "valid": True,
        "markets": [
            {
                "market": "SP500",
                "missing_expected_sessions": ["2020-07-07"],
            }
        ],
    }


def test_target_gap_and_origin_prior_gap_are_distinct():
    missing = {"SP500": pd.DatetimeIndex(["2020-07-07"])}
    target_gap = gap_metadata(
        market="SP500",
        origin_date="2020-07-06",
        target_dates=["2020-07-08"],
        previous_observed_date="2020-07-03",
        missing_by_market=missing,
    )
    assert target_gap["target_spans_archive_gap"] is True
    assert target_gap["target_missing_expected_session_count"] == 1
    assert target_gap["target_expected_session_span"] == 2
    assert target_gap["origin_follows_archive_gap"] is False
    assert target_gap["main_evaluation_eligible"] is False

    origin_gap = gap_metadata(
        market="SP500",
        origin_date="2020-07-08",
        target_dates=["2020-07-09"],
        previous_observed_date="2020-07-06",
        missing_by_market=missing,
    )
    assert origin_gap["target_spans_archive_gap"] is False
    assert origin_gap["origin_follows_archive_gap"] is True
    assert origin_gap["main_evaluation_eligible"] is True


def test_existing_loss_archive_can_be_annotated_without_dropping_rows():
    losses = pd.DataFrame(
        [
            {
                "market": "SP500",
                "origin_date": "2020-07-06",
                "horizon": 1,
                "target_start": "2020-07-08",
                "target_end": "2020-07-08",
                "mature_date": "2020-07-08",
                "target_dates_json": json.dumps(["2020-07-08"]),
            },
            {
                "market": "SP500",
                "origin_date": "2020-07-08",
                "horizon": 1,
                "target_start": "2020-07-09",
                "target_end": "2020-07-09",
                "mature_date": "2020-07-09",
                "target_dates_json": json.dumps(["2020-07-09"]),
            },
        ]
    )
    observed = {
        "SP500": pd.to_datetime(
            ["2020-07-03", "2020-07-06", "2020-07-08", "2020-07-09"]
        )
    }
    result = annotate_loss_table(
        losses,
        calendar_audit=_audit(),
        observed_dates_by_market=observed,
    )
    assert len(result) == len(losses)
    assert result["gap_policy"].eq(GAP_POLICY).all()
    assert result["target_spans_archive_gap"].tolist() == [True, False]
    assert result["origin_follows_archive_gap"].tolist() == [False, True]

    selected = select_evaluation_sample(result, "main_gap_excluded")
    assert selected["origin_date"].tolist() == ["2020-07-08"]
    with pytest.raises(ValueError, match="unsupported"):
        select_evaluation_sample(result, "all_rows")


def test_evaluation_policy_fails_closed_on_unannotated_losses():
    with pytest.raises(ValueError, match="not been exchange-gap annotated"):
        select_evaluation_sample(pd.DataFrame({"origin_date": ["2020-01-01"]}))


def test_calendar_audit_validates_required_fields(tmp_path: Path):
    audit = {
        "valid": True,
        "observed_non_sessions": [],
        "source_label_mismatches": [],
        "markets": [
            {"market": "SP500", "missing_expected_sessions": []}
        ],
    }
    path = tmp_path / "calendar_audit.json"
    path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    assert load_calendar_audit(path)["valid"] is True


def test_main_eligibility_cannot_contradict_target_gap():
    losses = pd.DataFrame(
        {
            "gap_policy": [GAP_POLICY],
            "target_spans_archive_gap": [True],
            "origin_follows_archive_gap": [False],
            "main_evaluation_eligible": [True],
        }
    )
    with pytest.raises(ValueError, match="contradicts"):
        select_evaluation_sample(losses)


def test_target_sequence_must_match_immutable_observed_dates():
    losses = pd.DataFrame(
        [
            {
                "market": "SP500",
                "origin_date": "2020-07-06",
                "horizon": 1,
                "target_start": "2020-07-09",
                "target_end": "2020-07-09",
                "mature_date": "2020-07-09",
                "target_dates_json": json.dumps(["2020-07-09"]),
            }
        ]
    )
    observed = {
        "SP500": pd.to_datetime(
            ["2020-07-03", "2020-07-06", "2020-07-08", "2020-07-09"]
        )
    }
    with pytest.raises(ValueError, match="next horizon observed dates"):
        annotate_loss_table(
            losses,
            calendar_audit=_audit(),
            observed_dates_by_market=observed,
        )


def test_string_false_is_not_treated_as_true():
    losses = pd.DataFrame(
        {
            "gap_policy": [GAP_POLICY, GAP_POLICY],
            "target_spans_archive_gap": ["False", "True"],
            "origin_follows_archive_gap": ["False", "False"],
            "main_evaluation_eligible": ["True", "False"],
        }
    )
    selected = select_evaluation_sample(losses, "main_gap_excluded")
    assert selected.index.tolist() == [0]
