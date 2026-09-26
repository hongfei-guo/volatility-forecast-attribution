"""Common-sample QLIKE comparison for the multiscale HAR-input pair."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast_features import HAR_INPUT_MODELS
from .mcs_dm import diebold_mariano
from .sample_alignment import select_evaluation_sample, strict_bool_series


def summarize_har_input_losses(losses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use gap-eligible, pairwise-common one-day CC losses for both means and DM."""
    keys = ["market", "origin_date", "horizon", "mature_date"]
    required = {*keys, "model_id", "qlike__rv_cc"}
    if missing := required - set(losses):
        raise ValueError(f"HAR loss archive lacks columns: {sorted(missing)}")
    pair = losses[losses["model_id"].isin(HAR_INPUT_MODELS)].copy()
    if pair.empty or pair["market"].isna().any():
        raise ValueError("HAR pair losses must identify at least one market")
    for date in ("origin_date", "mature_date"):
        pair[date] = pd.to_datetime(pair[date], errors="raise")
        if pair[date].isna().any() or not pair[date].eq(pair[date].dt.normalize()).all():
            raise ValueError("loss dates must be nonmissing session dates")
    pair["horizon"] = pd.to_numeric(pair["horizon"], errors="raise")
    if not pair["horizon"].eq(1).all():
        raise ValueError("HAR-input comparison is one-day only")
    if pair.duplicated(["market", "model_id", "origin_date", "horizon"]).any():
        raise ValueError("duplicate model-origin loss rows")
    if not (pair["mature_date"] > pair["origin_date"]).all():
        raise ValueError("target maturity must follow the forecast origin")
    if (pair.groupby(["market", "origin_date", "horizon"])["mature_date"].nunique() > 1).any():
        raise ValueError("the two models have different target maturity dates")
    selected = select_evaluation_sample(pair)
    gap = strict_bool_series(pair["target_spans_archive_gap"], name="target_spans_archive_gap")
    if (pair.assign(_gap=gap).groupby(keys)["_gap"].nunique() > 1).any():
        raise ValueError("the two models disagree on calendar-gap eligibility for one target")
    selected["qlike__rv_cc"] = pd.to_numeric(selected["qlike__rv_cc"], errors="raise")
    # NaN represents unavailable loss; infinity is a numerical failure, not a
    # supported missing-observation convention.
    observed = selected["qlike__rv_cc"].dropna().to_numpy(float)
    if np.any(~np.isfinite(observed)):
        raise ValueError("CC QLIKE contains infinite loss values")
    model_a, model_b = HAR_INPUT_MODELS
    summaries, common_frames = [], []
    for market in sorted(pair["market"].unique()):
        cell = selected[selected["market"] == market]
        if set(cell["model_id"]) != set(HAR_INPUT_MODELS):
            raise ValueError(f"both HAR models need eligible losses in {market}")
        common = cell.pivot(index=keys, columns="model_id", values="qlike__rv_cc")
        common = common.reindex(columns=HAR_INPUT_MODELS).dropna().sort_index()
        if len(common) < 5:
            raise ValueError(f"at least five common finite losses are required for DM in {market}")
        a, b = common[model_a].to_numpy(float), common[model_b].to_numpy(float)
        dm = diebold_mariano(a, b, horizon=1)
        summaries.append({
            "market": market, "horizon": 1, "loss_column": "qlike__rv_cc",
            "model_a": model_a, "model_b": model_b,
            "n_eligible_a": int((cell["model_id"] == model_a).sum()),
            "n_eligible_b": int((cell["model_id"] == model_b).sum()),
            "n_common": len(common),
            "mean_loss_a": float(a.mean()), "mean_loss_b": float(b.mean()),
            "mean_loss_difference": dm["mean_loss_difference"],
            "dm_statistic": dm["statistic"], "p_value_two_sided": dm["p_value"],
            "hac_lag": dm["hac_lag"], "negative_favors": "model_a",
            "reporting_role": "descriptive", "multiplicity_adjustment": "none",
        })
        common = common.rename(columns={model_a: "loss_a", model_b: "loss_b"}).reset_index()
        common["loss_difference"] = common["loss_a"] - common["loss_b"]
        common_frames.append(common)
    return pd.DataFrame(summaries), pd.concat(common_frames, ignore_index=True)
