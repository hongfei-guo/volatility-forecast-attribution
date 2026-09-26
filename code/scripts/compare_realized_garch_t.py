#!/usr/bin/env python3
"""Compare h=1 Realized GARCH-t with Gaussian Realized GARCH and RV-NN-SV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bnsv.calibration import pit_diagnostics
from bnsv.sample_alignment import (
    read_loss_archive,
    select_evaluation_sample,
)
from bnsv.mcs_dm import diebold_mariano


MODELS = (
    "Realized-GARCH-t-MLE",
    "Realized-GARCH-Gaussian-QMLE",
    "RV-NN-SV",
)
LOSSES = (
    "qlike",
    "return_log_score",
    "return_crps",
)


def _validate_targets(frame: pd.DataFrame) -> None:
    for origin, group in frame.groupby("origin_date", sort=False):
        if set(group["model_id"].astype(str)) != set(MODELS):
            continue
        for column in ("target_dates_json", "mature_date"):
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"target identity differs at {origin}: {column}")
        for column in ("realized_variance", "realized_cumulative_return"):
            values = group[column].to_numpy(float)
            if np.any(~np.isfinite(values)) or not np.allclose(
                values, values[0], rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"realized target differs at {origin}: {column}")


def _model_summary(group: pd.DataFrame) -> dict:
    report = {
        "market": str(group["market"].iloc[0]),
        "model_id": str(group["model_id"].iloc[0]),
        "observations": int(len(group)),
        **{f"mean_{loss}": float(group[loss].mean()) for loss in LOSSES},
        "pit": pit_diagnostics(group["return_pit"].to_numpy(float), lags=10),
    }
    return report


def _pairwise(frame: pd.DataFrame, comparator: str, loss: str) -> dict:
    pair = frame[frame["model_id"].isin((MODELS[0], comparator))].pivot(
        index="origin_date", columns="model_id", values=loss
    )
    complete = pair.dropna()
    if len(complete) < 20:
        raise ValueError(f"too few paired observations for {comparator}, {loss}")
    result = diebold_mariano(
        complete[MODELS[0]].to_numpy(float),
        complete[comparator].to_numpy(float),
        horizon=1,
    )
    return {
        "market": str(frame["market"].iloc[0]),
        "loss": loss,
        "model_a": MODELS[0],
        "model_b": comparator,
        "sign_convention": "negative_mean_difference_favors_realized_garch_t",
        "n_common": int(len(complete)),
        "mean_loss_model_a": float(complete[MODELS[0]].mean()),
        "mean_loss_model_b": float(complete[comparator].mean()),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-losses", required=True)
    parser.add_argument("--student-t-losses", required=True)
    parser.add_argument("--market", choices=("SP500", "FTSE100", "DAX"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    existing = read_loss_archive(args.existing_losses)
    student_t = read_loss_archive(args.student_t_losses)
    existing = existing[
        (existing["market"].astype(str) == args.market)
        & (existing["model_id"].astype(str).isin(MODELS[1:]))
        & (existing["horizon"].astype(int) == 1)
    ]
    student_t = student_t[
        (student_t["market"].astype(str) == args.market)
        & (student_t["model_id"].astype(str) == MODELS[0])
        & (student_t["horizon"].astype(int) == 1)
    ]
    frame = pd.concat([existing, student_t], ignore_index=True, sort=False)
    if frame.empty or set(frame["model_id"].astype(str)) != set(MODELS):
        raise ValueError("comparison does not contain the three required models")
    frame = select_evaluation_sample(frame, "main_gap_excluded")
    _validate_targets(frame)

    summaries = [
        _model_summary(group)
        for _, group in frame.groupby("model_id", sort=False)
    ]
    pairwise = [
        _pairwise(frame, comparator, loss)
        for comparator in MODELS[1:]
        for loss in LOSSES
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "market": args.market,
        "horizon": 1,
        "models": list(MODELS),
        "evaluation_sample_policy": "main_gap_excluded",
        "model_summaries": summaries,
        "pairwise_dm": pairwise,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_rows = []
    for item in summaries:
        summary_rows.append(
            {
                "market": item["market"],
                "model_id": item["model_id"],
                "observations": item["observations"],
                **{key: value for key, value in item.items() if key.startswith("mean_")},
                "pit_mean": item["pit"]["mean"],
                "pit_variance": item["pit"]["variance"],
                "pit_ks_p_value": item["pit"]["ks_p_value"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "model_metrics.csv", index=False)
    pd.DataFrame(pairwise).to_csv(output / "pairwise_dm.csv", index=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
