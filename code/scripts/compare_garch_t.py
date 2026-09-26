#!/usr/bin/env python3
"""Compare NN-SV with a return-only GARCH(1,1)-t model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bnsv.sample_alignment import read_loss_archive, select_evaluation_sample
from bnsv.mcs_dm import diebold_mariano


PAIRS = {
    1: ("qlike", "return_log_score"),
    5: ("qlike",),
    10: ("qlike",),
}
MODELS = ("NN-SV", "GARCH-t")


def _validate_pair_targets(subset: pd.DataFrame) -> None:
    identity_columns = ("target_dates_json", "mature_date")
    numeric_columns = ("realized_variance", "realized_cumulative_return")
    for origin, group in subset.groupby("origin_date", sort=False):
        if set(group["model_id"].astype(str)) != set(MODELS):
            continue
        for column in identity_columns:
            if column not in group or group[column].isna().any():
                raise ValueError(f"paired rows at {origin} lack {column}")
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"paired target identity differs at {origin}: {column}")
        for column in numeric_columns:
            if column not in group or group[column].isna().any():
                raise ValueError(f"paired rows at {origin} lack {column}")
            values = group[column].to_numpy(float)
            if np.any(~np.isfinite(values)) or not np.allclose(
                values, values[0], rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"paired realized target differs at {origin}: {column}")


def _pairwise(frame: pd.DataFrame, *, horizon: int, loss: str) -> dict[str, object]:
    subset = frame[
        (frame["horizon"].astype(int) == int(horizon))
        & frame["model_id"].astype(str).isin(MODELS)
    ].copy()
    _validate_pair_targets(subset)
    wide = subset.pivot(index="origin_date", columns="model_id", values=loss).reindex(
        columns=MODELS
    )
    complete = wide.dropna()
    if len(complete) < 20:
        raise ValueError(f"too few pairwise common observations for h={horizon}, {loss}")
    result = diebold_mariano(
        complete["NN-SV"].to_numpy(float),
        complete["GARCH-t"].to_numpy(float),
        horizon=int(horizon),
    )
    return {
        "horizon": int(horizon),
        "loss": loss,
        "model_a": "NN-SV",
        "model_b": "GARCH-t",
        "sign_convention": "negative_mean_difference_favors_NN_SV",
        "n_candidate_origins": int(len(wide)),
        "n_nn_sv_available": int(wide["NN-SV"].notna().sum()),
        "n_garch_t_available": int(wide["GARCH-t"].notna().sum()),
        "n_common": int(len(complete)),
        **result,
        "mean_loss_nn_sv": float(complete["NN-SV"].mean()),
        "mean_loss_garch_t": float(complete["GARCH-t"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-losses", required=True)
    parser.add_argument("--garch-losses", required=True)
    parser.add_argument("--market", choices=("SP500", "DAX", "FTSE100"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    existing = read_loss_archive(args.existing_losses)
    garch = read_loss_archive(args.garch_losses)
    frame = pd.concat([existing, garch], ignore_index=True, sort=False)
    frame = frame[frame["market"].astype(str) == args.market].copy()
    if frame.empty:
        raise ValueError(f"no loss rows found for market {args.market}")
    frame = select_evaluation_sample(frame, "main_gap_excluded")
    results = [
        _pairwise(frame, horizon=horizon, loss=loss)
        for horizon, losses in PAIRS.items()
        for loss in losses
    ]
    payload = {
        "market": args.market,
        "evaluation_sample_policy": "main_gap_excluded",
        "comparisons": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
