#!/usr/bin/env python3
"""Summarize the multiscale NN/LIN HAR pair without changing the primary MCS."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bnsv.har_input_summary import summarize_har_input_losses
from bnsv.sample_alignment import read_loss_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")
    losses = pd.concat([read_loss_archive(path) for path in args.losses], ignore_index=True)
    summary, common = summarize_har_input_losses(losses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "matched_qlike.csv", index=False)
    common.to_csv(args.output_dir / "common_losses.csv", index=False)
    print(f"Wrote {len(summary)} matched comparisons using {len(common)} common origins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
