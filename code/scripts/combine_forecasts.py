#!/usr/bin/env python3
"""Generate the documented prequential forecast combinations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bnsv.data_contract import read_daily_frame
from bnsv.forecast_combination import (
    generate_combination_panels,
    load_combination_design,
)


def _read_parquet(paths: list[str], *, label: str) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError(f"{label} must be existing Parquet files: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError(f"at least one {label} file is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate monthly prequential NN-SV forecast combinations"
    )
    parser.add_argument("--design", default="design/combination.yaml")
    parser.add_argument("--data", required=True, help="daily Parquet file")
    parser.add_argument("--forecasts", nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--packaged-weights",
        help="complete published monthly combination-weight path",
    )
    source.add_argument(
        "--losses",
        nargs="+",
        help="candidate loss panels regenerated from lawfully obtained data",
    )
    parser.add_argument(
        "--warmup-losses",
        nargs="+",
        help="candidate objective losses that originate and mature before 2018",
    )
    parser.add_argument(
        "--warmup-forecasts",
        nargs="+",
        help="prediction-only warmup panels; realised variance is read from --data",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("equal", "qlike", "log_score"),
        help="optional subset of the documented combination modes",
    )
    parser.add_argument(
        "--calendar-report",
        help="exchange-calendar report required when weights are regenerated",
    )
    parser.add_argument(
        "--output", required=True, help="ensemble forecast Parquet file"
    )
    parser.add_argument(
        "--weights-output", required=True, help="combination-weight CSV file"
    )
    parser.add_argument(
        "--predictive-root",
        help="root directory for candidate predictive_file paths",
    )
    parser.add_argument(
        "--components-output",
        help="directory for equal and log-score whole-path mixture files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="whole-path mixture seed; defaults to the documented design seed",
    )
    args = parser.parse_args()

    if args.warmup_losses is not None and args.warmup_forecasts is not None:
        parser.error("choose --warmup-losses or --warmup-forecasts")
    if (
        args.losses is not None
        and args.warmup_losses is None
        and args.warmup_forecasts is None
    ):
        parser.error("--losses requires one warmup input")
    if args.losses is None and (
        args.warmup_losses is not None or args.warmup_forecasts is not None
    ):
        parser.error("warmup inputs are used only with --losses")
    if args.losses is not None and args.calendar_report is None:
        parser.error("--losses requires --calendar-report")
    if (args.predictive_root is None) != (args.components_output is None):
        parser.error("--predictive-root and --components-output must be used together")
    if args.seed is not None and args.predictive_root is None:
        parser.error("--seed is used only with whole-path mixture components")

    design = load_combination_design(args.design)
    forecasts = _read_parquet(args.forecasts, label="forecast panel")
    losses = (
        None if args.losses is None else _read_parquet(args.losses, label="loss panel")
    )
    warmup = (
        None
        if args.warmup_losses is None
        else _read_parquet(args.warmup_losses, label="warmup-loss panel")
    )
    warmup_forecasts = (
        None
        if args.warmup_forecasts is None
        else _read_parquet(args.warmup_forecasts, label="warmup forecast panel")
    )
    weights = (
        None if args.packaged_weights is None else pd.read_csv(args.packaged_weights)
    )
    calendar_report = (
        None
        if args.calendar_report is None
        else json.loads(Path(args.calendar_report).read_text(encoding="utf-8"))
    )
    ensemble, weight_path = generate_combination_panels(
        daily=read_daily_frame(args.data),
        forecasts=forecasts,
        design=design,
        losses=losses,
        warmup_losses=warmup,
        warmup_forecasts=warmup_forecasts,
        packaged_weights=weights,
        calendar_report=calendar_report,
        selected_modes=args.modes,
        predictive_root=args.predictive_root,
        predictive_output_dir=args.components_output,
        mixture_seed=args.seed,
    )
    output = Path(args.output)
    weights_output = Path(args.weights_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(output, index=False)
    weight_path.to_csv(weights_output, index=False)
    print(f"Wrote {len(ensemble)} ensemble forecasts to {output}")
    print(f"Wrote {len(weight_path)} weight rows to {weights_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
