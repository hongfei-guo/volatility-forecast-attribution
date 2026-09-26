#!/usr/bin/env python3
"""Generate one empirical forecast panel from the documented daily data."""
from __future__ import annotations

import argparse

from bnsv.forecast_generation import generate_from_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="daily Parquet file")
    parser.add_argument(
        "--design", default="design/forecast_generation.yaml", help="forecast design"
    )
    parser.add_argument(
        "--market", required=True, choices=("SP500", "FTSE100", "DAX")
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("NN-SV", "RV-NN-SV", "RV-RA-NN-SV", "SV", "RV-LIN-SV"),
    )
    parser.add_argument("--output", required=True, help="forecast panel")
    parser.add_argument(
        "--work-dir", required=True, help="CmdStan sampling directory"
    )
    parser.add_argument(
        "--components-dir",
        help=(
            "optional root for compressed predictive components; "
            "predictive_file is relative to this directory"
        ),
    )
    parser.add_argument(
        "--origin-end", help="optional last forecast origin for a bounded run"
    )
    parser.add_argument(
        "--origin-start", help="optional first forecast origin"
    )
    parser.add_argument(
        "--refit-schedule", help="optional CSV schedule for a stated sample"
    )
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    panel = generate_from_design(
        data_path=args.data,
        design_path=args.design,
        market=args.market,
        model_id=args.model,
        output_path=args.output,
        work_dir=args.work_dir,
        origin_start=args.origin_start,
        origin_end=args.origin_end,
        refit_schedule=args.refit_schedule,
        show_progress=args.show_progress,
        components_dir=args.components_dir,
    )
    print(f"Wrote {len(panel)} forecasts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
