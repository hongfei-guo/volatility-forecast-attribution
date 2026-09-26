#!/usr/bin/env python3
"""Generate the external benchmark forecast panel for one market."""
from __future__ import annotations

import argparse

from bnsv.benchmarks import BENCHMARK_IDS, generate_from_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="daily Parquet file")
    parser.add_argument(
        "--design",
        default="design/external_benchmarks.yaml",
        help="benchmark design",
    )
    parser.add_argument(
        "--market", required=True, choices=("SP500", "FTSE100", "DAX")
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=BENCHMARK_IDS,
        help="benchmark to generate; repeat as needed (default: all)",
    )
    parser.add_argument("--output", required=True, help="output Parquet file")
    parser.add_argument(
        "--components-dir",
        help="optional directory for Gaussian predictive components",
    )
    parser.add_argument(
        "--origin-end", help="optional last forecast origin for a bounded run"
    )
    args = parser.parse_args()
    panel = generate_from_design(
        data_path=args.data,
        design_path=args.design,
        market=args.market,
        output_path=args.output,
        model_ids=args.model,
        origin_end=args.origin_end,
        components_dir=args.components_dir,
    )
    print(f"Wrote {len(panel)} benchmark forecasts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
