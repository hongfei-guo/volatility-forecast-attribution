#!/usr/bin/env python3
"""Generate a Student-t GARCH forecast panel for one market."""
from __future__ import annotations

import argparse

from bnsv.garch_models import MODEL_IDS, generate_garch_from_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="daily Parquet file")
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument(
        "--design",
        help="design YAML; defaults to the design for the selected model",
    )
    parser.add_argument(
        "--market", required=True, choices=("SP500", "FTSE100", "DAX")
    )
    parser.add_argument("--output", required=True, help="forecast-panel Parquet file")
    parser.add_argument(
        "--components-output",
        help="optional directory for predictive-component NPZ files",
    )
    parser.add_argument(
        "--origin-end", help="optional last forecast origin for a bounded run"
    )
    args = parser.parse_args()
    design = args.design
    if design is None:
        design = (
            "design/garch_t.yaml"
            if args.model == "GARCH-t"
            else "design/realized_garch_t.yaml"
        )
    panel = generate_garch_from_design(
        data_path=args.data,
        design_path=design,
        market=args.market,
        output_path=args.output,
        components_dir=args.components_output,
        origin_end=args.origin_end,
        expected_model_id=args.model,
    )
    print(f"Wrote {len(panel)} {args.model} forecasts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
