#!/usr/bin/env python3
"""Build and validate the three-market daily input archive."""
from __future__ import annotations

import argparse
import json

from bnsv.omi_builder import build_omi_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="lawfully obtained Oxford-Man CSV")
    parser.add_argument("--output", required=True, help="daily Parquet output")
    parser.add_argument("--report", required=True, help="validation report")
    args = parser.parse_args()
    report = build_omi_archive(
        source_path=args.source,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
