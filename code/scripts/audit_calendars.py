#!/usr/bin/env python3
"""Compare observed dates with the relevant exchange calendars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bnsv.calendar_audit import audit_exchange_calendars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_exchange_calendars(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
