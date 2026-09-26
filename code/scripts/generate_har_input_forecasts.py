#!/usr/bin/env python3
"""Generate the one-day multiscale-input neural/linear matched pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bnsv.forecast_generation import load_forecast_design, market_daily_frame
from bnsv.har_input_calibration import calibrate_har_input_scales
from bnsv.har_input_generation import generate_har_input_pair
from bnsv.har_posterior_cache import load_posterior_caches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="documented daily Parquet file")
    parser.add_argument("--market", default="SP500", choices=("SP500", "FTSE100", "DAX"))
    parser.add_argument("--design", default="design/forecast_generation.yaml")
    parser.add_argument("--output-dir", required=True, help="new directory for this matched run")
    parser.add_argument("--origin-start", help="optional first forecast origin")
    parser.add_argument("--origin-end", help="optional last forecast origin for a pilot")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--posterior-cache", action="append", default=[],
                        help="extracted POSTERIORS.json directory; may repeat for NN and LIN")
    parser.add_argument("--filter-policy", choices=("grid_agreement",),
                        default="grid_agreement")
    parser.add_argument("--model-workers", type=int, choices=(1, 2), default=1,
                        help="concurrent models per shared refit; each uses four parallel chains")
    parser.add_argument("--sampling-policy", choices=("matched_097_divergence_099_ess",),
                        default="matched_097_divergence_099_ess")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty")
    design_path = Path(args.design).resolve()
    design = load_forecast_design(design_path)
    if args.market not in design["markets"]:
        parser.error("market is not in the supplied forecast design")
    daily = market_daily_frame(args.data, market=args.market)
    calibration = calibrate_har_input_scales(daily)
    print(json.dumps(calibration, indent=2), flush=True)
    filters = design["state_filter"]
    cache = load_posterior_caches(args.posterior_cache, market=args.market,
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    panel = generate_har_input_pair(
        daily=daily, market=args.market,
        stan_dir=design_path.parent.parent / "code" / "models",
        output_dir=output, correction_scales=calibration["scales"],
        base_seed=int(design["seed"]),
        forecast_start=args.origin_start or design["sample"]["forecast_start"],
        forecast_end=args.origin_end or design["sample"]["forecast_end"],
        particle_count=int(filters["parameter_atoms"]),
        path_count=int(design["forecast"]["paths"]),
        grid_levels=tuple(int(v) for v in filters["certification_grid_levels"]),
        warning_ess_fraction=float(filters["warning_ess_fraction"]),
        show_progress=args.show_progress,
        calibration_record=calibration,
        model_workers=args.model_workers,
        sampling_policy=args.sampling_policy,
        posterior_cache=cache,
        filter_policy=args.filter_policy,
    )
    print(f"Wrote {len(panel)} forecasts to {output / 'forecasts.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
