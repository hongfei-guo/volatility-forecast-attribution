#!/usr/bin/env python3
"""Retrospective one-day OC-regressor HAR sensitivity; CC response and score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from bnsv.benchmarks import (
    benchmark_daily_frame, fit_ols, forecast_har_levels, har_design,
)
from bnsv.data_contract import read_daily_frame
from bnsv.mcs_dm import diebold_mariano
from bnsv.sample_alignment import read_loss_archive, select_evaluation_sample


CC, OC, NN = "HAR-RV", "HAR-RV-OC-input", "RV-NN-SV"
KEYS = ["market", "origin_date", "horizon", "mature_date"]


def reconstruct_pair(daily, reference, refit_dates):
    """Fit CC responses to CC/OC histories, using only observations through t."""
    grid = reference.loc[
        reference.model_id.eq(CC) & reference.horizon.eq(1)
    ].sort_values("origin_date").copy()
    for column in ("origin_date", "mature_date"):
        grid[column] = pd.to_datetime(grid[column], errors="raise")
    if grid.empty or grid.duplicated(KEYS).any():
        raise ValueError("HAR reference grid must be nonempty and unique")
    if daily.date.duplicated().any() or not daily.date.is_monotonic_increasing:
        raise ValueError("daily observations must be uniquely ordered")
    positions = {pd.Timestamp(d): i for i, d in enumerate(daily.date)}
    indices = [positions[d] for d in grid.origin_date]
    if np.any(np.diff(indices) != 1):
        raise ValueError("HAR reference grid must contain consecutive observed sessions")
    refits = set(pd.to_datetime(refit_dates))
    if grid.origin_date.iloc[0] not in refits:
        raise ValueError("first origin must be a documented refit")
    predictions = {CC: [], OC: []}
    fits = {}
    fit_rows = []
    for row, position in zip(grid.itertuples(index=False), indices):
        if position + 1 >= len(daily) or daily.date.iloc[position + 1] != row.mature_date:
            raise ValueError("reference maturity differs from next observed session")
        history = daily.iloc[:position + 1]
        for model, column in ((CC, "rv_cc"), (OC, "rv_oc")):
            values = history[column].to_numpy(float)
            if row.origin_date in refits:
                x, _ = har_design(values)
                fits[model] = fit_ols(x, history.rv_cc.to_numpy(float)[22:])
                fit_rows.append({
                    "market": row.market, "model_id": model,
                    "refit_date": row.origin_date, "n_training_rows": len(x),
                    **dict(zip(("intercept", "daily", "weekly", "monthly"),
                               fits[model].coefficients)),
                })
            value = float(forecast_har_levels(values, fits[model], 1)[0])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"nonpositive/nonfinite HAR forecast: {row.market} {row.origin_date} {model}")
            predictions[model].append(value)
    baseline = grid.variance_forecast.to_numpy(float)
    reproduced = np.asarray(predictions[CC])
    np.testing.assert_allclose(reproduced, baseline, rtol=1e-10, atol=1e-14)
    panels = []
    for model in (CC, OC):
        panel = grid.copy()
        panel["model_id"] = model
        panel["variance_forecast"] = predictions[model]
        panels.append(panel)
    check = {
        "market": str(grid.market.iloc[0]), "n_forecasts": len(grid),
        "n_refits": len(fit_rows) // 2,
        "max_absolute_baseline_difference": float(np.max(np.abs(reproduced - baseline))),
        "max_relative_baseline_difference": float(np.max(np.abs(reproduced / baseline - 1))),
    }
    return pd.concat(panels, ignore_index=True), pd.DataFrame(fit_rows), check


def summarize(losses):
    eligible = select_evaluation_sample(losses)
    rows, common_frames = [], []
    for market, cell in eligible.groupby("market", sort=True):
        wide = cell.pivot(index=KEYS, columns="model_id", values="qlike__rv_cc")
        for scope, models, contrasts in (
            ("har_pair", [CC, OC], ((OC, CC),)),
            ("three_model_context", [CC, OC, NN], ((NN, CC), (NN, OC))),
        ):
            common = wide.reindex(columns=models).dropna().sort_index()
            if len(common) < 5 or not np.isfinite(common.to_numpy()).all():
                raise ValueError("insufficient or invalid common losses")
            for a, b in contrasts:
                first, second = common[a].to_numpy(), common[b].to_numpy()
                dm = diebold_mariano(first, second, horizon=1)
                rows.append({
                    "market": market, "horizon": 1, "sample": scope,
                    "model_a": a, "model_b": b,
                    "n_common": len(common), "mean_loss_a": first.mean(),
                    "mean_loss_b": second.mean(), **dm,
                    "negative_difference_favors": "model_a",
                    "inference_role": "retrospective_descriptive",
                    "multiplicity_adjustment": "none",
                })
            common_frames.append(common.reset_index().assign(sample=scope))
    return pd.DataFrame(rows), pd.concat(common_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calendar-report", required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--refit-schedule", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("use a new output directory; preserve previous runs")
    data = read_daily_frame(args.data)
    schedule = pd.read_csv(args.refit_schedule)
    panels, coefficients, checks = [], [], []
    for market in ("SP500", "FTSE100", "DAX"):
        daily = benchmark_daily_frame(data, market=market)
        reference = pd.read_parquet(args.reference_dir / f"{market}.parquet")
        if set(reference.market) != {market}:
            raise ValueError("reference market identity mismatch")
        dates = pd.to_datetime(schedule.loc[schedule.market.eq(market), "refit_date"])
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("invalid refit calendar")
        pair, fits, check = reconstruct_pair(daily, reference, dates)
        context = reference.loc[reference.model_id.eq(NN) & reference.horizon.eq(1)].copy()
        for column in ("origin_date", "mature_date"):
            context[column] = pd.to_datetime(context[column], errors="raise")
        context = context.merge(pair.loc[pair.model_id.eq(CC), KEYS], on=KEYS,
                                how="inner", validate="one_to_one")
        panels.extend([pair, context])
        coefficients.append(fits)
        checks.append(check)
    args.output_dir.mkdir(parents=True)
    forecasts = args.output_dir / "forecasts.parquet"
    losses = args.output_dir / "losses.parquet"
    pd.concat(panels, ignore_index=True).to_parquet(forecasts, index=False)
    pd.concat(coefficients, ignore_index=True).to_csv(args.output_dir / "coefficients.csv", index=False)
    (args.output_dir / "baseline_reproduction.json").write_text(json.dumps(checks, indent=2) + "\n")
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("build_variance_losses.py")),
        "--data", args.data, "--calendar-report", args.calendar_report,
        "--forecasts", str(forecasts), "--output", str(losses),
    ], check=True)
    summary, common = summarize(read_loss_archive(losses))
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    common.to_csv(args.output_dir / "common_losses.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
