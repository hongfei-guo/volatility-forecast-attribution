#!/usr/bin/env python3
"""Rebuild the published forecast-evaluation tables from loss panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from bnsv.calibration import coverage_diagnostics, pit_diagnostics
from bnsv.mcs_dm import diebold_mariano, model_confidence_set
from bnsv.sample_alignment import select_evaluation_sample, strict_bool_series


MCS_COLUMNS = (
    "market",
    "horizon",
    "n_common",
    "alpha",
    "statistic",
    "mean_block_length",
    "model_universe_json",
    "retained_models_json",
    "elimination_order_json",
    "sequential_p_values_json",
)
MCS_MEAN_COLUMNS = ("market", "horizon", "model_id", "n_common", "mean_qlike")
DM_COLUMNS = (
    "market",
    "horizon",
    "loss_column",
    "model_a",
    "model_b",
    "n_common",
    "mean_loss_difference",
    "dm_statistic",
    "p_value_two_sided",
    "hac_lag",
    "loss_differential",
    "negative_favors",
    "reporting_role",
    "multiplicity_adjustment",
)
CALIBRATION_COLUMNS = (
    "market",
    "model_id",
    "horizon",
    "observations",
    "reporting_mode",
    "mean_return_crps",
    "mean_return_log_score",
    "pit_mean",
    "pit_variance",
    "pit_ks_p_value",
    "pit_ljung_box_p_value",
    "coverage_50",
    "coverage_50_exact_binomial_p_value",
    "coverage_50_mean_width",
    "coverage_90",
    "coverage_90_exact_binomial_p_value",
    "coverage_90_mean_width",
    "coverage_95",
    "coverage_95_exact_binomial_p_value",
    "coverage_95_mean_width",
)
FULL_LOSS_COLUMNS = {
    "return_crps",
    "return_log_score",
    "return_pit",
    *(
        f"return_interval_{level:02d}_{suffix}"
        for level in (50, 90, 95)
        for suffix in ("covered", "width")
    ),
}


def _ordered_unique(values: Sequence[Any], *, label: str) -> list[Any]:
    result = list(values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be a nonempty list without duplicates")
    return result


def _load_design(path: Path) -> dict[str, Any]:
    design = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("analysis design must be a mapping")
    markets = _ordered_unique(design.get("markets", []), label="markets")
    horizons = _ordered_unique(design.get("horizons", []), label="horizons")
    evaluation = design.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("analysis design lacks evaluation settings")
    mcs = evaluation.get("mcs")
    dm = evaluation.get("dm")
    if not isinstance(mcs, dict) or not isinstance(dm, dict):
        raise ValueError("analysis design lacks MCS or DM settings")
    _ordered_unique(mcs.get("models", []), label="MCS models")
    pairs = dm.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("analysis design lacks fixed DM pairs")
    allowed_horizons = set(int(value) for value in horizons)
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "loss_column",
            "model_a",
            "model_b",
            "horizons",
        }:
            raise ValueError("each DM pair must define loss, models, and horizons")
        pair_horizons = [int(value) for value in pair["horizons"]]
        if not pair_horizons or not set(pair_horizons) <= allowed_horizons:
            raise ValueError("DM pair contains an unsupported horizon")
        if pair["model_a"] == pair["model_b"]:
            raise ValueError("DM pair must contain two distinct models")
    return design


def _read_losses(paths: Sequence[Path]) -> pd.DataFrame:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("loss-panel paths must be unique")
    frames = []
    for path in resolved:
        if path.suffix.lower() not in {".parquet", ".pq"} or not path.is_file():
            raise ValueError(f"loss panel must be an existing Parquet file: {path}")
        frames.append(pd.read_parquet(path))
    losses = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "forecast_kind",
        "qlike",
    }
    missing = required - set(losses)
    if missing:
        raise ValueError(f"loss panels lack required columns: {sorted(missing)}")
    keys = ["market", "model_id", "origin_date", "horizon"]
    if losses.empty or losses.duplicated(keys).any():
        raise ValueError("loss panels are empty or contain duplicate forecast rows")
    return losses


def _validate_full_losses(selected: pd.DataFrame) -> None:
    missing = FULL_LOSS_COLUMNS - set(selected)
    if missing:
        raise ValueError(
            "full scope requires predictive-distribution columns: "
            f"{sorted(missing)}"
        )
    probabilistic = selected[
        selected["forecast_kind"].astype(str) == "probabilistic"
    ]
    if probabilistic.empty:
        raise ValueError("full scope requires probabilistic loss rows")
    for column in ("return_crps", "return_pit"):
        values = pd.to_numeric(probabilistic[column], errors="raise").to_numpy(float)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"full scope requires finite {column}")
    pit = probabilistic["return_pit"].to_numpy(float)
    if np.any((pit < 0) | (pit > 1)):
        raise ValueError("return_pit must lie in [0,1]")
    one_day = probabilistic[probabilistic["horizon"].astype(int) == 1]
    log_score = pd.to_numeric(one_day["return_log_score"], errors="raise").to_numpy(
        float
    )
    if one_day.empty or np.any(~np.isfinite(log_score)):
        raise ValueError("full scope requires finite one-day return_log_score")
    for level in (50, 90, 95):
        covered = f"return_interval_{level:02d}_covered"
        width = f"return_interval_{level:02d}_width"
        strict_bool_series(probabilistic[covered], name=covered)
        _finite_nonnegative_mean(probabilistic, width)


def _aligned_losses(
    data: pd.DataFrame,
    *,
    loss_column: str,
    models: Sequence[str],
) -> pd.DataFrame:
    requested = _ordered_unique([str(model) for model in models], label="models")
    if loss_column not in data:
        raise ValueError(f"loss panels lack {loss_column!r}")
    selected = data[data["model_id"].astype(str).isin(requested)].copy()
    observed = set(selected["model_id"].astype(str))
    missing = set(requested) - observed
    if missing:
        raise ValueError(f"requested models are absent: {sorted(missing)}")
    numeric = pd.to_numeric(selected[loss_column], errors="raise")
    finite = numeric.notna()
    if np.any(~np.isfinite(numeric[finite].to_numpy(float))):
        raise ValueError(f"{loss_column} contains non-finite values")
    selected[loss_column] = numeric
    wide = selected.pivot(
        index="origin_date", columns="model_id", values=loss_column
    ).reindex(columns=requested)
    complete = wide.dropna(axis=0, how="any")
    if complete.empty:
        raise ValueError("requested models have no common observations")
    return complete


def _evaluation_rows(
    selected: pd.DataFrame,
    *,
    design: dict[str, Any],
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = design["evaluation"]
    mcs = evaluation["mcs"]
    dm = evaluation["dm"]
    markets = [str(value) for value in design["markets"]]
    horizons = [int(value) for value in design["horizons"]]
    observed_markets = set(selected["market"].astype(str))
    unknown_markets = observed_markets - set(markets)
    if unknown_markets:
        raise ValueError(f"loss panels contain unknown markets: {sorted(unknown_markets)}")
    active_markets = [market for market in markets if market in observed_markets]
    universe = [str(value) for value in mcs["models"]]
    seed = int(design["seed"])
    mcs_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    for market in active_markets:
        market_data = selected[selected["market"].astype(str) == market]
        for horizon in horizons:
            cell = market_data[market_data["horizon"].astype(int) == horizon]
            if cell.empty:
                continue
            complete = _aligned_losses(cell, loss_column="qlike", models=universe)
            result = model_confidence_set(
                complete.to_numpy(float),
                universe,
                alpha=float(mcs["alpha"]),
                bootstrap_replications=int(mcs["replications"]),
                mean_block_length=float(mcs["mean_block_length"]),
                seed=seed,
            )
            n_common = int(len(complete))
            mcs_rows.append(
                {
                    "market": market,
                    "horizon": horizon,
                    "n_common": n_common,
                    "alpha": float(mcs["alpha"]),
                    "statistic": result.statistic,
                    "mean_block_length": result.mean_block_length,
                    "model_universe_json": json.dumps(universe),
                    "retained_models_json": json.dumps(list(result.retained_models)),
                    "elimination_order_json": json.dumps(list(result.elimination_order)),
                    "sequential_p_values_json": json.dumps(list(result.p_values)),
                }
            )
            for model in universe:
                mean_rows.append(
                    {
                        "market": market,
                        "horizon": horizon,
                        "model_id": model,
                        "n_common": n_common,
                        "mean_qlike": float(complete[model].mean()),
                    }
                )
            for pair in dm["pairs"]:
                if scope == "variance" and pair["loss_column"] != "qlike":
                    continue
                if horizon not in {int(value) for value in pair["horizons"]}:
                    continue
                model_a = str(pair["model_a"])
                model_b = str(pair["model_b"])
                paired = _aligned_losses(
                    cell,
                    loss_column=str(pair["loss_column"]),
                    models=[model_a, model_b],
                )
                comparison = diebold_mariano(
                    paired[model_a].to_numpy(float),
                    paired[model_b].to_numpy(float),
                    horizon=horizon,
                )
                dm_rows.append(
                    {
                        "market": market,
                        "horizon": horizon,
                        "loss_column": str(pair["loss_column"]),
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_common": int(len(paired)),
                        "mean_loss_difference": comparison["mean_loss_difference"],
                        "dm_statistic": comparison["statistic"],
                        "p_value_two_sided": comparison["p_value"],
                        "hac_lag": comparison["hac_lag"],
                        "loss_differential": "d = L_A - L_B",
                        "negative_favors": "model_a",
                        "reporting_role": "descriptive",
                        "multiplicity_adjustment": str(
                            dm["multiplicity_adjustment"]
                        ),
                    }
                )
    return mcs_rows, mean_rows, dm_rows


def _finite_nonnegative_mean(group: pd.DataFrame, column: str) -> float:
    values = group[column].to_numpy(float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{column} must contain finite nonnegative values")
    return float(np.mean(values))


def _calibration_rows(
    selected: pd.DataFrame,
    *,
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    markets = [str(value) for value in design["markets"]]
    horizons = [int(value) for value in design["horizons"]]
    probabilistic = selected[
        selected["forecast_kind"].astype(str) == "probabilistic"
    ].copy()
    rows: list[dict[str, Any]] = []
    for market in markets:
        market_data = probabilistic[
            probabilistic["market"].astype(str) == market
        ]
        if market_data.empty:
            continue
        models = sorted(market_data["model_id"].astype(str).unique())
        for model in models:
            model_data = market_data[market_data["model_id"].astype(str) == model]
            for horizon in horizons:
                group = model_data[model_data["horizon"].astype(int) == horizon]
                if group.empty:
                    continue
                one_day = horizon == 1
                pit = pit_diagnostics(
                    group["return_pit"].to_numpy(float),
                    lags=10,
                    include_reference_tests=one_day,
                )
                row: dict[str, Any] = {
                    "market": market,
                    "model_id": model,
                    "horizon": horizon,
                    "observations": int(len(group)),
                    "reporting_mode": (
                        "iid_reference_tests" if one_day else "descriptive_summary"
                    ),
                    "mean_return_crps": float(group["return_crps"].mean()),
                    "mean_return_log_score": (
                        float(group["return_log_score"].mean()) if one_day else np.nan
                    ),
                    "pit_mean": pit["mean"],
                    "pit_variance": pit["variance"],
                    "pit_ks_p_value": pit.get("ks_p_value", np.nan),
                    "pit_ljung_box_p_value": pit.get(
                        "ljung_box_p_value", np.nan
                    ),
                }
                for level in (50, 90, 95):
                    covered = f"return_interval_{level:02d}_covered"
                    width = f"return_interval_{level:02d}_width"
                    diagnostics = coverage_diagnostics(
                        strict_bool_series(group[covered], name=covered).to_numpy(),
                        level / 100,
                        include_reference_tests=one_day,
                    )
                    row[f"coverage_{level}"] = diagnostics["empirical_coverage"]
                    row[f"coverage_{level}_exact_binomial_p_value"] = diagnostics.get(
                        "exact_binomial_p_value", np.nan
                    )
                    row[f"coverage_{level}_mean_width"] = _finite_nonnegative_mean(
                        group, width
                    )
                rows.append(row)
    return rows


def _write_table(rows: list[dict[str, Any]], columns: Sequence[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"no rows were produced for {path.name}")
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the published forecast-evaluation CSV tables"
    )
    parser.add_argument("--losses", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("design/analysis.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("variance", "full"),
        default="variance",
        help=(
            "variance writes the three QLIKE tables; full also writes "
            "distribution calibration"
        ),
    )
    args = parser.parse_args()

    design = _load_design(args.config.resolve())
    losses = _read_losses(args.losses)
    selected = select_evaluation_sample(
        losses, str(design["evaluation"]["main_sample_policy"])
    )
    if args.scope == "full":
        _validate_full_losses(selected)
    mcs_rows, mean_rows, dm_rows = _evaluation_rows(
        selected, design=design, scope=args.scope
    )

    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"evaluation summary output must be a new or empty directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_table(mcs_rows, MCS_COLUMNS, output / "mcs_sets.csv")
    _write_table(
        mean_rows,
        MCS_MEAN_COLUMNS,
        output / "mcs_common_sample_mean_qlike.csv",
    )
    dm_name = "dm_results.csv" if args.scope == "full" else "dm_qlike.csv"
    _write_table(dm_rows, DM_COLUMNS, output / dm_name)
    if args.scope == "full":
        calibration_path = output / "distribution_calibration.csv"
        _write_table(
            _calibration_rows(selected, design=design),
            CALIBRATION_COLUMNS,
            calibration_path,
        )
        pd.read_csv(calibration_path).to_csv(calibration_path, index=False)
        print(f"Wrote four evaluation tables to {output}")
    else:
        print(f"Wrote three QLIKE evaluation tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
