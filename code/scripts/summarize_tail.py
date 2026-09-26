#!/usr/bin/env python3
"""Rebuild one-day tail-risk summary tables from analytic loss panels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from bnsv.calibration import var_backtests
from bnsv.mcs_dm import diebold_mariano
from bnsv.sample_alignment import select_evaluation_sample, strict_bool_series


MARKETS = ("SP500", "FTSE100", "DAX")
STUDENT_T_MODEL = "Realized-GARCH-t-MLE"
SUMMARY_COLUMNS = (
    "market",
    "model_id",
    "observations",
    "mean_var_es_01_fz0",
    "mean_var_01_quantile_loss",
    "var_01_exceedances",
    "var_01_exceedance_rate",
    "var_01_kupiec_p_value",
    "var_01_independence_p_value",
    "var_01_conditional_coverage_p_value",
    "mean_var_es_05_fz0",
    "mean_var_05_quantile_loss",
    "var_05_exceedances",
    "var_05_exceedance_rate",
    "var_05_kupiec_p_value",
    "var_05_independence_p_value",
    "var_05_conditional_coverage_p_value",
)
DM_COLUMNS = (
    "market",
    "alpha",
    "loss",
    "role",
    "model_a",
    "model_b",
    "sign_convention",
    "n_candidate_origins",
    "n_model_a_available",
    "n_model_b_available",
    "n_common",
    "mean_loss_model_a",
    "mean_loss_model_b",
    "alternative",
    "multiplicity_adjustment",
    "reporting_role",
    "statistic",
    "p_value",
    "mean_loss_difference",
    "hac_lag",
)


def _load_design(path: Path) -> dict[str, Any]:
    design = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError("tail design must be a mapping")
    if (
        design.get("horizon") != 1
        or design.get("evaluation_sample_policy") != "main_gap_excluded"
        or design.get("risk_levels") != [0.01, 0.05]
    ):
        raise ValueError("tail design differs from the reported evaluation")
    models = design.get("probabilistic_models")
    if not isinstance(models, dict) or len(models) != 9 or STUDENT_T_MODEL not in models:
        raise ValueError("tail design has an invalid model universe")
    core = design.get("main_table_models")
    if not isinstance(core, list) or not core or len(core) != len(set(core)):
        raise ValueError("tail design has an invalid main-table model list")
    if set(core) - set(models):
        raise ValueError("main-table models lie outside the tail model universe")
    pairs = design.get("fixed_dm_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("tail design lacks fixed DM pairs")
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"model_a", "model_b", "role"}:
            raise ValueError("each fixed DM pair must define two models and one role")
        if pair["model_a"] == pair["model_b"]:
            raise ValueError("a fixed DM pair must contain two distinct models")
        if pair["model_a"] not in models or pair["model_b"] not in models:
            raise ValueError("a fixed DM pair contains a model outside the tail universe")
    return design


def _read_losses(paths: Sequence[Path], design: dict[str, Any]) -> pd.DataFrame:
    if len(paths) != len(MARKETS):
        raise ValueError("provide exactly one analytic tail loss panel per market")
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("analytic tail loss-panel paths must be unique")
    required = {
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "target_dates_json",
        "mature_date",
        "realized_variance",
        "realized_cumulative_return",
        "target_spans_archive_gap",
        "origin_follows_archive_gap",
        "main_evaluation_eligible",
        "var_01_exceedance",
        "var_01_quantile_loss",
        "var_es_01_fz0",
        "var_05_exceedance",
        "var_05_quantile_loss",
        "var_es_05_fz0",
    }
    by_market: dict[str, pd.DataFrame] = {}
    expected_models = set(design["probabilistic_models"])
    for path in resolved:
        if path.suffix.lower() not in {".parquet", ".pq"} or not path.is_file():
            raise ValueError(f"tail loss panel must be an existing Parquet file: {path}")
        frame = pd.read_parquet(path)
        missing = required - set(frame)
        if missing:
            raise ValueError(f"tail loss panel {path} lacks columns: {sorted(missing)}")
        markets = frame["market"].dropna().astype(str).unique().tolist()
        if len(markets) != 1 or markets[0] not in MARKETS:
            raise ValueError(f"tail loss panel must contain one reported market: {path}")
        market = markets[0]
        if market in by_market:
            raise ValueError(f"multiple tail loss panels supplied for {market}")
        if set(frame["model_id"].dropna().astype(str)) != expected_models:
            raise ValueError(f"tail model universe differs for {market}")
        if set(pd.to_numeric(frame["horizon"], errors="raise").astype(int)) != {1}:
            raise ValueError(f"tail loss panel contains a non-one-day horizon: {market}")
        keys = ["market", "model_id", "origin_date", "horizon"]
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError(f"tail loss panel is empty or duplicated: {market}")
        frame["origin_date"] = pd.to_datetime(
            frame["origin_date"], errors="raise"
        ).dt.normalize()
        by_market[market] = frame.sort_values(
            ["model_id", "origin_date"]
        ).reset_index(drop=True)
    if set(by_market) != set(MARKETS):
        raise ValueError("tail loss panels do not cover all reported markets")
    return pd.concat([by_market[market] for market in MARKETS], ignore_index=True)


def _model_summaries(frame: pd.DataFrame, *, sample_policy: str) -> pd.DataFrame:
    selected = select_evaluation_sample(frame, sample_policy)
    rows: list[dict[str, Any]] = []
    for (market, model), group in selected.groupby(
        ["market", "model_id"], sort=False
    ):
        row: dict[str, Any] = {
            "market": str(market),
            "model_id": str(model),
            "observations": int(len(group)),
        }
        for alpha in (0.01, 0.05):
            label = f"{int(alpha * 100):02d}"
            exceedance = strict_bool_series(
                group[f"var_{label}_exceedance"],
                name=f"var_{label}_exceedance",
            ).to_numpy()
            backtest = var_backtests(exceedance, alpha)
            row.update(
                {
                    f"mean_var_es_{label}_fz0": float(
                        group[f"var_es_{label}_fz0"].mean()
                    ),
                    f"mean_var_{label}_quantile_loss": float(
                        group[f"var_{label}_quantile_loss"].mean()
                    ),
                    f"var_{label}_exceedances": int(backtest["exceedances"]),
                    f"var_{label}_exceedance_rate": float(
                        backtest["exceedance_rate"]
                    ),
                    f"var_{label}_kupiec_p_value": float(backtest["kupiec_p_value"]),
                    f"var_{label}_independence_p_value": float(
                        backtest["christoffersen_independence_p_value"]
                    ),
                    f"var_{label}_conditional_coverage_p_value": float(
                        backtest["conditional_coverage_p_value"]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def _validate_common_targets(subset: pd.DataFrame, models: tuple[str, str]) -> None:
    common_counts = subset.groupby("origin_date")["model_id"].nunique()
    common_origins = common_counts[common_counts == 2].index
    common = subset[subset["origin_date"].isin(common_origins)]
    for column in ("target_dates_json", "mature_date"):
        if (common.groupby("origin_date")[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"pairwise common targets disagree on {column}: {models}")
    for column in ("realized_variance", "realized_cumulative_return"):
        spread = common.groupby("origin_date")[column].agg(
            lambda values: np.ptp(values)
        )
        if (spread > 1e-12).any():
            raise ValueError(f"pairwise realized targets disagree on {column}: {models}")


def _fixed_dm(frame: pd.DataFrame, *, design: dict[str, Any]) -> pd.DataFrame:
    selected = select_evaluation_sample(
        frame, str(design["evaluation_sample_policy"])
    )
    rows: list[dict[str, Any]] = []
    for market in MARKETS:
        market_frame = selected[selected["market"].astype(str) == market]
        for pair in design["fixed_dm_pairs"]:
            model_a = str(pair["model_a"])
            model_b = str(pair["model_b"])
            subset = market_frame[
                market_frame["model_id"].astype(str).isin((model_a, model_b))
            ].copy()
            if set(subset["model_id"].astype(str)) != {model_a, model_b}:
                raise ValueError(
                    f"fixed DM pair is unavailable: {market} {model_a} {model_b}"
                )
            _validate_common_targets(subset, (model_a, model_b))
            origins_a = set(
                subset.loc[subset["model_id"] == model_a, "origin_date"].astype(str)
            )
            origins_b = set(
                subset.loc[subset["model_id"] == model_b, "origin_date"].astype(str)
            )
            for alpha in (0.01, 0.05):
                label = f"{int(alpha * 100):02d}"
                loss = f"var_es_{label}_fz0"
                wide = subset.pivot(
                    index="origin_date", columns="model_id", values=loss
                )
                complete = wide.reindex(columns=[model_a, model_b]).dropna()
                if len(complete) < 20:
                    raise ValueError(
                        f"too few common observations for {market} {pair['role']}"
                    )
                comparison = diebold_mariano(
                    complete[model_a].to_numpy(float),
                    complete[model_b].to_numpy(float),
                    horizon=1,
                )
                rows.append(
                    {
                        "market": market,
                        "alpha": alpha,
                        "loss": loss,
                        "role": str(pair["role"]),
                        "model_a": model_a,
                        "model_b": model_b,
                        "sign_convention": "negative_mean_difference_favors_model_a",
                        "n_candidate_origins": len(origins_a | origins_b),
                        "n_model_a_available": len(origins_a),
                        "n_model_b_available": len(origins_b),
                        "n_common": int(len(complete)),
                        "mean_loss_model_a": float(complete[model_a].mean()),
                        "mean_loss_model_b": float(complete[model_b].mean()),
                        "alternative": "two_sided",
                        "multiplicity_adjustment": "none",
                        "reporting_role": "descriptive",
                        **comparison,
                    }
                )
    return pd.DataFrame(rows, columns=list(DM_COLUMNS))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild one-day tail-risk tables from analytic loss panels"
    )
    parser.add_argument("--losses", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("design/tail_evaluation.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    design = _load_design(args.config.resolve())
    losses = _read_losses(args.losses, design)
    summary = _model_summaries(
        losses, sample_policy=str(design["evaluation_sample_policy"])
    )
    comparisons = _fixed_dm(losses, design=design)

    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"tail summary output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "all_models.csv", index=False)
    summary[summary["model_id"].isin(design["main_table_models"])].to_csv(
        output / "core_models.csv", index=False
    )
    comparisons.to_csv(output / "pairwise_dm.csv", index=False)
    print(f"Wrote three tail-risk tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
