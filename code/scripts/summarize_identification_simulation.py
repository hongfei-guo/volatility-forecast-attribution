#!/usr/bin/env python3
"""Summarize the identification simulation.

All 200 replications per DGP remain in the planned denominator. Continuous
metrics condition on successful fits. Coverage and declaration rates also use
the planned denominator, with unsuccessful fits treated as specified below.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import binom

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from bnsv.identification_simulation import clopper_pearson_interval  # noqa: E402


SCALARS = ("mu", "phi", "sigma_h", "sigma_eta", "nu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--records', type=Path, help='per-replication scientific records')
    inputs.add_argument('--preparation', type=Path, help='prepared synthetic datasets')
    parser.add_argument('--fits', type=Path, help='fit records and compact posteriors')
    parser.add_argument('--config', type=Path, default=PROJECT/'design/identification.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.preparation and not args.fits:
        parser.error('--preparation requires --fits')
    if args.records and args.fits:
        parser.error('--fits is only used with --preparation')
    return args



def resolve_dataset_path(path_value: str, preparation: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (preparation / path).resolve()


def crps_sample(draws: np.ndarray, truth: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    n = x.size
    weights = 2 * np.arange(1, n + 1) - n - 1
    mean_pairwise = 2.0 * float(np.dot(weights, x)) / (n * n)
    return float(np.mean(np.abs(x - truth)) - 0.5 * mean_pairwise)


def evaluate_function(
    compact: dict[str, np.ndarray],
    x: np.ndarray,
    truth: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    weights = compact["weights"]
    bias = compact["bias"]
    output = compact["output_weight"]
    centering = compact["centering"]
    x = np.asarray(x, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if x.shape[0] != truth.size or x.shape[1] != weights.shape[1]:
        raise RuntimeError("function-evaluation design is incompatible with posterior weights")
    posterior_mean = np.zeros(truth.size)
    covered = np.zeros(truth.size, dtype=bool)
    chunk = 256
    for start in range(0, truth.size, chunk):
        stop = min(truth.size, start + chunk)
        raw = np.einsum(
            "eth,eh->et",
            np.tanh(np.einsum("td,edh->eth", x[start:stop], weights) + bias[:, None, :]),
            output,
        )
        values = raw - centering[:, None]
        posterior_mean[start:stop] = np.mean(values, axis=0)
        lower, upper = np.quantile(values, [0.025, 0.975], axis=0)
        covered[start:stop] = (lower <= truth[start:stop]) & (truth[start:stop] <= upper)
    error = posterior_mean - truth
    variance = float(np.var(truth))
    metrics = {
        "function_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "function_nrmse": (
            float(np.sqrt(np.mean(np.square(error)) / variance)) if variance > 0 else math.nan
        ),
        "function_integrated_bias": float(np.mean(error)),
        "function_pointwise_coverage": float(np.mean(covered)),
        "posterior_mean_function_sd": float(np.std(posterior_mean, ddof=0)),
    }
    return metrics, posterior_mean


def scalar_rows(
    task: dict[str, Any],
    compact: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for name in SCALARS:
        draws = compact[name]
        truth = float(dataset[f"{name}_truth"][0])
        lower, upper = np.quantile(draws, [0.025, 0.975])
        estimate = float(np.mean(draws))
        rows.append(
            {
                "dgp_id": task["dgp_id"],
                "replication_id": task["replication_id"],
                "parameter": name,
                "truth": truth,
                "posterior_mean": estimate,
                "posterior_sd": float(np.std(draws, ddof=1)),
                "bias": estimate - truth,
                "squared_error": (estimate - truth) ** 2,
                "interval_lower": float(lower),
                "interval_upper": float(upper),
                "interval_width": float(upper - lower),
                "covered": bool(lower <= truth <= upper),
            }
        )
    return rows


def aggregate_scalar(
    rows: pd.DataFrame,
    *,
    planned_by_dgp: dict[str, int],
    unsuccessful_by_dgp: dict[str, int],
    coverage_rejection_cutoff: int,
) -> list[dict[str, Any]]:
    output = []
    for (dgp, parameter), group in rows.groupby(["dgp_id", "parameter"], sort=True):
        bias = group["bias"].to_numpy(float)
        sq = group["squared_error"].to_numpy(float)
        covered = int(group["covered"].sum())
        successful = len(group)
        planned = int(planned_by_dgp[dgp])
        unsuccessful = int(unsuccessful_by_dgp[dgp])
        if successful + unsuccessful != planned:
            raise RuntimeError(f"planned denominator mismatch for {dgp}")
        lower, upper = clopper_pearson_interval(covered, planned)
        mse = float(np.mean(sq))
        mse_mcse = float(np.std(sq, ddof=1) / math.sqrt(successful)) if successful > 1 else math.nan
        rmse = math.sqrt(mse)
        output.append(
            {
                "dgp_id": dgp,
                "parameter": parameter,
                "planned_replications": planned,
                "successful_replications": successful,
                "unsuccessful_replications": unsuccessful,
                "unsuccessful_rate": unsuccessful / planned,
                "mean_bias_conditional_successful": float(np.mean(bias)),
                "bias_mcse_conditional_successful": (
                    float(np.std(bias, ddof=1) / math.sqrt(successful)) if successful > 1 else math.nan
                ),
                "empirical_sd_conditional_successful": float(np.std(group["posterior_mean"], ddof=1)),
                "mean_posterior_sd_conditional_successful": float(np.mean(group["posterior_sd"])),
                "rmse_conditional_successful": rmse,
                "rmse_mcse_delta_conditional_successful": (
                    mse_mcse / (2 * rmse) if rmse > 0 else 0.0
                ),
                "mean_interval_width_conditional_successful": float(np.mean(group["interval_width"])),
                "covered": covered,
                "coverage_over_planned_replications": covered / planned,
                "coverage_clopper_pearson_95_planned_denominator": [lower, upper],
                "coverage_rejected": covered <= coverage_rejection_cutoff,
            }
        )
    return output


def median_order_interval(values: Iterable[float], level: float = 0.95) -> dict[str, Any]:
    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    if n == 0:
        return {"n": 0, "lower_order": None, "upper_order": None, "interval": [None, None], "attained_coverage": None}
    candidates = []
    for lower in range(1, n // 2 + 1):
        upper = n - lower + 1
        coverage = float(binom.cdf(upper - 1, n, 0.5) - binom.cdf(lower - 1, n, 0.5))
        if coverage >= level:
            candidates.append((lower, upper, coverage))
    if not candidates:
        return {"n": n, "lower_order": 1, "upper_order": n, "interval": [float(x[0]), float(x[-1])], "attained_coverage": float(1 - 2 * 0.5**n)}
    lower, upper, coverage = max(candidates, key=lambda item: item[0])
    return {
        "n": n,
        "lower_order": lower,
        "upper_order": upper,
        "interval": [float(x[lower - 1]), float(x[upper - 1])],
        "attained_coverage": coverage,
    }


def continuous_summary(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "mean_mcse": None, "median": None, "q25": None, "q75": None, "median_distribution_free_interval": median_order_interval([])}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "mean_mcse": float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else None,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "median_distribution_free_interval": median_order_interval(x),
    }


def aggregate_recovery_frame(
    frame: pd.DataFrame,
    *,
    planned_by_dgp: dict[str, int],
    unsuccessful_by_dgp: dict[str, int],
) -> list[dict[str, Any]]:
    output = []
    metrics = (
        "function_rmse",
        "function_nrmse",
        "function_integrated_bias",
        "posterior_mean_function_sd",
        "one_step_variance_bias",
        "one_step_variance_squared_error",
        "one_step_variance_interval_width",
        "crps_h1",
        "crps_h5",
        "crps_h10",
    )
    for dgp in ("DGP-N", "DGP-0"):
        group = frame[frame["dgp_id"] == dgp]
        planned = int(planned_by_dgp[dgp])
        unsuccessful = int(unsuccessful_by_dgp[dgp])
        if len(group) + unsuccessful != planned:
            raise RuntimeError(f"functional denominator mismatch for {dgp}")
        record: dict[str, Any] = {
            "dgp_id": dgp,
            "planned_replications": planned,
            "successful_replications": len(group),
            "unsuccessful_replications": unsuccessful,
            "unsuccessful_rate": unsuccessful / planned,
            "continuous_metrics_conditional_successful": {},
        }
        for metric in metrics:
            if metric in group:
                record["continuous_metrics_conditional_successful"][metric] = continuous_summary(group[metric])
        function_covered_sum = float(group["function_pointwise_coverage"].sum()) if len(group) else 0.0
        variance_covered = int(group["one_step_variance_covered"].sum()) if len(group) else 0
        record["function_pointwise_coverage_mean_conditional_successful"] = (
            float(group["function_pointwise_coverage"].mean()) if len(group) else None
        )
        record["function_pointwise_coverage_over_planned_replications"] = function_covered_sum / planned
        record["one_step_variance_covered"] = variance_covered
        record["one_step_variance_rmse_conditional_successful"] = (
            float(np.sqrt(group["one_step_variance_squared_error"].mean()))
            if len(group)
            else None
        )
        record["one_step_variance_coverage_over_planned_replications"] = variance_covered / planned
        record["one_step_variance_coverage_clopper_pearson_95"] = list(
            clopper_pearson_interval(variance_covered, planned)
        )
        output.append(record)
    return output


def descriptive_logistic_declaration_curve(
    truth: pd.DataFrame,
    successful: pd.DataFrame,
    *,
    q_c: float,
) -> dict[str, Any]:
    """Fit the unpenalised descriptive curve or report omission."""
    columns = [
        "replication_id",
        "material_correction_declaration",
        "posterior_probability_index_at_least_qc",
    ]
    merged = truth[[
        "replication_id", "true_relative_correction_variance_index"
    ]].merge(successful[columns], on="replication_id", how="left", validate="one_to_one")
    merged["unsuccessful_fit"] = merged["material_correction_declaration"].isna()
    merged["declaration"] = merged["material_correction_declaration"].eq(True)
    x = (
        merged["true_relative_correction_variance_index"].to_numpy(float) - q_c
    ) / q_c
    y = merged["declaration"].to_numpy(float)
    unsuccessful_positions = [
        {
            "replication_id": int(row.replication_id),
            "true_relative_correction_variance_index": float(
                row.true_relative_correction_variance_index
            ),
            "normalised_true_index": float(x[i]),
        }
        for i, row in enumerate(merged.itertuples(index=False))
        if bool(row.unsuccessful_fit)
    ]

    def omitted(reason: str) -> dict[str, Any]:
        return {
            "fit_outcome": "omitted",
            "reason": reason,
            "planned_replications": int(len(merged)),
            "unsuccessful_fits_enter_as_non_declarations": True,
            "unsuccessful_positions": unsuccessful_positions,
        }

    if not np.all(np.isfinite(x)) or q_c <= 0:
        return omitted("nonfinite_or_invalid_true_index")
    if np.unique(y).size < 2:
        return omitted("single_outcome_class")
    x_zero, x_one = x[y == 0], x[y == 1]
    if max(x_zero) <= min(x_one) or max(x_one) <= min(x_zero):
        return omitted("complete_or_quasi_complete_separation")

    design = np.column_stack((np.ones(len(x)), x))

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        probability = np.where(
            eta >= 0,
            1.0 / (1.0 + np.exp(-eta)),
            np.exp(eta) / (1.0 + np.exp(eta)),
        )
        return design.T @ (probability - y)

    fit = minimize(objective, np.zeros(2), jac=gradient, method="BFGS")
    beta = np.asarray(fit.x, dtype=float)
    if not fit.success:
        return omitted("ordinary_logistic_fit_did_not_converge")
    if not np.all(np.isfinite(beta)):
        return omitted("nonfinite_coefficient")

    eta = design @ beta
    probability = np.where(
        eta >= 0,
        1.0 / (1.0 + np.exp(-eta)),
        np.exp(eta) / (1.0 + np.exp(eta)),
    )
    weights = probability * (1.0 - probability)
    information = design.T @ (weights[:, None] * design)
    if np.linalg.matrix_rank(information) < 2:
        return omitted("singular_covariance_matrix")
    try:
        bread = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return omitted("singular_covariance_matrix")
    residual = y - probability
    meat = design.T @ (np.square(residual)[:, None] * design)
    covariance = bread @ meat @ bread * len(x) / (len(x) - design.shape[1])
    if not np.all(np.isfinite(covariance)) or np.linalg.matrix_rank(covariance) < 2:
        return omitted("singular_or_nonfinite_hc1_covariance")

    true_index = merged["true_relative_correction_variance_index"].to_numpy(float)
    order = np.argsort(true_index)
    x_plot = x[order]
    true_index_plot = true_index[order]
    plot_design = np.column_stack((np.ones(len(x_plot)), x_plot))
    eta_plot = plot_design @ beta
    fitted = np.where(
        eta_plot >= 0,
        1.0 / (1.0 + np.exp(-eta_plot)),
        np.exp(eta_plot) / (1.0 + np.exp(eta_plot)),
    )
    eta_variance = np.einsum("ij,jk,ik->i", plot_design, covariance, plot_design)
    if np.any(eta_variance < -1e-12) or not np.all(np.isfinite(eta_variance)):
        return omitted("nonfinite_standard_error")
    standard_error = fitted * (1.0 - fitted) * np.sqrt(np.maximum(eta_variance, 0.0))
    if not np.all(np.isfinite(standard_error)):
        return omitted("nonfinite_standard_error")
    rows = [
        {
            "true_relative_correction_variance_index": float(true_index_value),
            "normalised_true_index": float(x_value),
            "fitted_declaration_probability": float(p_value),
            "pointwise_95_lower": float(max(0.0, p_value - 1.96 * se_value)),
            "pointwise_95_upper": float(min(1.0, p_value + 1.96 * se_value)),
        }
        for true_index_value, x_value, p_value, se_value in zip(
            true_index_plot, x_plot, fitted, standard_error
        )
    ]
    return {
        "fit_outcome": "estimated",
        "method": "ordinary_unpenalised_logistic",
        "covariance": "HC1_sandwich",
        "interval": "pointwise_95_delta_method",
        "planned_replications": int(len(merged)),
        "unsuccessful_fits_enter_as_non_declarations": True,
        "unsuccessful_positions": unsuccessful_positions,
        "coefficients_or_tests_reported": False,
        "curve": rows,
    }


def write_dgpn_declaration_curve(
    output: Path,
    truth: pd.DataFrame,
    successful: pd.DataFrame,
    calibration: dict[str, Any],
    *,
    q_c: float,
) -> None:
    """Write the descriptive declaration curve and its data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if calibration["fit_outcome"] == "estimated":
        curve = pd.DataFrame(calibration["curve"])
        curve.to_csv(output / "dgp_n_logistic_declaration_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.fill_between(
            curve["true_relative_correction_variance_index"],
            curve["pointwise_95_lower"],
            curve["pointwise_95_upper"],
            color="0.85",
            linewidth=0,
            label="95% pointwise interval",
        )
        ax.plot(
            curve["true_relative_correction_variance_index"],
            curve["fitted_declaration_probability"],
            color="black",
            linewidth=1.4,
            label="Fitted declaration probability",
        )
        unsuccessful_x = [
            row["true_relative_correction_variance_index"]
            for row in calibration["unsuccessful_positions"]
        ]
        if unsuccessful_x:
            ax.plot(
                unsuccessful_x,
                np.zeros(len(unsuccessful_x)),
                "|",
                color="tab:red",
                markersize=8,
                label="Unsuccessful fits",
            )
        ax.axvline(
            q_c,
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            label="Materiality benchmark",
        )
        ax.set(
            xlabel="Pathwise true correction-variance index",
            ylabel="Declaration probability",
            ylim=(-0.03, 1.03),
        )
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(output / "dgp_n_logistic_declaration_curve.png", dpi=200)
        plt.close(fig)


def mcmc_diagnostic_row(
    task: dict[str, Any], fit_result: dict[str, Any], compact: dict[str, np.ndarray] | None
) -> dict[str, Any]:
    sampling_runs = fit_result.get("sampling_runs") or []
    last_run = sampling_runs[-1] if sampling_runs else {}
    assessment = last_run.get("assessment") or {}
    row: dict[str, Any] = {
        "dgp_id": task["dgp_id"],
        "replication_id": task["replication_id"],
        "fit_outcome": fit_result.get("fit_outcome"),
        "sampling_runs": len(sampling_runs),
        "last_adapt_delta": last_run.get("adapt_delta"),
    }
    for key in (
        "divergences",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "min_bfmi",
        "treedepth_fraction",
        "max_mcse_over_sd",
    ):
        row[key] = assessment.get(key)
    if compact is not None:
        leapfrog = compact.get("sampler__n_leapfrog")
        stepsize = compact.get("sampler__stepsize")
        row["median_n_leapfrog"] = float(np.median(leapfrog)) if leapfrog is not None else None
        row["median_stepsize"] = float(np.median(stepsize)) if stepsize is not None else None
    else:
        row["median_n_leapfrog"] = None
        row["median_stepsize"] = None
    return row


def build_simulation_summary(
    scalar_summary: list[dict[str, Any]],
    functional_summary: list[dict[str, Any]],
    replication_frame: pd.DataFrame,
    *,
    planned_by_dgp: dict[str, int],
    unsuccessful_by_dgp: dict[str, int],
) -> pd.DataFrame:
    """Return the compact simulation table reported with the paper."""
    scalar_map = {
        (str(row["dgp_id"]), str(row["parameter"])): row
        for row in scalar_summary
    }
    functional_map = {str(row["dgp_id"]): row for row in functional_summary}
    rows: list[dict[str, Any]] = []
    for dgp in ("DGP-N", "DGP-0"):
        phi = scalar_map[(dgp, "phi")]
        function = functional_map[dgp]
        continuous = function["continuous_metrics_conditional_successful"]
        successful = replication_frame[replication_frame["dgp_id"] == dgp]
        declarations = int(successful["material_correction_declaration"].sum())
        planned = int(planned_by_dgp[dgp])
        declaration_interval = clopper_pearson_interval(declarations, planned)
        phi_interval = phi["coverage_clopper_pearson_95_planned_denominator"]
        variance_interval = function[
            "one_step_variance_coverage_clopper_pearson_95"
        ]
        rows.append(
            {
                "dgp_id": dgp,
                "planned_replications": planned,
                "successful_fits": int(len(successful)),
                "unsuccessful_fits": int(unsuccessful_by_dgp[dgp]),
                "phi_mean_bias_conditional_successful": phi[
                    "mean_bias_conditional_successful"
                ],
                "phi_rmse_conditional_successful": phi[
                    "rmse_conditional_successful"
                ],
                "phi_coverage_planned_denominator": phi[
                    "coverage_over_planned_replications"
                ],
                "phi_coverage_lower_95": phi_interval[0],
                "phi_coverage_upper_95": phi_interval[1],
                "median_function_nrmse_conditional_successful": continuous[
                    "function_nrmse"
                ]["median"],
                "mean_function_pointwise_coverage_conditional_successful": function[
                    "function_pointwise_coverage_mean_conditional_successful"
                ],
                "one_step_variance_rmse_conditional_successful": function[
                    "one_step_variance_rmse_conditional_successful"
                ],
                "one_step_variance_coverage_planned_denominator": function[
                    "one_step_variance_coverage_over_planned_replications"
                ],
                "one_step_variance_coverage_lower_95": variance_interval[0],
                "one_step_variance_coverage_upper_95": variance_interval[1],
                "material_correction_declaration_count": declarations,
                "material_correction_declaration_rate_planned_denominator": (
                    declarations / planned
                ),
                "material_correction_declaration_lower_95": declaration_interval[0],
                "material_correction_declaration_upper_95": declaration_interval[1],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.records:
        config = yaml.safe_load(args.config.read_text())
        planned = int(config['replications']['per_dgp'])
        planned_by_dgp = {name: planned for name in config['dgp']['ids']}
        q_c = float(config['dgp']['anchor']['target_correction_share'])
        exact = {'coverage_rejection_cutoff': config['reporting']['coverage_rejection_at_or_below']}
        names = ['scalar_recovery', 'functional_recovery', 'fit_outcomes', 'mcmc_diagnostics', 'dgp_truth_by_replication']
        frames = {name: pd.read_csv(args.records/(name+'.csv')) for name in names}
        scalar_frame, replication_frame = frames['scalar_recovery'], frames['functional_recovery']
        outcome_frame, mcmc_frame, truth_frame = frames['fit_outcomes'], frames['mcmc_diagnostics'], frames['dgp_truth_by_replication']
        expected = {(name, i) for name in planned_by_dgp for i in range(1, planned+1)}
        def keys(frame):
            if frame.duplicated(['dgp_id', 'replication_id']).any():
                raise ValueError('duplicate simulation records')
            return set(zip(frame.dgp_id, frame.replication_id))
        for frame in [outcome_frame, mcmc_frame, truth_frame]:
            if keys(frame) != expected:
                raise ValueError('simulation records must retain all planned replications')
        if not set(outcome_frame.fit_outcome).issubset({'successful', 'unsuccessful'}):
            raise ValueError('unknown fit outcome')
        successful = outcome_frame[outcome_frame.fit_outcome == 'successful']
        successful_keys = keys(successful)
        if keys(replication_frame) != successful_keys:
            raise ValueError('functional metrics must match successful fits')
        if scalar_frame.duplicated(['dgp_id', 'replication_id', 'parameter']).any():
            raise ValueError('duplicate scalar record')
        if set(zip(scalar_frame.dgp_id, scalar_frame.replication_id, scalar_frame.parameter)) != {(d, i, q) for d, i in successful_keys for q in SCALARS}:
            raise ValueError('scalar records must cover every parameter of successful fits')
        unsuccessful_by_dgp = {name: int(((outcome_frame.dgp_id == name) & (outcome_frame.fit_outcome == 'unsuccessful')).sum()) for name in planned_by_dgp}
        tasks = outcome_frame.to_dict('records')
    else:
        preparation, fits = args.preparation.resolve(), args.fits.resolve()
        plan = json.loads((preparation / "tasks.json").read_text(encoding="utf-8"))
        tasks = plan["tasks"]
        exact = json.loads((preparation / "design_constants.json").read_text(encoding="utf-8"))
        if len(tasks) != int(exact["total_fits"]):
            raise RuntimeError("task count disagrees with exact design constants")
        missing = [
            t["task_id"]
            for t in tasks
            if not (fits / f"task_{int(t['task_id']):03d}" / "fit_result.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"simulation output is incomplete; missing fits: {missing}")

        scalar: list[dict[str, Any]] = []
        replication: list[dict[str, Any]] = []
        outcome_rows: list[dict[str, Any]] = []
        mcmc_rows: list[dict[str, Any]] = []
        truth_rows: list[dict[str, Any]] = []
        planned_by_dgp = {
            dgp: sum(task["dgp_id"] == dgp for task in tasks)
            for dgp in ("DGP-N", "DGP-0")
        }
        unsuccessful_by_dgp = {"DGP-N": 0, "DGP-0": 0}
        q_c = float(json.loads((preparation / "amplitude_calibration.json").read_text())["target_median_share"])

        for task in tasks:
            with np.load(resolve_dataset_path(task["dataset_path"], preparation), allow_pickle=False) as dataset:
                true_index = float(dataset["correction_share"][0])
            truth_rows.append(
                {
                    "dgp_id": task["dgp_id"],
                    "replication_id": int(task["replication_id"]),
                    "true_relative_correction_variance_index": true_index,
                    "true_index_at_least_q_c": bool(true_index >= q_c),
                }
            )

        for task in tasks:
            task_dir = fits / f"task_{int(task['task_id']):03d}"
            fit_result = json.loads((task_dir / "fit_result.json").read_text())
            fit_outcome = fit_result.get("fit_outcome")
            outcome_rows.append(
                {
                    "dgp_id": task["dgp_id"],
                    "replication_id": task["replication_id"],
                    "fit_outcome": fit_outcome,
                }
            )
            if fit_outcome == "unsuccessful":
                unsuccessful_by_dgp[task["dgp_id"]] += 1
                mcmc_rows.append(mcmc_diagnostic_row(task, fit_result, None))
                continue
            if fit_outcome != "successful":
                raise RuntimeError(f"task {task['task_id']} has unknown fit outcome {fit_outcome}")
            compact_path = task_dir / str(
                fit_result.get("posterior_file", "posterior_compact.npz")
            )
            if not compact_path.is_file():
                raise RuntimeError(f"task {task['task_id']} compact posterior is missing")
            dataset_path = resolve_dataset_path(task["dataset_path"], preparation)
            with np.load(compact_path, allow_pickle=False) as c, np.load(dataset_path, allow_pickle=False) as d:
                compact = {name: np.asarray(c[name]) for name in c.files}
                dataset = {name: np.asarray(d[name]) for name in d.files}
            mcmc_rows.append(mcmc_diagnostic_row(task, fit_result, compact))
            scalar.extend(scalar_rows(task, compact, dataset))
            n = int(dataset["estimation_rows"][0])
            x_eval = dataset["X"][:n]
            fm, profile = evaluate_function(compact, x_eval, dataset["correction_truth"][:n])
            del profile
            variance_draws = compact["exact_one_step_predictive_variance"]
            truth_variance = math.exp(
                float(dataset["mu_truth"][0])
                + float(dataset["phi_truth"][0]) * (float(dataset["b_truth"][n - 1]) - float(dataset["mu_truth"][0]))
                + float(dataset["correction_truth"][n])
                + 0.5 * float(dataset["sigma_eta_truth"][0]) ** 2
            )
            lo, hi = np.quantile(variance_draws, [0.025, 0.975])
            forecast_truth = compact["forecast_truth_cumulative_standardized_return"]
            forecast_draws = compact["forecast_cumulative_standardized_return"]
            row = {
                "dgp_id": task["dgp_id"],
                "replication_id": task["replication_id"],
                "true_relative_correction_variance_index": float(dataset["correction_share"][0]),
                "posterior_median_relative_correction_variance_index": float(np.median(compact["correction_share"])),
                "posterior_probability_index_at_least_qc": float(np.mean(compact["correction_share"] >= q_c)),
                "material_correction_declaration": bool(np.mean(compact["correction_share"] >= q_c) >= 0.95),
                "one_step_variance_truth": truth_variance,
                "one_step_variance_posterior_mean": float(np.mean(variance_draws)),
                "one_step_variance_bias": float(np.mean(variance_draws)) - truth_variance,
                "one_step_variance_squared_error": (float(np.mean(variance_draws)) - truth_variance) ** 2,
                "one_step_variance_covered": bool(lo <= truth_variance <= hi),
                "one_step_variance_interval_width": float(hi - lo),
                **fm,
            }
            for j, horizon in enumerate(compact["forecast_horizons"]):
                row[f"crps_h{int(horizon)}"] = crps_sample(forecast_draws[:, j], float(forecast_truth[j]))
            replication.append(row)

        scalar_frame = pd.DataFrame(scalar)
        replication_frame = pd.DataFrame(replication)
        outcome_frame = pd.DataFrame(outcome_rows)
        mcmc_frame = pd.DataFrame(mcmc_rows)
        truth_frame = pd.DataFrame(truth_rows)
    scalar_frame.to_csv(output / "scalar_recovery.csv", index=False)
    replication_frame.to_csv(output / "functional_recovery.csv", index=False)
    truth_frame.to_csv(output / "dgp_truth_by_replication.csv", index=False)
    outcome_frame.to_csv(output / "fit_outcomes.csv", index=False)
    mcmc_frame.to_csv(output / "mcmc_diagnostics.csv", index=False)

    scalar_summary = aggregate_scalar(
        scalar_frame,
        planned_by_dgp=planned_by_dgp,
        unsuccessful_by_dgp=unsuccessful_by_dgp,
        coverage_rejection_cutoff=int(exact["coverage_rejection_cutoff"]),
    )
    functional_summary = aggregate_recovery_frame(
        replication_frame,
        planned_by_dgp=planned_by_dgp,
        unsuccessful_by_dgp=unsuccessful_by_dgp,
    )

    dgpn_truth = truth_frame[truth_frame["dgp_id"] == "DGP-N"].copy()
    logistic_declaration_curve = descriptive_logistic_declaration_curve(
        dgpn_truth,
        replication_frame[replication_frame["dgp_id"] == "DGP-N"],
        q_c=q_c,
    )
    write_dgpn_declaration_curve(
        output,
        dgpn_truth,
        replication_frame,
        logistic_declaration_curve,
        q_c=q_c,
    )

    summary = build_simulation_summary(
        scalar_summary,
        functional_summary,
        replication_frame,
        planned_by_dgp=planned_by_dgp,
        unsuccessful_by_dgp=unsuccessful_by_dgp,
    )
    declarations = truth_frame.merge(
        replication_frame[['dgp_id', 'replication_id', 'material_correction_declaration']],
        on=['dgp_id', 'replication_id'], how='left', validate='one_to_one')
    declarations['material_correction_declaration'] = declarations['material_correction_declaration'].eq(True)
    for i, row in summary.iterrows():
        dgp = row['dgp_id']
        recovery = next(r for r in functional_summary if r['dgp_id'] == dgp)
        summary.loc[i, 'mean_function_pointwise_coverage_planned_denominator'] = recovery['function_pointwise_coverage_over_planned_replications']
        group = declarations[declarations.dgp_id == dgp]
        for label, mask in [('at_or_above_qc', group.true_relative_correction_variance_index >= q_c),
                            ('below_qc', group.true_relative_correction_variance_index < q_c)]:
            summary.loc[i, 'replications_' + label] = int(mask.sum())
            summary.loc[i, 'declarations_' + label] = int(group.loc[mask, 'material_correction_declaration'].sum())
    summary.to_csv(output / 'simulation_summary.csv', index=False)
    print(
        json.dumps(
            {"result": "written", "output": str(output), "replications": len(tasks)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
