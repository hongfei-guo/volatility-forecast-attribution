"""MCMC diagnostics used by the identification simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiagnosticThresholds:
    max_divergences: int = 0
    max_rhat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    min_bfmi: float = 0.30
    max_treedepth_fraction: float = 0.001
    max_mcse_over_sd: float = 0.05


def assess_summary_table(
    summary: Any,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> dict[str, Any]:
    """Assess a CmdStan summary DataFrame or compatible mapping."""
    columns = set(summary.columns)
    required = {"R_hat", "ESS_bulk", "ESS_tail", "MCSE", "StdDev"}
    missing = required - columns
    if missing:
        raise ValueError(f"summary table missing columns: {sorted(missing)}")
    rhat = float(np.nanmax(summary["R_hat"]))
    bulk = float(np.nanmin(summary["ESS_bulk"]))
    tail = float(np.nanmin(summary["ESS_tail"]))
    ratio = np.asarray(summary["MCSE"], float) / np.maximum(
        np.asarray(summary["StdDev"], float), 1e-15
    )
    mcse_ratio = float(np.nanmax(ratio))
    return {
        "max_rhat": rhat,
        "min_bulk_ess": bulk,
        "min_tail_ess": tail,
        "max_mcse_over_sd": mcse_ratio,
        "rhat_pass": rhat < thresholds.max_rhat,
        "bulk_ess_pass": bulk >= thresholds.min_bulk_ess,
        "tail_ess_pass": tail >= thresholds.min_tail_ess,
        "mcse_over_sd_pass": mcse_ratio < thresholds.max_mcse_over_sd,
        "summary_pass": (
            rhat < thresholds.max_rhat
            and bulk >= thresholds.min_bulk_ess
            and tail >= thresholds.min_tail_ess
            and mcse_ratio < thresholds.max_mcse_over_sd
        ),
    }


def assess_cmdstan_fit(
    fit: Any,
    *,
    max_treedepth: int,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> dict[str, Any]:
    summary = assess_summary_table(fit.summary(), thresholds)
    method_vars = fit.method_variables()
    divergent_values = np.asarray(method_vars.get("divergent__", 0), dtype=int)
    divergent = int(np.sum(divergent_values))
    if divergent_values.ndim == 2:
        divergences_by_chain = np.sum(divergent_values, axis=0).astype(int).tolist()
        post_warmup_draws_by_chain = [
            int(divergent_values.shape[0])
        ] * int(divergent_values.shape[1])
    elif divergent_values.ndim == 1:
        divergences_by_chain = [int(np.sum(divergent_values))]
        post_warmup_draws_by_chain = [int(divergent_values.shape[0])]
    else:
        divergences_by_chain = [divergent]
        post_warmup_draws_by_chain = [int(divergent_values.size)]
    treedepth = np.asarray(method_vars.get("treedepth__", []), dtype=float)
    treedepth_fraction = (
        float(np.mean(treedepth >= max_treedepth)) if treedepth.size else 0.0
    )
    energy = np.asarray(method_vars.get("energy__", []), dtype=float)
    if energy.ndim == 2 and energy.shape[0] > 1:
        numerator = np.mean(np.diff(energy, axis=0) ** 2, axis=0)
        denominator = np.var(energy, axis=0, ddof=1)
        bfmi = float(np.nanmin(numerator / np.maximum(denominator, 1e-15)))
    else:
        bfmi = float("nan")
    passed = (
        summary["summary_pass"]
        and divergent <= thresholds.max_divergences
        and treedepth_fraction < thresholds.max_treedepth_fraction
        and (np.isnan(bfmi) or bfmi > thresholds.min_bfmi)
    )
    return {
        **summary,
        "divergences": divergent,
        "divergences_by_chain": divergences_by_chain,
        "post_warmup_draws_by_chain": post_warmup_draws_by_chain,
        "divergence_pass": divergent <= thresholds.max_divergences,
        "treedepth_fraction": treedepth_fraction,
        "treedepth_pass": treedepth_fraction < thresholds.max_treedepth_fraction,
        "min_bfmi": bfmi,
        "bfmi_pass": bool(np.isnan(bfmi) or bfmi > thresholds.min_bfmi),
        "passed": bool(passed),
    }
