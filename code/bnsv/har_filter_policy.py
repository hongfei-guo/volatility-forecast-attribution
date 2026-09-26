"""Separate deterministic-grid agreement from posterior-weight degeneration."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .forecast_paths import filter_from_refit, observable_correction
from .forecast_filter import filter_bank_ess_diagnostic


POLICY = "grid_agreement"
TOLERANCE = 0.001
INITIAL_LEVELS = (2, 3, 4)
MAX_LEVEL = 6
WARNING_FRACTION = 0.20


class FilterAccuracyError(RuntimeError):
    def __init__(self, diagnostics):
        super().__init__("grid accuracy unresolved at the fixed maximum level")
        self.diagnostics = diagnostics


def variance_mean(filtered, *, model_id, next_x, return_scale):
    """Analytic h=1 variance mean, with correction inputs known at this close."""
    p = filtered.parameters
    features = np.broadcast_to(np.asarray(next_x, dtype=float),
                               (filtered.size, len(next_x)))
    drift = observable_correction(model_id, p, features)
    if filtered._state_probabilities is None:
        log_moment = p["phi"] * filtered.initial_state
    else:
        nodes = p["mu"][:, None] + filtered._sigma_h[:, None] * filtered._u_nodes
        log_probs = np.full_like(filtered._state_probabilities, -np.inf)
        np.log(filtered._state_probabilities, out=log_probs,
               where=filtered._state_probabilities > 0)
        log_moment = logsumexp(log_probs + p["phi"][:, None] * nodes, axis=1)
    log_variance = ((1 - p["phi"]) * p["mu"] + drift
                    + 0.5 * p["sigma_eta"] ** 2 + log_moment)
    log_weights = np.full_like(filtered.weights, -np.inf)
    np.log(filtered.weights, out=log_weights, where=filtered.weights > 0)
    value = float(np.exp(logsumexp(log_weights + log_variance)) * return_scale ** 2)
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("nonfinite one-day variance mean in grid diagnostic")
    return value


def assess_agreement(banks, *, model_id, next_x, return_scale):
    levels = tuple(sorted(banks))
    if len(levels) != 2 or levels[1] != levels[0] + 1:
        raise ValueError("direct agreement requires two adjacent grids")
    low, high = (banks[level] for level in levels)
    if low.size != high.size:
        raise ValueError("grid parameter clouds do not align")
    means = [variance_mean(b, model_id=model_id, next_x=next_x,
                          return_scale=return_scale) for b in (low, high)]
    tv = float(0.5 * np.abs(low.weights - high.weights).sum())
    ess_difference = abs(low.ess - high.ess) / high.size
    variance_difference = abs(means[1] - means[0]) / max(means)
    agreed = max(tv, ess_difference, variance_difference) <= TOLERANCE
    enough = min(low.ess, high.ess) >= WARNING_FRACTION * high.size
    return {"policy": POLICY, "grid_levels": list(levels),
            "ess_coarse": low.ess, "ess_fine": high.ess,
            "weight_tv": tv, "ess_fraction_difference": ess_difference,
            "variance_relative_difference": variance_difference,
            "variance_mean_coarse": means[0], "variance_mean_fine": means[1],
            "numerical_tolerance": TOLERANCE,
            "parameter_ess_threshold": WARNING_FRACTION * high.size,
            "numerical_agreement": agreed,
            "decision": "refine" if not agreed else "continue" if enough else "refit"}


def new_bank(refit, level):
    return filter_from_refit(refit, level=level, cache_transition_kernel=level <= 4,
                             transition_batch_size=32)


def initial_banks(refit):
    return {level: new_bank(refit, level) for level in INITIAL_LEVELS}


def replay_finer(refit, level, history, model_id):
    """Build each atom-batch kernel once while replaying the entire history."""
    full = new_bank(refit, level)
    if full._transition_kernel is not None:
        for x, z in history:
            full.step(x_t=x, observed_z=z,
                correction=lambda p, features: observable_correction(model_id, p, features))
        return full
    size = full.size
    probabilities = np.empty((size, len(full._u_nodes)))
    increments = np.empty((len(history), size))
    cumulative = np.empty(size)
    for first in range(0, size, 32):
        last = min(size, first + 32)
        partial = {key: value[first:last] if key.startswith("parameter__") or
                   key in {"weights", "state", "ancestry"} else value
                   for key, value in refit.items()}
        bank = filter_from_refit(partial, level=level, cache_transition_kernel=True,
                                 transition_batch_size=32)
        for index, (x, z) in enumerate(history):
            result = bank.step(x_t=x, observed_z=z,
                correction=lambda p, features: observable_correction(model_id, p, features))
            increments[index, first:last] = result["log_predictive_increment"]
        probabilities[first:last] = bank._state_probabilities
        cumulative[first:last] = bank._cumulative_log_likelihood
        del bank
    weights = np.asarray(refit["weights"], dtype=float).copy()
    weights /= weights.sum()
    for increment in increments:
        log_weights = np.log(np.maximum(weights, np.finfo(float).tiny)) + increment
        weights = np.exp(log_weights - logsumexp(log_weights))
    full._state_probabilities = probabilities
    full._cumulative_log_likelihood = cumulative
    full.weights = weights
    return full


def update_and_assess(banks, *, refit, history, model_id, x, observed_z,
                      next_x, return_scale):
    """Replay refinements from the refit; never initialize from a coarse state."""
    history.append((np.asarray(x, dtype=float).copy(), float(observed_z)))
    correction = lambda p, features: observable_correction(model_id, p, features)
    for bank in banks.values():
        bank.step(x_t=x, observed_z=observed_z, correction=correction)
    diagnostics = []
    while True:
        levels = tuple(sorted(banks))
        interval = filter_bank_ess_diagnostic(banks, initial_weights=refit["weights"], warning_fraction=WARNING_FRACTION)
        if interval["decision"] in {"continue", "refit"}:
            values = [banks[level].ess for level in levels]
            denominator = abs(values[1] - values[0])
            numerator = abs(values[2] - values[1])
            rho = numerator / denominator if denominator else (0.0 if numerator == 0 else None)
            diagnostics.append({"policy": POLICY, "grid_levels": list(levels),
                "ess_coarse": values[0], "ess_fine": values[-1], "ess_contraction_ratio": rho,
                "ess_lower": interval.get("ess_lower"), "ess_upper": interval.get("ess_upper"),
                "decision": interval["decision"],
                "acceptance_basis": "ess_extrapolation" if interval["decision"] == "continue" else "ess_interval_below_threshold"})
            return interval["decision"] == "continue", diagnostics
        finest = {level: banks[level] for level in levels[-2:]}
        assessment = assess_agreement(finest, model_id=model_id, next_x=next_x,
                                      return_scale=return_scale)
        assessment["acceptance_basis"] = "direct_agreement"
        diagnostics.append(assessment)
        if assessment["decision"] != "refine":
            return assessment["decision"] == "continue", diagnostics
        level = max(banks)
        if level >= MAX_LEVEL:
            raise FilterAccuracyError(diagnostics)
        finer = replay_finer(refit, level + 1, history, model_id)
        del banks[min(banks)]
        banks[level + 1] = finer
