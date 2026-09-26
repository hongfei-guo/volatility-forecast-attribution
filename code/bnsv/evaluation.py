"""Variance, observable-return density, calibration, and risk evaluation."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from .standardized_t import standardized_t_cdf, standardized_t_logpdf


def qlike(realized_variance: np.ndarray, forecast_variance: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    y = np.maximum(np.asarray(realized_variance, dtype=float), floor)
    f = np.maximum(np.asarray(forecast_variance, dtype=float), floor)
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def variance_mse(realized_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    return (np.asarray(realized_variance, dtype=float) - np.asarray(forecast_variance, dtype=float)) ** 2


def crps_ensemble(draws: np.ndarray, observation: float) -> float:
    x = np.asarray(draws, dtype=float).ravel()
    if x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("at least two finite draws required")
    first = np.mean(np.abs(x - observation))
    ordered = np.sort(x)
    n = ordered.size
    # Equivalent to 0.5*E|X-X'| without materialising an n-by-n matrix.
    pair = np.sum((2 * np.arange(1, n + 1) - n - 1) * ordered) / (n**2)
    return float(first - pair)


def randomized_pit(draws: np.ndarray, observation: float, rng: np.random.Generator) -> float:
    x = np.asarray(draws, dtype=float).ravel()
    if x.size < 2 or np.any(~np.isfinite(x)) or not np.isfinite(observation):
        raise ValueError("finite observation and at least two finite draws required")
    below = np.sum(x < observation)
    equal = np.sum(x == observation)
    # Randomized finite-ensemble rank. The observation itself occupies one of
    # M+1 exchangeable ranks; equal simulated values are randomized jointly.
    return float((below + rng.random() * (equal + 1)) / (x.size + 1))


def finite_mixture_standardized_t_log_density(
    observation: float,
    *,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    loc = np.asarray(locations, dtype=float)
    sd = np.asarray(conditional_sds, dtype=float)
    nu = np.asarray(degrees_of_freedom, dtype=float)
    if loc.shape != sd.shape or loc.shape != nu.shape or loc.ndim != 1 or loc.size < 2:
        raise ValueError("finite-mixture component arrays must be aligned one-dimensional vectors")
    if np.any(~np.isfinite(loc)) or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("invalid finite-mixture components")
    w = np.full(loc.size, 1.0 / loc.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != loc.shape or np.any(w < 0) or not np.isclose(np.sum(w), 1.0):
        raise ValueError("invalid finite-mixture weights")
    component_logs = standardized_t_logpdf(observation - loc, nu, sd)
    return float(logsumexp(component_logs + np.log(np.maximum(w, np.finfo(float).tiny))))


def finite_mixture_standardized_t_cdf(
    observation: float,
    *,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    loc = np.asarray(locations, dtype=float)
    sd = np.asarray(conditional_sds, dtype=float)
    nu = np.asarray(degrees_of_freedom, dtype=float)
    if loc.shape != sd.shape or loc.shape != nu.shape or loc.ndim != 1 or loc.size < 2:
        raise ValueError("finite-mixture component arrays must be aligned one-dimensional vectors")
    w = np.full(loc.size, 1.0 / loc.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != loc.shape or np.any(w < 0) or not np.isclose(np.sum(w), 1.0):
        raise ValueError("invalid finite-mixture weights")
    value = float(np.sum(w * standardized_t_cdf(observation - loc, nu, sd)))
    return float(np.clip(value, 0.0, 1.0))


def finite_mixture_gaussian_log_density(
    observation: float,
    *,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    loc = np.asarray(locations, dtype=float)
    sd = np.asarray(conditional_sds, dtype=float)
    if loc.shape != sd.shape or loc.ndim != 1 or loc.size < 2:
        raise ValueError("Gaussian mixture components must be aligned vectors")
    if np.any(~np.isfinite(loc)) or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("invalid Gaussian mixture components")
    w = np.full(loc.size, 1.0 / loc.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != loc.shape or np.any(w < 0) or not np.isclose(np.sum(w), 1.0):
        raise ValueError("invalid Gaussian mixture weights")
    standardized = (float(observation) - loc) / sd
    component_logs = -0.5 * (
        np.log(2.0 * np.pi) + 2.0 * np.log(sd) + standardized * standardized
    )
    return float(logsumexp(component_logs + np.log(np.maximum(w, np.finfo(float).tiny))))


def finite_mixture_gaussian_cdf(
    observation: float,
    *,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    loc = np.asarray(locations, dtype=float)
    sd = np.asarray(conditional_sds, dtype=float)
    if loc.shape != sd.shape or loc.ndim != 1 or loc.size < 2:
        raise ValueError("Gaussian mixture components must be aligned vectors")
    if np.any(~np.isfinite(loc)) or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("invalid Gaussian mixture components")
    w = np.full(loc.size, 1.0 / loc.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != loc.shape or np.any(w < 0) or not np.isclose(np.sum(w), 1.0):
        raise ValueError("invalid Gaussian mixture weights")
    value = float(np.sum(w * ndtr((float(observation) - loc) / sd)))
    return float(np.clip(value, 0.0, 1.0))


def interval_diagnostic(draws: np.ndarray, observation: float, level: float) -> dict[str, float | bool]:
    if not 0 < level < 1:
        raise ValueError("level must lie in (0,1)")
    alpha = 1 - level
    x = np.asarray(draws, dtype=float).ravel()
    if x.size < 2 or np.any(~np.isfinite(x)) or not np.isfinite(observation):
        raise ValueError("finite observation and at least two finite draws required")
    lower, upper = np.quantile(x, [alpha / 2, 1 - alpha / 2])
    score = upper - lower
    if observation < lower:
        score += 2 / alpha * (lower - observation)
    elif observation > upper:
        score += 2 / alpha * (observation - upper)
    return {
        "lower": float(lower),
        "upper": float(upper),
        "covered": bool(lower <= observation <= upper),
        "width": float(upper - lower),
        "interval_score": float(score),
    }


def quantile_loss(observation: float, quantile: float, alpha: float) -> float:
    if not 0 < alpha < 0.5 or not np.isfinite(observation) or not np.isfinite(quantile):
        raise ValueError("invalid quantile-loss inputs")
    return float((alpha - float(observation < quantile)) * (observation - quantile))


def fz0_var_es_score(observation: float, var: float, es: float, alpha: float) -> float:
    """FZ0 loss for a lower-tail return VaR/ES pair.

    The conventional FZ0 member requires a strictly negative ES. That is the
    relevant domain for demeaned daily equity-index returns; fail explicitly
    rather than silently changing the scoring rule if it is violated.
    """
    if not 0 < alpha < 0.5 or not all(np.isfinite(v) for v in (observation, var, es)):
        raise ValueError("invalid FZ0 inputs")
    if es >= 0:
        raise ValueError("FZ0 lower-tail score requires ES < 0")
    hit = float(observation <= var)
    return float(-(hit * (var - observation)) / (alpha * es) + var / es + np.log(-es) - 1.0)


def mixture_log_score(component_log_densities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logs = np.asarray(component_log_densities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if logs.shape[-1] != w.size or np.any(w < 0) or not np.isclose(np.sum(w), 1):
        raise ValueError("weights do not match component densities")
    return logsumexp(logs + np.log(np.maximum(w, np.finfo(float).tiny)), axis=-1)


def proxy_interval_coverage(draws: np.ndarray, rv_proxy: float, level: float) -> dict[str, float | bool | str]:
    result = interval_diagnostic(draws, rv_proxy, level)
    result["interpretation"] = "descriptive_proxy_coverage"
    return result
