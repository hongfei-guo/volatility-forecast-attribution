"""Observable-return calibration and risk diagnostics from saved loss rows."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def pit_diagnostics(
    pit: np.ndarray,
    lags: int = 10,
    *,
    include_reference_tests: bool = True,
) -> dict[str, float | int]:
    values = np.asarray(pit, dtype=float)
    values = values[np.isfinite(values)]
    minimum_size = lags + 3 if include_reference_tests else 2
    if values.size < minimum_size or np.any((values < 0) | (values > 1)):
        raise ValueError("insufficient valid PIT values")
    result: dict[str, float | int] = {
        "observations": int(values.size),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)),
    }
    if not include_reference_tests:
        return result

    ks = stats.kstest(values, "uniform")
    centered = values - np.mean(values)
    denominator = float(centered @ centered)
    autocorrelations = np.array(
        [
            float(centered[lag:] @ centered[:-lag] / denominator)
            for lag in range(1, lags + 1)
        ]
    )
    n = values.size
    q = float(n * (n + 2) * np.sum(autocorrelations**2 / (n - np.arange(1, lags + 1))))
    result.update(
        {
            "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "ljung_box_lags": int(lags),
            "ljung_box_statistic": q,
            "ljung_box_p_value": float(stats.chi2.sf(q, df=lags)),
        }
    )
    return result


def coverage_diagnostics(
    covered: np.ndarray,
    nominal: float,
    *,
    include_reference_tests: bool = True,
) -> dict[str, float | int]:
    hits = np.asarray(covered, dtype=bool)
    if hits.size == 0 or not 0 < nominal < 1:
        raise ValueError("invalid coverage inputs")
    count = int(np.sum(hits))
    result: dict[str, float | int] = {
        "observations": int(hits.size),
        "covered": count,
        "empirical_coverage": float(np.mean(hits)),
        "nominal_coverage": nominal,
    }
    if include_reference_tests:
        result["exact_binomial_p_value"] = float(
            stats.binomtest(count, hits.size, nominal).pvalue
        )
    return result


def var_backtests(
    exceedance: np.ndarray,
    alpha: float,
    *,
    include_reference_tests: bool = True,
) -> dict[str, float | int]:
    hits = np.asarray(exceedance, dtype=bool)
    minimum_size = 10 if include_reference_tests else 1
    if hits.size < minimum_size or not 0 < alpha < 0.5:
        raise ValueError("invalid VaR backtest inputs")
    n = hits.size
    x = int(np.sum(hits))
    rate = x / n
    result: dict[str, float | int] = {
        "observations": int(n),
        "exceedances": x,
        "exceedance_rate": float(rate),
        "nominal_alpha": alpha,
    }
    if not include_reference_tests:
        return result

    eps = np.finfo(float).tiny
    null_ll = (n - x) * math.log(max(1 - alpha, eps)) + x * math.log(max(alpha, eps))
    alt_ll = (n - x) * math.log(max(1 - rate, eps)) + x * math.log(max(rate, eps))
    lr_uc = max(0.0, -2 * (null_ll - alt_ll))
    n00 = int(np.sum((~hits[:-1]) & (~hits[1:])))
    n01 = int(np.sum((~hits[:-1]) & hits[1:]))
    n10 = int(np.sum(hits[:-1] & (~hits[1:])))
    n11 = int(np.sum(hits[:-1] & hits[1:]))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    common = (n01 + n11) / max(n - 1, 1)

    def bernoulli_ll(success: int, failure: int, probability: float) -> float:
        return success * math.log(max(probability, eps)) + failure * math.log(
            max(1 - probability, eps)
        )

    independent_ll = bernoulli_ll(n01 + n11, n00 + n10, common)
    markov_ll = bernoulli_ll(n01, n00, p01) + bernoulli_ll(n11, n10, p11)
    lr_ind = max(0.0, -2 * (independent_ll - markov_ll))
    result.update(
        {
            "kupiec_lr": lr_uc,
            "kupiec_p_value": float(stats.chi2.sf(lr_uc, 1)),
            "christoffersen_independence_lr": lr_ind,
            "christoffersen_independence_p_value": float(
                stats.chi2.sf(lr_ind, 1)
            ),
            "conditional_coverage_lr": lr_uc + lr_ind,
            "conditional_coverage_p_value": float(
                stats.chi2.sf(lr_uc + lr_ind, 2)
            ),
        }
    )
    return result
