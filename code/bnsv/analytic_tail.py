"""Deterministic h=1 VaR and ES for finite Gaussian and standardized-t mixtures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special


@dataclass(frozen=True)
class AnalyticTailResult:
    var: float
    es: float
    cdf_at_var: float
    components: int
    method: str


def _validate_common(
    alpha: float,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < float(alpha) < 0.5:
        raise ValueError("tail probability must lie in (0, 0.5)")
    location = np.asarray(locations, dtype=float)
    sd = np.asarray(conditional_sds, dtype=float)
    if location.ndim != 1 or sd.shape != location.shape or location.size < 1:
        raise ValueError("mixture locations and conditional SDs must be aligned vectors")
    if np.any(~np.isfinite(location)) or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("mixture locations and conditional SDs must be finite and positive")
    return location, sd


def _solve_quantile(
    *,
    alpha: float,
    component_quantiles: np.ndarray,
    cdf,
) -> tuple[float, float]:
    lower = float(np.min(component_quantiles))
    upper = float(np.max(component_quantiles))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("component quantile bracket is invalid")
    if np.isclose(lower, upper, rtol=0.0, atol=1e-15):
        value = 0.5 * (lower + upper)
    else:
        lower_error = float(cdf(lower) - alpha)
        upper_error = float(cdf(upper) - alpha)
        if lower_error > 1e-12 or upper_error < -1e-12:
            raise ValueError("component quantiles do not bracket the mixture quantile")
        value = float(
            optimize.brentq(
                lambda candidate: float(cdf(candidate) - alpha),
                lower,
                upper,
                xtol=1e-14,
                rtol=1e-13,
            )
        )
    probability = float(cdf(value))
    if not np.isfinite(probability) or abs(probability - alpha) > 1e-10:
        raise ValueError("analytic mixture quantile does not attain the target probability")
    return value, probability


def gaussian_mixture_var_es(
    alpha: float,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
) -> AnalyticTailResult:
    location, sd = _validate_common(alpha, locations, conditional_sds)
    normal_quantile = special.ndtri(alpha)
    component_quantiles = location + sd * normal_quantile

    def cdf(value: float) -> float:
        return float(np.mean(special.ndtr((value - location) / sd)))

    var, probability = _solve_quantile(
        alpha=alpha,
        component_quantiles=component_quantiles,
        cdf=cdf,
    )
    standardized = (var - location) / sd
    partial_first_moment = (
        location * special.ndtr(standardized)
        - sd
        * np.exp(-0.5 * standardized * standardized)
        / np.sqrt(2.0 * np.pi)
    )
    es = float(np.mean(partial_first_moment) / alpha)
    if not np.isfinite(es) or es > var + 1e-12:
        raise ValueError("analytic Gaussian mixture ES is invalid")
    return AnalyticTailResult(
        var=var,
        es=es,
        cdf_at_var=probability,
        components=int(location.size),
        method="finite_gaussian_mixture_analytic_var_es",
    )


def _student_t_pdf(values: np.ndarray, degrees_of_freedom: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    nu = np.asarray(degrees_of_freedom, dtype=float)
    return np.exp(
        special.gammaln(0.5 * (nu + 1.0))
        - special.gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1.0) * np.log1p(x * x / nu)
    )


def standardized_t_mixture_var_es(
    alpha: float,
    locations: np.ndarray,
    conditional_sds: np.ndarray,
    degrees_of_freedom: np.ndarray,
) -> AnalyticTailResult:
    location, sd = _validate_common(alpha, locations, conditional_sds)
    nu = np.asarray(degrees_of_freedom, dtype=float)
    if nu.shape != location.shape:
        raise ValueError("degrees of freedom must contain one value per mixture component")
    if np.any(~np.isfinite(nu)) or np.any(nu <= 2.0):
        raise ValueError("variance-standardized Student-t components require nu > 2")
    scale = np.sqrt((nu - 2.0) / nu)
    component_quantiles = location + sd * scale * special.stdtrit(nu, alpha)

    def cdf(value: float) -> float:
        standardized = (value - location) / (sd * scale)
        return float(np.mean(special.stdtr(nu, standardized)))

    var, probability = _solve_quantile(
        alpha=alpha,
        component_quantiles=component_quantiles,
        cdf=cdf,
    )
    standardized = (var - location) / (sd * scale)
    component_cdf = special.stdtr(nu, standardized)
    partial_first_moment = (
        location * component_cdf
        - sd
        * scale
        * ((nu + standardized * standardized) / (nu - 1.0))
        * _student_t_pdf(standardized, nu)
    )
    es = float(np.mean(partial_first_moment) / alpha)
    if not np.isfinite(es) or es > var + 1e-12:
        raise ValueError("analytic Student-t mixture ES is invalid")
    return AnalyticTailResult(
        var=var,
        es=es,
        cdf_at_var=probability,
        components=int(location.size),
        method="finite_variance_standardized_student_t_mixture_analytic_var_es",
    )


def analytic_tail_from_archive(
    path: str | Path,
    *,
    alpha: float,
    distribution: str,
) -> AnalyticTailResult:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"predictive archive is missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"raw_return_locations", "raw_conditional_sds"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"predictive archive lacks mixture arrays: {sorted(missing)}")
        locations = np.asarray(archive["raw_return_locations"], dtype=float)
        conditional_sds = np.asarray(archive["raw_conditional_sds"], dtype=float)
        if locations.ndim != 2 or locations.shape[1] != 1:
            raise ValueError("analytic tail evaluation is restricted to h=1 archives")
        if conditional_sds.shape != locations.shape:
            raise ValueError("return locations and conditional SDs do not align")
        if distribution == "gaussian_mixture":
            if "degrees_of_freedom" in archive.files:
                raise ValueError("Gaussian mixture archive unexpectedly contains degrees of freedom")
            return gaussian_mixture_var_es(
                alpha,
                locations[:, 0],
                conditional_sds[:, 0],
            )
        if distribution == "variance_standardized_student_t_mixture":
            if "degrees_of_freedom" not in archive.files:
                raise ValueError("Student-t mixture archive lacks degrees of freedom")
            return standardized_t_mixture_var_es(
                alpha,
                locations[:, 0],
                conditional_sds[:, 0],
                np.asarray(archive["degrees_of_freedom"], dtype=float),
            )
        raise ValueError(f"unsupported analytic tail distribution: {distribution}")
