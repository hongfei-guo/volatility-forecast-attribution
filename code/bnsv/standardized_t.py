"""One variance-standardized Student-t convention used throughout the project."""
from __future__ import annotations

import math
import numpy as np
from scipy import stats


def _validate_nu(nu: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(nu, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 2):
        raise ValueError("nu must be finite and strictly greater than 2")
    return arr


def standardized_t_scale(nu: np.ndarray | float) -> np.ndarray:
    nu_arr = _validate_nu(nu)
    return np.sqrt((nu_arr - 2.0) / nu_arr)


def standardized_t_rvs(
    nu: np.ndarray | float,
    size: int | tuple[int, ...] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw innovations with mean zero and variance one."""
    nu_arr = _validate_nu(nu)
    generator = np.random.default_rng() if rng is None else rng
    if nu_arr.ndim == 0:
        return generator.standard_t(float(nu_arr), size=size) * float(standardized_t_scale(nu_arr))
    if size is not None:
        requested = (size,) if isinstance(size, int) else tuple(size)
        if requested != nu_arr.shape:
            raise ValueError("for array-valued nu, size must be omitted or equal nu.shape")
    return generator.standard_t(nu_arr) * standardized_t_scale(nu_arr)


def standardized_t_logpdf(x: np.ndarray | float, nu: np.ndarray | float, sigma: np.ndarray | float = 1.0) -> np.ndarray:
    """Log density when sigma is the conditional standard deviation."""
    nu_arr = _validate_nu(nu)
    sigma_arr = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0):
        raise ValueError("sigma must be positive and finite")
    scipy_scale = sigma_arr * standardized_t_scale(nu_arr)
    return stats.t.logpdf(np.asarray(x, dtype=float), df=nu_arr, loc=0.0, scale=scipy_scale)


def standardized_t_pdf(x: np.ndarray | float, nu: np.ndarray | float, sigma: np.ndarray | float = 1.0) -> np.ndarray:
    return np.exp(standardized_t_logpdf(x, nu, sigma))


def standardized_t_cdf(x: np.ndarray | float, nu: np.ndarray | float, sigma: np.ndarray | float = 1.0) -> np.ndarray:
    nu_arr = _validate_nu(nu)
    sigma_arr = np.asarray(sigma, dtype=float)
    return stats.t.cdf(np.asarray(x, dtype=float), df=nu_arr, loc=0.0, scale=sigma_arr * standardized_t_scale(nu_arr))


def standardized_t_ppf(q: np.ndarray | float, nu: np.ndarray | float, sigma: np.ndarray | float = 1.0) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr <= 0) | (q_arr >= 1)):
        raise ValueError("q must lie strictly in (0,1)")
    nu_arr = _validate_nu(nu)
    sigma_arr = np.asarray(sigma, dtype=float)
    return stats.t.ppf(q_arr, df=nu_arr, loc=0.0, scale=sigma_arr * standardized_t_scale(nu_arr))


def standardized_t_es(alpha: float, nu: float, sigma: float = 1.0) -> float:
    """Left-tail expected shortfall under the standardized Student-t law."""
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0,0.5)")
    _validate_nu(nu)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    raw_q = stats.t.ppf(alpha, df=nu)
    raw_pdf = stats.t.pdf(raw_q, df=nu)
    raw_es = -((nu + raw_q**2) / ((nu - 1) * alpha)) * raw_pdf
    return float(sigma * standardized_t_scale(nu) * raw_es)
