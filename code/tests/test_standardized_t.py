from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bnsv.standardized_t import standardized_t_logpdf, standardized_t_rvs, standardized_t_scale


def test_standardized_t_monte_carlo_variance_is_one():
    draws = standardized_t_rvs(8.0, size=400_000, rng=np.random.default_rng(1))
    assert abs(np.var(draws) - 1.0) < 0.02


def test_standardized_t_logpdf_matches_scipy_scale_mapping():
    x, nu, sigma = 0.37, 7.5, 1.8
    expected = stats.t.logpdf(x, df=nu, scale=sigma * standardized_t_scale(nu))
    assert standardized_t_logpdf(x, nu, sigma) == pytest.approx(expected)


def test_standardized_t_rejects_nu_not_exceeding_two():
    with pytest.raises(ValueError, match="greater than 2"):
        standardized_t_rvs(2.0, size=10)
