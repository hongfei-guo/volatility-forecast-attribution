from __future__ import annotations

import numpy as np

from bnsv.calibration import coverage_diagnostics, pit_diagnostics, var_backtests


def test_descriptive_diagnostics_omit_iid_reference_tests() -> None:
    pit = pit_diagnostics(
        np.array([0.1, 0.3, 0.7, 0.9]), include_reference_tests=False
    )
    assert set(pit) == {"observations", "mean", "variance"}

    coverage = coverage_diagnostics(
        np.array([True, False, True]), 0.5, include_reference_tests=False
    )
    assert "exact_binomial_p_value" not in coverage
    assert coverage["empirical_coverage"] == 2 / 3

    risk = var_backtests(
        np.array([False, True, False]), 0.05, include_reference_tests=False
    )
    assert set(risk) == {
        "observations",
        "exceedances",
        "exceedance_rate",
        "nominal_alpha",
    }
