from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, special, stats

from bnsv.analytic_tail import (
    analytic_tail_from_archive,
    gaussian_mixture_var_es,
    standardized_t_mixture_var_es,
)
from scripts.build_predictive_losses import _tail_fields_from_saved_mixture


def test_single_gaussian_matches_closed_form() -> None:
    alpha = 0.01
    location = np.array([0.001])
    sd = np.array([0.02])
    result = gaussian_mixture_var_es(alpha, location, sd)
    z = special.ndtri(alpha)
    expected_var = location[0] + sd[0] * z
    expected_es = location[0] - sd[0] * np.exp(-0.5 * z * z) / (
        np.sqrt(2.0 * np.pi) * alpha
    )
    assert result.var == pytest.approx(expected_var, abs=1e-14)
    assert result.es == pytest.approx(expected_es, abs=1e-14)
    assert result.cdf_at_var == pytest.approx(alpha, abs=1e-14)


def test_single_standardized_t_matches_quadrature() -> None:
    alpha = 0.01
    location = np.array([0.001])
    sd = np.array([0.02])
    nu = np.array([8.0])
    result = standardized_t_mixture_var_es(alpha, location, sd, nu)
    scale = np.sqrt((nu[0] - 2.0) / nu[0])
    cutoff = stats.t.ppf(alpha, nu[0])
    numerical = integrate.quad(
        lambda value: (location[0] + sd[0] * scale * value)
        * stats.t.pdf(value, nu[0]),
        -np.inf,
        cutoff,
        epsabs=1e-13,
    )[0] / alpha
    assert result.var == pytest.approx(location[0] + sd[0] * scale * cutoff, abs=1e-14)
    assert result.es == pytest.approx(numerical, abs=1e-13)


def test_student_t_mixture_quantile_and_es_are_permutation_invariant() -> None:
    alpha = 0.05
    locations = np.array([-0.001, 0.002, 0.0])
    sds = np.array([0.01, 0.02, 0.015])
    nu = np.array([6.0, 9.0, 15.0])
    first = standardized_t_mixture_var_es(alpha, locations, sds, nu)
    order = np.array([2, 0, 1])
    second = standardized_t_mixture_var_es(
        alpha, locations[order], sds[order], nu[order]
    )
    assert first.var == pytest.approx(second.var, abs=1e-14)
    assert first.es == pytest.approx(second.es, abs=1e-14)
    assert first.cdf_at_var == pytest.approx(alpha, abs=1e-11)
    assert first.es < first.var


def test_archive_dispatches_student_t_and_rejects_higher_horizon(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    np.savez_compressed(
        valid,
        raw_return_locations=np.zeros((4, 1)),
        raw_conditional_sds=np.ones((4, 1)),
        degrees_of_freedom=np.full(4, 8.0),
    )
    result = analytic_tail_from_archive(
        valid,
        alpha=0.01,
        distribution="variance_standardized_student_t_mixture",
    )
    assert result.components == 4
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        raw_return_locations=np.zeros((4, 2)),
        raw_conditional_sds=np.ones((4, 2)),
        degrees_of_freedom=np.full(4, 8.0),
    )
    with pytest.raises(ValueError, match="restricted to h=1"):
        analytic_tail_from_archive(
            invalid,
            alpha=0.01,
            distribution="variance_standardized_student_t_mixture",
        )


def test_invalid_scale_and_degrees_of_freedom_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        gaussian_mixture_var_es(0.01, np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="nu > 2"):
        standardized_t_mixture_var_es(
            0.01,
            np.array([0.0]),
            np.array([1.0]),
            np.array([2.0]),
        )


def test_predictive_loss_builder_uses_analytic_h1_mixture_tails() -> None:
    locations = np.array([[-0.006], [0.001], [0.004]])
    conditional_sds = np.array([[0.012], [0.019], [0.025]])
    degrees_of_freedom = np.array([6.0, 9.0, 15.0])
    realized_return = -0.02
    expected = {
        "gaussian": {
            "var_05": -0.030697627689217282,
            "es_05": -0.04005559758066175,
            "var_es_05_fz0": -3.4511113737525325,
            "var_01": -0.045959019717130574,
            "es_01": -0.05449703549581285,
            "var_es_01_fz0": -3.0662783353033656,
        },
        "variance_standardized_student_t": {
            "var_05": -0.030434978463103023,
            "es_05": -0.04172562538152234,
            "var_es_05_fz0": -3.447232456359333,
            "var_01": -0.0485842302488876,
            "es_01": -0.05994757261034617,
            "var_es_01_fz0": -3.003839558864778,
        },
    }

    for distribution, expected_values in expected.items():
        fields = _tail_fields_from_saved_mixture(
            horizon=1,
            return_distribution=distribution,
            raw_return_locations=locations,
            raw_conditional_sds=conditional_sds,
            degrees_of_freedom=(
                degrees_of_freedom
                if distribution == "variance_standardized_student_t"
                else None
            ),
            realized_return=realized_return,
        )
        assert set(fields) == {
            "var_05",
            "es_05",
            "var_05_exceedance",
            "var_05_quantile_loss",
            "var_es_05_fz0",
            "var_01",
            "es_01",
            "var_01_exceedance",
            "var_01_quantile_loss",
            "var_es_01_fz0",
        }
        for field, value in expected_values.items():
            assert fields[field] == pytest.approx(value, abs=1e-12)

    assert _tail_fields_from_saved_mixture(
        horizon=5,
        return_distribution="variance_standardized_student_t",
        raw_return_locations=np.tile(locations, (1, 5)),
        raw_conditional_sds=np.tile(conditional_sds, (1, 5)),
        degrees_of_freedom=degrees_of_freedom,
        realized_return=realized_return,
    ) == {}
