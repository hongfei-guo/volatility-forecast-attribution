from __future__ import annotations

import numpy as np
import pytest

from bnsv.forecast_calculation import reconstruct_forecast_paths


def base_refit(*, dimension: int) -> dict[str, np.ndarray]:
    return {
        "ancestry": np.asarray([7, 3]),
        "feature_mean": np.zeros(dimension),
        "feature_scale": np.ones(dimension),
        "return_mean": np.asarray([0.01]),
        "return_scale": np.asarray([0.2]),
        "parameter__mu": np.asarray([-0.3, 0.2]),
        "parameter__phi": np.asarray([0.8, 0.5]),
    }


def base_inputs(horizon: int) -> dict[str, np.ndarray]:
    return {
        "source_parameter_ancestry": np.asarray([3, 7, 3]),
        "origin_baseline_states": np.asarray([0.1, -0.2, 0.3]),
        "origin_z_history_tail": np.asarray([-0.4, 0.25]),
        "origin_log_rv_history_tail": np.asarray([-1.1, -0.7]),
        "state_innovations": np.tile(
            np.asarray([[0.02], [-0.01], [0.03]]), (1, horizon)
        ),
        "return_innovations": np.tile(
            np.asarray([[0.4], [-0.2], [0.1]]), (1, horizon)
        ),
        "future_log_rv_paths": np.tile(
            np.linspace(-0.6, 0.3, horizon), (3, 1)
        ),
    }


def assert_path_identities(
    paths: dict[str, np.ndarray], inputs: dict[str, np.ndarray]
) -> None:
    np.testing.assert_allclose(
        paths["transition_means"] + inputs["state_innovations"],
        paths["baseline_states"],
    )
    np.testing.assert_allclose(
        paths["baseline_states"] + paths["observable_corrections"],
        paths["latent_states"],
    )
    np.testing.assert_allclose(
        paths["raw_conditional_sds"] ** 2, paths["raw_variances"]
    )
    np.testing.assert_allclose(
        paths["raw_return_locations"]
        + paths["raw_conditional_sds"] * inputs["return_innovations"],
        paths["raw_returns"],
    )


def test_sv_reconstructs_ten_step_state_and_return_paths() -> None:
    refit = base_refit(dimension=1)
    inputs = base_inputs(10)
    paths = reconstruct_forecast_paths("SV", refit, inputs)

    mu = np.asarray([0.2, -0.3, 0.2])
    phi = np.asarray([0.5, 0.8, 0.5])
    expected = np.empty((3, 10))
    state = inputs["origin_baseline_states"].copy()
    for step in range(10):
        state = mu + phi * (state - mu) + inputs["state_innovations"][:, step]
        expected[:, step] = state

    np.testing.assert_allclose(paths["baseline_states"], expected)
    np.testing.assert_array_equal(paths["observable_corrections"], 0.0)
    np.testing.assert_allclose(paths["latent_states"], expected)
    assert_path_identities(paths, inputs)


def test_nn_sv_reconstructs_one_step_neural_correction() -> None:
    refit = base_refit(dimension=1)
    refit.update(
        {
            "parameter__w1": np.asarray([[[0.5]], [[1.5]]]),
            "parameter__b1": np.asarray([[0.1], [-0.2]]),
            "parameter__output_weights": np.asarray([[0.7], [1.2]]),
            "parameter__centering_constant": np.asarray([0.05, -0.1]),
        }
    )
    inputs = base_inputs(1)
    paths = reconstruct_forecast_paths("NN-SV", refit, inputs)

    x = np.full(3, inputs["origin_z_history_tail"][-1])
    expected = np.asarray(
        [
            1.2 * np.tanh(1.5 * x[0] - 0.2) + 0.1,
            0.7 * np.tanh(0.5 * x[1] + 0.1) - 0.05,
            1.2 * np.tanh(1.5 * x[2] - 0.2) + 0.1,
        ]
    )
    np.testing.assert_allclose(paths["observable_corrections"][:, 0], expected)
    assert_path_identities(paths, inputs)


def test_rv_nn_sv_uses_recursive_realized_measure_inputs_at_horizon_ten() -> None:
    refit = base_refit(dimension=2)
    refit.update(
        {
            "parameter__w1": np.asarray(
                [[[0.0], [1.0]], [[0.0], [1.0]]]
            ),
            "parameter__b1": np.zeros((2, 1)),
            "parameter__output_weights": np.ones((2, 1)),
            "parameter__centering_constant": np.zeros(2),
        }
    )
    inputs = base_inputs(10)
    paths = reconstruct_forecast_paths("RV-NN-SV", refit, inputs)

    realized_inputs = np.concatenate(
        (
            [inputs["origin_log_rv_history_tail"][-1]],
            inputs["future_log_rv_paths"][0, :-1],
        )
    )
    expected = np.tanh(realized_inputs)
    np.testing.assert_allclose(
        paths["observable_corrections"],
        np.broadcast_to(expected, (3, 10)),
    )
    assert_path_identities(paths, inputs)


def test_rv_ra_adds_centered_lagged_asymmetry_to_the_rv_core() -> None:
    refit = base_refit(dimension=3)
    refit.update(
        {
            "parameter__w1": np.zeros((2, 2, 1)),
            "parameter__b1": np.zeros((2, 1)),
            "parameter__output_weights": np.zeros((2, 1)),
            "parameter__centering_constant": np.zeros(2),
            "parameter__gamma_A": np.asarray([2.0, -1.0]),
            "parameter__q_centering_mean": np.asarray([0.1, 0.1]),
        }
    )
    inputs = base_inputs(2)
    inputs["origin_asymmetry_history_tail"] = np.asarray([-0.5, 0.4])
    inputs["future_asymmetry_paths"] = np.asarray(
        [[0.8, -0.2], [0.8, -0.2], [0.8, -0.2]]
    )
    paths = reconstruct_forecast_paths("RV-RA-NN-SV", refit, inputs)

    # Ancestry [3, 7, 3] selects gamma [-1, 2, -1]. At step two the
    # first future asymmetry becomes the lagged input.
    expected = np.asarray(
        [
            [-1.0 * (0.4 - 0.1), -1.0 * (0.8 - 0.1)],
            [2.0 * (0.4 - 0.1), 2.0 * (0.8 - 0.1)],
            [-1.0 * (0.4 - 0.1), -1.0 * (0.8 - 0.1)],
        ]
    )
    np.testing.assert_allclose(paths["observable_corrections"], expected)
    assert_path_identities(paths, inputs)


def test_rv_lin_sv_reconstructs_one_step_linear_correction() -> None:
    refit = base_refit(dimension=2)
    refit.update(
        {
            "parameter__beta": np.asarray([[0.3, -0.2], [0.5, 0.4]]),
            "parameter__x_center": np.asarray([[0.1, -0.1], [-0.2, 0.2]]),
        }
    )
    inputs = base_inputs(1)
    paths = reconstruct_forecast_paths("RV-LIN-SV", refit, inputs)

    x = np.asarray(
        [inputs["origin_z_history_tail"][-1], inputs["origin_log_rv_history_tail"][-1]]
    )
    expected = np.asarray(
        [
            np.dot(x - refit["parameter__x_center"][1], refit["parameter__beta"][1]),
            np.dot(x - refit["parameter__x_center"][0], refit["parameter__beta"][0]),
            np.dot(x - refit["parameter__x_center"][1], refit["parameter__beta"][1]),
        ]
    )
    np.testing.assert_allclose(paths["observable_corrections"][:, 0], expected)
    assert_path_identities(paths, inputs)


def test_rv_lin_sv_rejects_unreported_horizons() -> None:
    refit = base_refit(dimension=2)
    refit.update(
        {
            "parameter__beta": np.zeros((2, 2)),
            "parameter__x_center": np.zeros((2, 2)),
        }
    )
    with pytest.raises(ValueError, match="horizon one"):
        reconstruct_forecast_paths("RV-LIN-SV", refit, base_inputs(10))
