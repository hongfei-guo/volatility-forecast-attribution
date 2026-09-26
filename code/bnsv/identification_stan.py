"""Stan-data adapter for the identification simulation."""
from __future__ import annotations

from typing import Any

import numpy as np


def nn_sv_stan_data(
    *,
    z: np.ndarray,
    x: np.ndarray,
    width: int,
    prior: dict[str, Any],
) -> dict[str, Any]:
    """Build data for the exponential-link NN-SV model."""
    persistence = float(prior["persistence_median"])
    return {
        "N": int(len(z)),
        "z": np.asarray(z, float),
        "mu_prior_mean": float(prior["latent_mean"]),
        "mu_prior_sd": float(prior["latent_sd"]),
        "phi_logit_loc": float(np.log(persistence / (1.0 - persistence))),
        "phi_logit_scale": float(prior["persistence_logit_sd"]),
        "sigma_h_prior_sd": float(prior["sigma_h_sd"]),
        "h0_prior_sd": float(prior["initial_state_sd"]),
        "nu_minus_two_log_mean": float(prior["nu_minus_two_log_mean"]),
        "nu_minus_two_log_sd": float(prior["nu_minus_two_log_sd"]),
        "D": int(x.shape[1]),
        "H": int(width),
        "X": np.asarray(x, float),
        "w1_base_sd": float(prior["first_layer_sd"]),
        "b1_prior_sd": float(prior["hidden_bias_sd"]),
        "output_log_mean": float(np.log(prior["output_base_scale"])),
        "output_log_sd": float(prior["output_log_sd"]),
    }
