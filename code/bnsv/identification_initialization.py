"""Deterministic initial values for the identification simulation."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np


_CHAIN_JITTER = (-0.15, -0.05, 0.05, 0.15)
_SIGMA_H_STD_INIT = (0.55, 0.65, 0.75, 0.85)


def make_prior_centered_nn_init(
    *,
    n: int,
    d: int,
    width: int,
    phi_logit_loc: float,
    phi_logit_scale: float,
    chain_id: int,
) -> dict[str, Any]:
    if n < 2 or d < 1 or width < 1:
        raise ValueError("n >= 2, d >= 1, and width >= 1 are required")
    if chain_id not in (1, 2, 3, 4):
        raise ValueError("chain_id must be 1, 2, 3, or 4")
    if not np.isfinite(phi_logit_loc) or not np.isfinite(phi_logit_scale):
        raise ValueError("phi prior coordinates must be finite")

    jitter = _CHAIN_JITTER[chain_id - 1]
    unit_offsets = np.linspace(-0.20, 0.20, width)
    if width == 5:
        ordered_quantiles = np.array([-1.18, -0.50, 0.0, 0.50, 1.18])
    else:
        normal = NormalDist()
        ordered_quantiles = np.array(
            [normal.inv_cdf((j + 1) / (width + 1)) for j in range(width)]
        )

    w1 = np.empty((d, width), dtype=float)
    for feature_index in range(d):
        w1[feature_index, :] = (
            0.05 * (feature_index + 1) * unit_offsets + 0.01 * jitter
        )

    phi_std = 0.5 * jitter
    phi_raw = phi_logit_loc + phi_logit_scale * phi_std
    log_phi = -float(np.logaddexp(0.0, -phi_raw))
    stationary_shrink = math.sqrt(-math.expm1(2.0 * log_phi))
    return {
        "mu_std": jitter,
        "phi_std": phi_std,
        "sigma_eta_std": _SIGMA_H_STD_INIT[chain_id - 1] * stationary_shrink,
        "nu_std": 0.25 * jitter,
        "w1_raw": w1.tolist(),
        "b1_std": (unit_offsets + 0.1 * jitter).tolist(),
        "log_output_weight_std": (ordered_quantiles + 0.05 * jitter).tolist(),
        "h0_raw": 0.10 * jitter,
        "alpha": [0.0] * (n - 1),
    }


def make_prior_centered_nn_inits(
    *,
    n: int,
    d: int,
    width: int,
    phi_logit_loc: float,
    phi_logit_scale: float,
    chains: int,
) -> list[dict[str, Any]]:
    if chains != 4:
        raise ValueError("the reported fit uses four chains")
    return [
        make_prior_centered_nn_init(
            n=n,
            d=d,
            width=width,
            phi_logit_loc=phi_logit_loc,
            phi_logit_scale=phi_logit_scale,
            chain_id=chain_id,
        )
        for chain_id in range(1, chains + 1)
    ]
