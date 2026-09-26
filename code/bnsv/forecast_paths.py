"""Posterior compression and forward paths for the five forecast models."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import invgamma

from .forecast_features import FeatureTransform, base_model_id, model_spec
from .forecast_filter import FixedParameterGridFilter
from .forecast_calculation import reconstruct_forecast_paths
from .standardized_t import standardized_t_ppf


def seed_from_context(base_seed: int, *parts: object) -> int:
    """Derive the fixed positive 32-bit seed used for one calculation."""
    payload = json.dumps(
        [int(base_seed), *parts], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return value % (2**31 - 1) + 1


def compact_refit(
    fit: Any,
    *,
    model_id: str,
    transform: FeatureTransform,
    estimation_features: np.ndarray,
    particle_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Select fixed posterior atoms and their last baseline states."""
    spec = model_spec(model_id)
    family = base_model_id(model_id)
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    available = len(np.asarray(fit.stan_variable("mu")))
    if available < particle_count:
        raise ValueError(
            f"posterior contains {available} draws; {particle_count} are required"
        )
    selected = rng.choice(available, size=particle_count, replace=False)
    arrays: dict[str, np.ndarray] = {
        "weights": np.full(particle_count, 1.0 / particle_count),
        "ancestry": np.arange(particle_count, dtype=int),
        "feature_mean": np.asarray(transform.feature_mean, dtype=float),
        "feature_scale": np.asarray(transform.feature_scale, dtype=float),
        "feature_names": np.asarray(transform.feature_names, dtype="U32"),
        "return_mean": np.asarray([transform.return_scaler.mean]),
        "return_scale": np.asarray([transform.return_scaler.scale]),
        "parameter__mu": np.asarray(fit.stan_variable("mu"))[selected],
        "parameter__phi": np.asarray(fit.stan_variable("phi"))[selected],
        "parameter__sigma_eta": np.asarray(fit.stan_variable("sigma_eta"))[selected],
        "parameter__nu": np.asarray(fit.stan_variable("nu"))[selected],
        "parameter__source_draw_index": selected,
    }
    state_name = "h" if model_id == "SV" else "b"
    arrays["state"] = np.asarray(fit.stan_variable(state_name))[selected, -1]
    arrays["state_kind"] = np.asarray(["baseline_b"], dtype="U16")
    if family in {"NN-SV", "RV-NN-SV", "RV-RA-NN-SV"}:
        arrays.update(
            {
                "parameter__w1": np.asarray(fit.stan_variable("W1"))[selected],
                "parameter__b1": np.asarray(fit.stan_variable("b1"))[selected],
                "parameter__output_weights": np.asarray(
                    fit.stan_variable("output_weight")
                )[selected],
                "parameter__centering_constant": np.asarray(
                    fit.stan_variable("centering_mean")
                )[selected],
            }
        )
        if model_id == "RV-RA-NN-SV":
            arrays.update(
                {
                    "parameter__gamma_A": np.asarray(fit.stan_variable("gamma_A"))[
                        selected
                    ],
                    "parameter__q_centering_mean": np.asarray(
                        fit.stan_variable("q_centering_mean")
                    )[selected],
                }
            )
    elif family == "RV-LIN-SV":
        x = np.asarray(estimation_features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(spec.feature_names):
            raise ValueError("linear estimation features must match the model dimension")
        beta = np.asarray(fit.stan_variable("beta"))[selected]
        arrays["parameter__beta"] = beta
        arrays["parameter__x_center"] = np.broadcast_to(
            np.mean(x, axis=0), beta.shape
        ).copy()
    return arrays


def filter_from_refit(
    refit: Mapping[str, np.ndarray], *, level: int = 4,
    cache_transition_kernel: bool = True, transition_batch_size: int = 128,
) -> FixedParameterGridFilter:
    parameters = {
        name.removeprefix("parameter__"): np.asarray(value).copy()
        for name, value in refit.items()
        if name.startswith("parameter__")
    }
    return FixedParameterGridFilter(
        parameters=parameters,
        initial_state=np.asarray(refit["state"], dtype=float),
        weights=np.asarray(refit["weights"], dtype=float).copy(),
        ancestry=np.asarray(refit["ancestry"], dtype=int),
        level=level,
        cache_transition_kernel=cache_transition_kernel,
        transition_batch_size=transition_batch_size,
    )


def filter_bank_from_refit(
    refit: Mapping[str, np.ndarray], *, levels: tuple[int, int, int],
    cache_transition_kernel: bool = True, transition_batch_size: int = 128,
) -> dict[int, FixedParameterGridFilter]:
    """Construct adjacent deterministic filters over identical parameter atoms."""
    normalized = tuple(int(level) for level in levels)
    if len(normalized) != 3 or any(
        right != left + 1
        for left, right in zip(normalized, normalized[1:])
    ):
        raise ValueError("filter levels must contain three adjacent integers")
    return {level: filter_from_refit(refit, level=level,
        cache_transition_kernel=cache_transition_kernel, transition_batch_size=transition_batch_size)
        for level in normalized}


def observable_correction(
    model_id: str,
    parameters: Mapping[str, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    """Evaluate the centred correction for one set of daily features."""
    model_id = base_model_id(model_id)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be a matrix")
    if model_id == "SV":
        return np.zeros(x.shape[0], dtype=float)
    if model_id == "RV-LIN-SV":
        return np.einsum(
            "nd,nd->n",
            x - np.asarray(parameters["x_center"], dtype=float),
            np.asarray(parameters["beta"], dtype=float),
        )
    core = x[:, :-1] if model_id == "RV-RA-NN-SV" else x
    hidden = np.tanh(
        np.einsum("nd,ndh->nh", core, np.asarray(parameters["w1"], dtype=float))
        + np.asarray(parameters["b1"], dtype=float)
    )
    correction = (
        np.einsum(
            "nh,nh->n",
            hidden,
            np.asarray(parameters["output_weights"], dtype=float),
        )
        - np.asarray(parameters["centering_constant"], dtype=float)
    )
    if model_id == "RV-RA-NN-SV":
        correction += np.asarray(parameters["gamma_A"], dtype=float) * (
            x[:, -1]
            - np.asarray(parameters["q_centering_mean"], dtype=float)
        )
    return correction


def asymmetry_to_unbounded(values: np.ndarray, shrink: float = 0.999) -> np.ndarray:
    """Map the bounded semivariance imbalance to the auxiliary HAR scale."""
    asymmetry = np.asarray(values, dtype=float)
    if not 0 < shrink < 1 or np.any(~np.isfinite(asymmetry)):
        raise ValueError("asymmetry transform inputs are invalid")
    clipped = np.clip(shrink * asymmetry, -1 + 1e-12, 1 - 1e-12)
    return np.arctanh(clipped)


def unbounded_to_asymmetry(values: np.ndarray, shrink: float = 0.999) -> np.ndarray:
    """Return auxiliary HAR paths to the defining [-1,1] support."""
    transformed = np.asarray(values, dtype=float)
    if not 0 < shrink < 1 or np.any(~np.isfinite(transformed)):
        raise ValueError("asymmetry inverse-transform inputs are invalid")
    return np.clip(np.tanh(transformed) / shrink, -1.0, 1.0)


def fit_log_har_posterior(
    log_rv: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
    prior_variance: float = 100.0,
    inverse_gamma_shape: float = 2.0,
    inverse_gamma_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the conjugate posterior for the auxiliary log-HAR model."""
    values = np.asarray(log_rv, dtype=float)
    if values.ndim != 1 or values.size < 30 or np.any(~np.isfinite(values)):
        raise ValueError("log_rv must contain at least 30 finite observations")
    if draws <= 0 or prior_variance <= 0:
        raise ValueError("draws and prior_variance must be positive")
    rows: list[list[float]] = []
    response: list[float] = []
    for index in range(22, values.size):
        history = values[:index]
        rows.append(
            [
                1.0,
                float(history[-1]),
                float(np.mean(history[-5:])),
                float(np.mean(history[-22:])),
            ]
        )
        response.append(float(values[index]))
    x = np.asarray(rows, dtype=float)
    y = np.asarray(response, dtype=float)
    prior_precision = np.eye(4) / prior_variance
    posterior_precision = prior_precision + x.T @ x
    posterior_variance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_variance @ (x.T @ y)
    shape = inverse_gamma_shape + x.shape[0] / 2
    scale = inverse_gamma_scale + 0.5 * (
        y @ y - posterior_mean @ posterior_precision @ posterior_mean
    )
    sigma2 = invgamma.rvs(shape, scale=scale, size=draws, random_state=rng)
    beta = np.stack(
        [
            rng.multivariate_normal(posterior_mean, value * posterior_variance)
            for value in sigma2
        ]
    )
    return beta, np.sqrt(sigma2)


def simulate_log_har_paths(
    history: np.ndarray,
    *,
    beta: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
    parameter_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate future realised variance with parameter uncertainty."""
    base = np.asarray(history, dtype=float)
    coefficients = np.asarray(beta, dtype=float)
    scales = np.asarray(sigma, dtype=float)
    if base.ndim != 1 or base.size < 22 or np.any(~np.isfinite(base)):
        raise ValueError("history must contain at least 22 finite observations")
    if coefficients.ndim != 2 or coefficients.shape[1] != 4:
        raise ValueError("beta must have four columns")
    if scales.shape != (coefficients.shape[0],) or np.any(scales <= 0):
        raise ValueError("sigma must align with beta and be positive")
    if horizon <= 0 or paths <= 0:
        raise ValueError("horizon and paths must be positive")
    indices = (
        rng.integers(0, coefficients.shape[0], size=paths)
        if parameter_indices is None
        else np.asarray(parameter_indices, dtype=int)
    )
    if indices.shape != (paths,) or np.any(
        (indices < 0) | (indices >= coefficients.shape[0])
    ):
        raise ValueError("auxiliary parameter indices are invalid")
    selected_beta = coefficients[indices]
    selected_sigma = scales[indices]
    work = np.broadcast_to(base[-22:], (paths, 22)).copy()
    result = np.empty((paths, horizon), dtype=float)
    for step in range(horizon):
        row = np.column_stack(
            (
                np.ones(paths),
                work[:, -1],
                np.mean(work[:, -5:], axis=1),
                np.mean(work, axis=1),
            )
        )
        value = np.einsum("ni,ni->n", row, selected_beta)
        value += selected_sigma * rng.normal(size=paths)
        result[:, step] = value
        work = np.column_stack((work[:, 1:], value))
    return result


def fit_rv_ra_auxiliary_posterior(
    log_rv: np.ndarray,
    asymmetry: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
    prior_variance: float = 100.0,
    inverse_gamma_shape: float = 2.0,
    inverse_gamma_scale: float = 0.1,
) -> dict[str, np.ndarray]:
    """Fit the two univariate HAR auxiliaries used by RV-RA-NN-SV."""
    rv_beta, rv_sigma = fit_log_har_posterior(
        log_rv,
        draws=draws,
        rng=rng,
        prior_variance=prior_variance,
        inverse_gamma_shape=inverse_gamma_shape,
        inverse_gamma_scale=inverse_gamma_scale,
    )
    a_beta, a_sigma = fit_log_har_posterior(
        asymmetry_to_unbounded(asymmetry),
        draws=draws,
        rng=rng,
        prior_variance=prior_variance,
        inverse_gamma_shape=inverse_gamma_shape,
        inverse_gamma_scale=inverse_gamma_scale,
    )
    return {
        "rv_beta": rv_beta,
        "rv_sigma": rv_sigma,
        "a_beta": a_beta,
        "a_sigma": a_sigma,
    }


def simulate_rv_ra_auxiliary_paths(
    log_rv_history: np.ndarray,
    asymmetry_history: np.ndarray,
    *,
    posterior: Mapping[str, np.ndarray],
    horizon: int,
    paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate realised variance and asymmetry with matched HAR draws."""
    count = np.asarray(posterior["rv_beta"]).shape[0]
    indices = rng.integers(0, count, size=paths)
    future_log_rv = simulate_log_har_paths(
        log_rv_history,
        beta=np.asarray(posterior["rv_beta"]),
        sigma=np.asarray(posterior["rv_sigma"]),
        horizon=horizon,
        paths=paths,
        rng=rng,
        parameter_indices=indices,
    )
    future_asymmetry_unbounded = simulate_log_har_paths(
        asymmetry_to_unbounded(asymmetry_history),
        beta=np.asarray(posterior["a_beta"]),
        sigma=np.asarray(posterior["a_sigma"]),
        horizon=horizon,
        paths=paths,
        rng=rng,
        parameter_indices=indices,
    )
    return future_log_rv, unbounded_to_asymmetry(
        future_asymmetry_unbounded
    )


def simulate_forecast_paths(
    *,
    model_id: str,
    state_filter: FixedParameterGridFilter,
    refit: Mapping[str, np.ndarray],
    standardized_return_history: np.ndarray,
    log_rv_history: np.ndarray,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
    future_log_rv: np.ndarray | None = None,
    asymmetry_history: np.ndarray | None = None,
    future_asymmetry: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one forecast origin using the filtered state distribution."""
    spec = model_spec(model_id)
    if horizon not in spec.horizons:
        raise ValueError(f"horizon {horizon} is not defined for {model_id}")
    z_history = np.asarray(standardized_return_history, dtype=float)
    rv_history = np.asarray(log_rv_history, dtype=float)
    if z_history.size < 22 or rv_history.size < 22:
        raise ValueError("forecast histories must contain at least 22 observations")
    if model_id in {"RV-NN-SV", "RV-RA-NN-SV"}:
        if future_log_rv is None or np.asarray(future_log_rv).shape != (
            paths,
            horizon,
        ):
            raise ValueError(
                f"{model_id} requires aligned future log-RV paths"
            )
    if model_id == "RV-RA-NN-SV":
        if asymmetry_history is None or np.asarray(asymmetry_history).size < 22:
            raise ValueError("RV-RA-NN-SV requires an asymmetry history")
        if future_asymmetry is None or np.asarray(future_asymmetry).shape != (
            paths,
            horizon,
        ):
            raise ValueError(
                "RV-RA-NN-SV requires aligned future asymmetry paths"
            )

    selected = rng.choice(
        state_filter.size,
        size=paths,
        replace=True,
        p=state_filter.weights,
    )
    origin_state = state_filter.sample_states(selected, rng)
    sigma_eta = np.asarray(state_filter.parameters["sigma_eta"], dtype=float)[
        selected
    ]
    nu = np.asarray(state_filter.parameters["nu"], dtype=float)[selected]
    state_innovations = np.empty((paths, horizon), dtype=float)
    return_innovations = np.empty((paths, horizon), dtype=float)
    for step in range(horizon):
        state_innovations[:, step] = sigma_eta * rng.normal(size=paths)
        return_innovations[:, step] = standardized_t_ppf(
            rng.uniform(size=paths), nu
        )

    inputs: dict[str, np.ndarray] = {
        "source_parameter_ancestry": state_filter.ancestry[selected],
        "origin_baseline_states": origin_state,
        "state_innovations": state_innovations,
        "return_innovations": return_innovations,
        "origin_z_history_tail": z_history[-22:],
        "origin_log_rv_history_tail": rv_history[-22:],
    }
    if future_log_rv is not None:
        inputs["future_log_rv_paths"] = np.asarray(future_log_rv, dtype=float)
    if asymmetry_history is not None:
        inputs["origin_asymmetry_history_tail"] = np.asarray(
            asymmetry_history, dtype=float
        )[-22:]
    if future_asymmetry is not None:
        inputs["future_asymmetry_paths"] = np.asarray(
            future_asymmetry, dtype=float
        )
    result = reconstruct_forecast_paths(model_id, refit, inputs)
    result["degrees_of_freedom"] = nu.copy()
    return result
