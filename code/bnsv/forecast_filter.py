"""Daily latent-state filtering with fixed posterior parameter atoms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from scipy.special import logsumexp, ndtr

from .standardized_t import standardized_t_logpdf


ParameterMap = dict[str, np.ndarray]
CorrectionFunction = Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray]


class NumericalFilterError(FloatingPointError):
    """Finite valid inputs could not produce a usable numerical filter update."""

    def __init__(self, message: str, *, level: int | None = None) -> None:
        super().__init__(message)
        self.level = level


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("weights must be a nonempty nonnegative vector")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have positive finite sum")
    normalized = values / total
    return float(1.0 / np.sum(np.square(normalized)))


def grid_shape(level: int) -> tuple[int, float]:
    """Return the node count and half-span for a deterministic grid level."""
    if int(level) != level or level < 0:
        raise ValueError("grid level must be a nonnegative integer")
    return 16 * (2 ** int(level)), 4.0 + 2.0 * int(level)


@dataclass
class FixedParameterGridFilter:
    """Integrate the one-dimensional baseline state for fixed parameters."""

    parameters: ParameterMap
    initial_state: np.ndarray
    weights: np.ndarray
    ancestry: np.ndarray
    level: int = 4
    transition_batch_size: int = 128
    cache_transition_kernel: bool = True
    _state_probabilities: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _transition_kernel: np.ndarray | None = field(init=False, repr=False)
    _cumulative_log_likelihood: np.ndarray = field(init=False, repr=False)
    _sigma_h: np.ndarray = field(init=False, repr=False)
    _u_nodes: np.ndarray = field(init=False, repr=False)
    _u_edges: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.transition_batch_size) != self.transition_batch_size or self.transition_batch_size < 1:
            raise ValueError("transition_batch_size must be a positive integer")
        self.transition_batch_size = int(self.transition_batch_size)
        self.initial_state = np.asarray(self.initial_state, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.ancestry = np.asarray(self.ancestry, dtype=int)
        size = self.initial_state.size
        if (
            size == 0
            or self.weights.shape != (size,)
            or self.ancestry.shape != (size,)
        ):
            raise ValueError("state, weights, and ancestry must align")
        if (np.any(~np.isfinite(self.initial_state))
                or np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0)):
            raise ValueError("invalid initial parameter cloud")
        total = float(self.weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights sum to zero")
        self.weights /= total
        for name, values in list(self.parameters.items()):
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != size:
                raise ValueError(f"parameter {name} does not align with atoms")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"parameter {name} must be finite")
            self.parameters[name] = array
        missing = {"mu", "phi", "sigma_eta", "nu"} - set(self.parameters)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        phi = np.asarray(self.parameters["phi"], dtype=float)
        sigma_eta = np.asarray(self.parameters["sigma_eta"], dtype=float)
        if np.any((phi <= 0) | (phi >= 1)) or np.any(sigma_eta <= 0):
            raise ValueError("phi and sigma_eta lie outside their support")
        if np.any(np.asarray(self.parameters["nu"]) <= 2):
            raise ValueError("nu must exceed two")
        shrink = np.sqrt(np.maximum(1.0 - np.square(phi), np.finfo(float).tiny))
        self._sigma_h = sigma_eta / shrink
        nodes, half_span = grid_shape(self.level)
        self._u_nodes = np.linspace(-half_span, half_span, nodes)
        midpoints = 0.5 * (self._u_nodes[:-1] + self._u_nodes[1:])
        self._u_edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
        self._cumulative_log_likelihood = np.zeros(size, dtype=float)
        self._transition_kernel = self._build_transition_kernel() if self.cache_transition_kernel else None

    @property
    def size(self) -> int:
        return self.initial_state.size

    @property
    def state(self) -> np.ndarray:
        if self._state_probabilities is None:
            return self.initial_state.copy()
        mean_u = self._state_probabilities @ self._u_nodes
        return np.asarray(self.parameters["mu"], dtype=float) + self._sigma_h * mean_u

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    @property
    def state_probabilities(self) -> np.ndarray | None:
        if self._state_probabilities is None:
            return None
        return self._state_probabilities.copy()

    @property
    def cumulative_log_likelihood(self) -> np.ndarray:
        return self._cumulative_log_likelihood.copy()

    def _build_transition_kernel(self, first: int = 0, last: int | None = None) -> np.ndarray:
        last = self.size if last is None else last
        nodes = self._u_nodes.size
        kernel = np.empty((last - first, nodes, nodes), dtype=float)
        phi = np.asarray(self.parameters["phi"], dtype=float)
        shrink = np.sqrt(np.maximum(1.0 - np.square(phi), np.finfo(float).tiny))
        batch = max(1, int(self.transition_batch_size))
        for start in range(first, last, batch):
            stop = min(last, start + batch)
            means = phi[start:stop, None] * self._u_nodes[None, :]
            z = (
                self._u_edges[None, None, :] - means[:, :, None]
            ) / shrink[start:stop, None, None]
            probabilities = np.maximum(np.diff(ndtr(z), axis=2), 0.0)
            row_sum = probabilities.sum(axis=2, keepdims=True)
            if np.any(row_sum <= 0) or np.any(~np.isfinite(row_sum)):
                raise FloatingPointError("invalid state transition probabilities")
            kernel[start-first:stop-first] = probabilities / row_sum
        return kernel

    def _predict_probabilities(self) -> np.ndarray:
        if self._state_probabilities is None:
            return self._first_prediction()
        if self._transition_kernel is not None:
            return np.einsum("mi,mij->mj", self._state_probabilities,
                             self._transition_kernel, optimize=True)
        predicted = np.empty_like(self._state_probabilities)
        for first in range(0, self.size, self.transition_batch_size):
            last = min(self.size, first + self.transition_batch_size)
            kernel = self._build_transition_kernel(first, last)
            predicted[first:last] = np.einsum("mi,mij->mj",
                self._state_probabilities[first:last], kernel, optimize=True)
        return predicted

    def _first_prediction(self) -> np.ndarray:
        mu = np.asarray(self.parameters["mu"], dtype=float)
        phi = np.asarray(self.parameters["phi"], dtype=float)
        shrink = np.sqrt(np.maximum(1.0 - np.square(phi), np.finfo(float).tiny))
        u0 = (self.initial_state - mu) / self._sigma_h
        means = phi * u0
        z = (self._u_edges[None, :] - means[:, None]) / shrink[:, None]
        probabilities = np.maximum(np.diff(ndtr(z), axis=1), 0.0)
        row_sum = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0) or np.any(~np.isfinite(row_sum)):
            raise FloatingPointError("invalid initial state prediction")
        return probabilities / row_sum

    def sample_states(
        self, particle_indices: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        indices = np.asarray(particle_indices, dtype=int)
        if np.any((indices < 0) | (indices >= self.size)):
            raise ValueError("particle index out of bounds")
        if self._state_probabilities is None:
            return self.initial_state[indices].copy()
        selected = self._state_probabilities[indices]
        uniforms = rng.random(indices.size)
        grid_indices = np.sum(
            np.cumsum(selected, axis=1) < uniforms[:, None], axis=1
        )
        grid_indices = np.minimum(grid_indices, self._u_nodes.size - 1)
        mu = np.asarray(self.parameters["mu"], dtype=float)[indices]
        return mu + self._sigma_h[indices] * self._u_nodes[grid_indices]

    def step(
        self,
        *,
        x_t: np.ndarray,
        observed_z: float,
        correction: CorrectionFunction,
    ) -> dict[str, float | np.ndarray]:
        """Update the conditional state distribution after one observation."""
        x = np.asarray(x_t, dtype=float)
        if x.ndim == 1:
            x = np.broadcast_to(x, (self.size, x.size))
        if x.ndim != 2 or x.shape[0] != self.size:
            raise ValueError("x_t must align with parameter atoms")
        if not np.isfinite(observed_z) or np.any(~np.isfinite(x)):
            raise ValueError("observed returns and feature inputs must be finite")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                return self._update(x, observed_z, correction)
        except NumericalFilterError:
            raise
        except FloatingPointError as exc:
            raise NumericalFilterError(str(exc), level=self.level) from exc

    def _update(
        self, x: np.ndarray, observed_z: float, correction: CorrectionFunction,
    ) -> dict[str, float | np.ndarray]:
        """Compute an update before committing any new state or outer weights."""
        drift = np.asarray(correction(self.parameters, x), dtype=float)
        if drift.shape != (self.size,):
            raise ValueError("correction must return one value per atom")
        if np.any(~np.isfinite(drift)):
            raise NumericalFilterError("non-finite correction", level=self.level)
        predicted = self._predict_probabilities()
        predicted = np.maximum(predicted, 0.0)
        predicted /= predicted.sum(axis=1, keepdims=True)
        mu = np.asarray(self.parameters["mu"], dtype=float)
        h = mu[:, None] + self._sigma_h[:, None] * self._u_nodes + drift[:, None]
        sigma = np.exp(0.5 * h)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise NumericalFilterError("unusable conditional scale", level=self.level)
        log_likelihood = standardized_t_logpdf(
            observed_z,
            np.asarray(self.parameters["nu"], dtype=float)[:, None],
            sigma,
        )
        if np.any(~np.isfinite(log_likelihood)):
            raise NumericalFilterError("non-finite predictive log likelihood", level=self.level)
        log_joint = np.log(np.maximum(predicted, np.finfo(float).tiny))
        log_joint += log_likelihood
        log_increment = logsumexp(log_joint, axis=1)
        probabilities = np.exp(log_joint - log_increment[:, None])
        cumulative = self._cumulative_log_likelihood + log_increment
        log_outer = np.log(np.maximum(self.weights, np.finfo(float).tiny))
        log_outer += log_increment
        log_outer -= logsumexp(log_outer)
        weights = np.exp(log_outer)
        if any(np.any(~np.isfinite(v)) for v in (probabilities, cumulative, weights)):
            raise NumericalFilterError("non-finite updated filter state", level=self.level)
        self._state_probabilities = probabilities
        self._cumulative_log_likelihood = cumulative
        self.weights = weights
        state = self.state
        return {
            "state": state,
            "log_variance": state + drift,
            "log_predictive_increment": log_increment.copy(),
            "parameter_ess": self.ess,
            "weights": self.weights.copy(),
        }


def _weighted_ess_interval(
    log_weights: np.ndarray, log_weight_errors: np.ndarray
) -> tuple[float, float]:
    weights = np.asarray(log_weights, dtype=float)
    errors = np.asarray(log_weight_errors, dtype=float)
    if (
        weights.ndim != 1
        or errors.shape != weights.shape
        or weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(~np.isfinite(errors))
        or np.any(errors < 0)
    ):
        raise ValueError("ESS error bounds must be aligned and finite")
    lower_log = 2 * logsumexp(weights - errors) - logsumexp(
        2 * weights + 2 * errors
    )
    upper_log = 2 * logsumexp(weights + errors) - logsumexp(
        2 * weights - 2 * errors
    )
    return (
        max(1.0, float(np.exp(max(lower_log, -745.0)))),
        min(float(weights.size), float(np.exp(min(upper_log, 709.0)))),
    )


def filter_bank_ess_diagnostic(
    filters: Mapping[int, FixedParameterGridFilter],
    *,
    initial_weights: np.ndarray,
    warning_fraction: float = 0.20,
) -> dict[str, object]:
    """Return both sides of the existing estimated-error ESS diagnostic."""
    levels = tuple(sorted(filters))
    if len(levels) != 3 or any(
        right != left + 1 for left, right in zip(levels, levels[1:])
    ):
        raise ValueError("filter continuation diagnostic requires three adjacent levels")
    coarse, previous, current = (filters[level] for level in levels)
    weights = np.asarray(initial_weights, dtype=float)
    if (
        weights.ndim != 1
        or weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("initial filter weights must be positive and normalized")
    if not 0 < warning_fraction < 1:
        raise ValueError("warning_fraction must lie in (0,1)")
    for filtered in (coarse, previous, current):
        if filtered.weights.shape != weights.shape or filtered.cumulative_log_likelihood.shape != weights.shape:
            raise ValueError("filter state does not align with initial atoms")
        if (np.any(~np.isfinite(filtered.weights)) or np.any(filtered.weights < 0)
                or filtered.weights.sum() <= 0
                or np.any(~np.isfinite(filtered.cumulative_log_likelihood))):
            raise NumericalFilterError("unusable filter state in ESS diagnostic", level=filtered.level)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            denominator = abs(previous.ess - coarse.ess)
            numerator = abs(current.ess - previous.ess)
            if denominator == 0:
                contraction = 0.0 if numerator == 0 else float("inf")
            else:
                contraction = numerator / denominator
            if not np.isfinite(contraction) or contraction >= 1:
                return {"decision": "unresolved", "ess_lower": 1.0,
                        "ess_upper": float(weights.size), "contraction": None if not np.isfinite(contraction) else contraction}
            adjacent_error = np.abs(
                current.cumulative_log_likelihood
                - previous.cumulative_log_likelihood
            )
            remainder_error = adjacent_error * contraction / (1 - contraction)
            if np.any(~np.isfinite(remainder_error)):
                raise NumericalFilterError("non-finite estimated likelihood remainder")
            lower, upper = _weighted_ess_interval(
                np.log(weights) + current.cumulative_log_likelihood,
                remainder_error,
            )
            threshold = warning_fraction * weights.size
            return {"decision": "continue" if lower >= threshold else "refit" if upper < threshold else "unresolved",
                    "ess_lower": lower, "ess_upper": upper, "contraction": contraction}
    except NumericalFilterError:
        raise
    except FloatingPointError as exc:
        raise NumericalFilterError("ESS diagnostic: " + str(exc)) from exc


def filter_bank_is_adequate(filters, *, initial_weights, warning_fraction=0.20) -> bool:
    """Return the continuation decision of the ESS diagnostic."""
    return filter_bank_ess_diagnostic(filters, initial_weights=initial_weights,
        warning_fraction=warning_fraction)["decision"] == "continue"
