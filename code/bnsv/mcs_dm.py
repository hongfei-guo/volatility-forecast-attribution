"""Block-bootstrap Model Confidence Set and descriptive Diebold-Mariano tests."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats


def _hac_variance(series: np.ndarray, lag: int) -> float:
    x = np.asarray(series, dtype=float)
    x = x - np.mean(x)
    n = x.size
    gamma0 = float(x @ x / n)
    value = gamma0
    for j in range(1, min(lag, n - 1) + 1):
        weight = 1 - j / (lag + 1)
        gamma = float(x[j:] @ x[:-j] / n)
        value += 2 * weight * gamma
    return max(value, 1e-15)


def automatic_hac_lag(series: np.ndarray, horizon: int = 1) -> int:
    """Conservative Newey-West plug-in lag, never shorter than h-1."""
    n = np.asarray(series).size
    if n < 5:
        raise ValueError("at least five observations are required")
    plug_in = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return min(n - 1, max(horizon - 1, plug_in))


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    hac_lag: int | None = None,
    harvey_correction: bool = True,
) -> dict[str, float | int]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 5:
        raise ValueError("loss vectors must align and contain at least five observations")
    d = a - b
    n = d.size
    selected_lag = automatic_hac_lag(d, horizon) if hac_lag is None else int(hac_lag)
    if selected_lag < horizon - 1 or selected_lag >= n:
        raise ValueError("hac_lag must lie between horizon-1 and n-1")
    variance = _hac_variance(d, selected_lag)
    stat = float(np.mean(d) / np.sqrt(variance / n))
    if harvey_correction:
        factor = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        stat *= float(max(factor, 0.0))
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {
        "statistic": stat,
        "p_value": p,
        "mean_loss_difference": float(np.mean(d)),
        "hac_lag": selected_lag,
    }


def stationary_bootstrap_indices(n: int, mean_block_length: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or mean_block_length < 1:
        raise ValueError("invalid bootstrap controls")
    p = 1.0 / mean_block_length
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices


@dataclass(frozen=True)
class MCSResult:
    retained_models: tuple[str, ...]
    elimination_order: tuple[str, ...]
    p_values: tuple[float, ...]
    mean_block_length: float
    statistic: str = "T_R"


def model_confidence_set(
    losses: np.ndarray,
    model_names: list[str],
    *,
    alpha: float = 0.10,
    bootstrap_replications: int = 1000,
    mean_block_length: float = 10.0,
    seed: int = 20260730,
) -> MCSResult:
    """Sequential studentized T_R range-statistic MCS.

    Studentisation uses the stationary-bootstrap variance of each pairwise
    sample mean. The test statistic is ``max_{i,j}|t_ij|`` and elimination
    removes the model with the largest standardized pairwise loss. This is the
    Hansen--Lunde--Nason ``T_R`` statistic and coherent ``e_R`` rule, not
    ``Tmax`` (which standardizes each model's average loss differential).
    """
    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 2 or loss.shape[1] != len(model_names):
        raise ValueError("loss matrix and model names do not align")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if loss.shape[0] < 20 or np.any(~np.isfinite(loss)):
        raise ValueError("MCS requires at least 20 complete finite observations")
    rng = np.random.default_rng(seed)
    active = list(range(loss.shape[1]))
    eliminated: list[str] = []
    p_values: list[float] = []
    running_p = 0.0
    while len(active) > 1:
        sub = loss[:, active]
        pair_diff = sub[:, :, None] - sub[:, None, :]
        mean_diff = np.mean(pair_diff, axis=0)
        bootstrap_means = np.empty((bootstrap_replications, len(active), len(active)))
        for b in range(bootstrap_replications):
            idx = stationary_bootstrap_indices(loss.shape[0], mean_block_length, rng)
            bootstrap_means[b] = np.mean(pair_diff[idx], axis=0)
        se = np.std(bootstrap_means, axis=0, ddof=1)
        se = np.maximum(se, 1e-12)
        t_matrix = mean_diff / se
        np.fill_diagonal(t_matrix, 0.0)
        t_obs = float(np.max(np.abs(t_matrix)))
        centered_bootstrap = bootstrap_means - mean_diff[None, :, :]
        boot_stats = np.max(np.abs(centered_bootstrap / se[None, :, :]), axis=(1, 2))
        p_value = float((1 + np.sum(boot_stats >= t_obs)) / (bootstrap_replications + 1))
        running_p = max(running_p, p_value)
        p_values.append(running_p)
        if p_value > alpha:
            break
        worst_local = int(np.argmax(np.max(t_matrix, axis=1)))
        eliminated.append(model_names[active[worst_local]])
        active.pop(worst_local)
    return MCSResult(
        retained_models=tuple(model_names[i] for i in active),
        elimination_order=tuple(eliminated),
        p_values=tuple(p_values),
        mean_block_length=float(mean_block_length),
        statistic="T_R",
    )
