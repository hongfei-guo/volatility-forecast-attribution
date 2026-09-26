"""Data-generating processes and summaries for the identification simulation."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import binom, beta, norm

STAN_SEED_MAX = 2_147_483_647
PRIMARY_REPLICATIONS = 200
AMPLITUDE_CALIBRATION_REPLICATIONS = 150
SEED_TABLE_PATH = Path(__file__).resolve().parents[2] / "design" / "identification_seeds.csv"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_npz(
    path: str | Path,
    arrays: dict[str, np.ndarray | float | int | str],
    *,
    compressed: bool,
) -> None:
    """Write the simulation arrays to an NPZ file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {name: np.asarray(value) for name, value in arrays.items()}
    if compressed:
        np.savez_compressed(destination, **values)
    else:
        np.savez(destination, **values)


@lru_cache(maxsize=1)
def _reported_seeds() -> dict[tuple[int, str], int]:
    """Read the random streams used by the reported simulation."""
    with SEED_TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["replication_id", "stream_name", "seed"]:
            raise ValueError("identification seed table has unexpected columns")
        result: dict[tuple[int, str], int] = {}
        for row in reader:
            key = (int(row["replication_id"]), str(row["stream_name"]))
            seed = int(row["seed"])
            if key in result or not 0 < seed <= STAN_SEED_MAX:
                raise ValueError("identification seed table contains an invalid row")
            result[key] = seed
    if not result:
        raise ValueError("identification seed table is empty")
    return result


def derived_seed(replication_id: int, stream_name: str) -> int:
    """Return the reported seed for one replication and calculation."""
    if replication_id < 1 or not stream_name:
        raise ValueError("invalid seed label")
    try:
        return _reported_seeds()[(replication_id, stream_name)]
    except KeyError as exc:
        raise ValueError("seed is not defined for this calculation") from exc


def standardized_student_t(rng: np.random.Generator, nu: float, size: int | tuple[int, ...]) -> np.ndarray:
    if nu <= 2:
        raise ValueError("nu must exceed two")
    return rng.standard_t(nu, size=size) * math.sqrt((nu - 2.0) / nu)


def collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape or x_arr.size < 2:
        raise ValueError("x and y must have the same nontrivial shape")
    order = np.argsort(x_arr, kind="mergesort")
    xs, ys = x_arr[order], y_arr[order]
    unique, start, count = np.unique(xs, return_index=True, return_counts=True)
    sums = np.add.reduceat(ys, start)
    return unique, sums / count


def natural_spline_with_tangent(x_nodes: np.ndarray, y_nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Natural cubic interpolation and linear endpoint-tangent extrapolation."""
    xn = np.asarray(x_nodes, dtype=float)
    yn = np.asarray(y_nodes, dtype=float)
    q = np.asarray(x, dtype=float)
    if xn.ndim != 1 or yn.shape != xn.shape or xn.size < 3 or np.any(np.diff(xn) <= 0):
        raise ValueError("spline nodes must be strictly increasing and nontrivial")
    spline = CubicSpline(xn, yn, bc_type="natural", extrapolate=False)
    out = np.asarray(spline(np.clip(q, xn[0], xn[-1])), dtype=float)
    left = q < xn[0]
    right = q > xn[-1]
    if np.any(left):
        out[left] = yn[0] + float(spline(xn[0], 1)) * (q[left] - xn[0])
    if np.any(right):
        out[right] = yn[-1] + float(spline(xn[-1], 1)) * (q[right] - xn[-1])
    return out


@dataclass(frozen=True)
class ReferenceCurve:
    x: np.ndarray
    y: np.ndarray
    left_slope: float
    right_slope: float
    source_records: tuple[dict[str, Any], ...]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return natural_spline_with_tangent(self.x, self.y, x)


@dataclass(frozen=True)
class DGPParameters:
    mu: float
    phi: float
    sigma_h: float
    sigma_eta: float
    nu: float
    target_correction_share: float
    estimation_rows: int
    forecast_rows: int
    burn_in: int


def parameters_from_config(config: dict[str, Any]) -> DGPParameters:
    p = config["dgp"]["anchor"]
    sample = config["dgp"]["sample"]
    return DGPParameters(
        mu=float(p["mu"]),
        phi=float(p["phi"]),
        sigma_h=float(p["sigma_h"]),
        sigma_eta=float(p["sigma_eta"]),
        nu=float(p["nu"]),
        target_correction_share=float(p["target_correction_share"]),
        estimation_rows=int(sample["estimation_rows"]),
        forecast_rows=int(sample["forecast_rows"]),
        burn_in=int(sample["burn_in"]),
    )


def _innovation_streams(
    params: DGPParameters,
    replication_id: int,
    *,
    seed_namespace: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n = params.burn_in + params.estimation_rows + params.forecast_rows
    if not seed_namespace:
        raise ValueError("seed namespace must be nonempty")
    state_rng = np.random.default_rng(
        derived_seed(replication_id, f"{seed_namespace}_state")
    )
    obs_rng = np.random.default_rng(
        derived_seed(replication_id, f"{seed_namespace}_observation")
    )
    alpha = state_rng.standard_normal(n - 1)
    eps = standardized_student_t(obs_rng, params.nu, n)
    b0 = params.mu + params.sigma_h * state_rng.standard_normal()
    z0 = math.exp(0.5 * params.mu) * float(eps[0])
    return alpha, eps, b0, z0


def simulate_replication(
    params: DGPParameters,
    curve: ReferenceCurve,
    *,
    replication_id: int,
    dgp_id: str,
    amplitude: float,
    seed_namespace: str = "experiment",
) -> dict[str, np.ndarray | float | int | str]:
    if dgp_id not in {"DGP-N", "DGP-0"}:
        raise ValueError("unknown DGP")
    if amplitude < 0:
        raise ValueError("amplitude must be nonnegative")
    n = params.burn_in + params.estimation_rows + params.forecast_rows
    alpha, eps, b0, z0 = _innovation_streams(
        params,
        replication_id,
        seed_namespace=seed_namespace,
    )
    b = np.empty(n, dtype=float)
    c = np.zeros(n, dtype=float)
    h = np.empty(n, dtype=float)
    z = np.empty(n, dtype=float)
    b[0], h[0], z[0] = b0, b0, z0
    nonlinear = dgp_id == "DGP-N"
    spline = CubicSpline(curve.x, curve.y, bc_type="natural", extrapolate=False)

    def evaluate_one(value: float) -> float:
        if value < curve.x[0]:
            return float(curve.y[0] + curve.left_slope * (value - curve.x[0]))
        if value > curve.x[-1]:
            return float(curve.y[-1] + curve.right_slope * (value - curve.x[-1]))
        return float(spline(value))

    for t in range(1, n):
        b[t] = params.mu + params.phi * (b[t - 1] - params.mu) + params.sigma_eta * alpha[t - 1]
        if nonlinear:
            c[t] = amplitude * evaluate_one(float(z[t - 1]))
        h[t] = b[t] + c[t]
        z[t] = math.exp(0.5 * h[t]) * eps[t]

    start = params.burn_in
    stop = start + params.estimation_rows + params.forecast_rows
    obs = slice(start, stop)
    lag = slice(start - 1, stop - 1)
    z_out = z[obs].copy()
    x_out = z[lag].copy()
    h_lag_out = h[lag].copy()
    b_out = b[obs].copy()
    c_out = c[obs].copy()
    h_out = h[obs].copy()

    # Exact empirical centring on estimation rows is a representation shift:
    # total h and observations remain unchanged.
    shift = float(np.mean(c_out[: params.estimation_rows]))
    c_out -= shift
    b_out += shift
    mu_truth = params.mu + shift
    if not np.array_equal(h_out, b_out + c_out):
        if not np.allclose(h_out, b_out + c_out, rtol=0.0, atol=2e-15):
            raise AssertionError("centred decomposition does not preserve h")

    share = float(np.var(c_out[: params.estimation_rows]) / np.var(h_out[: params.estimation_rows]))
    return {
        "dgp_id": np.asarray([dgp_id]),
        "replication_id": np.asarray([replication_id], dtype=np.int64),
        "z": z_out,
        "X": x_out[:, None],
        "h_lag_truth": h_lag_out,
        "b_truth": b_out,
        "correction_truth": c_out,
        "h_truth": h_out,
        "mu_truth": np.asarray([mu_truth]),
        "phi_truth": np.asarray([params.phi]),
        "sigma_h_truth": np.asarray([params.sigma_h]),
        "sigma_eta_truth": np.asarray([params.sigma_eta]),
        "nu_truth": np.asarray([params.nu]),
        "amplitude": np.asarray([amplitude if nonlinear else 0.0]),
        "correction_share": np.asarray([share]),
        "estimation_rows": np.asarray([params.estimation_rows], dtype=np.int64),
        "forecast_rows": np.asarray([params.forecast_rows], dtype=np.int64),
        "burn_in": np.asarray([params.burn_in], dtype=np.int64),
        "seed_namespace": np.asarray([seed_namespace]),
        "state_seed": np.asarray(
            [derived_seed(replication_id, f"{seed_namespace}_state")], dtype=np.int64
        ),
        "observation_seed": np.asarray(
            [derived_seed(replication_id, f"{seed_namespace}_observation")],
            dtype=np.int64,
        ),
    }


def median_correction_share(
    params: DGPParameters,
    curve: ReferenceCurve,
    amplitude: float,
    replications: int,
    *,
    seed_namespace: str,
) -> tuple[float, np.ndarray]:
    shares = np.asarray(
        [
            float(
                simulate_replication(
                    params,
                    curve,
                    replication_id=i,
                    dgp_id="DGP-N",
                    amplitude=amplitude,
                    seed_namespace=seed_namespace,
                )["correction_share"][0]
            )
            for i in range(1, replications + 1)
        ]
    )
    return float(np.median(shares)), shares


def solve_amplitude(
    params: DGPParameters,
    curve: ReferenceCurve,
    *,
    replications: int = AMPLITUDE_CALIBRATION_REPLICATIONS,
    seed_namespace: str = "calibration",
) -> dict[str, Any]:
    """Solve the empirical-share anchor on a calibration-only seed bank."""
    target = params.target_correction_share
    evaluations: list[dict[str, float]] = []

    def objective(a: float) -> float:
        median, _ = median_correction_share(
            params,
            curve,
            a,
            replications,
            seed_namespace=seed_namespace,
        )
        evaluations.append({"amplitude": float(a), "median_share": median})
        return median - target

    lo, hi = 0.0, 1.0
    f_lo, f_hi = objective(lo), objective(hi)
    while f_hi <= 0.0:
        hi *= 2.0
        if hi > 65536.0:
            raise RuntimeError("failed to bracket amplitude before numerical safety ceiling")
        f_hi = objective(hi)
    root = float(brentq(objective, lo, hi, xtol=np.finfo(float).eps * max(1.0, hi), rtol=1e-14))
    achieved, shares = median_correction_share(
        params,
        curve,
        root,
        replications,
        seed_namespace=seed_namespace,
    )
    ordered = sorted({(e["amplitude"], e["median_share"]) for e in evaluations})
    monotone = all(b[1] >= a[1] - 1e-14 for a, b in zip(ordered, ordered[1:]))
    if not monotone:
        raise RuntimeError("common-random median-share map is not monotone on evaluated points")
    return {
        "amplitude": root,
        "target_median_share": target,
        "achieved_median_share": achieved,
        "replications": replications,
        "seed_namespace": seed_namespace,
        "experiment_seed_namespace": "experiment",
        "seed_banks_disjoint_by_construction": seed_namespace != "experiment",
        "bracket": [lo, hi],
        "solver": "scipy.optimize.brentq",
        "absolute_error": abs(achieved - target),
        "share_min": float(np.min(shares)),
        "share_max": float(np.max(shares)),
        "share_q25": float(np.quantile(shares, 0.25)),
        "share_q75": float(np.quantile(shares, 0.75)),
        "evaluations": [{"amplitude": a, "median_share": s} for a, s in ordered],
        "monotone_on_evaluated_points": monotone,
        "common_random_numbers": True,
    }


def exact_design_constants(
    replications: int = PRIMARY_REPLICATIONS,
) -> dict[str, Any]:
    if replications != PRIMARY_REPLICATIONS:
        raise ValueError(f"the reported design uses {PRIMARY_REPLICATIONS} replications per DGP")
    coverage_cutoff = max(k for k in range(replications + 1) if binom.cdf(k, replications, 0.95) <= 0.05)
    false_cutoff = min(k for k in range(replications + 1) if binom.sf(k - 1, replications, 0.05) <= 0.05)
    z_975 = float(norm.ppf(0.975))
    precision_minimum = int(math.ceil(z_975**2 * 0.90 * 0.10 / 0.05**2))
    lower_order = max(
        k
        for k in range(1, replications // 2 + 1)
        if binom.cdf(replications - k, replications, 0.5)
        - binom.cdf(k - 1, replications, 0.5)
        >= 0.95
    )
    upper_order = replications - lower_order + 1
    median_interval_coverage = float(
        binom.cdf(upper_order - 1, replications, 0.5)
        - binom.cdf(lower_order - 1, replications, 0.5)
    )
    power_rates = (0.925, 0.90, 0.875, 0.85)
    return {
        "replications_per_dgp": replications,
        "exact_zero_event_minimum_replications": int(
            math.floor(math.log(0.05) / math.log(0.95)) + 1
        ),
        "coverage_precision_planning_rate": 0.90,
        "coverage_precision_target_95_half_width": 0.05,
        "coverage_precision_normal_quantile": z_975,
        "coverage_precision_minimum_replications": precision_minimum,
        "replication_count": replications,
        "zero_events_probability_at_rate_0_05": float(0.95**replications),
        "zero_event_one_sided_95_upper": float(1 - 0.05 ** (1 / replications)),
        "mcse_at_rate_0_05": float(math.sqrt(0.05 * 0.95 / replications)),
        "mcse_at_rate_0_90": float(math.sqrt(0.90 * 0.10 / replications)),
        "normal_planning_95_half_width_at_rate_0_90": float(
            z_975 * math.sqrt(0.90 * 0.10 / replications)
        ),
        "normal_planning_95_half_width_at_rate_0_95": float(
            z_975 * math.sqrt(0.95 * 0.05 / replications)
        ),
        "coverage_rejection_cutoff": coverage_cutoff,
        "coverage_cutoff_attained_probability": float(binom.cdf(coverage_cutoff, replications, 0.95)),
        "coverage_rejection_power_by_true_rate": {
            f"{rate:.3f}": float(binom.cdf(coverage_cutoff, replications, rate))
            for rate in power_rates
        },
        "false_declaration_rejection_cutoff": false_cutoff,
        "false_cutoff_attained_probability": float(binom.sf(false_cutoff - 1, replications, 0.05)),
        "median_distribution_free_95_interval_order_statistics": [
            lower_order,
            upper_order,
        ],
        "median_distribution_free_interval_attained_coverage": median_interval_coverage,
        "primary_fits": 2 * replications,
        "total_fits": 2 * replications,
    }


def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def dataset_record(
    path: Path,
    payload: dict[str, np.ndarray | float | int | str],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    stored_path = path.relative_to(root).as_posix() if root is not None else str(path)
    return {
        "path": stored_path,
        "dgp_id": str(np.asarray(payload["dgp_id"])[0]),
        "replication_id": int(np.asarray(payload["replication_id"])[0]),
        "correction_share": float(np.asarray(payload["correction_share"])[0]),
        "mu_truth": float(np.asarray(payload["mu_truth"])[0]),
        "state_seed": int(np.asarray(payload["state_seed"])[0]),
        "observation_seed": int(np.asarray(payload["observation_seed"])[0]),
        "seed_namespace": str(np.asarray(payload["seed_namespace"])[0]),
    }


def build_fit_plan(
    dataset_records: Iterable[dict[str, Any]],
    *,
    replications: int = PRIMARY_REPLICATIONS,
) -> list[dict[str, Any]]:
    records = list(dataset_records)
    nonlinear = [r for r in records if r["dgp_id"] == "DGP-N"]
    zero = [r for r in records if r["dgp_id"] == "DGP-0"]
    if replications != PRIMARY_REPLICATIONS:
        raise ValueError(f"the reported design uses {PRIMARY_REPLICATIONS} replications per DGP")
    if len(nonlinear) != replications or len(zero) != replications:
        raise ValueError(f"fit plan requires exactly {replications} immutable datasets per DGP")
    plan: list[dict[str, Any]] = []

    def add(record: dict[str, Any]) -> None:
        task_id = len(plan) + 1
        plan.append(
            {
                "task_id": task_id,
                "dgp_id": record["dgp_id"],
                "replication_id": int(record["replication_id"]),
                "dataset_path": record["path"],
            }
        )

    for record in sorted(records, key=lambda r: (r["dgp_id"], r["replication_id"])):
        add(record)
    if len(plan) != 2 * replications:
        raise AssertionError("fit-plan arithmetic drift")
    return plan


def dgp_parameters_dict(params: DGPParameters) -> dict[str, Any]:
    return asdict(params)
