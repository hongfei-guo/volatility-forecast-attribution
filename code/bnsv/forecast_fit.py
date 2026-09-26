"""Stan inputs and diagnostic sampling for the forecast models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping
from time import perf_counter

import numpy as np

from .diagnostics import DiagnosticThresholds, assess_cmdstan_fit
from .forecast_features import HAR_INPUT_MODELS, base_model_id, model_spec


NETWORK_WIDTH = 5
CHAINS = 4
PARTICLE_COUNT = 4000
FORECAST_PATH_COUNT = 4096

MU_PRIOR_MEAN = 0.0
MU_PRIOR_SD = 1.0
PHI_LOGIT_LOC = math.log(0.98 / 0.02)
PHI_LOGIT_SCALE = 1.0
SIGMA_H_PRIOR_SD = 0.9901178450473134
INITIAL_STATE_PRIOR_SD = 1.5
NU_MINUS_TWO_LOG_MEAN = math.log(6.0)
NU_MINUS_TWO_LOG_SD = 0.7

NN_OUTPUT_BASE_SCALE = {
    ("SP500", "NN-SV"): 0.2964544809008027,
    ("SP500", "RV-NN-SV"): 0.21887681468631867,
    ("SP500", "RV-RA-NN-SV"): 0.21887681468631867,
    ("DAX", "NN-SV"): 0.28826346770627714,
    ("DAX", "RV-NN-SV"): 0.21426550113931087,
    ("DAX", "RV-RA-NN-SV"): 0.21426550113931087,
    ("FTSE100", "NN-SV"): 0.28976347899431226,
    ("FTSE100", "RV-NN-SV"): 0.21620342057527236,
    ("FTSE100", "RV-RA-NN-SV"): 0.21620342057527236,
}

LINEAR_BASE_SCALE = {
    "SP500": 0.11910056202471753,
    "DAX": 0.11836624580668359,
    "FTSE100": 0.1191986763213542,
}


@dataclass(frozen=True)
class SamplingRun:
    adapt_delta: float
    max_treedepth: int
    warmup: int
    sampling: int


STANDARD_INITIAL_RUN = SamplingRun(0.95, 12, 1000, 2000)
STANDARD_LONG_RUN = SamplingRun(0.95, 12, 1000, 4000)
HIGH_ACCEPTANCE_RUN = SamplingRun(0.97, 12, 1000, 2000)
HIGH_ACCEPTANCE_LONG_RUN = SamplingRun(0.97, 12, 1000, 4000)
RV_LINEAR_RUN = SamplingRun(0.97, 12, 1000, 2000)
ORIGINAL_SAMPLING_POLICY = "original"
MATCHED_RETRY_SAMPLING_POLICY = "matched_097_divergence_099_ess"
MATCHED_SAMPLING_POLICIES = {MATCHED_RETRY_SAMPLING_POLICY}
DIVERGENCE_RETRY_RUN = SamplingRun(0.99, 12, 1000, 2000)
DIVERGENCE_RETRY_LONG_RUN = SamplingRun(0.99, 12, 1000, 4000)

DIAGNOSTIC_THRESHOLDS = DiagnosticThresholds(
    max_divergences=0,
    max_rhat=1.01,
    min_bulk_ess=400.0,
    min_tail_ess=400.0,
    min_bfmi=0.30,
    max_treedepth_fraction=0.001,
    max_mcse_over_sd=0.05,
)


class SamplingFailure(RuntimeError):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__("the posterior sample did not satisfy the stated diagnostics")
        self.records = records


def residual_asymmetry_prior_scale(
    standardized_asymmetry: np.ndarray,
    *,
    variance_multiplier: float = 1.25,
    prior_mass: float = 0.95,
) -> float:
    """Set the residual-asymmetry prior from its training-sample span."""
    values = np.asarray(standardized_asymmetry, dtype=float)
    if values.ndim != 1 or values.size < 30 or np.any(~np.isfinite(values)):
        raise ValueError(
            "standardized asymmetry must contain at least 30 finite values"
        )
    if variance_multiplier <= 1 or not 0 < prior_mass < 1:
        raise ValueError("the prior multiplier and probability are invalid")
    lower, upper = np.quantile(values, [0.10, 0.90])
    span = float(upper - lower)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("standardized asymmetry has no interquantile variation")
    normal_quantile = NormalDist().inv_cdf(0.5 + 0.5 * prior_mass)
    return float(np.log(variance_multiplier) / (normal_quantile * span))


def build_stan_data(
    *, model_id: str, market: str, z: np.ndarray, x: np.ndarray,
    correction_scale: float | None = None,
) -> dict[str, Any]:
    """Construct the data block for the current exponential-link Stan models."""
    spec = model_spec(model_id)
    family = base_model_id(model_id)
    if model_id in HAR_INPUT_MODELS:
        if correction_scale is None or not np.isfinite(correction_scale) or correction_scale <= 0:
            raise ValueError("HAR inputs require a positive pre-2017 calibrated correction scale")
    elif correction_scale is not None:
        raise ValueError("the original models retain their fixed correction scales")
    if market not in {"SP500", "FTSE100", "DAX"}:
        raise ValueError(f"unsupported market: {market}")
    response = np.asarray(z, dtype=float)
    features = np.asarray(x, dtype=float)
    expected_dimension = len(spec.feature_names)
    if response.ndim != 1 or response.size < 2 or np.any(~np.isfinite(response)):
        raise ValueError("z must contain at least two finite observations")
    if features.shape != (response.size, expected_dimension):
        raise ValueError("feature matrix does not match the model and response")
    if np.any(~np.isfinite(features)):
        raise ValueError("feature matrix must be finite")

    data: dict[str, Any] = {
        "N": int(response.size),
        "z": response,
        "mu_prior_mean": MU_PRIOR_MEAN,
        "mu_prior_sd": MU_PRIOR_SD,
        "phi_logit_loc": PHI_LOGIT_LOC,
        "phi_logit_scale": PHI_LOGIT_SCALE,
        "sigma_h_prior_sd": SIGMA_H_PRIOR_SD,
        "h0_prior_sd": INITIAL_STATE_PRIOR_SD,
        "nu_minus_two_log_mean": NU_MINUS_TWO_LOG_MEAN,
        "nu_minus_two_log_sd": NU_MINUS_TWO_LOG_SD,
    }
    if family in {"NN-SV", "RV-NN-SV", "RV-RA-NN-SV"}:
        core = features[:, :-1] if model_id == "RV-RA-NN-SV" else features
        data.update(
            {
                "D": int(core.shape[1]),
                "H": NETWORK_WIDTH,
                "X": core,
                "w1_base_sd": 1.0,
                "b1_prior_sd": 0.2,
                "output_log_mean": math.log(
                    correction_scale if correction_scale is not None
                    else NN_OUTPUT_BASE_SCALE[(market, model_id)]
                ),
                "output_log_sd": 0.2,
            }
        )
        if model_id == "RV-RA-NN-SV":
            data.update(
                {
                    "q": features[:, -1],
                    "gamma_A_prior_sd": residual_asymmetry_prior_scale(
                        features[:, -1]
                    ),
                }
            )
    elif family == "RV-LIN-SV":
        data.update(
            {
                "D": expected_dimension,
                "X": features,
                "beta_base_sd": (
                    correction_scale if correction_scale is not None
                    else LINEAR_BASE_SCALE[market]
                ),
            }
        )
    elif model_id != "SV":
        raise ValueError(f"unsupported forecast model: {model_id}")
    return data


_CHAIN_JITTER = (-0.15, -0.05, 0.05, 0.15)
_STATE_SCALE_INIT = (0.55, 0.65, 0.75, 0.85)


def _stationary_shrink(phi_std: float) -> float:
    phi_raw = PHI_LOGIT_LOC + PHI_LOGIT_SCALE * phi_std
    log_phi = -float(np.logaddexp(0.0, -phi_raw))
    return math.sqrt(-math.expm1(2.0 * log_phi))


def _nn_initial_values(n: int, d: int) -> list[dict[str, Any]]:
    unit_offsets = np.linspace(-0.20, 0.20, NETWORK_WIDTH)
    ordered_quantiles = np.array([-1.18, -0.50, 0.0, 0.50, 1.18])
    values: list[dict[str, Any]] = []
    for chain_id, jitter in enumerate(_CHAIN_JITTER):
        w1 = np.empty((d, NETWORK_WIDTH), dtype=float)
        for feature_index in range(d):
            w1[feature_index, :] = (
                0.05 * (feature_index + 1) * unit_offsets + 0.01 * jitter
            )
        phi_std = 0.5 * jitter
        values.append(
            {
                "mu_std": jitter,
                "phi_std": phi_std,
                "sigma_eta_std": _STATE_SCALE_INIT[chain_id]
                * _stationary_shrink(phi_std),
                "nu_std": 0.25 * jitter,
                "w1_raw": w1.tolist(),
                "b1_std": (unit_offsets + 0.1 * jitter).tolist(),
                "log_output_weight_std": (
                    ordered_quantiles + 0.05 * jitter
                ).tolist(),
                "h0_raw": 0.10 * jitter,
                "alpha": [0.0] * (n - 1),
            }
        )
    return values


def _linear_initial_values(n: int, d: int) -> list[dict[str, Any]]:
    axis = np.linspace(-1.0, 1.0, d)
    values: list[dict[str, Any]] = []
    for chain_id, jitter in enumerate(_CHAIN_JITTER):
        phi_std = 0.5 * jitter
        values.append(
            {
                "mu_std": jitter,
                "phi_std": phi_std,
                "sigma_eta_std": _STATE_SCALE_INIT[chain_id]
                * _stationary_shrink(phi_std),
                "nu_std": 0.25 * jitter,
                "h0_raw": 0.10 * jitter,
                "beta_raw": (0.04 * axis + 0.05 * jitter).tolist(),
                "alpha": [0.0] * (n - 1),
            }
        )
    return values


def initial_values(model_id: str, stan_data: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Return the four deterministic initial values used for neural and linear fits."""
    model_id = base_model_id(model_id)
    n = int(stan_data["N"])
    if model_id in {"NN-SV", "RV-NN-SV", "RV-RA-NN-SV"}:
        values = _nn_initial_values(n, int(stan_data["D"]))
        if model_id == "RV-RA-NN-SV":
            for value in values:
                value["gamma_A_std"] = 0.0
        return values
    if model_id == "RV-LIN-SV":
        return _linear_initial_values(n, int(stan_data["D"]))
    return None


def sampling_policy_for(model_id: str, requested: str | None = None) -> str:
    """Select the primary or multiscale sampling scheme."""
    family = base_model_id(model_id)
    policy = requested if requested is not None else (
        MATCHED_RETRY_SAMPLING_POLICY if model_id in HAR_INPUT_MODELS
        else ORIGINAL_SAMPLING_POLICY
    )
    if policy not in {ORIGINAL_SAMPLING_POLICY, *MATCHED_SAMPLING_POLICIES}:
        raise ValueError(f"unsupported sampling policy: {policy}")
    if policy in MATCHED_SAMPLING_POLICIES and family not in {
        "NN-SV", "RV-NN-SV", "RV-LIN-SV"
    }:
        raise ValueError("the matched sampling policy applies to NN/LIN comparisons")
    return policy


def first_sampling_run(
    model_id: str, *, sampling_policy: str | None = None
) -> SamplingRun:
    if sampling_policy_for(model_id, sampling_policy) in MATCHED_SAMPLING_POLICIES:
        return HIGH_ACCEPTANCE_RUN
    model_id = base_model_id(model_id)
    return RV_LINEAR_RUN if model_id == "RV-LIN-SV" else STANDARD_INITIAL_RUN


def _only_ess_is_insufficient(
    assessment: Mapping[str, Any], *, require_complete: bool = False
) -> bool:
    if require_complete:
        required = {
            "passed", "bulk_ess_pass", "tail_ess_pass", "rhat_pass",
            "mcse_over_sd_pass", "divergence_pass", "treedepth_pass", "bfmi_pass",
        }
        missing = required - assessment.keys()
        if missing:
            raise ValueError(f"incomplete diagnostic assessment: {sorted(missing)}")
    ess_failed = not bool(assessment.get("bulk_ess_pass", True)) or not bool(
        assessment.get("tail_ess_pass", True)
    )
    other_diagnostics_pass = all(
        bool(assessment.get(name, True))
        for name in (
            "rhat_pass",
            "mcse_over_sd_pass",
            "divergence_pass",
            "treedepth_pass",
            "bfmi_pass",
        )
    )
    return ess_failed and other_diagnostics_pass


def _sparse_divergences(assessment: Mapping[str, Any], run: SamplingRun) -> bool:
    divergences = int(assessment.get("divergences", 0))
    if divergences <= 0 or run.adapt_delta >= 0.97:
        return False
    by_chain = [int(value) for value in assessment.get("divergences_by_chain", [])]
    draws_by_chain = [
        int(value) for value in assessment.get("post_warmup_draws_by_chain", [])
    ]
    if not by_chain or len(by_chain) != len(draws_by_chain):
        return False
    other_diagnostics_pass = all(
        bool(assessment.get(name, True))
        for name in (
            "rhat_pass",
            "bulk_ess_pass",
            "tail_ess_pass",
            "mcse_over_sd_pass",
            "treedepth_pass",
            "bfmi_pass",
        )
    )
    total_limit = max(1, math.ceil(0.001 * sum(draws_by_chain)))
    chain_limits = [max(1, math.ceil(0.002 * draws)) for draws in draws_by_chain]
    return bool(
        other_diagnostics_pass
        and divergences <= total_limit
        and all(value <= limit for value, limit in zip(by_chain, chain_limits, strict=True))
    )


def next_sampling_run(
    *, model_id: str, current: SamplingRun, assessment: Mapping[str, Any],
    sampling_policy: str | None = None,
) -> SamplingRun | None:
    """Select the stated follow-up run from the diagnostic outcome."""
    policy = sampling_policy_for(model_id, sampling_policy)
    if policy == MATCHED_RETRY_SAMPLING_POLICY:
        if current not in {HIGH_ACCEPTANCE_RUN, HIGH_ACCEPTANCE_LONG_RUN,
                           DIVERGENCE_RETRY_RUN, DIVERGENCE_RETRY_LONG_RUN}:
            raise ValueError("the matched retry policy requires fixed 0.97/0.99 sampling runs")
        ess_only = _only_ess_is_insufficient(assessment, require_complete=True)
        if assessment["passed"]:
            return None
        if current.sampling == 2000 and ess_only:
            return (HIGH_ACCEPTANCE_LONG_RUN if current.adapt_delta == 0.97
                    else DIVERGENCE_RETRY_LONG_RUN)
        divergence_only = not assessment["divergence_pass"] and all(
            assessment[name] for name in (
                "rhat_pass", "bulk_ess_pass", "tail_ess_pass", "mcse_over_sd_pass",
                "treedepth_pass", "bfmi_pass",
            )
        )
        if current.adapt_delta == 0.97 and divergence_only:
            return (DIVERGENCE_RETRY_RUN if current.sampling == 2000
                    else DIVERGENCE_RETRY_LONG_RUN)
        return None
    model_id = base_model_id(model_id)
    if bool(assessment.get("passed", False)) or model_id == "RV-LIN-SV":
        return None
    if _sparse_divergences(assessment, current):
        return HIGH_ACCEPTANCE_RUN
    if _only_ess_is_insufficient(assessment) and current.sampling == 2000:
        return STANDARD_LONG_RUN if current.adapt_delta == 0.95 else HIGH_ACCEPTANCE_LONG_RUN
    return None


def fit_model(
    *,
    stan_file: str | Path,
    model_id: str,
    data: Mapping[str, Any],
    seed: int,
    output_dir: str | Path,
    show_progress: bool = False,
    sampling_policy: str | None = None,
    parallel_chains: int = CHAINS,
    phase_callback: Callable[[str], None] | None = None,
    previous_attempts: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Estimate one expanding-window refit and return an accepted posterior sample."""
    import cmdstanpy

    model_spec(model_id)
    policy = sampling_policy_for(model_id, sampling_policy)
    if parallel_chains not in range(1, CHAINS + 1):
        raise ValueError("parallel_chains must be between one and the fixed four chains")
    records = deepcopy(previous_attempts) if previous_attempts is not None else []
    run = first_sampling_run(model_id, sampling_policy=policy)
    used: set[SamplingRun] = set()
    if previous_attempts is not None:
        if policy != MATCHED_RETRY_SAMPLING_POLICY or not records:
            raise ValueError("historical continuation requires nonempty matched retry evidence")
        for record in records:
            prior = SamplingRun(**record["sampling"])
            if prior != run or prior in used or record.get("seed") != int(seed):
                raise ValueError("historical sampling sequence or seed differs")
            if record.get("sampling_policy") not in MATCHED_SAMPLING_POLICIES:
                raise ValueError("historical attempt has an incompatible policy")
            if record["diagnostics"]["passed"]:
                raise ValueError("accepted historical fits must not be resampled")
            used.add(prior)
            run = next_sampling_run(model_id=model_id, current=prior,
                                    assessment=record["diagnostics"], sampling_policy=policy)
            if run is None:
                raise ValueError("historical failure has no remaining allowed follow-up")
    if phase_callback is not None:
        phase_callback("compile")
    model = cmdstanpy.CmdStanModel(stan_file=str(Path(stan_file)))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    values = initial_values(model_id, data)
    while run not in used:
        used.add(run)
        sample_directory = destination
        if policy in MATCHED_SAMPLING_POLICIES:
            # Separate attempts retain the original unsuccessful CmdStan output.
            sample_directory = destination / f"attempt_{len(records) + 1:02d}"
            sample_directory.mkdir(exist_ok=False)
        if phase_callback is not None:
            phase_callback(f"sampling_attempt_{len(records) + 1:02d}")
        started = perf_counter()
        fit = model.sample(
            data=dict(data),
            seed=int(seed),
            chains=CHAINS,
            parallel_chains=parallel_chains,
            iter_warmup=run.warmup,
            iter_sampling=run.sampling,
            adapt_delta=run.adapt_delta,
            max_treedepth=run.max_treedepth,
            output_dir=str(sample_directory),
            show_progress=show_progress,
            inits=values,
        )
        sampling_seconds = perf_counter() - started
        if phase_callback is not None:
            phase_callback(f"diagnostics_attempt_{len(records) + 1:02d}")
        started = perf_counter()
        assessment = assess_cmdstan_fit(
            fit,
            max_treedepth=run.max_treedepth,
            thresholds=DIAGNOSTIC_THRESHOLDS,
        )
        records.append({"sampling": asdict(run), "diagnostics": assessment})
        records[-1]["sampling_seconds"] = sampling_seconds
        records[-1]["diagnostics_seconds"] = perf_counter() - started
        if policy in MATCHED_SAMPLING_POLICIES:
            records[-1]["sampling_policy"] = policy
            records[-1]["seed"] = int(seed)
        if bool(assessment["passed"]):
            return fit, records
        following = next_sampling_run(
            model_id=model_id, current=run, assessment=assessment,
            sampling_policy=policy,
        )
        if following is None:
            break
        if policy in MATCHED_SAMPLING_POLICIES:
            del fit
        run = following
    raise SamplingFailure(records)
