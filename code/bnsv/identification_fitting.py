"""CmdStan fitting for the identification simulation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

from .diagnostics import DiagnosticThresholds, assess_cmdstan_fit


@dataclass(frozen=True)
class SamplingSpecification:
    sequence_position: int
    adapt_delta: float
    max_treedepth: int
    warmup: int
    sampling: int


def sampling_sequence(config: dict[str, Any]) -> list[SamplingSpecification]:
    specifications = config.get("sampling_sequence")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("MCMC configuration requires a sampling sequence")
    return [
        SamplingSpecification(
            sequence_position=i,
            adapt_delta=float(item["adapt_delta"]),
            max_treedepth=int(item["max_treedepth"]),
            warmup=int(item["warmup"]),
            sampling=int(item["sampling"]),
        )
        for i, item in enumerate(specifications, start=1)
    ]


def classify_fit_assessment(
    assessment: dict[str, Any],
    *,
    adapt_delta: float,
    rules: dict[str, Any],
) -> str:
    if bool(assessment.get("passed", False)):
        return "successful"

    total_rate_limit = float(rules["sparse_divergence_max_total_fraction"])
    per_chain_rate_limit = float(rules["sparse_divergence_max_per_chain_fraction"])
    adapt_delta_ceiling = float(rules["adapt_delta_ceiling"])
    divergences = int(assessment.get("divergences", 0))
    by_chain = [
        int(value)
        for value in assessment.get("divergences_by_chain", [divergences])
    ]
    draws_by_chain = [
        int(value)
        for value in assessment.get(
            "post_warmup_draws_by_chain",
            [int(rules["reference_post_warmup_draws_per_chain"])] * len(by_chain),
        )
    ]
    if len(draws_by_chain) != len(by_chain) or any(value <= 0 for value in draws_by_chain):
        raise ValueError("post-warmup draw counts must align with chains")
    total_limit = max(1, math.ceil(total_rate_limit * sum(draws_by_chain)))
    chain_limits = [
        max(1, math.ceil(per_chain_rate_limit * value)) for value in draws_by_chain
    ]
    other_diagnostics_pass = all(
        bool(assessment.get(key, True))
        for key in (
            "rhat_pass",
            "bulk_ess_pass",
            "tail_ess_pass",
            "mcse_over_sd_pass",
            "treedepth_pass",
            "bfmi_pass",
        )
    )
    if divergences:
        if (
            not other_diagnostics_pass
            or divergences > total_limit
            or any(x > limit for x, limit in zip(by_chain, chain_limits, strict=True))
            or adapt_delta >= adapt_delta_ceiling - 1e-12
        ):
            return "unsuccessful"
        return "increase_adapt_delta"

    ess_failed = not bool(assessment.get("bulk_ess_pass", True)) or not bool(
        assessment.get("tail_ess_pass", True)
    )
    non_ess_diagnostics_pass = all(
        bool(assessment.get(key, True))
        for key in (
            "rhat_pass",
            "mcse_over_sd_pass",
            "divergence_pass",
            "treedepth_pass",
            "bfmi_pass",
        )
    )
    if ess_failed and non_ess_diagnostics_pass:
        return "increase_sampling"
    return "unsuccessful"


def next_sampling_specification(
    specifications: list[SamplingSpecification],
    *,
    current: SamplingSpecification,
    assessment_outcome: str,
    used_positions: set[int],
) -> SamplingSpecification | None:
    available = [
        item
        for item in specifications
        if item.sequence_position not in used_positions
    ]
    if assessment_outcome == "increase_sampling":
        candidates = [
            item
            for item in available
            if abs(item.adapt_delta - current.adapt_delta) < 1e-12
            and item.sampling > current.sampling
        ]
        return min(candidates, key=lambda item: item.sampling, default=None)
    if assessment_outcome == "increase_adapt_delta":
        candidates = [
            item
            for item in available
            if item.adapt_delta > current.adapt_delta
            and item.sampling <= current.sampling
        ]
        return min(
            candidates,
            key=lambda item: (item.adapt_delta, item.sampling),
            default=None,
        )
    return None


def _cmdstan_available() -> bool:
    if importlib.util.find_spec("cmdstanpy") is None:
        return False
    try:
        from cmdstanpy import cmdstan_path

        return Path(cmdstan_path()).is_dir()
    except Exception:
        return False


def fit_with_sampling_sequence(
    *,
    stan_file: str | Path,
    data: dict[str, Any],
    seed: int,
    chains: int,
    mcmc: dict[str, Any],
    output_dir: str | Path,
    inits: list[dict[str, Any]],
    show_progress: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    if not _cmdstan_available():
        raise RuntimeError("CmdStan is unavailable")
    if len(inits) != chains:
        raise ValueError("one initialization dictionary is required per chain")

    from cmdstanpy import CmdStanModel

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stan_source = Path(stan_file).resolve()
    model = CmdStanModel(stan_file=str(stan_source))
    thresholds_config = mcmc["thresholds"]
    thresholds = DiagnosticThresholds(
        max_divergences=int(thresholds_config["divergences_max"]),
        max_rhat=float(thresholds_config["rhat_max"]),
        min_bulk_ess=float(thresholds_config["bulk_ess_min"]),
        min_tail_ess=float(thresholds_config["tail_ess_min"]),
        min_bfmi=float(thresholds_config["bfmi_min"]),
        max_treedepth_fraction=float(
            thresholds_config["treedepth_saturation_max_fraction"]
        ),
        max_mcse_over_sd=float(thresholds_config["mcse_over_sd_max"]),
    )
    init_dir = output / "initializations"
    init_dir.mkdir(exist_ok=True)
    for chain_id, payload in enumerate(inits, start=1):
        path = init_dir / f"chain_{chain_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    specifications = sampling_sequence(mcmc)
    current = specifications[0]
    used: set[int] = set()
    records: list[dict[str, Any]] = []
    runs_path = output / "sampling_runs.json"
    while current.sequence_position not in used:
        used.add(current.sequence_position)
        started = time.monotonic()
        fit = model.sample(
            data=data,
            seed=seed,
            chains=chains,
            parallel_chains=chains,
            iter_warmup=current.warmup,
            iter_sampling=current.sampling,
            adapt_delta=current.adapt_delta,
            max_treedepth=current.max_treedepth,
            output_dir=str(output),
            show_progress=show_progress,
            inits=inits,
        )
        assessment = assess_cmdstan_fit(
            fit, max_treedepth=current.max_treedepth, thresholds=thresholds
        )
        assessment_outcome = classify_fit_assessment(
            assessment,
            adapt_delta=current.adapt_delta,
            rules=mcmc["sampling_adjustments"],
        )
        records.append(
            {
                **asdict(current),
                "seed": int(seed),
                "chains": int(chains),
                "runtime_seconds": time.monotonic() - started,
                "assessment": assessment,
                "assessment_outcome": assessment_outcome,
            }
        )
        runs_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        if assessment_outcome == "successful":
            return fit, records
        next_specification = next_sampling_specification(
            specifications,
            current=current,
            assessment_outcome=assessment_outcome,
            used_positions=used,
        )
        if next_specification is None:
            raise RuntimeError("fit diagnostics were unsuccessful")
        current = next_specification
    raise RuntimeError("the MCMC sampling sequence was unsuccessful")
