from __future__ import annotations

from pathlib import Path

import cmdstanpy
import numpy as np
import pytest

import bnsv.forecast_fit as fitting
from bnsv.forecast_features import HAR_INPUT_MODELS


def assessment(**updates: object) -> dict[str, object]:
    result = dict.fromkeys((
        "rhat_pass", "bulk_ess_pass", "tail_ess_pass", "mcse_over_sd_pass",
        "divergence_pass", "treedepth_pass", "bfmi_pass",
    ), True)
    result.update(passed=False, divergences=0)
    result.update(updates)
    return result


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("ess_flag", ["bulk_ess_pass", "tail_ess_pass"])
def test_both_members_receive_one_ess_only_followup(model_id: str, ess_flag: str) -> None:
    first = fitting.first_sampling_run(model_id)
    assert first == fitting.SamplingRun(0.97, 12, 1000, 2000)
    failed = assessment(**{ess_flag: False})
    following = fitting.next_sampling_run(model_id=model_id, current=first, assessment=failed)
    assert following == fitting.SamplingRun(0.97, 12, 1000, 4000)
    assert fitting.next_sampling_run(model_id=model_id, current=following, assessment=failed) is None


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("flag", [
    "divergence_pass", "rhat_pass", "bfmi_pass", "treedepth_pass", "mcse_over_sd_pass",
])
def test_other_failures_never_receive_an_extra_chance(model_id: str, flag: str) -> None:
    failed = assessment(bulk_ess_pass=False, **{flag: False})
    assert fitting.next_sampling_run(
        model_id=model_id, current=fitting.first_sampling_run(model_id), assessment=failed
    ) is None


def test_primary_and_multiscale_sampling_defaults() -> None:
    ess_only = assessment(bulk_ess_pass=False)
    assert fitting.first_sampling_run("RV-NN-SV") == fitting.STANDARD_INITIAL_RUN
    assert fitting.first_sampling_run("RV-LIN-SV") == fitting.RV_LINEAR_RUN
    assert fitting.next_sampling_run(
        model_id="RV-NN-SV", current=fitting.STANDARD_INITIAL_RUN, assessment=ess_only
    ) == fitting.STANDARD_LONG_RUN
    assert fitting.next_sampling_run(
        model_id="RV-LIN-SV", current=fitting.RV_LINEAR_RUN, assessment=ess_only
    ) is None
    for name in ("NN-SV", "RV-NN-SV", "RV-LIN-SV"):
        first = fitting.first_sampling_run(name, sampling_policy=fitting.MATCHED_RETRY_SAMPLING_POLICY)
        assert first == fitting.HIGH_ACCEPTANCE_RUN
        assert fitting.next_sampling_run(
            model_id=name, current=first, assessment=ess_only,
            sampling_policy=fitting.MATCHED_RETRY_SAMPLING_POLICY,
        ) == fitting.HIGH_ACCEPTANCE_LONG_RUN


def test_incomplete_diagnostics_or_wrong_policy_fail_clearly() -> None:
    with pytest.raises(ValueError, match="incomplete diagnostic"):
        fitting.next_sampling_run(
            model_id=HAR_INPUT_MODELS[0], current=fitting.HIGH_ACCEPTANCE_RUN,
            assessment={"passed": False, "bulk_ess_pass": False},
        )
    with pytest.raises(ValueError, match="fixed 0.97"):
        fitting.next_sampling_run(
            model_id=HAR_INPUT_MODELS[0], current=fitting.STANDARD_INITIAL_RUN,
            assessment=assessment(bulk_ess_pass=False),
        )
    for invalid in ("", "until_converged"):
        with pytest.raises(ValueError, match="unsupported sampling policy"):
            fitting.first_sampling_run(HAR_INPUT_MODELS[0], sampling_policy=invalid)


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("followup_succeeds", [False, True])
def test_sampling_loop_preserves_seed_initialization_and_both_attempts(
    model_id: str, followup_succeeds: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    class Model:
        def sample(self, **kwargs):
            calls.append(kwargs)
            (Path(kwargs["output_dir"]) / "retained-output.txt").write_text(str(len(calls)))
            return object()
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", lambda **kwargs: Model())
    diagnostics = iter([
        assessment(bulk_ess_pass=False),
        assessment(passed=True) if followup_succeeds else assessment(tail_ess_pass=False),
    ])
    monkeypatch.setattr(fitting, "assess_cmdstan_fit", lambda *args, **kwargs: next(diagnostics))
    data = fitting.build_stan_data(
        model_id=model_id, market="SP500", z=np.zeros(30), x=np.zeros((30, 4)),
        correction_scale=0.17,
    )
    historical = tmp_path / "old-failure.json"
    historical.write_text('{"status":"unsuccessful"}\n')
    phases = []
    kwargs = dict(stan_file="unused.stan", model_id=model_id, data=data,
                  seed=371, output_dir=tmp_path / "new-run", parallel_chains=2,
                  phase_callback=phases.append)
    if followup_succeeds:
        _, records = fitting.fit_model(**kwargs)
    else:
        with pytest.raises(fitting.SamplingFailure) as error:
            fitting.fit_model(**kwargs)
        records = error.value.records
    assert len(calls) == len(records) == 2
    assert [c["iter_sampling"] for c in calls] == [2000, 4000]
    for key in ("seed", "inits", "chains", "parallel_chains", "iter_warmup", "adapt_delta", "max_treedepth"):
        assert calls[0][key] == calls[1][key]
    assert calls[0]["adapt_delta"] == 0.97 and calls[0]["seed"] == 371
    assert all(c["chains"] == 4 and c["parallel_chains"] == 2 for c in calls)
    assert phases == ["compile", "sampling_attempt_01", "diagnostics_attempt_01",
                      "sampling_attempt_02", "diagnostics_attempt_02"]
    assert not records[0]["diagnostics"]["passed"]
    assert records[1]["diagnostics"]["passed"] == followup_succeeds
    assert all(r["sampling_policy"] == fitting.MATCHED_RETRY_SAMPLING_POLICY for r in records)
    assert (tmp_path / "new-run/attempt_01/retained-output.txt").read_text() == "1"
    assert (tmp_path / "new-run/attempt_02/retained-output.txt").read_text() == "2"
    assert historical.read_text() == '{"status":"unsuccessful"}\n'




@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("order", ["divergence_then_ess", "ess_then_divergence"])
def test_multiscale_policy_uses_each_allowance_once_and_keeps_failure_evidence(
    model_id, order, tmp_path, monkeypatch,
):
    calls = []
    class Model:
        def sample(self, **kwargs):
            calls.append(kwargs)
            (Path(kwargs["output_dir"]) / "evidence.txt").write_text(str(len(calls)))
            return object()
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", lambda **kwargs: Model())
    div = assessment(divergence_pass=False, divergences=2)
    ess = assessment(bulk_ess_pass=False)
    sequence = [div, ess] if order == "divergence_then_ess" else [ess, div]
    outcomes = iter([*sequence, assessment(passed=True)])
    monkeypatch.setattr(fitting, "assess_cmdstan_fit", lambda *a, **kw: next(outcomes))
    data = fitting.build_stan_data(model_id=model_id, market="FTSE100", z=np.zeros(30),
                                  x=np.zeros((30, 4)), correction_scale=0.17)
    _, records = fitting.fit_model(stan_file="unused.stan", model_id=model_id, data=data,
                                  seed=371, output_dir=tmp_path / "run")
    expected = ([(0.97, 2000), (0.99, 2000), (0.99, 4000)] if order == "divergence_then_ess"
                else [(0.97, 2000), (0.97, 4000), (0.99, 4000)])
    assert [(c["adapt_delta"], c["iter_sampling"]) for c in calls] == expected
    for key in ("seed", "inits", "chains", "parallel_chains", "iter_warmup", "max_treedepth"):
        assert all(c[key] == calls[0][key] for c in calls)
    assert all(c["chains"] == c["parallel_chains"] == 4 for c in calls)
    assert [r["diagnostics"]["passed"] for r in records] == [False, False, True]
    assert [p.read_text() for p in sorted((tmp_path / "run").glob("*/evidence.txt"))] == ["1", "2", "3"]
    for failure in (div, ess):
        assert fitting.next_sampling_run(model_id=model_id, current=fitting.DIVERGENCE_RETRY_LONG_RUN,
                                         assessment=failure) is None


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("other", ["rhat_pass", "bulk_ess_pass", "tail_ess_pass",
                                   "mcse_over_sd_pass", "treedepth_pass", "bfmi_pass"])
def test_divergence_with_another_failure_is_terminal(model_id, other):
    failed = assessment(divergence_pass=False, divergences=1, **{other: False})
    assert fitting.next_sampling_run(model_id=model_id, current=fitting.HIGH_ACCEPTANCE_RUN,
                                     assessment=failed) is None


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
def test_persistent_divergence_at_099_has_no_further_retry(model_id):
    failed = assessment(divergence_pass=False, divergences=1)
    assert fitting.next_sampling_run(model_id=model_id, current=fitting.HIGH_ACCEPTANCE_RUN,
                                     assessment=failed) == fitting.DIVERGENCE_RETRY_RUN
    assert fitting.next_sampling_run(model_id=model_id, current=fitting.DIVERGENCE_RETRY_RUN,
                                     assessment=failed) is None


@pytest.mark.parametrize("model_id", HAR_INPUT_MODELS)
@pytest.mark.parametrize("needs_ess_extension", [False, True])
def test_historical_divergence_starts_directly_at_099(model_id, needs_ess_extension, tmp_path, monkeypatch):
    from copy import deepcopy
    historical = [{"sampling": dict(adapt_delta=0.97, max_treedepth=12, warmup=1000, sampling=2000),
                   "diagnostics": assessment(divergence_pass=False, divergences=1),
                   "sampling_policy": fitting.MATCHED_RETRY_SAMPLING_POLICY, "seed": 371}]
    untouched = deepcopy(historical)
    calls = []
    class Model:
        def sample(self, **kwargs):
            calls.append(kwargs)
            return object()
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", lambda **kwargs: Model())
    outcomes = iter(([assessment(bulk_ess_pass=False)] if needs_ess_extension else []) + [assessment(passed=True)])
    monkeypatch.setattr(fitting, "assess_cmdstan_fit", lambda *a, **kw: next(outcomes))
    data = fitting.build_stan_data(model_id=model_id, market="SP500", z=np.zeros(30),
                                  x=np.zeros((30, 4)), correction_scale=0.17)
    _, records = fitting.fit_model(stan_file="unused.stan", model_id=model_id, data=data,
                                  seed=371, output_dir=tmp_path / "followup", previous_attempts=historical)
    assert all(c["adapt_delta"] == 0.99 and c["seed"] == 371 for c in calls)
    assert [c["iter_sampling"] for c in calls] == ([2000, 4000] if needs_ess_extension else [2000])
    assert Path(calls[0]["output_dir"]).name == "attempt_02"
    assert records[0] == historical[0] == untouched[0]
    assert not (tmp_path / "followup/attempt_01").exists()


@pytest.mark.parametrize("invalid", ["accepted", "seed", "mixed_failure", "exhausted"])
def test_ineligible_historical_attempts_never_launch_sampling(invalid, tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("sampling/compilation must not start")
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", forbidden)
    historical = [{"sampling": dict(adapt_delta=0.97, max_treedepth=12, warmup=1000, sampling=2000),
                   "diagnostics": assessment(divergence_pass=False, divergences=1),
                   "sampling_policy": fitting.MATCHED_RETRY_SAMPLING_POLICY, "seed": 371}]
    if invalid == "accepted":
        historical[0]["diagnostics"] = assessment(passed=True)
    elif invalid == "seed":
        historical[0]["seed"] = 372
    elif invalid == "mixed_failure":
        historical[0]["diagnostics"]["rhat_pass"] = False
    else:
        from copy import deepcopy
        historical.append(deepcopy(historical[0]))
        historical[1]["sampling"]["adapt_delta"] = 0.99
        historical[1]["sampling_policy"] = fitting.MATCHED_RETRY_SAMPLING_POLICY
    with pytest.raises(ValueError):
        fitting.fit_model(stan_file="unused.stan", model_id=HAR_INPUT_MODELS[1], data={},
                          seed=371, output_dir=tmp_path / "run", previous_attempts=historical)
