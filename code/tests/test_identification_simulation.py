from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bnsv.diagnostics import DiagnosticThresholds, assess_summary_table
from bnsv.identification_fitting import classify_fit_assessment, sampling_sequence
from bnsv.identification_initialization import make_prior_centered_nn_inits
from bnsv.identification_simulation import (
    DGPParameters,
    ReferenceCurve,
    build_fit_plan,
    derived_seed,
    exact_design_constants,
    simulate_replication,
    standardized_student_t,
)
from bnsv.identification_stan import nn_sv_stan_data


PROJECT = Path(__file__).resolve().parents[2]


def curve() -> ReferenceCurve:
    x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.asarray([-0.15, -0.05, -0.02, 0.08, 0.14])
    y -= y.mean()
    return ReferenceCurve(x=x, y=y, left_slope=0.1, right_slope=0.06, source_records=())


def parameters() -> DGPParameters:
    phi = 0.9
    sigma_h = 0.4
    return DGPParameters(
        mu=-0.5,
        phi=phi,
        sigma_h=sigma_h,
        sigma_eta=sigma_h * np.sqrt(1 - phi**2),
        nu=8.0,
        target_correction_share=0.02,
        estimation_rows=80,
        forecast_rows=10,
        burn_in=50,
    )


def load_summarizer():
    path = PROJECT / "code" / "scripts" / "summarize_identification_simulation.py"
    spec = importlib.util.spec_from_file_location("identification_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reported_random_stream_identity_is_preserved():
    assert derived_seed(1, "experiment_state") == 952002185
    assert derived_seed(1, "experiment_observation") == 794178605
    assert derived_seed(1, "experiment_state") != derived_seed(1, "calibration_state")
    assert derived_seed(1, "fit_task_001") == 898892196
    assert derived_seed(1, "forecast_task_001") == 853384225
    assert derived_seed(200, "fit_task_400") == 2044455209
    assert derived_seed(200, "forecast_task_400") == 774604312


def test_seed_table_covers_every_reported_calculation() -> None:
    path = PROJECT / "design" / "identification_seeds.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1500
    keys = {
        (int(row["replication_id"]), row["stream_name"])
        for row in rows
    }
    assert len(keys) == len(rows)
    assert all(0 < int(row["seed"]) <= 2_147_483_647 for row in rows)


def test_standardized_student_t_has_unit_variance():
    draws = standardized_student_t(np.random.default_rng(7), 8.0, 300_000)
    assert np.var(draws) == pytest.approx(1.0, abs=0.015)


def test_dgp_zero_nests_pure_sv_and_uses_paired_streams():
    nonlinear = simulate_replication(
        parameters(), curve(), replication_id=4, dgp_id="DGP-N", amplitude=1.0
    )
    zero = simulate_replication(
        parameters(), curve(), replication_id=4, dgp_id="DGP-0", amplitude=1.0
    )
    assert np.array_equal(zero["correction_truth"], np.zeros(90))
    assert np.array_equal(zero["h_truth"], zero["b_truth"])
    assert int(zero["state_seed"][0]) == int(nonlinear["state_seed"][0])
    assert int(zero["observation_seed"][0]) == int(nonlinear["observation_seed"][0])


def test_exact_rules_match_the_reported_400_task_design():
    constants = exact_design_constants()
    assert constants["replications_per_dgp"] == 200
    assert constants["coverage_rejection_cutoff"] == 184
    assert constants["false_declaration_rejection_cutoff"] == 16
    assert constants["median_distribution_free_95_interval_order_statistics"] == [86, 115]
    assert constants["total_fits"] == 400
    with pytest.raises(ValueError):
        exact_design_constants(150)


def test_task_list_contains_the_reported_replications():
    records = []
    for dgp in ("DGP-N", "DGP-0"):
        for replication_id in range(1, 201):
            records.append(
                {
                    "dgp_id": dgp,
                    "replication_id": replication_id,
                    "path": f"datasets/{dgp}_{replication_id}.npz",
                }
            )
    tasks = build_fit_plan(records)
    assert len(tasks) == 400
    assert all(
        set(task)
        == {
            "task_id",
            "dgp_id",
            "replication_id",
            "dataset_path",
        }
        for task in tasks
    )


def test_design_binds_scientific_counts_and_thresholds():
    config = yaml.safe_load((PROJECT / "design" / "identification.yaml").read_text())
    assert config["replications"]["per_dgp"] == 200
    assert config["model"]["width"] == 5
    assert config["dgp"]["ids"] == ["DGP-N", "DGP-0"]
    assert config["reporting"]["coverage_rejection_at_or_below"] == 184
    assert config["reporting"]["false_declaration_rejection_at_or_above"] == 16
    assert len(sampling_sequence(config["mcmc"])) == 4


def test_diagnostics_and_sampling_adjustment():
    summary = pd.DataFrame(
        {
            "R_hat": [1.001],
            "ESS_bulk": [900.0],
            "ESS_tail": [850.0],
            "MCSE": [0.01],
            "StdDev": [1.0],
        }
    )
    assert assess_summary_table(summary, DiagnosticThresholds())["summary_pass"] is True
    assessment = {
        "passed": False,
        "divergences": 2,
        "divergences_by_chain": [1, 1, 0, 0],
        "post_warmup_draws_by_chain": [2000] * 4,
        "rhat_pass": True,
        "bulk_ess_pass": True,
        "tail_ess_pass": True,
        "mcse_over_sd_pass": True,
        "treedepth_pass": True,
        "bfmi_pass": True,
    }
    rules = {
        "sparse_divergence_max_total_fraction": 0.001,
        "sparse_divergence_max_per_chain_fraction": 0.002,
        "reference_post_warmup_draws_per_chain": 2000,
        "adapt_delta_ceiling": 0.97,
    }
    assert classify_fit_assessment(assessment, adapt_delta=0.95, rules=rules) == "increase_adapt_delta"
    assert classify_fit_assessment(assessment, adapt_delta=0.97, rules=rules) == "unsuccessful"


def test_stan_data_and_four_initializations_are_finite():
    config = yaml.safe_load((PROJECT / "design" / "identification.yaml").read_text())
    data = nn_sv_stan_data(
        z=np.asarray([0.1, -0.2]),
        x=np.asarray([[0.0], [0.1]]),
        width=5,
        prior=config["model"]["prior"],
    )
    inits = make_prior_centered_nn_inits(
        n=2,
        d=1,
        width=5,
        phi_logit_loc=data["phi_logit_loc"],
        phi_logit_scale=data["phi_logit_scale"],
        chains=4,
    )
    assert set(data) == {
        "N",
        "z",
        "mu_prior_mean",
        "mu_prior_sd",
        "phi_logit_loc",
        "phi_logit_scale",
        "sigma_h_prior_sd",
        "h0_prior_sd",
        "nu_minus_two_log_mean",
        "nu_minus_two_log_sd",
        "D",
        "H",
        "X",
        "w1_base_sd",
        "b1_prior_sd",
        "output_log_mean",
        "output_log_sd",
    }
    assert data["N"] == 2 and data["D"] == 1 and data["H"] == 5
    assert len(inits) == 4
    assert all(np.isfinite(init["sigma_eta_std"]) for init in inits)


def test_unsuccessful_fit_is_counted_as_uncovered():
    module = load_summarizer()
    rows = pd.DataFrame(
        [
            {"dgp_id": "DGP-N", "parameter": "phi", "bias": 0.01, "squared_error": 0.0001, "covered": True, "posterior_mean": 0.90, "posterior_sd": 0.02, "interval_width": 0.08},
            {"dgp_id": "DGP-N", "parameter": "phi", "bias": -0.02, "squared_error": 0.0004, "covered": True, "posterior_mean": 0.87, "posterior_sd": 0.03, "interval_width": 0.09},
        ]
    )
    result = module.aggregate_scalar(
        rows,
        planned_by_dgp={"DGP-N": 3},
        unsuccessful_by_dgp={"DGP-N": 1},
        coverage_rejection_cutoff=1,
    )[0]
    assert result["successful_replications"] == 2
    assert result["unsuccessful_replications"] == 1
    assert result["coverage_over_planned_replications"] == pytest.approx(2 / 3)


def test_mcmc_diagnostic_uses_reader_facing_fields() -> None:
    module = load_summarizer()
    task = {"task_id": 1, "dgp_id": "DGP-N", "replication_id": 1}
    fit_result = {
        "fit_outcome": "unsuccessful",
        "sampling_runs": [
            {
                "adapt_delta": 0.97,
                "runtime_seconds": 1.0,
                "assessment": {"divergences": 2},
            }
        ],
    }
    row = module.mcmc_diagnostic_row(task, fit_result, compact=None)
    assert row["last_adapt_delta"] == 0.97
    assert set(row).isdisjoint({"task_id", "runtime_seconds"})
