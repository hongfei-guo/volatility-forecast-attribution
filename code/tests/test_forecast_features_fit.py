from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bnsv.forecast_features import (
    MODEL_SPECS,
    fit_feature_transform,
    prepare_estimation_data,
    transform_features,
)
from bnsv.forecast_fit import (
    HIGH_ACCEPTANCE_LONG_RUN,
    HIGH_ACCEPTANCE_RUN,
    RV_LINEAR_RUN,
    STANDARD_INITIAL_RUN,
    STANDARD_LONG_RUN,
    build_stan_data,
    first_sampling_run,
    initial_values,
    next_sampling_run,
)


def test_model_scope_is_exact() -> None:
    assert set(MODEL_SPECS) == {
        "NN-SV",
        "RV-NN-SV",
        "RV-RA-NN-SV",
        "RV-LIN-SV",
        "SV",
        "RV-NN-SV-HAR",
        "RV-LIN-SV-HAR",
    }
    assert MODEL_SPECS["NN-SV"].stan_file == "nn_sv.stan"
    assert MODEL_SPECS["RV-NN-SV"].stan_file == "nn_sv.stan"
    assert MODEL_SPECS["RV-RA-NN-SV"].stan_file == "rv_ra_nn_sv.stan"
    assert MODEL_SPECS["RV-LIN-SV"].horizons == (1,)
    assert MODEL_SPECS["SV"].feature_names == ("z_lag1",)


@pytest.mark.parametrize(
    ("model_id", "expected_names"),
    [
        ("NN-SV", ("z_lag1",)),
        ("RV-NN-SV", ("z_lag1", "log_rv_lag1")),
        (
            "RV-RA-NN-SV",
            ("z_lag1", "log_rv_lag1", "asymmetry_lag1"),
        ),
        ("RV-LIN-SV", ("z_lag1", "log_rv_lag1")),
        ("SV", ("z_lag1",)),
    ],
)
def test_expanding_feature_dimensions(
    synthetic_frame: pd.DataFrame,
    model_id: str,
    expected_names: tuple[str, ...],
) -> None:
    end = synthetic_frame.loc[79, "date"]
    transform = fit_feature_transform(
        synthetic_frame, model_id=model_id, training_end=end
    )
    data = prepare_estimation_data(synthetic_frame, transform)
    assert transform.feature_names == expected_names
    assert data.x.shape == (data.z.size, len(expected_names))
    assert data.dates.max() == end
    assert np.isfinite(data.z).all()
    assert np.isfinite(data.x).all()


def test_future_values_do_not_change_existing_features(
    synthetic_frame: pd.DataFrame,
) -> None:
    end = synthetic_frame.loc[79, "date"]
    transform = fit_feature_transform(
        synthetic_frame, model_id="RV-NN-SV", training_end=end
    )
    baseline = transform_features(synthetic_frame, transform)
    changed = synthetic_frame.copy()
    changed.loc[changed["date"] > end, "return_cc"] *= -25
    changed.loc[changed["date"] > end, "rv_oc"] *= 100
    perturbed = transform_features(changed, transform)
    mask = baseline["date"] <= end
    assert np.array_equal(
        baseline.loc[mask, list(transform.feature_names)].to_numpy(),
        perturbed.loc[mask, list(transform.feature_names)].to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "model_id", ["NN-SV", "RV-NN-SV", "RV-RA-NN-SV", "RV-LIN-SV", "SV"]
)
def test_stan_data_matches_current_models(
    synthetic_frame: pd.DataFrame, model_id: str
) -> None:
    transform = fit_feature_transform(
        synthetic_frame,
        model_id=model_id,
        training_end=synthetic_frame.loc[79, "date"],
    )
    estimation = prepare_estimation_data(synthetic_frame, transform)
    data = build_stan_data(
        model_id=model_id,
        market="SP500",
        z=estimation.z,
        x=estimation.x,
    )
    assert data["N"] == len(estimation.z)
    if model_id in {"NN-SV", "RV-NN-SV", "RV-RA-NN-SV"}:
        assert data["H"] == 5
        assert data["D"] == (
            len(transform.feature_names) - (model_id == "RV-RA-NN-SV")
        )
        assert math.isfinite(data["output_log_mean"])
        if model_id == "RV-RA-NN-SV":
            assert data["q"].shape == data["z"].shape
            assert data["gamma_A_prior_sd"] > 0
    elif model_id == "RV-LIN-SV":
        assert data["D"] == 2
        assert data["beta_base_sd"] > 0
    else:
        assert "D" not in data and "X" not in data


def test_initial_values_match_parameter_dimensions() -> None:
    z = np.linspace(-1, 1, 40)
    nn_data = build_stan_data(
        model_id="RV-NN-SV", market="DAX", z=z, x=np.zeros((40, 2))
    )
    nn_values = initial_values("RV-NN-SV", nn_data)
    assert nn_values is not None and len(nn_values) == 4
    assert np.asarray(nn_values[0]["w1_raw"]).shape == (2, 5)
    assert len(nn_values[0]["alpha"]) == 39
    assert np.all(np.diff(nn_values[0]["log_output_weight_std"]) > 0)

    rv_ra_x = np.zeros((40, 3))
    rv_ra_x[:, 2] = np.linspace(-1, 1, 40)
    rv_ra_data = build_stan_data(
        model_id="RV-RA-NN-SV", market="DAX", z=z, x=rv_ra_x
    )
    rv_ra_values = initial_values("RV-RA-NN-SV", rv_ra_data)
    assert rv_ra_values is not None
    assert np.asarray(rv_ra_values[0]["w1_raw"]).shape == (2, 5)
    assert {value["gamma_A_std"] for value in rv_ra_values} == {0.0}

    linear_data = build_stan_data(
        model_id="RV-LIN-SV", market="DAX", z=z, x=np.zeros((40, 2))
    )
    linear_values = initial_values("RV-LIN-SV", linear_data)
    assert linear_values is not None and len(linear_values[0]["beta_raw"]) == 2
    sv_data = build_stan_data(
        model_id="SV", market="DAX", z=z, x=np.zeros((40, 1))
    )
    assert initial_values("SV", sv_data) is None


def _assessment(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "passed": False,
        "rhat_pass": True,
        "bulk_ess_pass": True,
        "tail_ess_pass": True,
        "mcse_over_sd_pass": True,
        "divergence_pass": True,
        "treedepth_pass": True,
        "bfmi_pass": True,
        "divergences": 0,
        "divergences_by_chain": [0, 0, 0, 0],
        "post_warmup_draws_by_chain": [2000, 2000, 2000, 2000],
    }
    result.update(updates)
    return result


def test_diagnostic_sampling_sequence() -> None:
    assert first_sampling_run("NN-SV") == STANDARD_INITIAL_RUN
    assert first_sampling_run("RV-LIN-SV") == RV_LINEAR_RUN

    ess_only = _assessment(bulk_ess_pass=False)
    assert next_sampling_run(
        model_id="NN-SV", current=STANDARD_INITIAL_RUN, assessment=ess_only
    ) == STANDARD_LONG_RUN
    assert next_sampling_run(
        model_id="NN-SV", current=HIGH_ACCEPTANCE_RUN, assessment=ess_only
    ) == HIGH_ACCEPTANCE_LONG_RUN

    sparse = _assessment(
        divergence_pass=False,
        divergences=2,
        divergences_by_chain=[1, 1, 0, 0],
    )
    assert next_sampling_run(
        model_id="RV-NN-SV", current=STANDARD_INITIAL_RUN, assessment=sparse
    ) == HIGH_ACCEPTANCE_RUN
    assert next_sampling_run(
        model_id="RV-LIN-SV", current=RV_LINEAR_RUN, assessment=ess_only
    ) is None
