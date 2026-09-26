from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import bnsv.har_input_generation as generation
from bnsv.forecast_calculation import reconstruct_forecast_paths
from bnsv.forecast_features import (
    HAR_FEATURE_NAMES, HAR_INPUT_MODELS, base_model_id, fit_feature_transform,
    model_spec, prepare_estimation_data, transform_features,
)
from bnsv.forecast_filter import FixedParameterGridFilter
from bnsv.forecast_fit import (
    SamplingFailure, build_stan_data, first_sampling_run, initial_values,
    HIGH_ACCEPTANCE_RUN, MATCHED_RETRY_SAMPLING_POLICY,
)
from bnsv.forecast_paths import compact_refit, observable_correction
from bnsv.har_input_calibration import calibrate_har_input_scales


MODELS = Path(__file__).resolve().parents[1] / "models"


class SyntheticFit:
    def __init__(self, data: dict[str, Any], model_id: str) -> None:
        n, d, draws = data["N"], data["D"], 8
        axis = np.linspace(-0.1, 0.1, draws)
        self.values = {
            "mu": axis, "phi": np.linspace(0.85, 0.92, draws),
            "sigma_eta": np.linspace(0.10, 0.14, draws),
            "nu": np.linspace(7.0, 10.0, draws),
            "b": np.linspace(-0.3, 0.2, n)[None, :] + axis[:, None],
        }
        if base_model_id(model_id) == "RV-LIN-SV":
            self.values["beta"] = np.broadcast_to(np.arange(1, d + 1) * 0.015, (draws, d)).copy()
        else:
            rng = np.random.default_rng(103)
            w1 = rng.normal(0, 0.2, (draws, d, 5))
            bias = rng.normal(0, 0.05, (draws, 5))
            w = np.full((draws, 5), 0.05)
            raw = (np.tanh(np.einsum("nd,mdh->mnh", data["X"], w1) + bias[:, None, :]) * w[:, None, :]).sum(axis=2)
            self.values.update(W1=w1, b1=bias, output_weight=w, centering_mean=raw.mean(axis=1))

    def stan_variable(self, name: str) -> np.ndarray:
        return self.values[name]

    def stan_variables(self) -> dict[str, np.ndarray]:
        return self.values


def test_har_inputs_use_only_lagged_oc_and_match_between_models(synthetic_frame: pd.DataFrame) -> None:
    end = synthetic_frame.loc[79, "date"]
    transforms = [fit_feature_transform(synthetic_frame, model_id=name, training_end=end) for name in HAR_INPUT_MODELS]
    inputs = [prepare_estimation_data(synthetic_frame, transform) for transform in transforms]
    np.testing.assert_array_equal(inputs[0].x, inputs[1].x)
    np.testing.assert_array_equal(inputs[0].z, inputs[1].z)
    np.testing.assert_array_equal(inputs[0].dates, inputs[1].dates)
    assert inputs[0].dates[0] == synthetic_frame.loc[22, "date"]
    assert inputs[0].x.shape == (58, 4)
    transform = transforms[0]
    assert transform.feature_names == HAR_FEATURE_NAMES
    frame = transform_features(synthetic_frame, transform)
    i = 55
    y = np.log(np.maximum(synthetic_frame["rv_oc"], transform.rv_floor))
    expected = np.array([
        transform.return_scaler.standardize(synthetic_frame.loc[i - 1, "return_cc"]),
        y.iloc[i - 1], y.iloc[i - 5:i].mean(), y.iloc[i - 22:i].mean(),
    ])
    actual = frame.loc[i, list(HAR_FEATURE_NAMES)].to_numpy(float) * transform.feature_scale + transform.feature_mean
    np.testing.assert_allclose(actual, expected)
    assert abs(actual[2] - np.log(np.exp(y.iloc[i - 5:i]).mean())) > 1e-6
    changed = synthetic_frame.copy()
    changed["rv_cc"] *= 1000
    changed.loc[i:, "rv_oc"] *= 100
    changed.loc[i:, "return_cc"] *= -10
    altered = transform_features(changed, transform)
    pd.testing.assert_frame_equal(frame.loc[:i], altered.loc[:i], check_exact=True)
    future = synthetic_frame.copy()
    future.loc[future["date"] > end, "rv_oc"] *= 1000
    future.loc[future["date"] > end, "return_cc"] *= -1000
    future_transform = fit_feature_transform(future, model_id=HAR_INPUT_MODELS[0], training_end=end)
    future_inputs = prepare_estimation_data(future, future_transform)
    np.testing.assert_array_equal(future_inputs.x, inputs[0].x)
    np.testing.assert_array_equal(future_inputs.z, inputs[0].z)


@pytest.mark.parametrize("name", HAR_INPUT_MODELS)
def test_har_stan_and_initialization_are_the_parent_model(name: str, synthetic_frame: pd.DataFrame) -> None:
    parent = base_model_id(name)
    assert model_spec(name).stan_file == model_spec(parent).stan_file
    assert model_spec(name).horizons == (1,)
    transform = fit_feature_transform(synthetic_frame, model_id=name, training_end=synthetic_frame.loc[79, "date"])
    inputs = prepare_estimation_data(synthetic_frame, transform)
    data = build_stan_data(model_id=name, market="SP500", z=inputs.z, x=inputs.x, correction_scale=0.17)
    old = build_stan_data(model_id=parent, market="SP500", z=inputs.z, x=inputs.x[:, :2])
    changed_keys = {"D", "X", "output_log_mean", "beta_base_sd"}
    assert set(data) == set(old)
    for key in data.keys() - changed_keys:
        np.testing.assert_array_equal(data[key], old[key])
    assert data["D"] == 4
    assert first_sampling_run(name) == HIGH_ACCEPTANCE_RUN
    values = initial_values(name, data)
    expected = initial_values(parent, data)
    assert values == expected
    assert np.asarray(values[0]["w1_raw" if parent == "RV-NN-SV" else "beta_raw"]).shape[0] == 4
    with pytest.raises(ValueError, match="calibrated"):
        build_stan_data(model_id=name, market="SP500", z=inputs.z, x=inputs.x)
    with pytest.raises(ValueError, match="fixed correction"):
        build_stan_data(model_id=parent, market="SP500", z=inputs.z, x=inputs.x[:, :2], correction_scale=0.17)


def test_prior_calibration_cannot_use_post_2016_data(synthetic_frame: pd.DataFrame) -> None:
    original = calibrate_har_input_scales(synthetic_frame)
    altered = synthetic_frame.copy()
    mask = altered["date"] > pd.Timestamp("2016-12-31")
    assert mask.any()
    altered.loc[mask, "rv_oc"] *= 100000
    altered.loc[mask, "return_cc"] *= -100000
    assert original == calibrate_har_input_scales(altered)
    assert original["ddof"] == 0
    assert original["neural_draws"] == 500 and original["linear_draws"] == 2000
    assert original["last_reference_row"] <= "2016-12-31"
    for name in HAR_INPUT_MODELS:
        assert original["scales"][name] > 0


@pytest.mark.parametrize("name", HAR_INPUT_MODELS)
def test_nonzero_correction_matches_filter_and_forecast(name: str, synthetic_frame: pd.DataFrame) -> None:
    origin = 59
    transform = fit_feature_transform(synthetic_frame, model_id=name, training_end=synthetic_frame.loc[origin, "date"])
    data = prepare_estimation_data(synthetic_frame, transform)
    stan_data = build_stan_data(model_id=name, market="SP500", z=data.z, x=data.x, correction_scale=0.17)
    fit = SyntheticFit(stan_data, name)
    refit = compact_refit(fit, model_id=name, transform=transform, estimation_features=data.x,
                         particle_count=4, rng=np.random.default_rng(12))
    history = synthetic_frame.iloc[:origin + 1]
    forecast_inputs = {
        "source_parameter_ancestry": refit["ancestry"], "origin_baseline_states": refit["state"],
        "state_innovations": np.zeros((4, 1)), "return_innovations": np.zeros((4, 1)),
        "origin_z_history_tail": transform.return_scaler.standardize(history["return_cc"].to_numpy())[-22:],
        "origin_log_rv_history_tail": np.log(np.maximum(history["rv_oc"], transform.rv_floor)).to_numpy()[-22:],
    }
    calculated = reconstruct_forecast_paths(name, refit, forecast_inputs)
    x = transform_features(synthetic_frame, transform).loc[origin + 1, list(HAR_FEATURE_NAMES)].to_numpy(float)
    parameters = {k.removeprefix("parameter__"): v for k, v in refit.items() if k.startswith("parameter__")}
    correction = observable_correction(name, parameters, np.broadcast_to(x, (4, 4)))
    np.testing.assert_allclose(calculated["observable_corrections"][:, 0], correction, atol=1e-14)
    assert np.max(np.abs(correction)) > 1e-5
    expected_h = parameters["mu"] + parameters["phi"] * (refit["state"] - parameters["mu"]) + correction
    np.testing.assert_allclose(calculated["raw_variances"][:, 0], transform.return_scaler.scale**2 * np.exp(expected_h))
    for key in ("state_innovations", "return_innovations"):
        forecast_inputs[key] = np.zeros((4, 5))
    with pytest.raises(ValueError, match="horizon one"):
        reconstruct_forecast_paths(name, refit, forecast_inputs)


def lightweight_banks(refit):
    from bnsv.forecast_paths import filter_bank_from_refit
    return filter_bank_from_refit(refit, levels=(0, 1, 2))


def update_for_test(banks, *, model_id, x, observed_z, **kwargs):
    for bank in banks.values():
        bank.step(x_t=x, observed_z=observed_z,
                  correction=lambda p, features: observable_correction(model_id, p, features))
    return True, []


def test_pair_refits_both_models_after_either_trigger(
    synthetic_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_frame = synthetic_frame.copy()
    synthetic_frame["date"] += pd.DateOffset(years=1)
    calls, events = [], []
    def fit_function(**kwargs: Any):
        calls.append(kwargs)
        events.append("fit")
        return SyntheticFit(kwargs["data"], kwargs["model_id"]), []
    original_forecast = generation.simulate_forecast_paths
    def forecast(**kwargs: Any):
        events.append("forecast")
        return original_forecast(**kwargs)
    original_step = FixedParameterGridFilter.step
    def step(self, **kwargs: Any):
        events.append("filter")
        return original_step(self, **kwargs)
    checks = []
    def adequate(*args, **kwargs):
        checks.append(None)
        update_for_test(*args, **kwargs)
        return len(checks) != 2, []  # Linear member triggers the shared refit.
    monkeypatch.setattr(generation, "simulate_forecast_paths", forecast)
    monkeypatch.setattr(FixedParameterGridFilter, "step", step)
    monkeypatch.setattr(generation, "initial_banks", lightweight_banks)
    monkeypatch.setattr(generation, "update_and_assess", adequate)
    output = tmp_path / "pair"
    panel = generation.generate_har_input_pair(
        daily=synthetic_frame, market="SP500", stan_dir=MODELS, output_dir=output,
        correction_scales=dict.fromkeys(HAR_INPUT_MODELS, 0.17),
        forecast_start=synthetic_frame.loc[59, "date"], forecast_end=synthetic_frame.loc[61, "date"],
        particle_count=4, path_count=8, grid_levels=(2, 3, 4), fit_function=fit_function,
    )
    assert len(panel) == 6 and len(calls) == 4
    assert events[:5] == ["fit", "fit", "forecast", "forecast", "filter"]
    assert events[10:14] == ["fit", "fit", "forecast", "forecast"]
    for first, second in zip(calls[::2], calls[1::2]):
        np.testing.assert_array_equal(first["data"]["X"], second["data"]["X"])
        np.testing.assert_array_equal(first["data"]["z"], second["data"]["z"])
        assert first["seed"] == second["seed"]
    records = [json.loads(p.read_text()) for p in sorted(output.glob("refits/*/fit.json"))]
    assert all(r["sampling_policy"] == MATCHED_RETRY_SAMPLING_POLICY for r in records)
    assert json.loads((output / "run.json").read_text())["sampling_policy"] == MATCHED_RETRY_SAMPLING_POLICY
    assert sum(r["reason"] == "adaptive" for r in records) == 2
    assert all(r["trigger_models"] == [HAR_INPUT_MODELS[1]] for r in records if r["reason"] == "adaptive")
    assert len(list(output.glob("refits/*/posterior.npz"))) == 4
    for row in panel.itertuples():
        with np.load(output / "components" / row.predictive_file) as arrays:
            assert arrays["raw_variances"].shape == (8, 1)
            assert arrays["raw_variances"].mean() == row.variance_forecast
    pd.testing.assert_frame_equal(panel, pd.read_parquet(output / "forecasts.parquet"))
    with pytest.raises(FileExistsError):
        generation.generate_har_input_pair(
            daily=synthetic_frame, market="SP500", stan_dir=MODELS, output_dir=output,
            correction_scales=dict.fromkeys(HAR_INPUT_MODELS, 0.17),
            forecast_start=synthetic_frame.loc[59, "date"], forecast_end=synthetic_frame.loc[61, "date"],
            fit_function=fit_function,
        )


def test_unsuccessful_fit_is_preserved_without_extra_retries(
    synthetic_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_frame = synthetic_frame.copy()
    synthetic_frame["date"] += pd.DateOffset(years=1)
    calls = []
    def fit_function(**kwargs: Any):
        calls.append(kwargs["model_id"])
        if kwargs["model_id"] == HAR_INPUT_MODELS[0]:
            raise SamplingFailure([{"diagnostics": {"passed": False}}])
        return SyntheticFit(kwargs["data"], kwargs["model_id"]), []
    monkeypatch.setattr(generation, "initial_banks", lightweight_banks)
    monkeypatch.setattr(generation, "update_and_assess", update_for_test)
    panel = generation.generate_har_input_pair(
        daily=synthetic_frame, market="SP500", stan_dir=MODELS, output_dir=tmp_path / "failed",
        correction_scales=dict.fromkeys(HAR_INPUT_MODELS, 0.17),
        forecast_start=synthetic_frame.loc[59, "date"], forecast_end=synthetic_frame.loc[61, "date"],
        particle_count=4, path_count=8, grid_levels=(2, 3, 4), fit_function=fit_function,
    )
    assert calls == list(HAR_INPUT_MODELS)
    assert set(panel["model_id"]) == {HAR_INPUT_MODELS[1]}
    records = [json.loads(p.read_text()) for p in (tmp_path / "failed").glob("refits/*/fit.json")]
    assert sum(r["status"] == "unsuccessful" for r in records) == 1
    availability = [json.loads(line) for line in (tmp_path / "failed/availability.jsonl").read_text().splitlines()]
    assert sum(not r["available"] for r in availability) == 3


def test_pair_refits_on_first_observed_origin_of_new_month(
    synthetic_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = synthetic_frame.copy()
    frame["date"] += pd.DateOffset(years=1)
    first_new_month = int(frame.index[frame["date"] >= pd.Timestamp("2017-12-01")][0])
    calls = []
    def fit_function(**kwargs: Any):
        calls.append(kwargs["model_id"])
        return SyntheticFit(kwargs["data"], kwargs["model_id"]), []
    monkeypatch.setattr(generation, "initial_banks", lightweight_banks)
    monkeypatch.setattr(generation, "update_and_assess", update_for_test)
    output = tmp_path / "monthly"
    generation.generate_har_input_pair(
        daily=frame, market="SP500", stan_dir=MODELS, output_dir=output,
        correction_scales=dict.fromkeys(HAR_INPUT_MODELS, 0.17),
        forecast_start=frame.loc[first_new_month - 1, "date"],
        forecast_end=frame.loc[first_new_month, "date"],
        particle_count=4, path_count=8, grid_levels=(2, 3, 4), fit_function=fit_function,
    )
    assert calls == list(HAR_INPUT_MODELS) * 2
    records = [json.loads(p.read_text()) for p in output.glob("refits/*/fit.json")]
    assert len(records) == 4 and all(r["reason"] == "monthly" for r in records)


def test_extension_does_not_enter_primary_evaluation_or_combination() -> None:
    design = MODELS.parents[1] / "design"
    analysis = yaml.safe_load((design / "analysis.yaml").read_text())
    combination = yaml.safe_load((design / "combination.yaml").read_text())
    assert not set(HAR_INPUT_MODELS) & set(analysis["evaluation"]["mcs"]["models"])
    assert not set(HAR_INPUT_MODELS) & set(combination["candidate_models"])
