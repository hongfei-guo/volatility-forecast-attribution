#!/usr/bin/env python3
"""Analyze the conditional forecast value of the RV-RA component in 2017.

The calculation uses evaluated loss panels and compact RV-RA refit and
predictive archives. It performs no model fitting and reads no observations
from the 2018--2022 evaluation sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


EVALUATION_START = "2018-01-01"
EXPECTED_ORIGINS = 241
EXPECTED_HORIZONS = (1, 5, 10)
COUNTERFACTUAL_HORIZONS = (5, 10)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def qlike(realized: np.ndarray | float, forecast: np.ndarray | float) -> np.ndarray:
    y = np.maximum(np.asarray(realized, dtype=float), 1e-15)
    f = np.maximum(np.asarray(forecast, dtype=float), 1e-15)
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def circular_block_indices(
    n: int, *, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if n <= 0 or block_length <= 0:
        raise ValueError("n and block_length must be positive")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def fit_fixed_effects(
    outcome: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(covariates, dtype=float)
    g = np.asarray(groups).astype(str)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size or g.shape != y.shape:
        raise ValueError("fixed-effect regression inputs are not aligned")
    levels = np.unique(g)
    fixed = (g[:, None] == levels[None, :]).astype(float)
    design = np.column_stack([fixed, x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("fixed-effect regression is rank deficient")
    return coefficients[-x.shape[1] :]


def varying_group_count(values: np.ndarray, groups: np.ndarray) -> int:
    x = np.asarray(values)
    g = np.asarray(groups).astype(str)
    if x.ndim != 1 or g.shape != x.shape:
        raise ValueError("regime support inputs are not aligned")
    return sum(np.unique(x[g == level]).size > 1 for level in np.unique(g))


def validate_config(config: Mapping[str, Any]) -> None:
    sample = config.get("sample", {})
    if (
        sample.get("origin_count") != EXPECTED_ORIGINS
        or tuple(sample.get("horizons", [])) != EXPECTED_HORIZONS
        or tuple(sample.get("counterfactual_horizons", []))
        != COUNTERFACTUAL_HORIZONS
        or sample.get("origin_start") != "2017-01-03"
        or sample.get("origin_end") != "2017-12-14"
        or sample.get("last_target_date") != "2017-12-29"
        or sample.get("evaluation_start") != EVALUATION_START
    ):
        raise ValueError("analysis sample identity changed")
    bootstrap = config.get("bootstrap", {})
    if (
        bootstrap.get("method") != "circular_moving_block"
        or bootstrap.get("block_length") != 10
        or bootstrap.get("replications") != 2000
        or bootstrap.get("seed") != 20260730
    ):
        raise ValueError("bootstrap design changed")
    if (
        config.get("conditional_regression", {}).get(
            "unsupported_regime_reporting"
        )
        != "unavailable_with_support_counts"
    ):
        raise ValueError("unsupported-regime reporting definition changed")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_rv_ra_records(
    rv_ra_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Path]]:
    root = rv_ra_root.resolve()
    month_roots = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "forecast_records.jsonl").is_file()
    )
    if len(month_roots) != 12:
        raise ValueError("RV-RA input must contain 12 monthly forecast directories")
    by_cell: dict[tuple[str, int], dict[str, Any]] = {}
    refit_paths: dict[str, Path] = {}
    for task_root in month_roots:
        records_path = task_root / "forecast_records.jsonl"
        refit_candidates = sorted((task_root / "refits").glob("*.npz"))
        if len(refit_candidates) != 1:
            raise ValueError(f"expected one RV-RA refit in {task_root / 'refits'}")
        refit_path = refit_candidates[0]
        for record in _read_records(records_path):
            origin = str(record["origin_date"])
            horizon = int(record["horizon"])
            if not origin.startswith("2017-"):
                raise ValueError("monthly forecast records contain a non-2017 origin")
            if record.get("model_id") != "RV-RA-NN-SV":
                raise ValueError(f"unexpected model for {origin}, h={horizon}")
            target_dates = [str(value) for value in record.get("target_dates", [])]
            if (
                horizon not in EXPECTED_HORIZONS
                or len(target_dates) != horizon
                or any(value >= EVALUATION_START for value in target_dates)
            ):
                raise ValueError(
                    f"forecast target enters the evaluation sample: {origin}, h={horizon}"
                )
            key = (origin, horizon)
            if key in by_cell:
                raise ValueError(f"duplicate RV-RA forecast cell: {key}")
            record = dict(record)
            record["task_root"] = str(task_root)
            by_cell[key] = record
            refit_id = str(record["refit_id"])
            previous = refit_paths.get(refit_id)
            if previous is not None and previous != refit_path:
                raise ValueError(f"refit {refit_id} has multiple source files")
            refit_paths[refit_id] = refit_path
    return by_cell, refit_paths


def predictive_array_path(record: Mapping[str, Any]) -> Path:
    origin = str(record["origin_date"]).replace("-", "")
    horizon = int(record["horizon"])
    return Path(str(record["task_root"])) / "draws" / f"{origin}_h{horizon}.npz"


def load_paired_losses(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    required = {
        "origin_date",
        "horizon",
        "qlike__RV-NN-SV",
        "qlike__RV-RA-NN-SV",
    }
    if required - set(frame.columns):
        raise ValueError("paired-loss input lacks required columns")
    frame["origin_date"] = frame["origin_date"].astype(str)
    frame["horizon"] = frame["horizon"].astype(int)
    if frame.duplicated(["origin_date", "horizon"]).any():
        raise ValueError("paired-loss input contains duplicate cells")
    origin_sets = {
        horizon: set(frame.loc[frame["horizon"] == horizon, "origin_date"])
        for horizon in EXPECTED_HORIZONS
    }
    origins = sorted(set.intersection(*(origin_sets[h] for h in EXPECTED_HORIZONS)))
    selected = frame[
        frame["origin_date"].isin(origins)
        & frame["horizon"].isin(EXPECTED_HORIZONS)
    ].copy()
    counts = selected.groupby("horizon")["origin_date"].nunique().to_dict()
    if (
        len(origins) != EXPECTED_ORIGINS
        or origins[0] != "2017-01-03"
        or origins[-1] != "2017-12-14"
        or counts != {1: 241, 5: 241, 10: 241}
        or len(selected) != 723
    ):
        raise ValueError("common-origin identity changed")
    return selected.sort_values(["origin_date", "horizon"]), origins


def load_member_losses(path: Path, model_id: str, origins: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "origin_date",
        "horizon",
        "model_id",
        "variance_forecast",
        "realized_variance",
        "qlike",
        "target_dates_json",
        "realized_asymmetry_path_json",
    }
    if required - set(frame.columns):
        raise ValueError(f"{model_id} loss archive lacks required columns")
    frame["origin_date"] = frame["origin_date"].astype(str)
    frame["horizon"] = frame["horizon"].astype(int)
    selected = frame[
        frame["origin_date"].isin(origins) & frame["horizon"].isin(EXPECTED_HORIZONS)
    ].copy()
    if len(selected) != EXPECTED_ORIGINS * len(EXPECTED_HORIZONS):
        raise ValueError(f"{model_id} does not cover the 241-by-3 sample")
    if set(selected["model_id"].astype(str)) != {model_id}:
        raise ValueError(f"unexpected model id in {model_id} loss archive")
    if selected.duplicated(["origin_date", "horizon"]).any():
        raise ValueError(f"duplicate cells in {model_id} loss archive")
    for value in selected["target_dates_json"]:
        dates = [str(item) for item in json.loads(str(value))]
        if any(item >= EVALUATION_START for item in dates):
            raise ValueError(f"{model_id} loss archive enters the evaluation sample")
    return selected.set_index(["origin_date", "horizon"]).sort_index()


def load_gamma_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "task_id",
        "origin",
        "slice",
        "width",
        "draws",
        "gamma_A_mean",
        "gamma_A_sd",
        "gamma_A_q025",
        "gamma_A_median",
        "gamma_A_q975",
    }
    if required - set(frame.columns):
        raise ValueError("gamma_A posterior summary lacks required columns")
    selected = frame[
        (frame["slice"] == "calendar_month_refit")
        & (frame["width"] == 5)
        & frame["task_id"].isin(range(1, 13))
    ].copy()
    selected["origin"] = selected["origin"].astype(str)
    origins = sorted(selected["origin"])
    if (
        len(selected) != 12
        or selected["origin"].duplicated().any()
        or set(selected["task_id"].astype(int)) != set(range(1, 13))
        or set(selected["draws"].astype(int)) != {8000}
        or origins[0] != "2017-01-03"
        or origins[-1] != "2017-12-01"
    ):
        raise ValueError("calendar-2017 gamma_A summary identity changed")
    return selected.set_index("origin").sort_index()


def load_refit(path: Path, daily: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    required = {
        "ancestry",
        "feature_names",
        "parameter__mu",
        "parameter__phi",
        "parameter__w1",
        "parameter__b1",
        "parameter__output_weights",
        "parameter__centering_constant",
        "parameter__gamma_A",
        "parameter__q_centering_mean",
        "return_mean",
        "return_scale",
    }
    with np.load(path, allow_pickle=False) as arrays:
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"compact refit lacks arrays: {sorted(missing)}")
        payload = {name: np.asarray(arrays[name]).copy() for name in required}
        if "training_features" in arrays.files:
            payload["training_features"] = arrays["training_features"].copy()
        elif daily is None:
            raise ValueError("compact conditional components require the daily data")
        else:
            from bnsv.forecast_features import fit_feature_transform, prepare_estimation_data

            transform = fit_feature_transform(
                daily, model_id="RV-RA-NN-SV",
                training_end=pd.Timestamp(path.stem.rsplit("_", 1)[-1]),
            )
            for key, value in (("feature_mean", transform.feature_mean),
                               ("feature_scale", transform.feature_scale),
                               ("return_mean", transform.return_scaler.mean),
                               ("return_scale", transform.return_scaler.scale)):
                np.testing.assert_allclose(arrays[key], value, rtol=1e-12, atol=1e-12)
            payload["training_features"] = prepare_estimation_data(daily, transform).x
            payload["rv_floor"] = np.asarray(transform.rv_floor)
    if payload["feature_names"].tolist() != [
        "z_lag1",
        "log_rv_lag1",
        "asymmetry_lag1",
    ]:
        raise ValueError("RV-RA feature order changed")
    q_center = payload["parameter__q_centering_mean"]
    if not np.allclose(q_center, q_center[0], atol=1e-14, rtol=0.0):
        raise ValueError("q centering constant varies across retained draws")
    return payload


def load_predictive_arrays(record, refit, daily=None):
    with np.load(predictive_array_path(record), allow_pickle=False) as arrays:
        payload = {name: arrays[name].copy() for name in arrays.files}
    names = ("origin_z_history_tail", "origin_log_rv_history_tail",
             "origin_asymmetry_history_tail")
    present = [name in payload for name in names]
    if any(present) and not all(present):
        raise ValueError("incomplete origin-history fields")
    if not any(present):
        if daily is None or "rv_floor" not in refit:
            raise ValueError("compact conditional components require the daily data")
        origin = pd.Timestamp(record["origin_date"])
        tail = daily[daily.date <= origin].tail(22)
        if len(tail) != 22 or tail.date.iloc[-1] != origin:
            raise ValueError("origin history is incomplete or misaligned")
        payload[names[0]] = (tail.return_cc.to_numpy(float) - refit["return_mean"][0]) / refit["return_scale"][0]
        payload[names[1]] = np.log(np.maximum(tail.rv_oc.to_numpy(float), float(refit["rv_floor"])))
        payload[names[2]] = tail.asymmetry.to_numpy(float)
    return payload


def origin_state_inputs(
    *,
    origins: list[str],
    records: Mapping[tuple[str, int], Mapping[str, Any]],
    refit_paths: Mapping[str, Path],
    daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    refit_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for origin in origins:
        record = records[(origin, 5)]
        refit_id = str(record["refit_id"])
        if refit_id not in refit_cache:
            refit_cache[refit_id] = load_refit(refit_paths[refit_id], daily)
        refit = refit_cache[refit_id]
        draw_path = predictive_array_path(record)
        if not draw_path.is_file():
            raise FileNotFoundError(f"RV-RA predictive array is missing: {draw_path}")
        arrays = load_predictive_arrays(record, refit, daily)
        raw_q = float(arrays["origin_asymmetry_history_tail"][-1])
        raw_v = float(arrays["origin_log_rv_history_tail"][-1])
        mean = np.asarray(arrays["feature_mean"], dtype=float)
        scale = np.asarray(arrays["feature_scale"], dtype=float)
        q_center = float(refit["parameter__q_centering_mean"][0])
        q_value = (raw_q - mean[2]) / scale[2] - q_center
        v_value = (raw_v - mean[1]) / scale[1]
        training = np.asarray(refit["training_features"], dtype=float)
        q_threshold = float(np.quantile(training[:, 2] - q_center, 0.75))
        v_threshold = float(np.quantile(training[:, 1], 0.75))
        rows.append(
            {
                "origin_date": origin,
                "refit_id": refit_id,
                "q_centered": q_value,
                "standardized_log_rv": v_value,
                "high_asymmetry": q_value > q_threshold,
                "high_log_rv": v_value > v_threshold,
            }
        )
    return pd.DataFrame(rows).set_index("origin_date").loc[origins], refit_cache


def _counterfactual_from_arrays(
    draws: Mapping[str, np.ndarray],
    refit: Mapping[str, np.ndarray],
    *,
    mode: str,
    realized_asymmetry: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if mode not in {
        "predicted_asymmetry",
        "realized_asymmetry",
        "no_asymmetry_term",
    }:
        raise ValueError(f"unsupported asymmetry comparison: {mode}")
    particle_indices = np.asarray(draws["source_particle_indices"], dtype=int)
    source_ancestry = np.asarray(draws["source_parameter_ancestry"], dtype=int)
    n_paths = particle_indices.size
    horizon = np.asarray(draws["state_innovations"]).shape[1]
    refit_ancestry = np.asarray(refit["ancestry"], dtype=int)
    if source_ancestry.shape != particle_indices.shape:
        raise ValueError("source parameter ancestry is not path-aligned")
    if np.any(particle_indices < 0) or np.any(particle_indices >= len(refit_ancestry)):
        raise ValueError("source particle index is outside the compact refit")
    if len(np.unique(refit_ancestry)) != len(refit_ancestry):
        raise ValueError("compact refit ancestry is not unique")
    order = np.argsort(refit_ancestry)
    sorted_ancestry = refit_ancestry[order]
    positions = np.searchsorted(sorted_ancestry, source_ancestry)
    if np.any(positions >= len(sorted_ancestry)):
        raise ValueError("source parameter ancestry is absent from the compact refit")
    if not np.array_equal(sorted_ancestry[positions], source_ancestry):
        raise ValueError("source parameter ancestry is absent from the compact refit")
    parameter_indices = order[positions]

    parameters = {
        name: np.asarray(refit[f"parameter__{name}"])[parameter_indices]
        for name in (
            "mu",
            "phi",
            "w1",
            "b1",
            "output_weights",
            "centering_constant",
            "gamma_A",
            "q_centering_mean",
        )
    }
    state = np.asarray(draws["origin_baseline_states"], dtype=float).copy()
    z_history = np.broadcast_to(
        np.asarray(draws["origin_z_history_tail"], dtype=float), (n_paths, 22)
    ).copy()
    rv_history = np.broadcast_to(
        np.asarray(draws["origin_log_rv_history_tail"], dtype=float), (n_paths, 22)
    ).copy()
    a_history = np.broadcast_to(
        np.asarray(draws["origin_asymmetry_history_tail"], dtype=float), (n_paths, 22)
    ).copy()
    future_rv = np.asarray(draws["future_log_rv_paths"], dtype=float)
    if mode == "realized_asymmetry":
        if realized_asymmetry is None or np.asarray(realized_asymmetry).shape != (horizon,):
            raise ValueError(
                "realized-asymmetry comparison requires one value per target date"
            )
        future_a = np.broadcast_to(
            np.asarray(realized_asymmetry, dtype=float), (n_paths, horizon)
        )
    else:
        future_a = np.asarray(draws["future_asymmetry_paths"], dtype=float)
    mean = np.asarray(draws["feature_mean"], dtype=float)
    scale = np.asarray(draws["feature_scale"], dtype=float)
    state_innovations = np.asarray(draws["state_innovations"], dtype=float)
    return_innovations = np.asarray(draws["return_innovations"], dtype=float)
    return_scale = float(np.asarray(refit["return_scale"])[0])
    return_mean = float(np.asarray(refit["return_mean"])[0])

    outputs = {
        name: np.empty((n_paths, horizon), dtype=float)
        for name in (
            "baseline_states",
            "latent_states",
            "transition_means",
            "ar_components",
            "observable_corrections",
            "standardized_returns",
            "standardized_variances",
            "raw_returns",
            "raw_variances",
            "raw_conditional_sds",
        )
    }
    for step in range(horizon):
        raw_x = np.column_stack(
            [z_history[:, -1], rv_history[:, -1], a_history[:, -1]]
        )
        x = (raw_x - mean) / scale
        hidden = np.tanh(
            np.einsum("nd,ndh->nh", x[:, :2], parameters["w1"])
            + parameters["b1"]
        )
        rv_correction = (
            np.einsum("nh,nh->n", hidden, parameters["output_weights"])
            - parameters["centering_constant"]
        )
        branch = np.zeros(n_paths)
        if mode != "no_asymmetry_term":
            branch = parameters["gamma_A"] * (
                x[:, 2] - parameters["q_centering_mean"]
            )
        correction = rv_correction + branch
        ar_component = parameters["phi"] * (state - parameters["mu"])
        transition_mean = parameters["mu"] + ar_component
        state = transition_mean + state_innovations[:, step]
        latent = state + correction
        standardized_variance = np.exp(latent)
        standardized_return = np.sqrt(standardized_variance) * return_innovations[:, step]
        raw_variance = return_scale**2 * standardized_variance
        raw_return = return_mean + return_scale * standardized_return

        outputs["baseline_states"][:, step] = state
        outputs["latent_states"][:, step] = latent
        outputs["transition_means"][:, step] = transition_mean
        outputs["ar_components"][:, step] = ar_component
        outputs["observable_corrections"][:, step] = correction
        outputs["standardized_returns"][:, step] = standardized_return
        outputs["standardized_variances"][:, step] = standardized_variance
        outputs["raw_returns"][:, step] = raw_return
        outputs["raw_variances"][:, step] = raw_variance
        outputs["raw_conditional_sds"][:, step] = np.sqrt(raw_variance)

        z_history = np.column_stack([z_history[:, 1:], standardized_return])
        rv_history = np.column_stack([rv_history[:, 1:], future_rv[:, step]])
        a_history = np.column_stack([a_history[:, 1:], future_a[:, step]])
    return outputs


def counterfactual_cell(
    *,
    record: Mapping[str, Any],
    refit: Mapping[str, np.ndarray],
    realized_asymmetry: np.ndarray,
    daily: pd.DataFrame | None = None,
) -> dict[str, float]:
    draw_path = predictive_array_path(record)
    if not draw_path.is_file():
        raise FileNotFoundError(f"RV-RA predictive array is missing: {draw_path}")
    draws = load_predictive_arrays(record, refit, daily)
    results: dict[str, float] = {}
    modes = (
        "predicted_asymmetry",
        "realized_asymmetry",
        "no_asymmetry_term",
    )
    for mode in modes:
        counterfactual = _counterfactual_from_arrays(
            draws,
            refit,
            mode=mode,
            realized_asymmetry=(
                realized_asymmetry if mode == "realized_asymmetry" else None
            ),
        )
        results[mode] = float(
            np.mean(np.sum(counterfactual["raw_variances"], axis=1))
        )
    return results


def loss_matrices(
    *,
    origins: list[str],
    paired: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    rv = paired.pivot(index="origin_date", columns="horizon", values="qlike__RV-NN-SV")
    ra = paired.pivot(index="origin_date", columns="horizon", values="qlike__RV-RA-NN-SV")
    return (
        rv.loc[origins, list(EXPECTED_HORIZONS)].to_numpy(float),
        ra.loc[origins, list(EXPECTED_HORIZONS)].to_numpy(float),
    )


def primary_outcome(rv_losses: np.ndarray, ra_losses: np.ndarray) -> np.ndarray:
    baselines = np.mean(rv_losses, axis=0)
    if np.any(baselines <= 0):
        raise ValueError("RV QLIKE horizon baseline must be positive")
    return np.mean((ra_losses - rv_losses) / baselines, axis=1)


def aggregate_paired_difference(values: np.ndarray, indices: np.ndarray | None = None) -> float:
    matrix = np.asarray(values, dtype=float)
    if indices is not None:
        matrix = matrix[np.asarray(indices, dtype=int)]
    return float(np.mean(np.mean(matrix, axis=0)))


def run_analysis(
    *,
    config: Mapping[str, Any],
    paired_losses: Path,
    nn_losses: Path,
    rv_nn_losses: Path,
    gamma_summary_path: Path,
    rv_ra_root: Path,
    daily_path: Path | None = None,
) -> dict[str, Any]:
    daily = None
    if daily_path is not None:
        from prepare_rv_ra_conditional_inputs import _daily_sp500
        daily = _daily_sp500(daily_path)
    paired, origins = load_paired_losses(paired_losses)
    nn_member_losses = load_member_losses(nn_losses, "NN-SV", origins)
    rv_member_losses = load_member_losses(rv_nn_losses, "RV-NN-SV", origins)
    gamma_summary = load_gamma_summary(gamma_summary_path)
    records, refit_paths = load_rv_ra_records(rv_ra_root)
    expected_cells = {(origin, horizon) for origin in origins for horizon in EXPECTED_HORIZONS}
    if expected_cells - set(records):
        raise ValueError("RV-RA forecast collection lacks one or more common cells")
    states, refits = origin_state_inputs(
        origins=origins, records=records, refit_paths=refit_paths, daily=daily
    )
    rv_array, ra_array = loss_matrices(origins=origins, paired=paired)
    d_point = primary_outcome(rv_array, ra_array)

    comparison_variance = {
        mode: np.empty((EXPECTED_ORIGINS, len(COUNTERFACTUAL_HORIZONS)), dtype=float)
        for mode in (
            "predicted_asymmetry",
            "realized_asymmetry",
            "no_asymmetry_term",
        )
    }
    comparison_qlike = {
        name: values.copy() for name, values in comparison_variance.items()
    }
    for row_index, origin in enumerate(origins):
        for horizon_index, horizon in enumerate(COUNTERFACTUAL_HORIZONS):
            record = records[(origin, horizon)]
            rv_row = rv_member_losses.loc[(origin, horizon)]
            realized_a = np.asarray(
                json.loads(str(rv_row["realized_asymmetry_path_json"])), dtype=float
            )
            record_target_dates = [str(value) for value in record["target_dates"]]
            loss_target_dates = [
                str(value) for value in json.loads(str(rv_row["target_dates_json"]))
            ]
            if (
                realized_a.shape != (horizon,)
                or len(record_target_dates) != horizon
                or record_target_dates != loss_target_dates
            ):
                raise ValueError(
                    f"RV comparison targets are misaligned for {origin}, h={horizon}"
                )
            variances = counterfactual_cell(
                record=record,
                refit=refits[str(record["refit_id"])],
                realized_asymmetry=realized_a,
                daily=daily,
            )
            realized_variance = float(rv_row["realized_variance"])
            for mode in comparison_variance:
                comparison_variance[mode][row_index, horizon_index] = variances[mode]
                comparison_qlike[mode][row_index, horizon_index] = float(
                    qlike(realized_variance, variances[mode])
                )

    two_equal = np.empty((EXPECTED_ORIGINS, len(EXPECTED_HORIZONS)), dtype=float)
    three_equal = np.empty_like(two_equal)
    for i, origin in enumerate(origins):
        for j, horizon in enumerate(EXPECTED_HORIZONS):
            nn_row = nn_member_losses.loc[(origin, horizon)]
            rv_row = rv_member_losses.loc[(origin, horizon)]
            realized = float(rv_row["realized_variance"])
            if not np.isclose(realized, float(nn_row["realized_variance"]), atol=1e-15, rtol=1e-12):
                raise ValueError("member archives disagree on realized variance")
            ra_forecast = float(records[(origin, horizon)]["variance_mean"])
            two_forecast = 0.5 * (
                float(nn_row["variance_forecast"]) + float(rv_row["variance_forecast"])
            )
            three_forecast = (
                float(nn_row["variance_forecast"])
                + float(rv_row["variance_forecast"])
                + ra_forecast
            ) / 3.0
            two_equal[i, j] = float(qlike(realized, two_forecast))
            three_equal[i, j] = float(qlike(realized, three_forecast))

    result = summarize_conditional(config, states, rv_array, ra_array,
                                   comparison_qlike, two_equal, three_equal, gamma_summary)
    result["counterfactual_variances"] = pd.DataFrame([
        dict(origin_date=origin, horizon=h, **{
            mode: float(values[i, j]) for mode, values in comparison_variance.items()})
        for i, origin in enumerate(origins)
        for j, h in enumerate(COUNTERFACTUAL_HORIZONS)
    ])
    return result


def summarize_conditional(config, states, rv_array, ra_array, comparison_qlike,
                          two_equal, three_equal, gamma_summary):
    """Apply the same normalization, regression and bootstrap to either input route."""
    origins = states.index.astype(str).tolist()
    if len(origins) != EXPECTED_ORIGINS or states.index.has_duplicates:
        raise ValueError("conditional records require 241 unique origins")
    for values in (rv_array, ra_array, two_equal, three_equal):
        if values.shape != (EXPECTED_ORIGINS, 3) or not np.isfinite(values).all():
            raise ValueError("conditional loss matrix is invalid")
    d_point = primary_outcome(rv_array, ra_array)
    q = states["q_centered"].to_numpy(float)
    v = states["standardized_log_rv"].to_numpy(float)
    groups = states["refit_id"].to_numpy(str)
    continuous_x = np.column_stack([q, v, q * v])
    continuous_estimate = fit_fixed_effects(d_point, continuous_x, groups)
    high_a = states["high_asymmetry"].to_numpy(float)[:, None]
    high_v = states["high_log_rv"].to_numpy(float)[:, None]
    regime_support = {
        "high_asymmetry": {
            "high_state_count": int(high_a.sum()),
            "low_state_count": int(len(high_a) - high_a.sum()),
            "varying_refit_count": varying_group_count(high_a[:, 0], groups),
        },
        "high_log_rv": {
            "high_state_count": int(high_v.sum()),
            "low_state_count": int(len(high_v) - high_v.sum()),
            "varying_refit_count": varying_group_count(high_v[:, 0], groups),
        },
    }
    regime_a_estimate = (
        float(fit_fixed_effects(d_point, high_a, groups)[0])
        if regime_support["high_asymmetry"]["varying_refit_count"] > 0
        else float("nan")
    )
    regime_v_estimate = (
        float(fit_fixed_effects(d_point, high_v, groups)[0])
        if regime_support["high_log_rv"]["varying_refit_count"] > 0
        else float("nan")
    )

    asymmetry_differences = {
        "predicted_minus_realized_asymmetry": (
            comparison_qlike["predicted_asymmetry"]
            - comparison_qlike["realized_asymmetry"]
        ),
        "predicted_minus_no_asymmetry_term": (
            comparison_qlike["predicted_asymmetry"]
            - comparison_qlike["no_asymmetry_term"]
        ),
        "realized_minus_no_asymmetry_term": (
            comparison_qlike["realized_asymmetry"]
            - comparison_qlike["no_asymmetry_term"]
        ),
    }
    equal_difference = three_equal - two_equal
    point = {
        "beta_asymmetry": float(continuous_estimate[0]),
        "beta_log_rv": float(continuous_estimate[1]),
        "beta_interaction": float(continuous_estimate[2]),
        "high_asymmetry": regime_a_estimate,
        "high_log_rv": regime_v_estimate,
        **{
            name: aggregate_paired_difference(values)
            for name, values in asymmetry_differences.items()
        },
        "equal_weight_three_minus_two": aggregate_paired_difference(equal_difference),
    }

    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    replications = int(bootstrap["replications"])
    draws = {name: np.full(replications, np.nan, dtype=float) for name in point}
    for replication in range(replications):
        indices = circular_block_indices(
            EXPECTED_ORIGINS,
            block_length=int(bootstrap["block_length"]),
            rng=rng,
        )
        d_boot = primary_outcome(rv_array[indices], ra_array[indices])
        beta = fit_fixed_effects(d_boot, continuous_x[indices], groups[indices])
        draws["beta_asymmetry"][replication] = beta[0]
        draws["beta_log_rv"][replication] = beta[1]
        draws["beta_interaction"][replication] = beta[2]
        if np.isfinite(regime_a_estimate):
            draws["high_asymmetry"][replication] = fit_fixed_effects(
                d_boot, high_a[indices], groups[indices]
            )[0]
        if np.isfinite(regime_v_estimate):
            draws["high_log_rv"][replication] = fit_fixed_effects(
                d_boot, high_v[indices], groups[indices]
            )[0]
        for name, values in asymmetry_differences.items():
            draws[name][replication] = aggregate_paired_difference(values, indices)
        draws["equal_weight_three_minus_two"][replication] = aggregate_paired_difference(
            equal_difference, indices
        )

    lower, upper = [float(value) for value in bootstrap["interval_percentiles"]]
    table_rows = []
    for name, estimate in point.items():
        estimated = bool(np.isfinite(estimate))
        support = regime_support.get(name, {})
        table_rows.append(
            {
                "estimand": name,
                "estimate": estimate,
                "ci_lower": (
                    float(np.percentile(draws[name], lower))
                    if estimated
                    else float("nan")
                ),
                "ci_upper": (
                    float(np.percentile(draws[name], upper))
                    if estimated
                    else float("nan")
                ),
                "origin_count": EXPECTED_ORIGINS,
                "status": (
                    "estimated"
                    if estimated
                    else "not_estimable_no_within_refit_high_state_variation"
                ),
                "high_state_count": support.get("high_state_count"),
                "low_state_count": support.get("low_state_count"),
                "varying_refit_count": support.get("varying_refit_count"),
                "interval": (
                    "paired circular moving-block percentile"
                    if estimated
                    else "not_computed"
                ),
            }
        )
    table = pd.DataFrame(table_rows)

    origin_level = states.copy()
    origin_level["D_t"] = d_point
    for j, horizon in enumerate(EXPECTED_HORIZONS):
        origin_level[f"qlike_rv_h{horizon}"] = rv_array[:, j]
        origin_level[f"qlike_rv_ra_h{horizon}"] = ra_array[:, j]
        origin_level[f"equal_weight_two_h{horizon}"] = two_equal[:, j]
        origin_level[f"equal_weight_three_h{horizon}"] = three_equal[:, j]
    for j, horizon in enumerate(COUNTERFACTUAL_HORIZONS):
        for mode in comparison_qlike:
            origin_level[f"qlike_{mode}_h{horizon}"] = comparison_qlike[mode][
                :, j
            ]

    monthly_rows = []
    for refit_id in sorted(states["refit_id"].unique()):
        mask = origin_level["refit_id"] == refit_id
        refit_date_compact = refit_id.rsplit("_", 1)[-1]
        refit_date = pd.Timestamp(refit_date_compact).strftime("%Y-%m-%d")
        if refit_date not in gamma_summary.index:
            raise ValueError(
                f"gamma_A summary lacks monthly re-estimation date {refit_date}"
            )
        gamma = gamma_summary.loc[refit_date]
        monthly_rows.append(
            {
                "refit_id": refit_id,
                "refit_date": refit_date,
                "origin_count": int(mask.sum()),
                "gamma_A_draws": int(gamma["draws"]),
                "gamma_A_mean": float(gamma["gamma_A_mean"]),
                "gamma_A_sd": float(gamma["gamma_A_sd"]),
                "gamma_A_median": float(gamma["gamma_A_median"]),
                "gamma_A_ci_lower": float(gamma["gamma_A_q025"]),
                "gamma_A_ci_upper": float(gamma["gamma_A_q975"]),
                "mean_D_t": float(origin_level.loc[mask, "D_t"].mean()),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    return {
        "status": "complete",
        "origins": origins,
        "table": table,
        "origin_level": origin_level.reset_index(),
        "monthly": monthly,
        "non_estimable_estimands": [
            name for name, estimate in point.items() if not np.isfinite(estimate)
        ],
    }


def write_outputs(result: Mapping[str, Any], config: Mapping[str, Any], output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=False)
    output = config["output"]
    table_path = output_root / str(output["table"])
    monthly_path = output_root / str(output["monthly_panel"])
    figure_path = output_root / str(output["figure"])
    result["table"].to_csv(table_path, index=False)
    result["monthly"].to_csv(monthly_path, index=False)

    monthly = result["monthly"].copy()
    x = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].errorbar(
        x,
        monthly["gamma_A_median"],
        yerr=np.vstack(
            [
                monthly["gamma_A_median"] - monthly["gamma_A_ci_lower"],
                monthly["gamma_A_ci_upper"] - monthly["gamma_A_median"],
            ]
        ),
        fmt="o-",
        color="#1f4e79",
        capsize=3,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"Posterior $\gamma_A$")
    axes[0].set_title("Residual-asymmetry coefficient by re-estimation date")
    axes[1].plot(x, monthly["mean_D_t"], "o-", color="#a64b2a")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"Monthly mean $D_t$")
    axes[1].set_title("RV-RA minus RV normalized QLIKE by re-estimation date")
    axes[1].set_xticks(x, monthly["refit_date"], rotation=45, ha="right")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "description": "Descriptive RV-RA conditional-value results for 2017",
        "origin_count": len(result["origins"]),
        "origin_start": result["origins"][0],
        "origin_end": result["origins"][-1],
        "last_target_date": config["sample"]["last_target_date"],
        "counterfactual_horizons": list(COUNTERFACTUAL_HORIZONS),
        "non_estimable_estimands": result["non_estimable_estimands"],
        "outputs": [table_path.name, monthly_path.name, figure_path.name],
    }
    (output_root / str(output["summary"])).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the descriptive RV-RA conditional-value comparisons."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paired-losses", required=True)
    parser.add_argument("--nn-losses", required=True)
    parser.add_argument("--rv-nn-losses", required=True)
    parser.add_argument("--gamma-summary", required=True)
    parser.add_argument("--rv-ra-root", required=True)
    parser.add_argument("--daily", type=Path,
                        help="lawful daily data required by compact conditional components")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config(config)
    result = run_analysis(
        config=config,
        paired_losses=Path(args.paired_losses).resolve(),
        nn_losses=Path(args.nn_losses).resolve(),
        rv_nn_losses=Path(args.rv_nn_losses).resolve(),
        gamma_summary_path=Path(args.gamma_summary).resolve(),
        rv_ra_root=Path(args.rv_ra_root).resolve(),
        daily_path=args.daily,
    )
    output_root = Path(args.output_root)
    write_outputs(result, config, output_root)
    print(
        json.dumps(
            {
                "output": output_root.name,
                "files": ["estimates.csv", "monthly_panel.csv", "figure.png", "summary.json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
