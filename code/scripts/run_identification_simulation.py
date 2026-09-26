#!/usr/bin/env python3
"""Run one task from the identification simulation."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from bnsv.identification_fitting import fit_with_sampling_sequence  # noqa: E402
from bnsv.identification_initialization import make_prior_centered_nn_inits  # noqa: E402
from bnsv.identification_simulation import (  # noqa: E402
    canonical_json_bytes,
    derived_seed,
    standardized_student_t,
    write_npz,
)
from bnsv.identification_stan import nn_sv_stan_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "design" / "identification.yaml")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--dry-run-all", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset(path_value: str, preparation: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        raise ValueError("dataset paths must be relative to the preparation directory")
    resolved = (preparation / path).resolve()
    if not resolved.is_relative_to(preparation.resolve()):
        raise ValueError("dataset path leaves the preparation directory")
    return resolved


def validate_preparation(config: dict[str, Any], preparation: Path) -> list[dict[str, Any]]:
    datasets = load_json(preparation / "datasets.json")
    plan = load_json(preparation / "tasks.json")
    expected = 2 * int(config["replications"]["per_dgp"])
    if datasets.get("records_count") != expected or plan.get("task_count") != expected:
        raise ValueError("prepared dataset and task counts differ from the design")
    records = {record["path"]: record for record in datasets["records"]}
    expected_pairs = {
        (str(dgp_id), replication_id)
        for dgp_id in config["dgp"]["ids"]
        for replication_id in range(1, int(config["replications"]["per_dgp"]) + 1)
    }
    observed_pairs = {
        (str(task["dgp_id"]), int(task["replication_id"])) for task in plan["tasks"]
    }
    if observed_pairs != expected_pairs:
        raise ValueError("prepared tasks do not match the DGP and replication design")
    for task in plan["tasks"]:
        record = records.get(task["dataset_path"])
        if record is None:
            raise ValueError("task has no corresponding dataset")
        if (
            str(record["dgp_id"]) != str(task["dgp_id"])
            or int(record["replication_id"]) != int(task["replication_id"])
        ):
            raise ValueError("task and dataset labels differ")
        path = resolve_dataset(task["dataset_path"], preparation)
        if not path.is_file():
            raise ValueError(f"dataset is missing: {task['dataset_path']}")
        with np.load(path, allow_pickle=False) as payload:
            if (
                str(payload["dgp_id"][0]) != str(task["dgp_id"])
                or int(payload["replication_id"][0])
                != int(task["replication_id"])
            ):
                raise ValueError(f"dataset labels differ: {task['dataset_path']}")
    return plan["tasks"]


def task_stan_data(
    task: dict[str, Any],
    preparation: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    model = config["model"]
    if model.get("id") != "NN-SV" or int(model.get("width", -1)) != 5:
        raise ValueError("every task must use the reported H=5 NN-SV specification")
    dataset = resolve_dataset(task["dataset_path"], preparation)
    with np.load(dataset, allow_pickle=False) as payload:
        n = int(payload["estimation_rows"][0])
        z = np.asarray(payload["z"][:n], dtype=float)
        x = np.asarray(payload["X"][:n], dtype=float)
    data = nn_sv_stan_data(
        z=z,
        x=x,
        width=5,
        prior=model["prior"],
    )
    inits = make_prior_centered_nn_inits(
        n=int(data["N"]),
        d=int(data["D"]),
        width=int(data["H"]),
        phi_logit_loc=float(data["phi_logit_loc"]),
        phi_logit_scale=float(data["phi_logit_scale"]),
        chains=int(config["mcmc"]["chains"]),
    )
    stan_file = (PROJECT / model["stan_file"]).resolve()
    if not stan_file.is_file():
        raise ValueError(f"Stan model is missing: {stan_file}")
    return data, stan_file, inits


def _network_output(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "...h,...h->...",
        np.tanh(np.einsum("...d,...dh->...h", x, weights) + bias),
        output,
    )


def compact_successful_fit(
    fit: Any,
    *,
    task: dict[str, Any],
    dataset_path: Path,
    destination: Path,
    config: dict[str, Any],
) -> str:
    common = ("mu", "phi", "sigma_h", "sigma_eta", "nu")
    compact: dict[str, np.ndarray] = {
        name: np.asarray(fit.stan_variable(name)) for name in common
    }
    h = np.asarray(fit.stan_variable("h"))
    correction = np.asarray(fit.stan_variable("correction"))
    compact["correction_share"] = np.var(correction, axis=1) / np.var(h, axis=1)
    compact["h_last"] = h[:, -1]
    compact["correction_mean"] = np.mean(correction, axis=1)
    compact["weights"] = np.asarray(fit.stan_variable("W1"))
    compact["bias"] = np.asarray(fit.stan_variable("b1"))
    compact["output_weight"] = np.asarray(fit.stan_variable("output_weight"))
    compact["centering"] = np.asarray(fit.stan_variable("centering_mean"))
    compact["state_last"] = np.asarray(fit.stan_variable("b"))[:, -1]
    for name, values in fit.method_variables().items():
        compact[f"sampler__{name.rstrip('_')}"] = np.asarray(values)

    with np.load(dataset_path, allow_pickle=False) as dataset:
        n = int(dataset["estimation_rows"][0])
        next_x = float(dataset["X"][n, 0])
    c_next = _network_output(
        np.full((compact["mu"].size, 1), next_x),
        compact["weights"],
        compact["bias"],
        compact["output_weight"],
    ) - compact["centering"]
    mean_h_next = compact["mu"] + compact["phi"] * (
        compact["state_last"] - compact["mu"]
    ) + c_next
    compact["exact_one_step_predictive_variance"] = np.exp(
        mean_h_next + 0.5 * np.square(compact["sigma_eta"])
    )

    path_count = int(config["prediction"]["paths"])
    horizons = np.asarray(config["prediction"]["horizons"], dtype=int)
    if path_count > compact["mu"].size or horizons.tolist() != [1, 5, 10]:
        raise ValueError("prediction settings differ from the reported simulation")
    rng = np.random.default_rng(
        derived_seed(int(task["replication_id"]), f"forecast_task_{int(task['task_id']):03d}")
    )
    draw_index = rng.choice(compact["mu"].size, size=path_count, replace=False)
    mu = compact["mu"][draw_index]
    phi = compact["phi"][draw_index]
    sigma_eta = compact["sigma_eta"][draw_index]
    nu = compact["nu"][draw_index]
    state = compact["state_last"][draw_index].copy()
    weights = compact["weights"][draw_index]
    bias = compact["bias"][draw_index]
    output_weight = compact["output_weight"][draw_index]
    centering = compact["centering"][draw_index]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        n = int(dataset["estimation_rows"][0])
        lag_z = np.full(path_count, float(dataset["z"][n - 1]))
        withheld = np.asarray(dataset["z"][n : n + int(horizons[-1])], dtype=float)
    alpha = rng.standard_normal((path_count, int(horizons[-1])))
    eps = np.column_stack(
        [standardized_student_t(rng, float(value), int(horizons[-1])) for value in nu]
    ).T
    cumulative = np.zeros(path_count)
    predictive = np.empty((path_count, len(horizons)), dtype=float)
    positions = {int(horizon): i for i, horizon in enumerate(horizons)}
    for step in range(1, int(horizons[-1]) + 1):
        state = mu + phi * (state - mu) + sigma_eta * alpha[:, step - 1]
        correction_next = _network_output(
            lag_z[:, None], weights, bias, output_weight
        ) - centering
        lag_z = np.exp(0.5 * (state + correction_next)) * eps[:, step - 1]
        cumulative += lag_z
        if step in positions:
            predictive[:, positions[step]] = cumulative
    compact["forecast_draw_index"] = draw_index
    compact["forecast_horizons"] = horizons
    compact["forecast_cumulative_standardized_return"] = predictive
    compact["forecast_truth_cumulative_standardized_return"] = np.asarray(
        [np.sum(withheld[:horizon]) for horizon in horizons], dtype=float
    )

    compact_path = destination / "posterior_compact.npz"
    write_npz(compact_path, compact, compressed=False)
    with np.load(compact_path, allow_pickle=False) as restored:
        if set(restored.files) != set(compact):
            raise RuntimeError("compact posterior round-trip changed variables")
        for name, expected in compact.items():
            if not np.array_equal(restored[name], expected, equal_nan=True):
                raise RuntimeError(f"compact posterior round-trip changed {name}")
    csv_files = getattr(getattr(fit, "runset", None), "csv_files", ()) or ()
    for item in csv_files:
        csv_path = Path(item).resolve()
        if not csv_path.is_relative_to(destination.resolve()):
            raise RuntimeError("CmdStan CSV path leaves the task directory")
        csv_path.unlink()
    return compact_path.name


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("cmdstanpy", "numpy", "scipy", "xarray", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        from cmdstanpy import cmdstan_version

        cmdstan = str(cmdstan_version())
    except Exception:
        cmdstan = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cmdstan": cmdstan,
        "packages": packages,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    preparation = args.preparation.resolve()
    tasks = validate_preparation(config, preparation)
    if args.dry_run_all:
        for task in tasks:
            task_stan_data(task, preparation, config)
        print(json.dumps({"status": "PASS", "tasks": len(tasks)}, indent=2))
        return 0
    if args.task_id is None:
        raise SystemExit("--task-id is required unless --dry-run-all is used")
    task = next((item for item in tasks if int(item["task_id"]) == args.task_id), None)
    if task is None:
        raise SystemExit("unknown task id")
    output = args.output.resolve() / f"task_{args.task_id:03d}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty task directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset(task["dataset_path"], preparation)
    data, stan_file, inits = task_stan_data(task, preparation, config)
    try:
        fit, sampling_runs = fit_with_sampling_sequence(
            stan_file=stan_file,
            data=data,
            seed=derived_seed(int(task["replication_id"]), f"fit_task_{args.task_id:03d}"),
            chains=int(config["mcmc"]["chains"]),
            mcmc=config["mcmc"],
            output_dir=output / "cmdstan",
            inits=inits,
            show_progress=args.show_progress,
        )
    except RuntimeError as exc:
        runs_path = output / "cmdstan" / "sampling_runs.json"
        if not runs_path.is_file():
            raise
        sampling_runs = load_json(runs_path)
        fit_result = {
            "fit_outcome": "unsuccessful",
            "task": task,
            "sampling_runs": sampling_runs,
            "reason": str(exc),
            "runtime": runtime_environment(),
        }
        (output / "fit_result.json").write_bytes(canonical_json_bytes(fit_result))
        print(json.dumps({"fit_outcome": "unsuccessful", "task_id": args.task_id}, indent=2))
        return 2
    posterior_file = compact_successful_fit(
        fit,
        task=task,
        dataset_path=dataset_path,
        destination=output,
        config=config,
    )
    fit_result = {
        "fit_outcome": "successful",
        "task": task,
        "sampling_runs": sampling_runs,
        "posterior_file": posterior_file,
        "runtime": runtime_environment(),
    }
    (output / "fit_result.json").write_bytes(canonical_json_bytes(fit_result))
    print(json.dumps({"fit_outcome": "successful", "task_id": args.task_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
