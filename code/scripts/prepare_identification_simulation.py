#!/usr/bin/env python3
"""Generate the synthetic datasets used by the identification simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml
from scipy.interpolate import CubicSpline


PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from bnsv.identification_simulation import (  # noqa: E402
    ReferenceCurve,
    build_fit_plan,
    canonical_json_bytes,
    dataset_record,
    dgp_parameters_dict,
    exact_design_constants,
    parameters_from_config,
    simulate_replication,
    solve_amplitude,
    write_npz,
)
from bnsv.file_identity import sha256_file  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "design" / "identification.yaml")
    parser.add_argument("--reference-curve", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_reference_curve(path: Path, expected_sha256: str) -> ReferenceCurve:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("reference curve is missing or differs from the reported input")
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("reference curve has invalid nodes")
    if abs(float(np.mean(y))) >= 1e-14:
        raise ValueError("reference curve is not centered")
    spline = CubicSpline(x, y, bc_type="natural")
    return ReferenceCurve(
        x=x,
        y=y,
        left_slope=float(spline(x[0], 1)),
        right_slope=float(spline(x[-1], 1)),
        source_records=(),
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    datasets_dir = output / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    curve_spec = config["dgp"]["reference_curve"]
    curve_path = (
        args.reference_curve.resolve()
        if args.reference_curve
        else (PROJECT / curve_spec["file"]).resolve()
    )
    curve = load_reference_curve(curve_path, str(curve_spec["sha256"]))
    shutil.copyfile(curve_path, output / "reference_curve.npz")

    params = parameters_from_config(config)
    amplitude = solve_amplitude(
        params,
        curve,
        replications=int(config["dgp"]["amplitude_calibration_replications"]),
        seed_namespace="calibration",
    )
    amplitude["dgp_parameters"] = dgp_parameters_dict(params)
    write_json(output / "amplitude_calibration.json", amplitude)

    replications = int(config["replications"]["per_dgp"])
    records: list[dict[str, object]] = []
    for dgp_id in config["dgp"]["ids"]:
        for replication_id in range(1, replications + 1):
            payload = simulate_replication(
                params,
                curve,
                replication_id=replication_id,
                dgp_id=str(dgp_id),
                amplitude=float(amplitude["amplitude"]),
                seed_namespace="experiment",
            )
            path = datasets_dir / f"{dgp_id}_rep{replication_id:03d}.npz"
            write_npz(path, payload, compressed=True)
            records.append(dataset_record(path, payload, root=output))

    tasks = build_fit_plan(records, replications=replications)
    write_json(
        output / "datasets.json",
        {
            "records": records,
            "records_count": len(records),
            "dgp_counts": {
                dgp: sum(record["dgp_id"] == dgp for record in records)
                for dgp in config["dgp"]["ids"]
            },
            "paired_common_random_numbers": True,
        },
    )
    write_json(output / "tasks.json", {"tasks": tasks, "task_count": len(tasks)})
    constants = exact_design_constants(replications)
    write_json(output / "design_constants.json", constants)

    checks = {
        "reference_grid_strictly_increasing": bool(np.all(np.diff(curve.x) > 0)),
        "reference_grid_centered": abs(float(np.mean(curve.y))) < 1e-14,
        "amplitude_solver_monotone": bool(amplitude["monotone_on_evaluated_points"]),
        "amplitude_share_error_below_1e_12": float(amplitude["absolute_error"]) < 1e-12,
        "dataset_count": len(records) == 2 * replications,
        "task_count": len(tasks) == 2 * replications,
        "dgp0_share_zero": all(
            float(record["correction_share"]) == 0.0
            for record in records
            if record["dgp_id"] == "DGP-0"
        ),
        "paired_state_streams": all(
            next(r for r in records if r["dgp_id"] == "DGP-N" and r["replication_id"] == i)["state_seed"]
            == next(r for r in records if r["dgp_id"] == "DGP-0" and r["replication_id"] == i)["state_seed"]
            for i in range(1, replications + 1)
        ),
        "calibration_stream_is_disjoint": bool(amplitude["seed_banks_disjoint_by_construction"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"identification preparation failed: {checks}")
    write_json(output / "checks.json", checks)
    print(json.dumps({"output": str(output), "datasets": len(records), "tasks": len(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
