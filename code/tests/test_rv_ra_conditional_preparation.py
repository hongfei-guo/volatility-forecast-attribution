from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/scripts/prepare_rv_ra_conditional_inputs.py"
SPEC = importlib.util.spec_from_file_location("rv_ra_preparation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_observable_conditional_inputs_are_prepared_from_daily_data_and_panels(
    tmp_path: Path,
) -> None:
    schedule_dates = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2017-01-03",
                "2017-02-01",
                "2017-03-01",
                "2017-04-03",
                "2017-05-01",
                "2017-06-01",
                "2017-07-03",
                "2017-08-01",
                "2017-09-01",
                "2017-10-02",
                "2017-11-01",
                "2017-12-01",
            ]
        )
    )
    candidates = list(pd.bdate_range("2017-01-03", "2017-12-14"))
    removable = [date for date in candidates if date not in set(schedule_dates)]
    removed = set(removable[:7])
    origins = pd.DatetimeIndex([date for date in candidates if date not in removed])
    assert len(origins) == 241
    history = pd.bdate_range(end="2016-12-30", periods=80)
    targets = pd.bdate_range("2017-12-15", periods=10)
    dates = history.append(origins).append(targets)
    index = np.arange(len(dates), dtype=float)
    rv_oc = np.exp(-9.0 + 0.1 * np.sin(index / 11))
    asymmetry = 0.5 * np.sin(index / 7)
    daily = pd.DataFrame(
        {
            "date": dates,
            "market": "SP500",
            "return_cc": 0.01 * np.sin(index / 5),
            "rv_oc": rv_oc,
            "rv_cc": rv_oc + 1e-6,
            "asymmetry": asymmetry,
        }
    )
    data_path = tmp_path / "daily.parquet"
    daily.to_parquet(data_path, index=False)
    positions = {date: position for position, date in enumerate(dates)}
    panel_paths: dict[str, Path] = {}
    for model_index, model_id in enumerate(MODULE.MODELS, start=1):
        rows = []
        for origin in origins:
            position = positions[origin]
            for horizon in MODULE.HORIZONS:
                rows.append(
                    {
                        "market": "SP500",
                        "model_id": model_id,
                        "origin_date": str(origin.date()),
                        "horizon": horizon,
                        "mature_date": str(dates[position + horizon].date()),
                        "variance_forecast": model_index * horizon * 1e-4,
                    }
                )
        path = tmp_path / f"{model_id}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        panel_paths[model_id] = path
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        {"market": "SP500", "refit_date": schedule_dates.strftime("%Y-%m-%d")}
    ).to_csv(schedule_path, index=False)

    outputs = MODULE.prepare_inputs(
        data_path=data_path,
        nn_forecasts=panel_paths["NN-SV"],
        rv_nn_forecasts=panel_paths["RV-NN-SV"],
        rv_ra_forecasts=panel_paths["RV-RA-NN-SV"],
        refit_schedule=schedule_path,
        output_dir=tmp_path / "prepared",
    )

    paired = pd.read_csv(outputs["paired_losses"])
    states = pd.read_csv(outputs["origin_states"])
    assert len(paired) == 723
    assert paired.groupby("horizon")["origin_date"].nunique().to_dict() == {
        1: 241,
        5: 241,
        10: 241,
    }
    assert len(states) == 241
    assert states["refit_id"].nunique() == 12
    assert states[["q_centered", "standardized_log_rv"]].notna().all().all()
