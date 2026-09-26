#!/usr/bin/env python3
"""Prepare the observable 2017 inputs for the RV-RA conditional analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bnsv.data_contract import read_daily_frame
from bnsv.evaluation import qlike
from bnsv.forecast_features import (
    FeatureTransform,
    fit_feature_transform,
    prepare_estimation_data,
)


MODELS = ("NN-SV", "RV-NN-SV", "RV-RA-NN-SV")
HORIZONS = (1, 5, 10)
ORIGIN_START = "2017-01-03"
ORIGIN_END = "2017-12-14"
EXPECTED_ORIGINS = 241


def _forecast_panel(path: str | Path, model_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "mature_date",
        "variance_forecast",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"forecast panel lacks columns: {sorted(missing)}")
    selected = frame[
        (frame["market"].astype(str) == "SP500")
        & (frame["model_id"].astype(str) == model_id)
    ].copy()
    selected["origin_date"] = pd.to_datetime(
        selected["origin_date"], errors="raise"
    ).dt.normalize()
    selected["mature_date"] = pd.to_datetime(
        selected["mature_date"], errors="raise"
    ).dt.normalize()
    selected["horizon"] = pd.to_numeric(
        selected["horizon"], errors="raise"
    ).astype(int)
    selected = selected[
        selected["origin_date"].between(ORIGIN_START, ORIGIN_END)
        & selected["horizon"].isin(HORIZONS)
    ]
    if selected.duplicated(["origin_date", "horizon"]).any():
        raise ValueError(f"{model_id} forecast panel contains duplicate cells")
    return selected.sort_values(["origin_date", "horizon"]).reset_index(
        drop=True
    )


def _daily_sp500(path: str | Path) -> pd.DataFrame:
    frame = read_daily_frame(path)
    selected = frame[frame["market"].astype(str) == "SP500"].copy()
    required = {"date", "return_cc", "rv_oc", "rv_cc", "asymmetry"}
    missing = required - set(selected)
    if missing or selected.empty:
        raise ValueError(f"daily data lack S&P 500 fields: {sorted(missing)}")
    selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
    selected = selected.sort_values("date").reset_index(drop=True)
    if selected["date"].duplicated().any():
        raise ValueError("daily S&P 500 dates are duplicated")
    return selected


def _loss_panel(
    forecasts: pd.DataFrame, daily: pd.DataFrame, model_id: str
) -> pd.DataFrame:
    positions = {
        pd.Timestamp(date): position
        for position, date in enumerate(daily["date"])
    }
    rows: list[dict[str, object]] = []
    for forecast in forecasts.itertuples(index=False):
        origin = pd.Timestamp(forecast.origin_date)
        horizon = int(forecast.horizon)
        if origin not in positions:
            raise ValueError(f"daily data lack forecast origin {origin.date()}")
        target = daily.iloc[
            positions[origin] + 1 : positions[origin] + horizon + 1
        ]
        if len(target) != horizon:
            raise ValueError(f"target window is incomplete at {origin.date()}")
        target_dates = [str(pd.Timestamp(value).date()) for value in target["date"]]
        if pd.Timestamp(forecast.mature_date) != pd.Timestamp(target["date"].iloc[-1]):
            raise ValueError(f"mature date differs at {origin.date()}, h={horizon}")
        realized = float(target["rv_cc"].sum())
        variance = float(forecast.variance_forecast)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("variance forecasts must be finite and positive")
        rows.append(
            {
                "origin_date": str(origin.date()),
                "horizon": horizon,
                "model_id": model_id,
                "variance_forecast": variance,
                "realized_variance": realized,
                "qlike": float(qlike(realized, variance)),
                "target_dates_json": json.dumps(
                    target_dates, separators=(",", ":")
                ),
                "realized_asymmetry_path_json": json.dumps(
                    target["asymmetry"].to_numpy(float).tolist(),
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def _origin_states(
    daily: pd.DataFrame,
    origins: list[str],
    refit_schedule: str | Path,
) -> pd.DataFrame:
    schedule = pd.read_csv(refit_schedule)
    required = {"market", "refit_date"}
    if required - set(schedule):
        raise ValueError("refit schedule lacks market or refit_date")
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            schedule.loc[
                schedule["market"].astype(str) == "SP500", "refit_date"
            ],
            errors="raise",
        )
    ).normalize()
    if len(dates) != 12 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("the 2017 conditional analysis requires 12 ordered refits")
    cache: dict[pd.Timestamp, tuple[FeatureTransform, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for raw_origin in origins:
        origin = pd.Timestamp(raw_origin).normalize()
        eligible = dates[dates <= origin]
        if eligible.empty:
            raise ValueError(f"no refit precedes {raw_origin}")
        refit = pd.Timestamp(eligible[-1])
        if refit not in cache:
            transform = fit_feature_transform(
                daily, model_id="RV-RA-NN-SV", training_end=refit
            )
            estimation = prepare_estimation_data(daily, transform)
            cache[refit] = (transform, estimation.x)
        transform, training = cache[refit]
        state = daily.loc[daily["date"] == origin, ["rv_oc", "asymmetry"]]
        if len(state) != 1 or state.isna().any(axis=None):
            raise ValueError(f"conditional state is unavailable at {raw_origin}")
        q_center = float(np.mean(training[:, 2]))
        raw_q = float(state.iloc[0]["asymmetry"])
        raw_log_rv = float(
            np.log(max(float(state.iloc[0]["rv_oc"]), transform.rv_floor))
        )
        q_value = float(
            (raw_q - transform.feature_mean[2]) / transform.feature_scale[2]
            - q_center
        )
        v_value = float(
            (raw_log_rv - transform.feature_mean[1])
            / transform.feature_scale[1]
        )
        rows.append(
            {
                "origin_date": raw_origin,
                "refit_id": f"SP500_RV-RA-NN-SV_{refit:%Y%m%d}",
                "q_centered": q_value,
                "standardized_log_rv": v_value,
                "high_asymmetry": bool(
                    q_value > np.quantile(training[:, 2] - q_center, 0.75)
                ),
                "high_log_rv": bool(
                    v_value > np.quantile(training[:, 1], 0.75)
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_inputs(
    *,
    data_path: str | Path,
    nn_forecasts: str | Path,
    rv_nn_forecasts: str | Path,
    rv_ra_forecasts: str | Path,
    refit_schedule: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    daily = _daily_sp500(data_path)
    paths = dict(
        zip(MODELS, (nn_forecasts, rv_nn_forecasts, rv_ra_forecasts), strict=True)
    )
    losses = {
        model_id: _loss_panel(
            _forecast_panel(path, model_id), daily, model_id
        )
        for model_id, path in paths.items()
    }
    cell_sets = {
        model_id: set(zip(frame["origin_date"], frame["horizon"], strict=True))
        for model_id, frame in losses.items()
    }
    common = set.intersection(*cell_sets.values())
    origins = sorted({origin for origin, horizon in common if horizon in HORIZONS})
    expected = {(origin, horizon) for origin in origins for horizon in HORIZONS}
    if len(origins) != EXPECTED_ORIGINS or common != expected:
        raise ValueError("forecast panels do not contain the stated 241-by-3 sample")
    for model_id in MODELS:
        losses[model_id] = losses[model_id][
            losses[model_id].apply(
                lambda row: (row["origin_date"], int(row["horizon"])) in expected,
                axis=1,
            )
        ].sort_values(["origin_date", "horizon"])
    rv = losses["RV-NN-SV"].loc[:, ["origin_date", "horizon", "qlike"]].rename(
        columns={"qlike": "qlike__RV-NN-SV"}
    )
    ra = losses["RV-RA-NN-SV"].loc[:, ["origin_date", "horizon", "qlike"]].rename(
        columns={"qlike": "qlike__RV-RA-NN-SV"}
    )
    paired = rv.merge(ra, on=["origin_date", "horizon"], validate="one_to_one")
    states = _origin_states(daily, origins, refit_schedule)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_losses": destination / "paired_losses.csv",
        "nn_losses": destination / "nn_sv_losses.csv",
        "rv_nn_losses": destination / "rv_nn_sv_losses.csv",
        "rv_ra_losses": destination / "rv_ra_nn_sv_losses.csv",
        "origin_states": destination / "origin_states.csv",
    }
    paired.to_csv(outputs["paired_losses"], index=False)
    losses["NN-SV"].to_csv(outputs["nn_losses"], index=False)
    losses["RV-NN-SV"].to_csv(outputs["rv_nn_losses"], index=False)
    losses["RV-RA-NN-SV"].to_csv(outputs["rv_ra_losses"], index=False)
    states.to_csv(outputs["origin_states"], index=False)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--nn-forecasts", required=True)
    parser.add_argument("--rv-nn-forecasts", required=True)
    parser.add_argument("--rv-ra-forecasts", required=True)
    parser.add_argument("--refit-schedule", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = prepare_inputs(
        data_path=args.data,
        nn_forecasts=args.nn_forecasts,
        rv_nn_forecasts=args.rv_nn_forecasts,
        rv_ra_forecasts=args.rv_ra_forecasts,
        refit_schedule=args.refit_schedule,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: path.name for name, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
