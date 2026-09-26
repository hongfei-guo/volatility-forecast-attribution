#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bnsv.scoring_components import load_components, ensemble_pit

from bnsv.analytic_tail import (
    gaussian_mixture_var_es,
    standardized_t_mixture_var_es,
)
from bnsv.data_contract import (
    EVALUATION_PROXY_COLUMNS,
    read_daily_frame,
)
from bnsv.evaluation import (
    crps_ensemble,
    finite_mixture_gaussian_cdf,
    finite_mixture_gaussian_log_density,
    finite_mixture_standardized_t_cdf,
    finite_mixture_standardized_t_log_density,
    fz0_var_es_score,
    interval_diagnostic,
    quantile_loss,
    qlike,
    randomized_pit,
    variance_mse,
)
from bnsv.sample_alignment import (
    gap_metadata,
    load_calendar_audit,
    missing_sessions_by_market,
    write_loss_archive,
)


def _producer_variance_mean(record: dict) -> tuple[float, str] | None:
    """Read an exact producer-supplied variance mean when present."""
    for name in ("variance_forecast", "variance_mean"):
        if name not in record or record[name] is None:
            continue
        if np.isscalar(record[name]) and bool(pd.isna(record[name])):
            continue
        value = float(record[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be finite and strictly positive when supplied"
            )
        method = record.get("variance_mean_method")
        if method is None or (np.isscalar(method) and bool(pd.isna(method))):
            method = name
        return value, str(method)
    return None


def _variance_mean_for_evaluation(
    record: dict,
    variance_draws: np.ndarray,
) -> tuple[float, str]:
    """Use the producer-supplied exact mean when available."""
    draw_values = np.asarray(variance_draws, dtype=float)
    if draw_values.ndim != 1 or draw_values.size == 0 or np.any(~np.isfinite(draw_values)):
        raise ValueError("integrated variance draws must be a finite nonempty vector")
    supplied = _producer_variance_mean(record)
    if supplied is None:
        variance_mean = float(np.mean(draw_values))
        source = "predictive_draw_mean"
    else:
        variance_mean, source = supplied
        if source == "mean_integrated_path_variance" and not np.isclose(
            variance_mean,
            float(np.mean(draw_values)),
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "mean_integrated_path_variance does not match predictive components"
            )
    if not np.isfinite(variance_mean) or variance_mean <= 0:
        raise ValueError(
            "probabilistic forecast variance_mean must be finite and strictly positive"
        )
    return variance_mean, source


def _tail_fields_from_saved_mixture(
    *,
    horizon: int,
    return_distribution: str,
    raw_return_locations: np.ndarray,
    raw_conditional_sds: np.ndarray,
    degrees_of_freedom: np.ndarray | None,
    realized_return: float,
    risk_levels: tuple[float, ...] = (0.05, 0.01),
) -> dict[str, float | bool]:
    """Evaluate h=1 VaR and ES from the saved finite mixture."""
    if horizon != 1:
        return {}
    locations = np.asarray(raw_return_locations, dtype=float)
    conditional_sds = np.asarray(raw_conditional_sds, dtype=float)
    if (
        locations.ndim != 2
        or locations.shape[1] != 1
        or conditional_sds.shape != locations.shape
    ):
        raise ValueError("h=1 mixture locations and conditional SDs must align")

    fields: dict[str, float | bool] = {}
    for alpha in risk_levels:
        if return_distribution == "gaussian":
            tail = gaussian_mixture_var_es(
                alpha,
                locations[:, 0],
                conditional_sds[:, 0],
            )
        elif return_distribution == "variance_standardized_student_t":
            if degrees_of_freedom is None:
                raise ValueError("Student-t mixture requires degrees of freedom")
            tail = standardized_t_mixture_var_es(
                alpha,
                locations[:, 0],
                conditional_sds[:, 0],
                np.asarray(degrees_of_freedom, dtype=float),
            )
        else:
            raise ValueError(f"unsupported return distribution: {return_distribution}")

        label = f"{int(round(100 * alpha)):02d}"
        var = float(tail.var)
        es = float(tail.es)
        fields[f"var_{label}"] = var
        fields[f"es_{label}"] = es
        fields[f"var_{label}_exceedance"] = bool(realized_return < var)
        fields[f"var_{label}_quantile_loss"] = quantile_loss(
            realized_return, var, alpha
        )
        fields[f"var_es_{label}_fz0"] = (
            fz0_var_es_score(realized_return, var, es, alpha) if es < 0 else np.nan
        )
    return fields


def _load_forecast_records(paths: list[str | Path]) -> list[dict]:
    """Load one or more generated Parquet panels or JSONL record files."""
    records: list[dict] = []
    for source in paths:
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            records.extend(pd.read_parquet(path).to_dict("records"))
            continue
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not records:
        raise ValueError("forecast inputs contain no records")
    return records


def _predictive_file(record: dict) -> str | None:
    value = record.get("predictive_file")
    if value is None:
        return None
    if not isinstance(value, str) and np.isscalar(value) and bool(pd.isna(value)):
        return None
    if not isinstance(value, str):
        raise ValueError("predictive_file must be a relative path")
    path = value.strip()
    if not path:
        return None
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "predictive_file must be relative to forecast-root without parent traversal"
        )
    return relative.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align trading-day targets and build variance and return losses"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--forecasts",
        "--forecast-records",
        dest="forecast_records",
        required=True,
        nargs="+",
        help="one or more generated Parquet panels or JSONL record files",
    )
    parser.add_argument(
        "--forecast-root",
        required=True,
        help="directory containing the relative predictive_file entries",
    )
    parser.add_argument(
        "--calendar-audit",
        required=True,
        help="report produced by audit_calendars.py",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    records = _load_forecast_records(args.forecast_records)
    data = read_daily_frame(args.data)
    calendar_audit = load_calendar_audit(args.calendar_audit)
    missing_sessions = missing_sessions_by_market(calendar_audit)
    interval_levels = (0.50, 0.90, 0.95)
    variance_interval_levels = (0.50, 0.90, 0.95)
    risk_levels = (0.05, 0.01)
    rng = np.random.default_rng(args.seed)
    output_rows = []
    market_frames = {str(market): group.sort_values('date').reset_index(drop=True)
                     for market, group in data.groupby('market')}
    origin_positions = {}
    for market, frame in market_frames.items():
        if frame.date.duplicated().any():
            raise ValueError('daily data contain duplicate market dates')
        origin_positions[market] = {pd.Timestamp(date): i for i, date in enumerate(frame.date)}
    for record in records:
        market_data = market_frames[record['market']]
        origin = pd.Timestamp(record['origin_date'])
        if origin not in origin_positions[record['market']]:
            raise ValueError(f'origin not uniquely found: {origin}')
        pos = origin_positions[record['market']][origin]
        h = int(record["horizon"])
        target = market_data.iloc[pos + 1 : pos + h + 1]
        if len(target) != h:
            raise ValueError("incomplete forecast target")
        missing_proxies = set(EVALUATION_PROXY_COLUMNS) - set(target.columns)
        if missing_proxies:
            raise ValueError(
                "evaluation data are missing required variance proxies: "
                f"{sorted(missing_proxies)}"
            )
        proxy_paths = {
            proxy: target[proxy].to_numpy(dtype=float)
            for proxy in EVALUATION_PROXY_COLUMNS
        }
        if any(
            np.any(~np.isfinite(values)) or np.any(values < 0)
            for values in proxy_paths.values()
        ):
            raise ValueError("variance-proxy target paths must be finite and non-negative")
        realized_by_proxy = {
            proxy: float(np.sum(values)) for proxy, values in proxy_paths.items()
        }
        if any(value <= 0 for value in realized_by_proxy.values()):
            raise ValueError("integrated variance-proxy targets must be strictly positive")
        realized_variance = realized_by_proxy["rv_cc"]
        realized_return = float(target["return_cc"].sum())
        target_dates = [str(value.date()) for value in target["date"].tolist()]
        gap_fields = gap_metadata(
            market=str(record["market"]),
            origin_date=origin,
            target_dates=target_dates,
            previous_observed_date=(
                market_data.iloc[pos - 1]["date"] if pos > 0 else None
            ),
            missing_by_market=missing_sessions,
        )
        if "target_dates" in record and list(record["target_dates"]) != target_dates:
            raise ValueError("recorded target dates do not match evaluation data")
        row = {
            **record,
            **gap_fields,
            "target_start": str(target.iloc[0]["date"].date()),
            "target_end": str(target.iloc[-1]["date"].date()),
            "mature_date": str(target.iloc[-1]["date"].date()),
            "target_dates_json": json.dumps(target_dates, separators=(",", ":")),
            "variance_proxy_columns_json": json.dumps(
                list(EVALUATION_PROXY_COLUMNS), separators=(",", ":")
            ),
            "realized_return_path_json": json.dumps(
                target["return_cc"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_variance_path_json": json.dumps(
                target["rv_cc"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_intraday_rv_path_json": json.dumps(
                target["rv_oc"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_overnight_return_path_json": json.dumps(
                target["return_overnight"].to_numpy(float).tolist(),
                separators=(",", ":"),
            ),
            "realized_upside_semivariance_path_json": json.dumps(
                target["rv_up"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_downside_semivariance_path_json": json.dumps(
                target["rv_down"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_asymmetry_path_json": json.dumps(
                target["asymmetry"].to_numpy(float).tolist(), separators=(",", ":")
            ),
            "realized_variance": realized_variance,
            "realized_cumulative_return": realized_return,
        }
        for proxy, values in proxy_paths.items():
            row[f"realized_variance_path__{proxy}_json"] = json.dumps(
                values.tolist(), separators=(",", ":")
            )
            row[f"realized_variance__{proxy}"] = realized_by_proxy[proxy]
        predictive_file = _predictive_file(record)
        if predictive_file is None:
            supplied = _producer_variance_mean(record)
            if supplied is None:
                raise ValueError(
                    "a point forecast requires variance_forecast or variance_mean"
                )
            variance_mean, _ = supplied
            row.update(
                {
                    "forecast_kind": "variance_point_only",
                    "variance_forecast": variance_mean,
                    "qlike": float(qlike(realized_variance, variance_mean)),
                    "variance_mse": float(variance_mse(realized_variance, variance_mean)),
                }
            )
            for proxy, realized_proxy in realized_by_proxy.items():
                row[f"qlike__{proxy}"] = float(qlike(realized_proxy, variance_mean))
                row[f"variance_mse__{proxy}"] = float(
                    variance_mse(realized_proxy, variance_mean)
                )
            output_rows.append(row)
            if len(output_rows) % 1000 == 0:
                print(f"Scored {len(output_rows)}/{len(records)} forecasts", flush=True)
            continue

        components = load_components(args.forecast_root, predictive_file, h)
        return_distribution = str(record.get('return_distribution', 'variance_standardized_student_t'))
        if return_distribution not in {'gaussian', 'variance_standardized_student_t'}:
            raise ValueError('unsupported return distribution')
        variance_draws = components['integrated_variances']
        return_draws = components['cumulative_returns']
        raw_return_locations = components['locations'][:, None] if h == 1 else None
        raw_conditional_sds = components['conditional_sds'][:, None] if h == 1 else None
        degrees_of_freedom = components.get('degrees_of_freedom')
        if h == 1 and return_distribution == 'variance_standardized_student_t' and degrees_of_freedom is None:
            raise ValueError('Student-t evaluation requires degrees of freedom')
        variance_mean, variance_mean_source = _variance_mean_for_evaluation(
            record, variance_draws
        )
        row.update(
            {
                "forecast_kind": "probabilistic",
                "forecast_draw_count": int(return_draws.size),
                "variance_forecast": variance_mean,
                "variance_forecast_source": variance_mean_source,
                "qlike": float(qlike(realized_variance, variance_mean)),
                "variance_mse": float(variance_mse(realized_variance, variance_mean)),
                "return_crps": crps_ensemble(return_draws, realized_return),
            }
        )
        for proxy, realized_proxy in realized_by_proxy.items():
            row[f"qlike__{proxy}"] = float(qlike(realized_proxy, variance_mean))
            row[f"variance_mse__{proxy}"] = float(
                variance_mse(realized_proxy, variance_mean)
            )
        if h == 1:
            locations = raw_return_locations[:, 0]
            component_sds = raw_conditional_sds[:, 0]
            if return_distribution == "gaussian":
                log_density = finite_mixture_gaussian_log_density(
                    realized_return,
                    locations=locations,
                    conditional_sds=component_sds,
                )
                pit = finite_mixture_gaussian_cdf(
                    realized_return,
                    locations=locations,
                    conditional_sds=component_sds,
                )
                pit_method = "finite_mixture_gaussian_cdf"
            else:
                assert degrees_of_freedom is not None
                log_density = finite_mixture_standardized_t_log_density(
                    realized_return,
                    locations=locations,
                    conditional_sds=component_sds,
                    degrees_of_freedom=degrees_of_freedom,
                )
                pit = finite_mixture_standardized_t_cdf(
                    realized_return,
                    locations=locations,
                    conditional_sds=component_sds,
                    degrees_of_freedom=degrees_of_freedom,
                )
                pit_method = "finite_mixture_standardized_t_cdf"
            row["return_log_density"] = log_density
            row["return_log_score"] = -log_density
            row["return_pit"] = pit
            row["return_pit_method"] = pit_method
        else:
            row["return_pit"] = (ensemble_pit(return_draws, realized_return, components["pit_uniform"])
                                 if "pit_uniform" in components else randomized_pit(return_draws, realized_return, rng))
            row["return_pit_method"] = "randomized_finite_ensemble_rank"

        row.update(
            _tail_fields_from_saved_mixture(
                horizon=h,
                return_distribution=return_distribution,
                raw_return_locations=raw_return_locations,
                raw_conditional_sds=raw_conditional_sds,
                degrees_of_freedom=degrees_of_freedom,
                realized_return=realized_return,
                risk_levels=risk_levels,
            )
        )

        for level in interval_levels:
            label = f"{int(round(100 * level)):02d}"
            diagnostic = interval_diagnostic(return_draws, realized_return, level)
            row[f"return_interval_{label}_lower"] = diagnostic["lower"]
            row[f"return_interval_{label}_upper"] = diagnostic["upper"]
            row[f"return_interval_{label}_covered"] = diagnostic["covered"]
            row[f"return_interval_{label}_width"] = diagnostic["width"]
            row[f"return_interval_{label}_score"] = diagnostic["interval_score"]
        for level in variance_interval_levels:
            label = f"{int(round(100 * level)):02d}"
            diagnostic = interval_diagnostic(variance_draws, realized_variance, level)
            row[f"proxy_variance_interval_{label}_lower"] = diagnostic["lower"]
            row[f"proxy_variance_interval_{label}_upper"] = diagnostic["upper"]
            row[f"proxy_variance_interval_{label}_covered"] = diagnostic["covered"]
            row[f"proxy_variance_interval_{label}_width"] = diagnostic["width"]
        row["proxy_variance_interval_interpretation"] = (
            "descriptive_proxy_coverage"
        )
        output_rows.append(row)
        if len(output_rows) % 1000 == 0:
            print(f"Scored {len(output_rows)}/{len(records)} forecasts", flush=True)
    out = pd.DataFrame(output_rows)
    destination = Path(args.output)
    write_loss_archive(out, destination)
    print(f"Wrote {len(out)} loss rows to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
