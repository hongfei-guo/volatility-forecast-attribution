"""One-day matched HAR-input forecasts using the existing SV calculations."""
from __future__ import annotations
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping
import numpy as np
import pandas as pd
from .forecast_features import HAR_INPUT_MODELS, fit_feature_transform, model_spec, prepare_estimation_data, transform_features
from .forecast_fit import SamplingFailure, build_stan_data, fit_model, sampling_policy_for
from .har_posterior_cache import read_cached_refit
from .har_filter_policy import POLICY as DIRECT_FILTER_POLICY, INITIAL_LEVELS, MAX_LEVEL, TOLERANCE, FilterAccuracyError, initial_banks, update_and_assess
from .forecast_filter import NumericalFilterError
from .forecast_generation import FitFunction, forecast_record
from .forecast_paths import compact_refit, seed_from_context, simulate_forecast_paths

def generate_har_input_pair(*, daily: pd.DataFrame, market: str, stan_dir: str | Path, output_dir: str | Path, correction_scales: Mapping[str, float], forecast_start: str | pd.Timestamp, forecast_end: str | pd.Timestamp, base_seed: int=20260730, particle_count: int=4000, path_count: int=4096, grid_levels: tuple[int, int, int]=(2, 3, 4), warning_ess_fraction: float=0.2, fit_function: FitFunction=fit_model, show_progress: bool=False, calibration_record: Mapping[str, object] | None=None, model_workers: int=1, sampling_policy: str | None=None, posterior_cache: Mapping | None=None, filter_policy: str=DIRECT_FILTER_POLICY, allow_empty_forecasts: bool=False) -> pd.DataFrame:
    """Fit both models on every monthly or pair-triggered re-estimation date.

    Both models use the common 0.97 / 0.99 retry policy and retain the
    existing hard diagnostics, state grids and ESS continuation rule. A failed
    model stays unavailable until the next shared re-estimation date; its
    unsuccessful fit is retained. No old forecast or benchmark is replaced.
    """
    if set(correction_scales) != set(HAR_INPUT_MODELS):
        raise ValueError('provide calibrated scales for exactly the two HAR-input models')
    if model_workers not in (1, 2):
        raise ValueError('model_workers must be one or two; each model uses four chains')
    policy = sampling_policy_for(HAR_INPUT_MODELS[0], sampling_policy)
    if filter_policy != DIRECT_FILTER_POLICY:
        raise ValueError('the multiscale pair requires grid agreement')
    if grid_levels != (2, 3, 4) or warning_ess_fraction != 0.2:
        raise ValueError('direct agreement uses the specified grid and ESS settings')
    if any((not np.isfinite(v) or v <= 0 for v in correction_scales.values())):
        raise ValueError('correction scales must be positive and finite')
    if len(grid_levels) != 3 or any((b != a + 1 for (a, b) in zip(grid_levels, grid_levels[1:]))):
        raise ValueError('the continuation rule requires three adjacent grid levels')
    if particle_count < 1 or path_count < 1 or (not 0 < warning_ess_fraction < 1):
        raise ValueError('invalid forecast or filter settings')
    if 'market' in daily and set(daily['market'].astype(str)) != {market}:
        raise ValueError('the matched run requires one market')
    frame = daily.loc[:, ['date', 'return_cc', 'rv_oc']].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    if frame.empty or frame['date'].duplicated().any() or (not frame['date'].is_monotonic_increasing):
        raise ValueError('daily observations must be nonempty, ordered and unique')
    frame = frame.reset_index(drop=True)
    (start, end) = (pd.Timestamp(forecast_start), pd.Timestamp(forecast_end))
    if start > end or start <= pd.Timestamp('2016-12-31'):
        raise ValueError('HAR forecast dates must follow the pre-2017 calibration sample')
    positions = [i for (i, date) in enumerate(frame['date']) if start <= date <= end and i + 1 < len(frame)]
    if not positions:
        raise ValueError('no complete one-day forecast origins')
    first_origin = frame.loc[positions[0], 'date']
    fit_feature_transform(frame, model_id=HAR_INPUT_MODELS[0], training_end=first_origin)
    for name in HAR_INPUT_MODELS:
        if not (Path(stan_dir) / model_spec(name).stan_file).is_file():
            raise FileNotFoundError(Path(stan_dir) / model_spec(name).stan_file)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new HAR output directory; existing evidence is preserved')
    output.mkdir(parents=True, exist_ok=True)
    components = output / 'components'
    components.mkdir()
    (output / 'run.json').write_text(json.dumps({'market': market, 'models': list(HAR_INPUT_MODELS), 'horizons': [1], 'forecast_start': str(start.date()), 'forecast_end': str(end.date()), 'seed': base_seed, 'correction_scales': dict(correction_scales), 'parameter_atoms': particle_count, 'forecast_paths': path_count, 'grid_levels': list(grid_levels), 'warning_ess_fraction': warning_ess_fraction, 'measurement_variable': 'rv_oc', 'evaluation_target': 'rv_cc', 'refit_rule': 'monthly_or_either_model_ess_or_numerical_filter_failure', 'sampling_policy': policy, 'model_workers': model_workers, 'chains_per_model': 4, 'parallel_chains_per_model': 4, 'filter_policy': filter_policy, **{'initial_filter_levels': list(INITIAL_LEVELS), 'maximum_filter_level': MAX_LEVEL, 'numerical_agreement_tolerance': TOLERANCE}}, indent=2) + '\n')
    if calibration_record is not None:
        (output / 'prior_calibration.json').write_text(json.dumps(dict(calibration_record), indent=2) + '\n')
    (refits, transforms, transformed, banks) = ({}, {}, {}, {})
    filter_history = {}
    rows: list[dict[str, object]] = []
    previous_month = None
    trigger_models: list[str] = []
    with (output / 'forecast_records.jsonl').open('x') as forecasts, (output / 'availability.jsonl').open('x') as availability, (output / 'filter_events.jsonl').open('x') as filter_events:
        for position in positions:
            origin = pd.Timestamp(frame.loc[position, 'date'])
            month = origin.to_period('M')
            monthly = month != previous_month
            if monthly or trigger_models:
                reason = 'monthly' if monthly else 'adaptive'
                banks.clear()
                refits.clear()
                transforms.clear()
                transformed.clear()
                filter_history.clear()
                common_transform = fit_feature_transform(frame, model_id=HAR_INPUT_MODELS[0], training_end=origin)
                estimation = prepare_estimation_data(frame, common_transform)
                cached = {}
                for name in HAR_INPUT_MODELS:
                    key = (name, str(origin.date()))
                    if posterior_cache and key in posterior_cache:
                        if posterior_cache[key].get('unavailable'):
                            cached[name] = (None, [])
                            continue
                        transform = replace(common_transform, model_id=name)
                        expected_data = build_stan_data(model_id=name, market=market, z=estimation.z, x=estimation.x, correction_scale=correction_scales[name])
                        cached[name] = read_cached_refit(posterior_cache[key], market=market, model_id=name, origin=str(origin.date()), seed=seed_from_context(base_seed, market, str(origin.date()), 'nuts'), data=expected_data, transform=transform, particle_count=particle_count)

                def estimate(name):
                    transform = replace(common_transform, model_id=name)
                    data = build_stan_data(model_id=name, market=market, z=estimation.z, x=estimation.x, correction_scale=correction_scales[name])
                    fit_dir = output / 'refits' / f'{market}_{name}_{origin:%Y%m%d}'
                    fit_dir.mkdir(parents=True)
                    fit_seed = seed_from_context(base_seed, market, str(origin.date()), 'nuts')
                    record = {'model_id': name, 'origin_date': str(origin.date()), 'reason': reason, 'trigger_models': trigger_models.copy(), 'seed': fit_seed, 'training_rows': len(estimation.z), 'training_start': str(estimation.dates[0].date()), 'training_end': str(estimation.dates[-1].date()), 'feature_names': list(transform.feature_names), 'return_mean': transform.return_scaler.mean, 'return_scale': transform.return_scaler.scale, 'feature_mean': transform.feature_mean.tolist(), 'feature_scale': transform.feature_scale.tolist(), 'rv_floor': transform.rv_floor, 'correction_scale': correction_scales[name], 'sampling_policy': policy}
                    started = perf_counter()
                    if name in cached:
                        (compact, attempts) = cached[name]
                        if compact is None:
                            record.update(status='unsuccessful', attempts=[], source_action='retained_unavailable', historical_failure_source=str(posterior_cache[name, str(origin.date())]['manifest']))
                            (fit_dir / 'fit.json').write_text(json.dumps(record, indent=2) + '\n')
                            return (name, None, transform)
                        np.savez_compressed(fit_dir / 'posterior.npz', **compact)
                        record.update(status='accepted', attempts=attempts, source_action='reused', posterior_source=str(posterior_cache[name, str(origin.date())]['posterior.npz']), elapsed_seconds=perf_counter() - started)
                        (fit_dir / 'fit.json').write_text(json.dumps(record, indent=2) + '\n')
                        return (name, compact, transform)
                    compact = None
                    try:
                        (fit, attempts) = fit_function(stan_file=Path(stan_dir) / model_spec(name).stan_file, model_id=name, data=data, seed=fit_seed, output_dir=fit_dir / 'cmdstan', show_progress=show_progress, parallel_chains=4, sampling_policy=policy)
                    except SamplingFailure as exc:
                        record.update(status='unsuccessful', attempts=exc.records)
                    else:
                        compact = compact_refit(fit, model_id=name, transform=transform, estimation_features=estimation.x, particle_count=particle_count, rng=np.random.default_rng(seed_from_context(base_seed, market, str(origin.date()), 'posterior_particles')))
                        del fit
                        np.savez_compressed(fit_dir / 'posterior.npz', **compact)
                        record.update(status='accepted', attempts=attempts)
                    record['elapsed_seconds'] = perf_counter() - started
                    record['source_action'] = 'estimated'
                    (fit_dir / 'fit.json').write_text(json.dumps(record, indent=2) + '\n')
                    return (name, compact, transform)
                with ThreadPoolExecutor(max_workers=model_workers) as pool:
                    for (name, compact, transform) in pool.map(estimate, HAR_INPUT_MODELS):
                        if compact is not None:
                            refits[name] = compact
                            transforms[name] = transform
                            transformed[name] = transform_features(frame, transform)
                for (name, compact) in refits.items():
                    started = perf_counter()
                    banks[name] = initial_banks(compact)
                    filter_history[name] = []
                    filter_events.write(json.dumps({'event': 'bank_built', 'model_id': name, 'date': str(origin.date()), 'elapsed_seconds': perf_counter() - started}) + '\n')
                filter_events.flush()
                trigger_models = []
            previous_month = month
            for name in HAR_INPUT_MODELS:
                available = name in refits
                availability.write(json.dumps({'model_id': name, 'origin_date': str(origin.date()), 'available': available}) + '\n')
                if not available:
                    continue
                transform = transforms[name]
                history = frame.iloc[:position + 1]
                paths = simulate_forecast_paths(model_id=name, state_filter=banks[name][max(banks[name])], refit=refits[name], standardized_return_history=transform.return_scaler.standardize(history['return_cc'].to_numpy(float)), log_rv_history=np.log(np.maximum(history['rv_oc'].to_numpy(float), transform.rv_floor)), horizon=1, paths=path_count, rng=np.random.default_rng(seed_from_context(base_seed, market, str(origin.date()), 1, 'forecast_paths')))
                row = forecast_record(paths=paths, market=market, model_id=name, origin=origin, mature_date=pd.Timestamp(frame.loc[position + 1, 'date']), horizon=1, path_count=path_count, component_root=components)
                rows.append(row)
                forecasts.write(json.dumps(row) + '\n')
            forecasts.flush()
            availability.flush()
            if position == positions[-1]:
                continue
            for name in HAR_INPUT_MODELS:
                if name not in refits:
                    continue
                transform = transforms[name]
                x_next = transformed[name].loc[position + 1, list(transform.feature_names)].to_numpy(float)
                next_z = float(transform.return_scaler.standardize(frame.loc[position + 1, 'return_cc']))
                event = {'model_id': name, 'forecast_origin': str(origin.date()), 'observation_date': str(pd.Timestamp(frame.loc[position + 1, 'date']).date())}
                try:
                    next_forecast_x = transformed[name].loc[position + 2, list(transform.feature_names)].to_numpy(float)
                    (adequate, details) = update_and_assess(banks[name], refit=refits[name], history=filter_history[name], model_id=name, x=x_next, observed_z=next_z, next_x=next_forecast_x, return_scale=transform.return_scaler.scale)
                    event['numerical_checks'] = details
                except FilterAccuracyError as exc:
                    event.update(event='accuracy_unresolved', numerical_checks=exc.diagnostics)
                    filter_events.write(json.dumps(event) + '\n')
                    filter_events.flush()
                    raise
                except NumericalFilterError as exc:
                    event.update(event='numerical_failure', message=str(exc), grid_level=exc.level)
                    filter_events.write(json.dumps(event) + '\n')
                    filter_events.flush()
                    raise
                event.update(event='filter_update', adequate=adequate)
                filter_events.write(json.dumps(event) + '\n')
                filter_events.flush()
                if not adequate:
                    trigger_models.append(name)
                    banks.clear()
                    break
    if not rows and (not allow_empty_forecasts):
        raise RuntimeError('no accepted HAR forecasts; unsuccessful fits remain in the output')
    columns = ['market', 'model_id', 'origin_date', 'horizon', 'mature_date', 'forecast_kind', 'variance_forecast', 'cumulative_return_mean', 'forecast_draw_count', 'return_distribution', 'variance_mean_method', 'predictive_file']
    panel = pd.DataFrame(rows, columns=columns).sort_values(['origin_date', 'model_id']).reset_index(drop=True)
    panel.to_parquet(output / 'forecasts.parquet', index=False)
    return panel
