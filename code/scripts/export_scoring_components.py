"""Extract scoring inputs and verify them against retained predictive arrays."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from bnsv.scoring_components import from_paths, load_components, replay_garch, ensemble_pit
from bnsv.evaluation import (crps_ensemble, qlike, finite_mixture_gaussian_log_density,
                             finite_mixture_standardized_t_log_density)


PUBLIC_COLUMNS = ['market', 'model_id', 'origin_date', 'mature_date', 'horizon',
                  'scope', 'forecast_kind', 'variance_forecast', 'variance_mean_method',
                  'return_distribution', 'forecast_draw_count', 'target_dates_json',
                  'main_evaluation_eligible', 'predictive_file']


def write_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(arrays):
                raise ValueError('existing component has different fields')
            for key, value in arrays.items():
                np.testing.assert_array_equal(z[key], value)
    else:
        temporary = path.with_suffix('.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)


def export_record(row, output):
    horizon = int(row['horizon'])
    path = output/row['predictive_file']
    kind = row['component_kind']
    if kind == 'garch_parameters':
        payload = {k: np.asarray(v) for k, v in json.loads(row['garch_parameters']).items()}
        expected = replay_garch(payload)
    else:
        source = Path(row['source_path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row['source_sha256']:
            raise ValueError(f'source identity mismatch: {source.name}')
        with np.load(source, allow_pickle=False) as z:
            expected = from_paths(z, horizon)
            if kind == 'selection':
                models, draws = z['source_model_indices'], z['source_draw_indices']
                if not (models.min() >= 0 and models.max() < 256 and draws.min() >= 0 and draws.max() < 65536):
                    raise ValueError('indices cannot be represented without loss')
                payload = {'source_files': np.asarray(json.loads(row['source_candidates'])),
                           'source_model_indices': models.astype(np.uint8),
                           'source_draw_indices': draws.astype(np.uint16)}
            elif kind == 'draws':
                payload = dict(expected)
            else:
                raise ValueError('unknown component representation')
    target = float(row['realized_cumulative_return'])
    if horizon > 1 and pd.notna(row.get('return_pit')):
        values = expected['cumulative_returns']
        below, equal = np.sum(values < target), np.sum(values == target)
        u = (float(row['return_pit'])*(len(values)+1)-below)/(equal+1)
        if not -1e-10 <= u <= 1+1e-10:
            raise ValueError('stored PIT cannot be represented by the predictive ensemble')
        payload['pit_uniform'] = np.asarray(np.clip(u, 0, 1))
    write_arrays(path, payload)
    loaded = load_components(output, row['predictive_file'], horizon)
    for key, value in expected.items():
        np.testing.assert_array_equal(loaded[key], value)
    errors = {}
    if pd.notna(row.get('return_crps')):
        score = float(crps_ensemble(loaded['cumulative_returns'], target))
        np.testing.assert_allclose(score, row['return_crps'], rtol=1e-9, atol=1e-12)
        errors['crps_error'] = abs(score-float(row['return_crps']))
    if pd.notna(row.get('qlike')):
        score = float(qlike(float(row['realized_variance']), float(row['variance_forecast'])))
        np.testing.assert_allclose(score, row['qlike'], rtol=1e-10, atol=1e-12)
        errors['qlike_error'] = abs(score-float(row['qlike']))
    if horizon == 1 and pd.notna(row.get('return_log_score')):
        args = {'locations': loaded['locations'], 'conditional_sds': loaded['conditional_sds']}
        if row['return_distribution'] == 'gaussian':
            score = -float(finite_mixture_gaussian_log_density(target, **args))
        else:
            score = -float(finite_mixture_standardized_t_log_density(target, degrees_of_freedom=loaded['degrees_of_freedom'], **args))
        np.testing.assert_allclose(score, row['return_log_score'], rtol=1e-9, atol=1e-10)
        errors['log_score_error'] = abs(score-float(row['return_log_score']))
    if 'pit_uniform' in loaded:
        pit = ensemble_pit(loaded['cumulative_returns'], target, loaded['pit_uniform'])
        np.testing.assert_allclose(pit, row['return_pit'], rtol=1e-12, atol=1e-12)
        errors['pit_error'] = abs(pit-float(row['return_pit']))
    public = {key: row[key] for key in PUBLIC_COLUMNS}
    public['predictive_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    check = {key: row[key] for key in ['market','model_id','origin_date','horizon','scope']}
    check.update(component_kind=kind, component_bytes=path.stat().st_size, **errors)
    return public, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    records = pd.read_parquet(args.records)
    keys = ['market','model_id','origin_date','horizon']
    if 'qlike' not in records or records.qlike.isna().any():
        raise ValueError('reference QLIKE is required for every forecast')
    if records.empty or records.duplicated(keys).any() or records.predictive_file.duplicated().any():
        raise ValueError('forecast records must be nonempty and unique')
    for name in records.predictive_file:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('component paths must be relative')
    args.output.mkdir(parents=True, exist_ok=True)
    args.checks.mkdir(parents=True, exist_ok=True)
    outputs, checks = [], []
    # Candidate files must exist before selections can be reconstructed.
    for selection in [False, True]:
        group = records[records.component_kind.eq('selection') == selection]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for public, check in pool.map(lambda r: export_record(r, args.output), group.to_dict('records')):
                outputs.append(public);checks.append(check)
                if len(outputs) % 1000 == 0:
                    print(json.dumps({'verified':len(outputs),'total':len(records)}),flush=True)
    pd.DataFrame(outputs).sort_values(keys).to_parquet(args.output/'forecasts.parquet', index=False)
    pd.DataFrame(checks).to_csv(args.checks/'checks.csv', index=False)
    print(json.dumps({'verified':len(outputs),'component_bytes':sum(r['component_bytes'] for r in checks)},indent=2),flush=True)


if __name__ == '__main__':
    main()
