#!/usr/bin/env python3
"""Extract prediction and posterior components for the 2017 RV-RA comparisons."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import h5py

from analyze_rv_ra_conditional_value import load_paired_losses, load_rv_ra_records, predictive_array_path

DRAW_FIELDS = (
    'source_particle_indices', 'source_parameter_ancestry', 'origin_baseline_states',
    'future_log_rv_paths', 'future_asymmetry_paths', 'state_innovations',
    'return_innovations', 'feature_mean', 'feature_scale',
)
REFIT_FIELDS = (
    'ancestry', 'feature_names', 'feature_mean', 'feature_scale', 'return_mean',
    'return_scale', *('parameter__' + name for name in (
        'mu', 'phi', 'w1', 'b1', 'output_weights', 'centering_constant',
        'gamma_A', 'q_centering_mean')),
)
RECORD_FIELDS = ('market', 'model_id', 'origin_date', 'mature_date', 'horizon',
                 'target_dates', 'refit_id', 'variance_mean')


def gamma_row(values, month, origin):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != 8000 or not np.isfinite(values).all():
        raise ValueError('monthly gamma posterior must contain 8000 finite draws')
    return dict(task_id=month, origin=origin, slice='calendar_month_refit', width=5,
                draws=len(values), gamma_A_mean=values.mean(),
                gamma_A_sd=values.std(ddof=1), gamma_A_q025=np.quantile(values, .025),
                gamma_A_median=np.median(values), gamma_A_q975=np.quantile(values, .975))


def export(source, output, checks):
    records, refits = load_rv_ra_records(source / 'rv_ra')
    _, origins = load_paired_losses(source / 'tables/paired_losses.csv')
    cells = [(origin, h) for origin in origins for h in (1, 5, 10)]
    if set(cells) - records.keys():
        raise ValueError('source forecasts do not cover the comparison sample')
    output.mkdir(parents=True, exist_ok=False)
    checks.mkdir(parents=True, exist_ok=True)
    source_checks = []
    gamma = []
    seen_refits = set()
    by_month = {}
    for origin, h in cells:
        r = records[(origin, h)]
        month = Path(r['task_root']).name
        target = output / 'rv_ra' / month
        by_month.setdefault(month, []).append({k: r[k] for k in RECORD_FIELDS})
        refit_id = r['refit_id']
        if refit_id not in seen_refits:
            path = refits[refit_id]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != r['refit_artifact_sha256']:
                raise ValueError(f'refit identity mismatch: {refit_id}')
            with np.load(path, allow_pickle=False) as arrays:
                values = {k: arrays[k].copy() for k in REFIT_FIELDS}
            with h5py.File(path.with_suffix('.nc'), 'r') as ds:
                values['posterior_gamma_A'] = ds['posterior/gamma_A'][...].reshape(-1)
            date = pd.Timestamp(refit_id.rsplit('_', 1)[-1]).strftime('%Y-%m-%d')
            gamma.append(gamma_row(values['posterior_gamma_A'], int(month.split('_')[-1]), date))
            dest = target / 'refits' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **values)
            source_checks.append(dict(kind='refit',file=str(path),sha256=digest))
            seen_refits.add(refit_id)
        if h == 1:
            continue
        path = predictive_array_path(r)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != r['draw_artifact_sha256']:
            raise ValueError(f'predictive identity mismatch: {origin}, h={h}')
        with np.load(path, allow_pickle=False) as arrays:
            values = {k: arrays[k].copy() for k in DRAW_FIELDS}
        dest = target / 'draws' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **values)
        with np.load(dest, allow_pickle=False) as compact:
            for k, value in values.items():
                np.testing.assert_array_equal(value, compact[k])
        source_checks.append(dict(kind='predictive',file=str(path),sha256=digest))
        if len(source_checks) % 50 == 0:
            print(f'Extracted {len(source_checks)} source files', flush=True)
    for month, rows in by_month.items():
        path = output / 'rv_ra' / month / 'forecast_records.jsonl'
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    gamma = pd.DataFrame(gamma).sort_values('origin')
    reference = pd.read_csv(source / 'tables/gamma_a_summary.csv')
    reference = reference[reference.task_id.isin(range(1,13)) & reference.slice.eq('calendar_month_refit') & reference.width.eq(5)]
    cols = list(gamma.columns)
    pd.testing.assert_frame_equal(gamma.set_index('origin').sort_index(),
        reference[cols].set_index('origin').sort_index(),check_dtype=False,rtol=1e-12,atol=1e-14)
    gamma.to_csv(output / 'gamma_summary.csv', index=False)
    for model, name in [('NN-SV','nn_sv'),('RV-NN-SV','rv_nn_sv')]:
        frame = pd.read_csv(source / 'tables' / f'{name}_losses.csv')
        frame = frame[frame.origin_date.isin(origins) & frame.horizon.isin((1,5,10))]
        cols = ['market','model_id','origin_date','mature_date','horizon','variance_forecast']
        panel = frame[cols].copy()
        if len(panel) != 723 or set(panel.model_id) != {model} or panel.duplicated(['origin_date','horizon']).any():
            raise ValueError(f'invalid member prediction panel: {model}')
        panel.to_parquet(output / f'{name}_forecasts.parquet',index=False)
    frame = pd.DataFrame([dict(market=r['market'],model_id=r['model_id'],
        origin_date=r['origin_date'],mature_date=r['mature_date'],horizon=r['horizon'],
        variance_forecast=r['variance_mean']) for r in (records[cell] for cell in cells)])
    frame.to_parquet(output / 'rv_ra_forecasts.parquet',index=False)
    (output / 'README.md').write_text('''# RV-RA conditional-analysis components

Prediction panels cover the common 241 S&P 500 origins in 2017 at horizons 1, 5 and 10. The collection contains twelve parameter sets, the 8,000 posterior asymmetry-coefficient draws for each month, and 482 stochastic decompositions at horizons 5 and 10. No estimation is performed when these objects are read.

The parameter subsets retain the original particle ancestry. The full coefficient draws used for posterior intervals are stored separately from the parameter atoms used in prediction. Prediction files retain the origin baseline state, simulated future inputs and the original state/return innovations. Training matrices, historical observed inputs and evaluation outcomes are excluded.

Use the accompanying code's `prepare_rv_ra_conditional_inputs.py` with these three forecast panels and lawful daily data. Then call `analyze_rv_ra_conditional_value.py` with the prepared loss tables, `gamma_summary.csv`, this collection's `rv_ra` directory, and `--daily` pointing to the same daily data. The code reconstructs origin histories and monthly feature transformations from that input, including the training-sample RV floor.

`gamma_summary.csv` is computed from `posterior_gamma_A` in each monthly parameter file using the mean, sample standard deviation and 2.5%, 50%, 97.5% quantiles. The package's analysis design fixes the bootstrap and sample. Availability of this separate component collection and permission to use the underlying daily data must be established independently.
''')
    pd.DataFrame(source_checks).to_csv(checks / 'source_files.csv',index=False)
    summary=dict(origins=len(origins),forecast_rows=len(cells)*3,refits=len(seen_refits),
                 predictive_files=sum(r['kind']=='predictive' for r in source_checks),
                 gamma_draws=len(gamma)*8000,
                 component_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    (checks/'export.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='source directory containing rv_ra/ and tables/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True,help='local verification directory, outside the component collection')
    a=p.parse_args()
    export(a.source,a.output,a.checks)


if __name__=='__main__':
    main()
