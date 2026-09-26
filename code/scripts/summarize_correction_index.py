#!/usr/bin/env python3
"""Summarise correction-index recovery from stored simulation records; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEY = ['dgp_id', 'replication_id']


def checked(frame, expected):
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('Missing or duplicate replication keys')
    if set(map(tuple, frame[KEY].to_numpy())) != expected:
        raise ValueError('Unexpected replication keys')


def export_records(args, qc, expected):
    source = args.source_records
    outcomes = pd.read_csv(source / 'fit_outcomes.csv')
    truth = pd.read_csv(source / 'dgp_truth_by_replication.csv')
    old = pd.read_csv(source / 'functional_recovery.csv').set_index(KEY)
    checked(outcomes, expected)
    checked(truth, expected)
    tasks = pd.DataFrame(json.loads(args.tasks.read_text())['tasks'])
    checked(tasks, expected)
    frame = outcomes.merge(truth, on=KEY, validate='one_to_one').merge(tasks[KEY + ['task_id']], on=KEY, validate='one_to_one')
    rows = []
    for r in frame.itertuples(index=False):
        row = dict(dgp_id=r.dgp_id, replication_id=r.replication_id, fit_outcome=r.fit_outcome,
                   true_index=r.true_relative_correction_variance_index, q_c=qc,
                   posterior_draws=0, posterior_median=np.nan, interval_lower=np.nan,
                   interval_upper=np.nan, probability_at_least_qc=np.nan, covered=False,
                   material_declaration=False)
        if r.fit_outcome == 'successful':
            with np.load(args.fits / f'task_{r.task_id:03d}' / 'posterior_compact.npz', allow_pickle=False) as data:
                draws = data['correction_share']
            if draws.ndim != 1 or len(draws) == 0 or not np.isfinite(draws).all() or (draws < 0).any():
                raise ValueError(f'Invalid index draws: {r.dgp_id}, {r.replication_id}')
            lo, med, hi = np.quantile(draws, [0.025, 0.5, 0.975], method='linear')
            prob = float(np.mean(draws >= qc))
            prior = old.loc[(r.dgp_id, r.replication_id)]
            np.testing.assert_allclose([med, prob, row['true_index']],
                [prior.posterior_median_relative_correction_variance_index,
                 prior.posterior_probability_index_at_least_qc,
                 prior.true_relative_correction_variance_index], rtol=0, atol=1e-15)
            row.update(posterior_draws=len(draws), posterior_median=med, interval_lower=lo,
                       interval_upper=hi, probability_at_least_qc=prob,
                       covered=bool(lo <= row['true_index'] <= hi), material_declaration=bool(prob >= .95))
        elif r.fit_outcome != 'unsuccessful':
            raise ValueError('Unknown fit outcome')
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(KEY)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.records, index=False, float_format='%.17g')


def summarise(frame, qc, expected):
    checked(frame, expected)
    if not np.allclose(frame.q_c, qc, rtol=0, atol=1e-16):
        raise ValueError('Threshold differs from the simulation design')
    rows = []
    for dgp, group in frame.groupby('dgp_id', sort=False):
        good = group[group.fit_outcome == 'successful']
        bad = group[group.fit_outcome == 'unsuccessful']
        if len(good) + len(bad) != 200:
            raise ValueError('Unknown fit status or missing repetitions')
        cols = ['posterior_median', 'interval_lower', 'interval_upper', 'probability_at_least_qc']
        if not np.isfinite(good[cols].to_numpy()).all() or not bad[cols].isna().all().all():
            raise ValueError('Posterior summaries disagree with fit status')
        if not ((good.interval_lower <= good.posterior_median) & (good.posterior_median <= good.interval_upper)).all():
            raise ValueError('Invalid quantile ordering')
        covered = ((group.interval_lower <= group.true_index) & (group.true_index <= group.interval_upper) & (group.fit_outcome == 'successful'))
        if not np.array_equal(covered.to_numpy(), group.covered.to_numpy()):
            raise ValueError('Coverage flags disagree with interval endpoints')
        def put(name, value, n):
            rows.append(dict(dgp_id=dgp, quantity=name, value=float(value), n=int(n)))
        put('q_c', qc, 200)
        put('successful_fits', len(good), 200)
        if dgp == 'DGP-N':
            put('covered_fits', covered.sum(), 200)
            put('coverage_all_replications', covered.mean(), 200)
            put('coverage_successful_fits', covered.sum()/len(good), len(good))
            put('truth_above_interval', (good.true_index > good.interval_upper).sum(), len(good))
            put('truth_below_interval', (good.true_index < good.interval_lower).sum(), len(good))
        put('interval_upper_below_qc', (good.interval_upper < qc).sum(), len(good))
        put('probability_at_least_qc_max', good.probability_at_least_qc.max(), len(good))
        for label, values in [('posterior_median', good.posterior_median),
                              ('true_index', group.true_index),
                              ('probability_at_least_qc', good.probability_at_least_qc)]:
            for suffix, q in [('q25', .25), ('median', .5), ('q75', .75), ('q95', .95)]:
                put(label + '_' + suffix, np.quantile(values, q, method='linear'), len(values))
        for name, mask in [('truth_at_least_qc', group.true_index >= qc), ('truth_below_qc', group.true_index < qc)]:
            put('declarations_' + name, ((group.probability_at_least_qc >= .95) & mask).sum(), mask.sum())
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, default=ROOT/'inputs/identification/correction_index.csv')
    p.add_argument('--output', type=Path, default=ROOT/'results/identification/correction_index_summary.csv')
    p.add_argument('--config', type=Path, default=ROOT/'design/identification.yaml')
    p.add_argument('--fits', type=Path, help='Optional stored posterior directory, to export per-replication records')
    p.add_argument('--tasks', type=Path)
    p.add_argument('--source-records', type=Path, default=ROOT/'inputs/identification')
    args = p.parse_args()
    qc = float(yaml.safe_load(args.config.read_text())['dgp']['anchor']['target_correction_share'])
    expected = {(d, r) for d in ['DGP-N', 'DGP-0'] for r in range(1, 201)}
    if args.fits:
        if not args.tasks:
            p.error('--fits requires --tasks')
        export_records(args, qc, expected)
    frame = pd.read_csv(args.records, float_precision='round_trip')
    summary = summarise(frame, qc, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format='%.17g')
    print(summary.pivot(index='quantity', columns='dgp_id', values='value').to_string())


if __name__ == '__main__':
    main()
