"""Forecast comparisons from user-supplied frozen-forecast score panels.

Loss schema follows build_predictive_losses.py. The original and reconstructed
panels must have identical forecast keys, predictions and eligibility flags.
"""
from pathlib import Path
import argparse, json, yaml
import numpy as np, pandas as pd
from bnsv.mcs_dm import diebold_mariano, model_confidence_set
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--original', type=Path, required=True)
p.add_argument('--reconstructed', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mcs', action='store_true')
p.add_argument('--fluctuation', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
K = ['market', 'model_id', 'origin_date', 'mature_date', 'horizon']
frames = {'original': pd.read_parquet(a.original)}
if a.reconstructed:
    frames['reconstructed'] = pd.read_parquet(a.reconstructed)
for (name, d) in frames.items():
    assert len(d) > 0 and (not d.duplicated(K).any())
    for c in ['origin_date', 'mature_date']:
        d[c] = pd.to_datetime(d[c])
    frames[name] = d.sort_values(K).reset_index(drop=True)
for d in frames.values():
    pd.testing.assert_frame_equal(frames['original'][K + ['variance_forecast', 'main_evaluation_eligible']], d[K + ['variance_forecast', 'main_evaluation_eligible']], check_exact=True)
frames = {n: d[d.main_evaluation_eligible].copy() for (n, d) in frames.items()}

def pair(d, m, h, x, y, metric):
    q = d[(d.market == m) & (d.horizon == h) & d.model_id.isin([x, y])].pivot(index=['origin_date', 'mature_date'], columns='model_id', values=metric).reindex(columns=[x, y]).dropna().sort_index()
    if len(q) < 20:
        raise ValueError(f'Insufficient paired observations: {m}, {h}, {x}, {y}, {metric}')
    assert np.isfinite(q.to_numpy()).all()
    return q
C = [('information', 'RV-NN-SV', 'NN-SV', 'qlike', [1, 5, 10]), ('neural_form', 'RV-NN-SV', 'RV-LIN-SV', 'qlike', [1]), ('combination', 'NN-ENSEMBLE-QLIKE', 'RV-NN-SV', 'qlike', [1, 5, 10]), ('information', 'RV-NN-SV', 'NN-SV', 'return_log_score', [1]), ('combination', 'NN-ENSEMBLE-LOG_SCORE', 'RV-NN-SV', 'return_log_score', [1]), ('information', 'RV-NN-SV', 'NN-SV', 'return_crps', [1, 5, 10])]
rows = []
sub = []
ms = []
series = []
concentration = []
largest = []
for (name, d) in frames.items():
    for m in ['SP500', 'FTSE100', 'DAX']:
        for (role, x, y, metric, hs) in C:
            if metric not in d:
                continue
            for h in hs:
                q = pair(d, m, h, x, y, metric)
                for (period, lo, hi) in [('full', '1900', '2100'), ('2018-2019', '2018', '2020'), ('2020', '2020', '2021'), ('2021-2022Feb', '2021', '2022-03-01')]:
                    z = q[(q.index.get_level_values(0) >= pd.Timestamp(lo)) & (q.index.get_level_values(0) < pd.Timestamp(hi))]
                    row = dict(variant=name, market=m, role=role, metric=metric, horizon=h, period=period, n=len(z), mean_a=z[x].mean(), mean_b=z[y].mean(), **diebold_mariano(z[x].to_numpy(), z[y].to_numpy(), horizon=h))
                    rows.append(row)
                if name == 'original' and role == 'information' and (metric == 'qlike'):
                    z = q.reset_index()
                    z['loss_difference'] = z[x] - z[y]
                    z['cumulative_qlike_difference'] = z.loss_difference.cumsum()
                    z['market'] = m
                    z['horizon'] = h
                    if h == 1:
                        series.append(z)
                    if m == 'SP500':
                        z = z[z.origin_date.dt.year == 2020].sort_values(['loss_difference', 'origin_date'], ascending=[False, True])
                        net = z.loss_difference.sum()
                        for k in [1, 3, 5, 10]:
                            concentration.append(dict(horizon=h, top_k=k, n=len(z), net_differential=net, top_sum=z.head(k).loss_difference.sum()))
                        z = z.head(10).copy()
                        z['rank'] = np.arange(1, len(z) + 1)
                        largest.append(z)
        for (role, x, y) in [('form', 'RV-NN-SV-HAR', 'RV-LIN-SV-HAR'), ('NN_inputs', 'RV-NN-SV-HAR', 'RV-NN-SV'), ('LIN_inputs', 'RV-LIN-SV-HAR', 'RV-LIN-SV')]:
            if x not in set(d.model_id):
                continue
            z = pair(d, m, 1, x, y, 'qlike')
            ms.append(dict(variant=name, market=m, role=role, n=len(z), **diebold_mariano(z[x].to_numpy(), z[y].to_numpy(), horizon=1)))
res = pd.DataFrame(rows)
res.to_csv(a.output / 'comparisons.csv', index=False)
pd.concat(series).to_csv(a.output / 'information_series.csv', index=False)
pd.DataFrame(concentration).to_csv(a.output / 'concentration.csv', index=False)
pd.concat(largest).to_csv(a.output / 'largest_origins.csv', index=False)
if ms:
    ms = pd.DataFrame(ms)
    for name in frames:
        for form in [True, False]:
            idx = ms.index[(ms.variant == name) & (ms.role == 'form' if form else ms.role != 'form')]
            v = ms.loc[idx, 'p_value'].to_numpy()
            order = np.argsort(v)
            adj = np.empty(len(v))
            adj[order] = np.minimum(1, np.maximum.accumulate((len(v) - np.arange(len(v))) * v[order]))
            ms.loc[idx, 'p_holm'] = adj
    ms.to_csv(a.output / 'ms_comparisons.csv', index=False)
if a.mcs:
    root = Path(__file__).resolve().parents[2]
    design = yaml.safe_load((root / 'design/analysis.yaml').read_text())
    from types import SimpleNamespace
    ref = [SimpleNamespace(market=m, horizon=h) for m in design['markets'] for h in design['horizons']]
    result = []
    for (name, d) in frames.items():
        for r in ref:
            for expanded in [False, True] if r.horizon == 1 and 'RV-NN-SV-HAR' in set(d.model_id) else [False]:
                models = list(design['evaluation']['mcs']['models']) + (['RV-NN-SV-HAR', 'RV-LIN-SV-HAR'] if expanded else [])
                z = d[(d.market == r.market) & (d.horizon == r.horizon) & d.model_id.isin(models)].pivot(index=['origin_date', 'mature_date'], columns='model_id', values='qlike')[models].dropna().sort_index()
                assert len(z) > 0 and np.isfinite(z.to_numpy()).all()
                fit = model_confidence_set(z.to_numpy(), models, alpha=0.1, bootstrap_replications=2000, mean_block_length=10, seed=20260730)
                result.append(dict(variant=name, market=r.market, horizon=r.horizon, universe='expanded_eleven' if expanded else 'main_nine', n=len(z), winner=z.mean().idxmin(), retained=list(fit.retained_models), means=z.mean().to_dict()))
    (a.output / 'mcs.json').write_text(json.dumps(result, indent=2))
if a.fluctuation:
    stats = []
    paths = []
    for m in ['SP500', 'FTSE100', 'DAX']:
        q = pair(frames['original'], m, 1, 'RV-NN-SV', 'NN-SV', 'qlike')
        d = (q.iloc[:, 0] - q.iloc[:, 1]).to_numpy()
        n = len(d)
        window = int(np.floor(0.3 * n))
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        z = d - d.mean()
        v = np.dot(z, z) / n + 2 * sum(((1 - j / (lag + 1)) * np.dot(z[j:], z[:-j]) / n for j in range(1, lag + 1)))
        assert v > 0
        rolling = np.convolve(d, np.ones(window), 'valid') / np.sqrt(window * v)
        rng = np.random.default_rng(20260913)
        null = []
        for start in range(0, 50000, 500):
            z = rng.normal(size=(500, n))
            walk = np.c_[np.zeros(len(z)), z.cumsum(axis=1)]
            null.extend(np.max(abs((walk[:, window:] - walk[:, :-window]) / np.sqrt(window)), axis=1))
        cv = np.quantile(null, 0.95)
        stats.append(dict(market=m, n=n, window=window, hac_lag=lag, critical_95_mc=cv, max_abs_stat=abs(rolling).max()))
        paths.append(pd.DataFrame(dict(market=m, window_end_origin=q.index.get_level_values(0)[window - 1:], statistic=rolling, critical95=cv)))
    pd.DataFrame(stats).to_csv(a.output / 'fluctuation_summary.csv', index=False)
    pd.concat(paths).to_csv(a.output / 'fluctuation_series.csv', index=False)
print('Forecast comparisons written.')
