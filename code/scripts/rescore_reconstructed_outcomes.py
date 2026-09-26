"""Rescore retained forecasts with reconstructed FTSE/DAX evaluation outcomes."""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from bnsv.evaluation import qlike, variance_mse, crps_ensemble, finite_mixture_standardized_t_log_density, finite_mixture_standardized_t_cdf, finite_mixture_gaussian_log_density, finite_mixture_gaussian_cdf, quantile_loss, fz0_var_es_score
from contextlib import nullcontext
from bnsv.scoring_components import load_components
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--daily', type=Path, required=True)
parser.add_argument('--reconstructed-daily', type=Path, required=True)
parser.add_argument('--scores', type=Path, required=True)
parser.add_argument('--components', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
ROOT = options.output
ROOT.mkdir(parents=True, exist_ok=True)
K = ['market', 'model_id', 'origin_date', 'mature_date', 'horizon']
source = pd.read_parquet(options.daily)
source.date = pd.to_datetime(source.date)
patched = pd.read_parquet(options.reconstructed_daily)
patched.date = pd.to_datetime(patched.date)
assert not source.duplicated(['market', 'date']).any() and (not patched.duplicated(['market', 'date']).any())
olddata = source.set_index(['market', 'date']).sort_index()
newdata = patched.set_index(['market', 'date']).sort_index()
pd.testing.assert_index_equal(olddata.index, newdata.index)
changed_days = ~np.isclose(olddata[['return_cc', 'rv_cc']], newdata[['return_cc', 'rv_cc']], rtol=1e-12, atol=1e-15).all(axis=1)
change_dates = {m: set(g.index.get_level_values('date')) for (m, g) in newdata.loc[changed_days].groupby(level='market')}
d = pd.read_parquet(options.scores)
for c in ['origin_date', 'mature_date']:
    d[c] = pd.to_datetime(d[c])
assert len(d) > 0 and (not d.duplicated(K).any())
keep = K + ['target_dates_json', 'main_evaluation_eligible', 'target_spans_archive_gap', 'variance_forecast', 'realized_variance', 'realized_cumulative_return', 'qlike', 'variance_mse', 'return_crps', 'return_log_density', 'return_log_score', 'return_pit', 'return_pit_method']
keep += [c for c in d if c.startswith(('var_', 'es_', 'return_interval_', 'proxy_variance_interval_'))]
keep = list(dict.fromkeys((c for c in keep if c in d)))
out = d[keep].copy()
orig = out.copy()
out['evaluation_variant'] = 'reconstructed'
orig['evaluation_variant'] = 'original'
out['target_variance_changed'] = False
out['target_return_changed'] = False
changed = []
npz_count = 0
max_old_nls_error = 0.0
for (i, r) in d[d.market.isin(change_dates)].iterrows():
    ds = pd.to_datetime(json.loads(r.target_dates_json))
    assert len(ds) == r.horizon and ds[-1] == r.mature_date and (ds.min() > r.origin_date)
    if not any((dt in change_dates[r.market] for dt in ds)):
        continue
    keys = [(r.market, dt) for dt in ds]
    y0 = float(olddata.loc[keys, 'rv_cc'].sum())
    y1 = float(newdata.loc[keys, 'rv_cc'].sum())
    r0 = float(olddata.loc[keys, 'return_cc'].sum())
    r1 = float(newdata.loc[keys, 'return_cc'].sum())
    np.testing.assert_allclose(y0, r.realized_variance, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(qlike(y0, r.variance_forecast), r.qlike, rtol=1e-10, atol=1e-12)
    if np.isfinite(r.realized_cumulative_return):
        np.testing.assert_allclose(r0, r.realized_cumulative_return, rtol=1e-10, atol=1e-14)
    cy = not np.isclose(y0, y1, rtol=1e-12, atol=1e-15)
    cr = not np.isclose(r0, r1, rtol=1e-12, atol=1e-14)
    out.loc[i, ['realized_variance', 'realized_cumulative_return', 'qlike', 'variance_mse', 'target_variance_changed', 'target_return_changed']] = [y1, r1, float(qlike(y1, r.variance_forecast)), float(variance_mse(y1, r.variance_forecast)), cy, cr]
    for level in ['50', '90', '95']:
        for (prefix, val) in [('return_interval_', r1), ('proxy_variance_interval_', y1)]:
            lo = prefix + level + '_lower'
            hi = prefix + level + '_upper'
            cov = prefix + level + '_covered'
            sc = prefix + level + '_score'
            if lo in out and pd.notna(r.get(lo)):
                (low, high) = (float(r[lo]), float(r[hi]))
                alpha = 1 - int(level) / 100
                out.loc[i, cov] = bool(low <= val <= high)
                if sc in out:
                    out.loc[i, sc] = high - low + 2 / alpha * max(low - val, 0) + 2 / alpha * max(val - high, 0)
    for level in ['01', '05']:
        vcol = 'var_' + level
        ecol = 'es_' + level
        if vcol in out and np.isfinite(r.get(vcol, np.nan)):
            (v, e) = (float(r[vcol]), float(r[ecol]))
            alpha = int(level) / 100
            out.loc[i, 'var_' + level + '_exceedance'] = bool(r1 < v)
            out.loc[i, 'var_' + level + '_quantile_loss'] = float(quantile_loss(r1, v, alpha))
            out.loc[i, 'var_es_' + level + '_fz0'] = float(fz0_var_es_score(r1, v, e, alpha))
    probabilistic = np.isfinite(r.get('return_crps', np.nan)) or np.isfinite(r.get('return_log_score', np.nan))
    if cr and probabilistic:
        name = r.predictive_file
        if not isinstance(name, str) or not name:
            raise ValueError(f'Missing predictive component for {r.market}, {r.model_id}, {r.origin_date}, h={r.horizon}')
        path = options.components / name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = r.get('predictive_sha256')
        if isinstance(expected, str) and expected:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        loader = nullcontext(load_components(options.components, name, int(r.horizon)))
        with loader as z:
            ret = z['cumulative_returns']
            if np.isfinite(r.get('return_crps', np.nan)):
                np.testing.assert_allclose(crps_ensemble(ret, r0), r.return_crps, rtol=1e-09, atol=1e-12)
                out.loc[i, 'return_crps'] = crps_ensemble(ret, r1)
            if r.horizon == 1:
                args = dict(locations=z['locations'], conditional_sds=z['conditional_sds'])
                if r.return_distribution == 'gaussian':
                    ld = finite_mixture_gaussian_log_density
                    cdf = finite_mixture_gaussian_cdf
                else:
                    args['degrees_of_freedom'] = z['degrees_of_freedom']
                    ld = finite_mixture_standardized_t_log_density
                    cdf = finite_mixture_standardized_t_cdf
                check = -ld(r0, **args)
                np.testing.assert_allclose(check, r.return_log_score, rtol=1e-09, atol=1e-10)
                max_old_nls_error = max(max_old_nls_error, abs(check - r.return_log_score))
                logd = ld(r1, **args)
                out.loc[i, ['return_log_density', 'return_log_score', 'return_pit']] = [logd, -logd, cdf(r1, **args)]
            elif np.isfinite(r.get('return_pit', np.nan)):
                u = (r.return_pit * (len(ret) + 1) - np.sum(ret < r0)) / (np.sum(ret == r0) + 1)
                assert -1e-08 <= u <= 1 + 1e-08, (r.model_id, r.origin_date, u)
                out.loc[i, 'return_pit'] = (np.sum(ret < r1) + u * (np.sum(ret == r1) + 1)) / (len(ret) + 1)
        npz_count += 1
    changed.append(dict(market=r.market, model_id=r.model_id, origin_date=str(r.origin_date.date()), mature_date=str(r.mature_date.date()), horizon=int(r.horizon), variance_changed=cy, return_changed=cr, y_old=y0, y_new=y1, return_old=r0, return_new=r1, qlike_old=float(r.qlike), qlike_new=float(out.loc[i, 'qlike'])))
for col in K + ['variance_forecast', 'main_evaluation_eligible', 'target_spans_archive_gap']:
    pd.testing.assert_series_equal(orig[col], out[col], check_exact=True)
for col in ['qlike', 'variance_mse', 'return_crps', 'return_log_score']:
    mask = ~d.market.isin(change_dates)
    pd.testing.assert_series_equal(orig.loc[mask, col], out.loc[mask, col], check_exact=True)
orig.to_parquet(ROOT / 'original_scores.parquet', index=False)
out.to_parquet(ROOT / 'reconstructed_scores.parquet', index=False)
pd.DataFrame(changed).to_csv(ROOT / 'changed_forecasts.csv', index=False)
(ROOT / 'rescore_summary.json').write_text(json.dumps(dict(rows=len(out), affected_row_records=len(changed), forecast_arrays_read=npz_count, max_old_log_score_reproduction_error=max_old_nls_error, predictions_weights_and_eligibility_unchanged=True, source_data_unchanged=True, scope='Primary RV-CC variance scores, return scores/PIT/intervals and stored VaR-ES scores. Alternative tick/subsampled variance proxies are not repaired.'), indent=2) + '\n')
print(json.dumps(json.loads((ROOT / 'rescore_summary.json').read_text()), indent=2))
