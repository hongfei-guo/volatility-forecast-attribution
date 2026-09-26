"""Apply the two coverage screens and reconstruct fixed-forecast outcomes.

Inputs are user-provided daily data and the output of audit_source_coverage.py.
No model fit, prediction, availability flag or combination weight is changed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--daily', type=Path, required=True)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
s = pd.read_parquet(a.daily)
s.date = pd.to_datetime(s.date)
assert not s.duplicated(['market', 'date']).any()
s = s.sort_values(['market', 'date']).set_index(['market', 'date'])
c = pd.read_csv(a.audit, parse_dates=['date'])
assert len(c) > 0 and (not c.duplicated(['market', 'date']).any())
assert c[['nobs', 'expected_hours', 'span_fraction']].notna().all().all()
assert (c.nobs > 0).all() and (c.expected_hours > 0).all()
c['year'] = c.date.dt.year
c['rate'] = c.nobs / c.expected_hours
c['count_ratio'] = c.rate / c.groupby(['market', 'year']).rate.transform('median')
c['span_flag'] = c.span_fraction < 0.95
c['count_flag'] = c.count_ratio < 0.5
flags = c[(c.phase == 'evaluation') & (c.span_flag | c.count_flag)].copy()
assert len(flags) == 22 and flags.groupby('market').size().to_dict() == {'DAX': 1, 'FTSE100': 21}
assert flags.hf_complete.eq(True).all() and flags.missing_bars.eq(0).all()
assert flags[['hf_open', 'hf_continuous_close', 'hf_rv5']].notna().all().all()
flags.to_csv(a.output / 'coverage_flags.csv', index=False)
rows = []
for (m, g) in c[c.phase == 'evaluation'].groupby('market'):
    q = g[~g.span_flag & ~g.count_flag & g.hf_complete.eq(True) & (g.rv5 > 0) & (g.hf_rv5 > 0)]
    v = q.hf_rv5 / q.rv5
    assert len(v) > 0
    rows.append(dict(market=m, n=len(v), q25=v.quantile(0.25), median=v.median(), q75=v.quantile(0.75), iqr=v.quantile(0.75) - v.quantile(0.25)))
pd.DataFrame(rows).to_csv(a.output / 'complete_day_scale_iqr.csv', index=False)
new = s.copy()
for r in flags.itertuples():
    key = (r.market, r.date)
    for (f, value) in [('open_price', r.hf_open), ('close_price', r.hf_continuous_close), ('rv_oc', r.hf_rv5)]:
        new.loc[key, f] = value
changes = []
for (market, g) in new.groupby(level=0):
    dates = g.index.get_level_values('date')
    direct = set(flags.loc[flags.market == market, 'date'])
    affected = direct | {dates[i + 1] for (i, d) in enumerate(dates[:-1]) if d in direct}
    prev = g.close_price.shift(1)
    for date in affected:
        k = (market, date)
        new.loc[k, 'return_cc'] = np.log(new.loc[k, 'close_price'] / prev.loc[k])
        new.loc[k, 'return_overnight'] = np.log(new.loc[k, 'open_price'] / prev.loc[k])
        new.loc[k, 'rv_cc'] = new.loc[k, 'rv_oc'] + new.loc[k, 'return_overnight'] ** 2
    for date in direct:
        k = (market, date)
        new.loc[k, 'return_oc'] = np.log(new.loc[k, 'close_price'] / new.loc[k, 'open_price'])
    for date in affected:
        k = (market, date)
        for f in ['open_price', 'close_price', 'rv_oc', 'return_cc', 'return_overnight', 'return_oc', 'rv_cc']:
            if not np.isclose(s.loc[k, f], new.loc[k, f], rtol=1e-12, atol=1e-15):
                changes.append(dict(market=market, date=date, field=f, original=s.loc[k, f], reconstructed=new.loc[k, f]))
new.reset_index().to_parquet(a.output / 'daily_reconstructed.parquet', index=False)
pd.DataFrame(changes).to_csv(a.output / 'outcome_changes.csv', index=False)
print('Reconstructed 22 source sessions; forecasts and inputs to fitted models unchanged.')
