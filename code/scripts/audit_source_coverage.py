"""Read-only OMI/FirstRate source audit; diagnostics are not exclusion rules."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np, pandas as pd
import exchange_calendars as xcals
p = argparse.ArgumentParser()
p.add_argument('--data-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
source = a.data_root / 'raw/external/omi/oxfordmanrealizedvolatilityindices.csv'
fields = ['Unnamed: 0', 'Symbol', 'open_time', 'close_time', 'open_price', 'close_price', 'nobs', 'rv5', 'rsv']
r = pd.read_csv(source, usecols=fields)
mp = {'.SPX': ('SP500', 'XNYS', 'SPX', 'America/New_York'), '.FTSE': ('FTSE100', 'XLON', 'UKX', 'Europe/London'), '.GDAXI': ('DAX', 'XETR', 'DAX', 'Europe/Berlin')}
r = r[r.Symbol.isin(mp)].copy()
r['date'] = pd.to_datetime(r['Unnamed: 0'].str[:10])
r['market'] = r.Symbol.map({s: v[0] for (s, v) in mp.items()})
assert not r.duplicated(['market', 'date']).any()

def secs(v):
    v = v.astype(int)
    assert (v % 100 < 60).all() and (v // 100 % 100 < 60).all()
    return v // 10000 * 3600 + v // 100 % 100 * 60 + v % 100
r['observed_hours'] = (secs(r.close_time) - secs(r.open_time)) / 3600
source_records = []
hf_records = []
summaries = []
meta = []
non_sessions = []
for (symbol, (market, calendar, ticker, localtz)) in mp.items():
    s = r[r.market == market].copy().sort_values('date')
    cal = xcals.get_calendar(calendar, start=s.date.min() - pd.Timedelta(days=7), end=s.date.max() + pd.Timedelta(days=7))
    sch = cal.schedule
    s['expected_hours'] = s.date.map((sch.close - sch.open).dt.total_seconds() / 3600)
    assert s.expected_hours.notna().all()
    s['span_fraction'] = s.observed_hours / s.expected_hours
    s['coverage_flag'] = np.select([s.span_fraction < 0.5, s.span_fraction < 0.95], ['severe_short', 'shortened'], default='near_full_or_longer')
    s['phase'] = np.select([s.date < '2017-01-01', s.date < '2018-01-01'], ['through_2016', '2017'], default='evaluation')
    source_records.append(s)
    hf_path = a.data_root / f'raw/hf/firstrate/{ticker}_full_5min.txt'
    h = pd.read_csv(hf_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close'])
    dt = pd.to_datetime(h.timestamp)
    assert not dt.duplicated().any() and dt.is_monotonic_increasing
    assert np.isfinite(h[['open', 'high', 'low', 'close']]).all().all() and (h[['open', 'high', 'low', 'close']] > 0).all().all()
    assert ((h.high >= h[['open', 'close']].max(axis=1)) & (h.low <= h[['open', 'close']].min(axis=1)) & (h.high >= h.low)).all()
    h['utc'] = dt.dt.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise').dt.tz_convert('UTC')
    h['date'] = h.utc.dt.tz_convert(localtz).dt.tz_localize(None).dt.normalize()
    (first, last) = (h.date.min(), h.date.max())
    h = h[h.date <= s.date.max()]
    recs = []
    for (day, g) in h.groupby('date', sort=True):
        if day not in sch.index:
            non_sessions.append(dict(market=market, date=str(day.date()), rows=len(g), flat=bool(g[['open', 'high', 'low', 'close']].to_numpy().max() == g[['open', 'high', 'low', 'close']].to_numpy().min())))
            continue
        row = sch.loc[day]
        grid = pd.date_range(row.open, row.close, freq='5min', inclusive='left')
        core = g[(g.utc >= row.open) & (g.utc < row.close)].sort_values('utc')
        missing = grid.difference(pd.DatetimeIndex(core.utc))
        extra = pd.DatetimeIndex(core.utc).difference(grid)
        assert len(extra) == 0
        complete = len(missing) == 0
        vals = np.r_[core.open.iloc[0], core.close.to_numpy()] if len(core) else np.array([])
        rv = float(np.square(np.diff(np.log(vals))).sum()) if complete else np.nan
        recs.append(dict(market=market, date=day, hf_bars=len(core), expected_bars=len(grid), missing_bars=len(missing), hf_complete=complete, hf_open=core.open.iloc[0] if len(core) else np.nan, hf_continuous_close=core.close.iloc[-1] if len(core) else np.nan, hf_last_close=g.close.iloc[-1], hf_rv5=rv, missing_grid_utc_json=json.dumps([str(v) for v in missing])))
    hh = pd.DataFrame(recs)
    x = s.merge(hh, on=['market', 'date'], how='left', validate='one_to_one')
    x['rv_ratio_hf_omi'] = np.where(x.rv5 > 0, x.hf_rv5 / x.rv5, np.nan)
    x['close_difference_bps'] = 10000.0 * (x.hf_continuous_close / x.close_price - 1)
    hf_records.append(x)
    for (phase, z) in x.groupby('phase'):
        for (flag, g) in z.groupby('coverage_flag'):
            full = g[g.hf_complete == True]
            v = full[full.rv5 > 0]
            summaries.append(dict(market=market, phase=phase, coverage_flag=flag, n_omi=len(g), n_hf_complete=len(full), median_hf_omi_rv_ratio=v.rv_ratio_hf_omi.median(), median_abs_close_difference_bps=full.close_difference_bps.abs().median(), log_rv_correlation=v[['rv5', 'hf_rv5']].apply(np.log).corr().iloc[0, 1] if len(v) > 2 else np.nan))
    meta.append(dict(market=market, relative_file=str(hf_path.relative_to(a.data_root)), bytes=hf_path.stat().st_size, sha256=hashlib.sha256(hf_path.read_bytes()).hexdigest(), first_day=str(first.date()), last_day=str(last.date())))
    print(market, 'source', len(s), 'evaluation flags', s[s.phase == 'evaluation'].coverage_flag.value_counts().to_dict(), flush=True)
allr = pd.concat(source_records, ignore_index=True)
allh = pd.concat(hf_records, ignore_index=True)
allr.to_csv(a.output / 'all_session_metadata.csv', index=False)
allh.to_csv(a.output / 'source_hf_comparison.csv', index=False)
pd.DataFrame(summaries).to_csv(a.output / 'coverage_summary.csv', index=False)
pd.DataFrame(non_sessions).to_csv(a.output / 'hf_non_session_records.csv', index=False)
allh[allh.coverage_flag != 'near_full_or_longer'].to_csv(a.output / 'flagged_sessions.csv', index=False)
allh[(allh.phase == 'evaluation') & (allh.coverage_flag != 'near_full_or_longer')].to_csv(a.output / 'evaluation_flagged_sessions.csv', index=False)
(a.output / 'audit_scope.json').write_text(json.dumps(dict(source_relative_path=str(source.relative_to(a.data_root)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_date_rule='First 10 characters of vendor session label; no UTC date relabeling', assets=['SP500', 'FTSE100', 'DAX'], not_audited_other_assets=True, source_metadata_bands={'severe_short': 'duration < 50% of calendar expected session', 'shortened': '50% <= duration < 95%', 'near_full_or_longer': '>=95%'}, calendar_version=xcals.__version__, hf_timestamp='America/New_York, bar start', hf_rv_rule='first regular-bar open plus consecutive regular-bar closes, complete calendar grid only, excluding post-close auction bars', source_files=meta, limitations='Coverage bands are diagnostics, not validated universal exclusion rules. Historical calendar session hours and unusual closures require contextual review. Complete first/last spans do not rule out intraday holes.', canonical_source_modified=False), indent=2) + '\n')
