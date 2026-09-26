"""Exact evaluation records for the reported outcomes and predictive functionals."""
import hashlib
import json
import numpy as np
import pandas as pd
from .evaluation import quantile_loss, fz0_var_es_score, qlike, variance_mse

KEYS=['market','model_id','origin_date','mature_date','horizon']
SCORES=['return_log_score','return_crps','return_pit']
METADATA=KEYS+['forecast_kind','variance_forecast','variance_mean_method','return_distribution',
               'forecast_draw_count','target_dates_json','main_evaluation_eligible',
               'target_spans_archive_gap','origin_follows_archive_gap','return_pit_method']
FUNCTIONALS=[f'{v}_{l}' for l in ['01','05'] for v in ['var','es']]+[
    f'return_interval_{l}_{s}' for l in ['50','90','95'] for s in ['lower','upper']]


def ordered(frame):
    frame=frame.copy()
    for c in ['origin_date','mature_date']:frame[c]=pd.to_datetime(frame[c],errors='raise').dt.normalize()
    if frame[KEYS].isna().any(axis=None) or frame.duplicated(KEYS).any():
        raise ValueError('evaluation records have missing or duplicate forecast keys')
    return frame.sort_values(KEYS).reset_index(drop=True)


def target_identity(frame):
    """Bind stored target-dependent scores to one complete ordered outcome panel."""
    d=ordered(frame);h=hashlib.sha256()
    key=d[KEYS].copy()
    for c in ['origin_date','mature_date']:key[c]=key[c].dt.strftime('%Y-%m-%d')
    h.update(key.to_csv(index=False).encode())
    values=d[['realized_variance','realized_cumulative_return']].to_numpy(dtype='<f8')
    if not np.isfinite(values).all():raise ValueError('outcome identity requires finite values')
    h.update(values.tobytes())
    return h.hexdigest()


def attach_functionals(records, variance_losses, daily, expected_target_identity, patch=None):
    records=ordered(records);result=ordered(variance_losses)
    pd.testing.assert_frame_equal(records[KEYS],result[KEYS],check_exact=True)
    for c in ['variance_forecast','main_evaluation_eligible','target_spans_archive_gap','origin_follows_archive_gap']:
        pd.testing.assert_series_equal(records[c],result[c],check_names=False,check_exact=True)
    for x,y in zip(records.target_dates_json,result.target_dates_json):
        if json.loads(x)!=json.loads(y):raise ValueError('target window differs from evaluation record')
    frames={m:g.sort_values('date').set_index('date') for m,g in daily.groupby('market')}
    cache={};returns=[];variances=[]
    for r in result.itertuples():
        key=(r.market,r.origin_date,int(r.horizon))
        if key not in cache:
            dates=pd.to_datetime(json.loads(r.target_dates_json))
            if len(dates)!=r.horizon or dates[-1]!=r.mature_date:
                raise ValueError('incomplete or misaligned outcome window')
            cache[key]=(float(frames[r.market].loc[dates,'return_cc'].sum()),
                        float(frames[r.market].loc[dates,'rv_cc'].sum()))
        returns.append(cache[key][0]);variances.append(cache[key][1])
    result['realized_cumulative_return']=returns
    result['realized_variance']=variances
    result['realized_variance__rv_cc']=variances
    result['qlike']=qlike(np.asarray(variances),result.variance_forecast.to_numpy(float))
    result['variance_mse']=variance_mse(np.asarray(variances),result.variance_forecast.to_numpy(float))
    result['qlike__rv_cc']=result.qlike
    result['variance_mse__rv_cc']=result.variance_mse
    if target_identity(result)!=expected_target_identity:
        raise ValueError('stored distribution scores belong to a different outcome panel')
    for c in ['forecast_draw_count','return_pit_method',*SCORES,*FUNCTIONALS]:result[c]=records[c]
    if patch is not None:
        patch=ordered(patch)
        indexes=pd.MultiIndex.from_frame(result[KEYS]);positions=indexes.get_indexer(pd.MultiIndex.from_frame(patch[KEYS]))
        if (positions<0).any():raise ValueError('reconstructed score record has an unknown forecast key')
        result.loc[positions,SCORES]=patch[SCORES].to_numpy()
    result['return_log_density']=-result.return_log_score
    y=result.realized_cumulative_return.to_numpy(float)
    for level in ['50','90','95']:
        prefix=f'return_interval_{level}_';lo=result[prefix+'lower'].to_numpy(float);hi=result[prefix+'upper'].to_numpy(float)
        valid=np.isfinite(lo)&np.isfinite(hi)
        if (lo[valid]>hi[valid]).any():raise ValueError('invalid interval endpoints')
        result[prefix+'covered']=pd.Series([None]*len(result),dtype=object)
        result.loc[valid,prefix+'covered']=(lo[valid]<=y[valid])&(y[valid]<=hi[valid])
        result[prefix+'width']=hi-lo
        alpha=1-int(level)/100
        result[prefix+'score']=hi-lo+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0)
    for level in ['01','05']:
        v=result[f'var_{level}'].to_numpy(float);e=result[f'es_{level}'].to_numpy(float);valid=np.isfinite(v)&np.isfinite(e);alpha=int(level)/100
        result[f'var_{level}_exceedance']=pd.Series([None]*len(result),dtype=object)
        result.loc[valid,f'var_{level}_exceedance']=y[valid]<v[valid]
        result[f'var_{level}_quantile_loss']=np.nan
        result.loc[valid,f'var_{level}_quantile_loss']=[quantile_loss(yy,vv,alpha) for yy,vv in zip(y[valid],v[valid])]
        result[f'var_es_{level}_fz0']=np.nan;ok=valid&(e<0)
        result.loc[ok,f'var_es_{level}_fz0']=[fz0_var_es_score(yy,vv,ee,alpha) for yy,vv,ee in zip(y[ok],v[ok],e[ok])]
    return result
