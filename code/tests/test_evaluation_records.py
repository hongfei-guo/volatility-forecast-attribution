import json
import numpy as np
import pandas as pd
import pytest
from bnsv.evaluation_records import attach_functionals,target_identity,FUNCTIONALS


def inputs():
    key=dict(market='SP500',model_id='NN-SV',origin_date='2020-01-02',mature_date='2020-01-03',horizon=1)
    common=dict(**key,variance_forecast=.02,main_evaluation_eligible=True,
                target_spans_archive_gap=False,origin_follows_archive_gap=False,
                target_dates_json=json.dumps(['2020-01-03']))
    record=dict(**common,forecast_draw_count=4096,return_pit_method='cdf',
                return_log_score=.7,return_crps=.03,return_pit=.4)
    for c in FUNCTIONALS:record[c]=-.1 if c.endswith('_lower') or c.startswith('var_') else (-.2 if c.startswith('es_') else .1)
    records=pd.DataFrame([record])
    losses=pd.DataFrame([dict(**common,realized_variance=.03)])
    daily=pd.DataFrame([dict(market='SP500',date=pd.Timestamp('2020-01-03'),return_cc=-.05,rv_cc=.03)])
    expected=losses.assign(realized_cumulative_return=-.05)
    return records,losses,daily,target_identity(expected)


def test_functionals_recompute_coverage_and_preserve_distribution_scores():
    records,losses,daily,identity=inputs()
    result=attach_functionals(records,losses,daily,identity)
    assert result.return_log_score.iloc[0]==.7
    assert result.return_interval_90_covered.iloc[0]
    assert result.return_interval_90_width.iloc[0]==.2
    assert not result.var_01_exceedance.iloc[0]
    assert np.isfinite(result.var_es_01_fz0.iloc[0])


def test_records_reject_a_changed_outcome_even_when_dates_match():
    records,losses,daily,identity=inputs()
    daily.loc[0,'return_cc']=.06
    with pytest.raises(ValueError,match='different outcome panel'):
        attach_functionals(records,losses,daily,identity)


def test_reconstructed_scores_cannot_be_attached_to_an_unknown_forecast():
    records,losses,daily,identity=inputs()
    patch=records[['market','model_id','origin_date','mature_date','horizon',
                   'return_log_score','return_crps','return_pit']].copy()
    patch['model_id']='SV'
    with pytest.raises(ValueError,match='unknown forecast key'):
        attach_functionals(records,losses,daily,identity,patch)
