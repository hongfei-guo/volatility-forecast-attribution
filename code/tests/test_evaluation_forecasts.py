import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_prepare_excludes_warmup_and_does_not_duplicate_probability_panels():
    script = Path(__file__).resolve().parents[1]/'scripts/prepare_evaluation_forecasts.py'
    spec = importlib.util.spec_from_file_location('prepare_evaluation_forecasts', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    components = pd.DataFrame([
        dict(market='DAX', model_id='SV', origin_date='2017-12-28', mature_date='2017-12-29', horizon=1, scope='warmup', forecast_kind='probabilistic', predictive_file='a.npz'),
        dict(market='DAX', model_id='SV', origin_date='2018-01-02', mature_date='2018-01-03', horizon=1, scope='evaluation', forecast_kind='probabilistic', predictive_file='b.npz')])
    points = pd.DataFrame([
        dict(market='DAX', model_id='SV', origin_date='2018-01-02', mature_date='2018-01-03', horizon=1, forecast_kind='probabilistic'),
        dict(market='DAX', model_id='HAR-RV', origin_date='2018-01-02', mature_date='2018-01-03', horizon=1, forecast_kind='variance_point_only')])
    result = module.prepare(components, [points], pd.Timestamp('2018-01-01'), pd.Timestamp('2022-02-25'))
    assert len(result) == 2 and set(result.model_id) == {'SV', 'HAR-RV'}
    assert result.origin_date.min() == pd.Timestamp('2018-01-02')
    with pytest.raises(ValueError, match='unique'):
        module.prepare(components, [points, points], pd.Timestamp('2018-01-01'), pd.Timestamp('2022-02-25'))
