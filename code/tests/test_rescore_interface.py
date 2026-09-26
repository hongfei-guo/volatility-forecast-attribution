from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from bnsv.evaluation import crps_ensemble, finite_mixture_gaussian_log_density, qlike


def test_rescore_multiple_density_records_from_explicit_inputs(tmp_path):
    dates = pd.date_range('2020-01-01', periods=3)
    daily = pd.DataFrame({'market': 'DAX', 'date': dates, 'return_cc': [0.0, 0.1, -0.1], 'rv_cc': 1.0})
    corrected = daily.copy()
    corrected.loc[1:, 'return_cc'] = [0.2, -0.2]
    corrected.loc[1:, 'rv_cc'] = [1.2, 1.3]
    rows = []
    components = tmp_path / 'components'
    components.mkdir()
    returns = np.linspace(-1, 1, 20)
    for i in [0, 1]:
        filename = f'forecast_{i}.npz'
        np.savez_compressed(components / filename, raw_returns=returns[:, None], raw_variances=np.ones((20, 1)),
                            raw_return_locations=np.zeros((20, 1)), raw_conditional_sds=np.ones((20, 1)))
        target = float(daily.loc[i + 1, 'return_cc'])
        rows.append(dict(market='DAX', model_id='SV', origin_date=dates[i],
                         mature_date=dates[i + 1], horizon=1,
                         target_dates_json=json.dumps([str(dates[i + 1].date())]),
                         main_evaluation_eligible=True, target_spans_archive_gap=False,
                         variance_forecast=1.0, realized_variance=1.0,
                         realized_cumulative_return=target, qlike=float(qlike(1.0, 1.0)),
                         variance_mse=0.0, return_crps=crps_ensemble(returns, target),
                         return_log_score=-finite_mixture_gaussian_log_density(
                             target, locations=np.zeros(20), conditional_sds=np.ones(20)),
                         return_distribution='gaussian', predictive_file=filename))
    daily.to_parquet(tmp_path / 'daily.parquet', index=False)
    corrected.to_parquet(tmp_path / 'corrected.parquet', index=False)
    pd.DataFrame(rows).to_parquet(tmp_path / 'scores.parquet', index=False)
    script = Path(__file__).resolve().parents[1] / 'scripts/rescore_reconstructed_outcomes.py'
    subprocess.run([sys.executable, str(script), '--daily', str(tmp_path / 'daily.parquet'),
                    '--reconstructed-daily', str(tmp_path / 'corrected.parquet'),
                    '--scores', str(tmp_path / 'scores.parquet'), '--components', str(components),
                    '--output', str(tmp_path / 'results')], check=True, capture_output=True)
    result = pd.read_parquet(tmp_path / 'results/reconstructed_scores.parquet')
    np.testing.assert_allclose(result.qlike, qlike(np.array([1.2, 1.3]), np.ones(2)))
    np.testing.assert_allclose(result.return_log_score, 0.5 * np.log(2 * np.pi) + 0.5 * .2**2)
    assert result.variance_forecast.eq(1.0).all()
