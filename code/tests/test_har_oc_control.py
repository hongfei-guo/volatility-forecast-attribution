import numpy as np
import pandas as pd

from bnsv.benchmarks import fit_har_levels, forecast_har_levels
from bnsv.sample_alignment import GAP_POLICY
from scripts.run_har_oc_control import CC, OC, NN, reconstruct_pair, summarize


def test_oc_regressors_keep_cc_response_and_do_not_read_future():
    rng = np.random.default_rng(731)
    dates = pd.bdate_range("2016-01-01", periods=110)
    oc = rng.uniform(0.0001, 0.0006, len(dates))
    cc = oc + rng.uniform(0.00001, 0.0002, len(dates))
    daily = pd.DataFrame({"date": dates, "rv_cc": cc, "rv_oc": oc})
    refits = dates[[75, 80]]
    rows = []
    for i in range(75, 88):
        if dates[i] in refits:
            fit = fit_har_levels(cc[:i + 1])
        rows.append({"market": "SP500", "model_id": CC, "origin_date": dates[i],
                     "horizon": 1, "mature_date": dates[i + 1],
                     "variance_forecast": forecast_har_levels(cc[:i + 1], fit, 1)[0]})
    reference = pd.DataFrame(rows)
    panel, coefficients, check = reconstruct_pair(daily, reference, refits)
    # Independent direct regression: the dependent variable is CC, never OC.
    x = np.array([[1, oc[i - 1], oc[i - 5:i].mean(), oc[i - 22:i].mean()]
                  for i in range(22, 76)])
    beta = np.linalg.lstsq(x, cc[22:76], rcond=None)[0]
    expected = np.array([1, oc[75], oc[71:76].mean(), oc[54:76].mean()]) @ beta
    assert np.isclose(panel.loc[panel.model_id.eq(OC), "variance_forecast"].iloc[0], expected,
                      rtol=1e-13, atol=0)
    changed = daily.copy()
    changed.loc[88:, ["rv_cc", "rv_oc"]] *= 100
    after, _, _ = reconstruct_pair(changed, reference, refits)
    pd.testing.assert_frame_equal(panel, after)
    assert len(coefficients) == 4 and check["max_absolute_baseline_difference"] == 0


def test_primary_pair_keeps_dates_when_context_model_is_missing_and_excludes_gaps():
    rows = []
    dates = pd.bdate_range("2018-01-02", periods=14)
    for m, offset in ((CC, 0), (OC, 0.1), (NN, 0.2)):
        for i in range(13):
            if m == NN and i == 3:
                continue
            rows.append({"market": "SP500", "origin_date": dates[i],
                         "mature_date": dates[i+1], "horizon": 1, "model_id": m,
                         "qlike__rv_cc": 0.5 + offset + i * 0.02,
                         "gap_policy": GAP_POLICY, "target_spans_archive_gap": i == 2,
                         "main_evaluation_eligible": i != 2})
    summary, _ = summarize(pd.DataFrame(rows))
    assert summary.n_common.tolist() == [12, 11, 11]
    assert np.isclose(summary.iloc[0].mean_loss_difference, 0.1)
