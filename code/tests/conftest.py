from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    n = 120
    dates = pd.bdate_range("2016-09-01", periods=n)
    overnight = rng.normal(0, 0.002, n)
    rv_oc = np.exp(rng.normal(-9.0, 0.35, n))
    share = rng.uniform(0.35, 0.65, n)
    rv_down = rv_oc * share
    rv_up = rv_oc - rv_down
    asym = (rv_down - rv_up) / (rv_down + rv_up + 1e-12)
    return_cc = overnight + rng.normal(0, np.sqrt(rv_oc))
    frame = pd.DataFrame(
        {
            "date": dates,
            "market": "SP500",
            "return_cc": return_cc,
            "return_overnight": overnight,
            "return_oc": return_cc - overnight,
            "rv_oc": rv_oc,
            "rv_cc": overnight**2 + rv_oc,
            "rv_down": rv_down,
            "rv_up": rv_up,
            "asymmetry": asym,
            "rv5": rv_oc,
            "rv5_ss": rv_oc * 1.01,
            "rv10": rv_oc * 0.98,
            "rv10_ss": rv_oc * 0.99,
            "rsv": rv_down,
            "rsv_ss": rv_down * 1.01,
            "bv": rv_oc * 0.90,
            "bv_ss": rv_oc * 0.91,
            "medrv": rv_oc * 0.88,
            "rk_parzen": rv_oc * 1.05,
            "rk_twoscale": rv_oc * 1.03,
            "rk_th2": rv_oc * 1.02,
            "rv5_ss_oc": rv_oc * 1.01,
            "rv10_oc": rv_oc * 0.98,
            "rv10_ss_oc": rv_oc * 0.99,
            "bv_oc": rv_oc * 0.90,
            "bv_ss_oc": rv_oc * 0.91,
            "medrv_oc": rv_oc * 0.88,
            "rk_parzen_oc": rv_oc * 1.05,
            "rk_twoscale_oc": rv_oc * 1.03,
            "rk_th2_oc": rv_oc * 1.02,
            "rv_down_ss": rv_down * 1.01,
            "source_symbol": ".SPX",
            "source_timestamp": dates.strftime("%Y-%m-%d 00:00:00+00:00"),
            "open_price": np.full(n, 100.0),
            "close_price": np.full(n, 100.0),
            "open_time": np.full(n, 93000.0),
            "close_time": np.full(n, 160000.0),
            "nobs": np.full(n, 1000.0),
        }
    )
    overnight_square = overnight**2
    for full_day, intraday in {
        "rv5_ss_cc": "rv5_ss_oc",
        "rv10_cc": "rv10_oc",
        "rv10_ss_cc": "rv10_ss_oc",
        "bv_cc": "bv_oc",
        "bv_ss_cc": "bv_ss_oc",
        "medrv_cc": "medrv_oc",
        "rk_parzen_cc": "rk_parzen_oc",
        "rk_twoscale_cc": "rk_twoscale_oc",
        "rk_th2_cc": "rk_th2_oc",
    }.items():
        frame[full_day] = frame[intraday] + overnight_square
    return frame
