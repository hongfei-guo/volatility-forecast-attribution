from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/scripts/analyze_rv_ra_conditional_value.py"
SPEC = importlib.util.spec_from_file_location("rv_ra_conditional_value", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_asymmetry_comparisons_preserve_lag_one_timing() -> None:
    paths = 2
    horizon = 2
    refit = {
        "ancestry": np.array([7, 9]),
        "parameter__mu": np.array([0.0, 0.0]),
        "parameter__phi": np.array([0.0, 0.0]),
        "parameter__w1": np.zeros((2, 2, 1)),
        "parameter__b1": np.zeros((2, 1)),
        "parameter__output_weights": np.zeros((2, 1)),
        "parameter__centering_constant": np.array([0.0, 0.0]),
        "parameter__gamma_A": np.array([99.0, 1.0]),
        "parameter__q_centering_mean": np.array([0.0, 0.0]),
        "return_mean": np.array([0.0, 0.0]),
        "return_scale": np.array([2.0, 2.0]),
    }
    draws = {
        "source_particle_indices": np.zeros(paths, dtype=int),
        "source_parameter_ancestry": np.full(paths, 9, dtype=int),
        "origin_baseline_states": np.zeros(paths),
        "origin_z_history_tail": np.zeros(22),
        "origin_log_rv_history_tail": np.zeros(22),
        "origin_asymmetry_history_tail": np.r_[np.zeros(21), 0.5],
        "future_log_rv_paths": np.zeros((paths, horizon)),
        "future_asymmetry_paths": np.broadcast_to(
            np.array([0.2, 0.4]), (paths, horizon)
        ),
        "feature_mean": np.zeros(3),
        "feature_scale": np.ones(3),
        "state_innovations": np.zeros((paths, horizon)),
        "return_innovations": np.zeros((paths, horizon)),
    }

    predicted = MODULE._counterfactual_from_arrays(
        draws, refit, mode="predicted_asymmetry", realized_asymmetry=None
    )
    realized = MODULE._counterfactual_from_arrays(
        draws,
        refit,
        mode="realized_asymmetry",
        realized_asymmetry=np.array([-0.3, 0.9]),
    )
    no_asymmetry = MODULE._counterfactual_from_arrays(
        draws, refit, mode="no_asymmetry_term", realized_asymmetry=None
    )

    np.testing.assert_allclose(
        predicted["raw_variances"],
        np.broadcast_to(4.0 * np.exp([0.5, 0.2]), (paths, horizon)),
    )
    np.testing.assert_allclose(
        realized["raw_variances"],
        np.broadcast_to(4.0 * np.exp([0.5, -0.3]), (paths, horizon)),
    )
    np.testing.assert_allclose(no_asymmetry["raw_variances"], 4.0)


def test_fixed_effects_use_within_refit_variation() -> None:
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    x = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])[:, None]
    group_effect = np.where(groups == "a", 10.0, -4.0)
    y = group_effect + 2.5 * x[:, 0]
    coefficient = MODULE.fit_fixed_effects(y, x, groups)
    np.testing.assert_allclose(coefficient, [2.5], atol=1e-12)


def test_regime_support_requires_within_refit_variation() -> None:
    groups = np.array(["a", "a", "b", "b"])
    assert MODULE.varying_group_count(np.array([0, 1, 0, 0]), groups) == 1
    assert MODULE.varying_group_count(np.array([0, 0, 1, 1]), groups) == 0


def test_circular_blocks_are_deterministic_and_in_range() -> None:
    first = MODULE.circular_block_indices(
        17, block_length=5, rng=np.random.default_rng(20260730)
    )
    second = MODULE.circular_block_indices(
        17, block_length=5, rng=np.random.default_rng(20260730)
    )
    assert first.shape == (17,)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first < 17))


def test_equal_horizon_aggregation_does_not_weight_by_cell_count() -> None:
    values = np.array([[1.0, 10.0], [3.0, 14.0]])
    assert MODULE.aggregate_paired_difference(values) == 7.0


def test_paired_loss_loader_selects_the_common_grid(tmp_path: Path) -> None:
    origins = pd.date_range("2017-01-03", "2017-12-14", periods=241).strftime(
        "%Y-%m-%d"
    )
    rows = [
        {
            "origin_date": origin,
            "horizon": horizon,
            "qlike__RV-NN-SV": 1.0 + horizon / 100,
            "qlike__RV-RA-NN-SV": 0.99 + horizon / 100,
        }
        for origin in origins
        for horizon in (1, 5, 10)
    ]
    source = tmp_path / "paired_losses.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    selected, origins = MODULE.load_paired_losses(source)
    assert len(selected) == 723
    assert len(origins) == 241
    assert origins[0] == "2017-01-03"
    assert origins[-1] == "2017-12-14"
    assert selected.groupby("horizon")["origin_date"].nunique().to_dict() == {
        1: 241,
        5: 241,
        10: 241,
    }


def test_gamma_summary_uses_all_draws_for_12_calendar_refits(tmp_path: Path) -> None:
    origins = pd.date_range("2017-01-03", "2017-12-01", periods=12).strftime(
        "%Y-%m-%d"
    )
    source = tmp_path / "gamma_a_summary.csv"
    pd.DataFrame(
        {
            "task_id": np.arange(1, 13),
            "origin": origins,
            "slice": "calendar_month_refit",
            "width": 5,
            "draws": 8000,
            "gamma_A_mean": np.linspace(-0.01, 0.01, 12),
            "gamma_A_sd": 0.03,
            "gamma_A_q025": -0.05,
            "gamma_A_median": 0.0,
            "gamma_A_q975": 0.05,
        }
    ).to_csv(source, index=False)
    summary = MODULE.load_gamma_summary(source)
    assert len(summary) == 12
    assert set(summary["draws"].astype(int)) == {8000}
    assert summary.index[0] == "2017-01-03"
    assert summary.index[-1] == "2017-12-01"


def test_compact_histories_use_monthly_rv_floor_and_past_data(tmp_path):
    path = tmp_path / 'draws'
    path.mkdir()
    np.savez_compressed(path / '20170103_h5.npz', feature_mean=np.zeros(3))
    record = dict(task_root=str(tmp_path), origin_date='2017-01-03', horizon=5)
    dates = pd.date_range(end='2017-01-03', periods=22)
    daily = pd.DataFrame(dict(date=dates, return_cc=np.arange(22.) / 100,
                              rv_oc=np.full(22, .1), asymmetry=np.full(22, .3)))
    future = pd.DataFrame(dict(date=[pd.Timestamp('2017-01-04')],
                              return_cc=[999.], rv_oc=[999.], asymmetry=[999.]))
    refit = dict(return_mean=np.array([.01]), return_scale=np.array([.02]),
                 rv_floor=np.array(.2))
    a = MODULE.load_predictive_arrays(record, refit, daily)
    b = MODULE.load_predictive_arrays(record, refit, pd.concat([daily, future]))
    for name in ('origin_z_history_tail', 'origin_log_rv_history_tail',
                 'origin_asymmetry_history_tail'):
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_allclose(a['origin_log_rv_history_tail'], np.log(.2))
    np.testing.assert_allclose(a['origin_z_history_tail'], (daily.return_cc-.01)/.02)
    np.testing.assert_allclose(a['origin_asymmetry_history_tail'], .3)
