from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bnsv.forecast_combination import (
    WEIGHT_COLUMNS,
    exponential_discount_weights,
    generate_combination_panels,
    load_combination_design,
)


CANDIDATES = ("NN-SV", "RV-NN-SV")


def _design() -> dict[str, object]:
    return {
        "seed": 20260730,
        "markets": ["M"],
        "horizons": [1, 5, 10],
        "candidate_models": list(CANDIDATES),
        "sample": {
            "objective_start": "2017-01-01",
            "forecast_start": "2018-01-01",
            "forecast_end": "2018-02-28",
        },
        "weighting": {
            "half_life_sessions": 2,
            "update_frequency": "calendar_month",
            "evaluation_sample_policy": "main_gap_excluded",
            "eligibility_column": "main_evaluation_eligible",
            "availability_column": "available",
            "modes": {
                "equal": [1, 5, 10],
                "qlike": [1, 5, 10],
                "log_score": [1],
            },
        },
    }


def _daily() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2017-12-27",
            "2017-12-28",
            "2017-12-29",
            "2018-01-02",
            "2018-01-03",
            "2018-01-04",
            "2018-02-01",
            "2018-02-02",
        ]
    )
    return pd.DataFrame(
        {
            "market": "M",
            "date": dates,
            "rv_cc": [0.5, 1.0, 0.8, 1.2, 2.0, 3.0, 1.5, 2.5],
        }
    )


def _calendar_report() -> dict[str, object]:
    daily = _daily()
    return {
        "valid": True,
        "markets": [
            {
                "market": "M",
                "first_date": str(daily["date"].min().date()),
                "last_date": str(daily["date"].max().date()),
                "expected_sessions": len(daily) + 1,
                "missing_expected_sessions": ["2018-01-05"],
                "observed_non_sessions": [],
                "source_label_mismatches": [],
            }
        ],
    }


def _candidate_forecasts() -> pd.DataFrame:
    rows = []
    for origin, mature in (
        ("2018-01-02", "2018-01-03"),
        ("2018-01-03", "2018-01-04"),
        ("2018-02-01", "2018-02-02"),
    ):
        for model_index, model in enumerate(CANDIDATES):
            if origin == "2018-01-03" and model == "RV-NN-SV":
                continue
            rows.append(
                {
                    "market": "M",
                    "model_id": model,
                    "origin_date": origin,
                    "horizon": 1,
                    "mature_date": mature,
                    "variance_forecast": 1.0 + 3.0 * model_index,
                    "cumulative_return_mean": 0.01 + 0.02 * model_index,
                }
            )
    return pd.DataFrame(rows)


def _warmup_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": "M",
                "model_id": model,
                "origin_date": "2017-12-27",
                "horizon": 1,
                "mature_date": "2017-12-28",
                "variance_forecast": forecast,
                "available": True,
            }
            for model, forecast in zip(CANDIDATES, (1.0, 4.0), strict=True)
        ]
    )


def _main_losses() -> pd.DataFrame:
    rows = []
    for origin, mature, realized in (
        ("2018-01-02", "2018-01-03", 2.0),
        ("2018-01-03", "2018-01-04", 3.0),
    ):
        for model_index, model in enumerate(CANDIDATES):
            if origin == "2018-01-03" and model == "RV-NN-SV":
                continue
            rows.append(
                {
                    "market": "M",
                    "model_id": model,
                    "origin_date": origin,
                    "horizon": 1,
                    "mature_date": mature,
                    "realized_variance": realized,
                    "variance_forecast": realized + 3.0 * model_index,
                    "main_evaluation_eligible": True,
                    "return_log_density": -0.1 - model_index,
                }
            )
    return pd.DataFrame(rows)


def test_qlike_weights_use_matured_session_discounting_and_availability() -> None:
    ensemble, weights = generate_combination_panels(
        daily=_daily(),
        forecasts=_candidate_forecasts(),
        design=_design(),
        losses=_main_losses(),
        warmup_forecasts=_warmup_forecasts(),
        calendar_report=_calendar_report(),
        selected_modes=("equal", "qlike"),
    )

    assert len(ensemble) == 6
    assert len(weights) == 12
    january = weights[
        (weights["mode"] == "qlike") & (weights["update_date"] == "2018-01-02")
    ].set_index("model_id")
    assert set(january["matured_observations"]) == {1}
    assert january.loc["NN-SV", "prequential_weight"] > 0.999
    availability_change = weights[
        (weights["mode"] == "equal") & (weights["update_date"] == "2018-01-03")
    ].set_index("model_id")
    assert availability_change.loc["NN-SV", "availability_adjusted_weight"] == 1.0
    assert availability_change.loc["RV-NN-SV", "availability_adjusted_weight"] == 0.0
    february = weights[
        (weights["mode"] == "qlike") & (weights["update_date"] == "2018-02-01")
    ]
    assert set(february["matured_observations"]) == {2}
    missing_member_row = ensemble[
        (ensemble["model_id"] == "NN-ENSEMBLE-EQUAL")
        & (ensemble["origin_date"] == "2018-01-03")
    ].iloc[0]
    assert missing_member_row["combination_weights"] == [1.0, 0.0]
    assert missing_member_row["variance_forecast"] == 1.0
    assert missing_member_row["forecast_kind"] == "variance_point_only"

    session_weights = exponential_discount_weights(np.asarray([2, 1]), half_life=1.0)
    calendar_weights = exponential_discount_weights(np.asarray([4, 1]), half_life=1.0)
    assert not np.allclose(session_weights, calendar_weights)


def _packaged_weights(*, include_february: bool = True) -> pd.DataFrame:
    rows = []
    for update in ["2018-01-02", "2018-02-01"] if include_february else ["2018-01-02"]:
        for model, weight in zip(CANDIDATES, (0.25, 0.75), strict=True):
            rows.append(
                {
                    "market": "M",
                    "horizon": 1,
                    "mode": "log_score",
                    "update_date": update,
                    "model_id": model,
                    "available": True,
                    "discount_age_unit": "exchange_session",
                    "evaluation_sample_policy": "main_gap_excluded",
                    "matured_observations": 10,
                    "prequential_weight": weight,
                    "availability_adjusted_weight": weight,
                }
            )
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


def _component_forecasts(
    root: Path,
    *,
    horizon: int,
    origins: tuple[tuple[str, str], ...],
    draws: int = 12,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    archives: dict[str, dict[str, np.ndarray]] = {}
    for origin, mature in origins:
        for model_index, model in enumerate(CANDIDATES):
            offset = 1000.0 * model_index
            draw = np.arange(draws, dtype=float)[:, None]
            step = np.arange(horizon, dtype=float)[None, :]
            arrays = {
                "raw_returns": offset + 10.0 * draw + step,
                "raw_variances": 1.0 + offset + 10.0 * draw + step,
                "raw_return_locations": offset + 20.0 * draw + step,
                "raw_conditional_sds": 0.5 + offset + 10.0 * draw + step,
                "degrees_of_freedom": 5.0 + offset + np.arange(draws, dtype=float),
            }
            filename = f"{model}_{origin}_h{horizon}.npz"
            np.savez_compressed(root / filename, **arrays)
            archives[f"{model}_{origin}"] = arrays
            rows.append(
                {
                    "market": "M",
                    "model_id": model,
                    "origin_date": origin,
                    "horizon": horizon,
                    "mature_date": mature,
                    "variance_forecast": 2.0 + model_index,
                    "cumulative_return_mean": 0.1 + model_index,
                    "return_distribution": "variance_standardized_student_t",
                    "predictive_file": filename,
                }
            )
    return pd.DataFrame(rows), archives


def _component_weights(
    *,
    horizon: int,
    modes: tuple[str, ...],
    updates: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for mode in modes:
        for update in updates:
            for model, weight in zip(CANDIDATES, (0.5, 0.5), strict=True):
                rows.append(
                    {
                        "market": "M",
                        "horizon": horizon,
                        "mode": mode,
                        "update_date": update,
                        "model_id": model,
                        "available": True,
                        "discount_age_unit": "exchange_session",
                        "evaluation_sample_policy": "main_gap_excluded",
                        "matured_observations": 10,
                        "prequential_weight": weight,
                        "availability_adjusted_weight": weight,
                    }
                )
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


def test_packaged_weight_path_generates_conditional_log_score_mean_panel() -> None:
    forecasts = _candidate_forecasts()
    forecasts = forecasts[forecasts["origin_date"].isin(["2018-01-02", "2018-02-01"])]
    ensemble, weights = generate_combination_panels(
        daily=_daily(),
        forecasts=forecasts,
        design=_design(),
        packaged_weights=_packaged_weights(),
        selected_modes=("log_score",),
    )

    assert len(ensemble) == 2
    assert len(weights) == 4
    assert set(ensemble["forecast_kind"]) == {"variance_point_only"}
    assert np.allclose(ensemble["variance_forecast"], 3.25)
    assert np.allclose(ensemble["cumulative_return_mean"], 0.025)
    assert "predictive_file" not in ensemble


def test_whole_path_mixture_selects_complete_candidate_paths(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    mixture_root = tmp_path / "mixtures"
    candidate_root.mkdir()
    origins = (
        ("2018-01-02", "2018-01-09"),
        ("2018-02-01", "2018-02-08"),
    )
    forecasts, archives = _component_forecasts(
        candidate_root,
        horizon=5,
        origins=origins,
    )
    ensemble, _ = generate_combination_panels(
        daily=_daily(),
        forecasts=forecasts,
        design=_design(),
        packaged_weights=_component_weights(
            horizon=5,
            modes=("equal",),
            updates=("2018-01-02", "2018-02-01"),
        ),
        selected_modes=("equal",),
        predictive_root=candidate_root,
        predictive_output_dir=mixture_root,
    )

    assert set(ensemble["forecast_kind"]) == {"probabilistic"}
    assert set(ensemble["return_distribution"]) == {
        "variance_standardized_student_t"
    }
    assert set(ensemble["forecast_draw_count"]) == {12}
    rng = np.random.default_rng(20260730)
    for origin, _ in origins:
        expected_members = rng.choice(2, size=12, p=np.asarray([0.5, 0.5]))
        expected_sources = rng.integers(0, 12, size=12)
        record = ensemble[ensemble["origin_date"] == origin].iloc[0]
        with np.load(mixture_root / record["predictive_file"]) as mixture:
            assert np.array_equal(
                mixture["source_model_indices"], expected_members
            )
            assert np.array_equal(mixture["source_draw_indices"], expected_sources)
            for name in (
                "raw_returns",
                "raw_variances",
                "raw_return_locations",
                "raw_conditional_sds",
                "degrees_of_freedom",
            ):
                stacked = np.stack(
                    [
                        archives[f"{model}_{origin}"][name]
                        for model in CANDIDATES
                    ]
                )
                expected = stacked[expected_members, expected_sources]
                assert np.array_equal(mixture[name], expected)
            assert record["cumulative_return_mean"] == float(
                np.mean(np.sum(mixture["raw_returns"], axis=1))
            )


def test_each_combination_cell_starts_from_the_documented_seed(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidates"
    mixture_root = tmp_path / "mixtures"
    candidate_root.mkdir()
    forecasts, _ = _component_forecasts(
        candidate_root,
        horizon=1,
        origins=(("2018-01-02", "2018-01-03"),),
    )
    ensemble, _ = generate_combination_panels(
        daily=_daily(),
        forecasts=forecasts,
        design=_design(),
        packaged_weights=_component_weights(
            horizon=1,
            modes=("equal", "log_score"),
            updates=("2018-01-02",),
        ),
        selected_modes=("equal", "log_score"),
        predictive_root=candidate_root,
        predictive_output_dir=mixture_root,
    )

    files = {
        row.combination_mode: mixture_root / row.predictive_file
        for row in ensemble.itertuples(index=False)
    }
    with np.load(files["equal"]) as equal, np.load(files["log_score"]) as log_score:
        assert np.array_equal(
            equal["source_model_indices"], log_score["source_model_indices"]
        )
        assert np.array_equal(
            equal["source_draw_indices"], log_score["source_draw_indices"]
        )


def test_conditional_generation_requires_every_monthly_weight_update() -> None:
    forecasts = _candidate_forecasts()
    forecasts = forecasts[forecasts["origin_date"].isin(["2018-01-02", "2018-02-01"])]
    with pytest.raises(ValueError, match="lack required monthly updates"):
        generate_combination_panels(
            daily=_daily(),
            forecasts=forecasts,
            design=_design(),
            packaged_weights=_packaged_weights(include_february=False),
            selected_modes=("log_score",),
        )


def test_prediction_only_warmup_cannot_regenerate_log_score_weights() -> None:
    with pytest.raises(ValueError, match="evaluated log densities"):
        generate_combination_panels(
            daily=_daily(),
            forecasts=_candidate_forecasts(),
            design=_design(),
            losses=_main_losses(),
            warmup_forecasts=_warmup_forecasts(),
            selected_modes=("log_score",),
        )


def test_packaged_warmup_contains_predictions_only() -> None:
    package = Path(__file__).resolve().parents[2]
    panel = pd.read_parquet(package / "forecasts" / "combination_warmup.parquet")
    assert list(panel.columns) == [
        "market",
        "model_id",
        "origin_date",
        "horizon",
        "target_start",
        "target_end",
        "mature_date",
        "origin_session_index",
        "mature_session_index",
        "available",
        "variance_forecast",
    ]
    assert len(panel) == 4434
    assert int(panel["available"].sum()) == 4368
    assert panel.loc[panel["available"], "variance_forecast"].notna().all()
    assert panel.loc[~panel["available"], "variance_forecast"].isna().all()
    forbidden = {
        "realized_variance",
        "return_log_density",
        "return_cc",
        "rv_cc",
        "data_file_sha256",
    }
    assert not (forbidden & set(panel.columns))


def test_packaged_weights_reconstruct_all_variance_mean_combinations() -> None:
    package = Path(__file__).resolve().parents[2]
    panels = pd.concat(
        [
            pd.read_parquet(package / "forecasts" / f"{market}.parquet")
            for market in ("SP500", "FTSE100", "DAX")
        ],
        ignore_index=True,
    )
    candidate_panels = panels[panels["model_id"].isin(CANDIDATES)].copy()
    reproduced, _ = generate_combination_panels(
        daily=pd.DataFrame(),
        forecasts=candidate_panels,
        design=load_combination_design(package / "design" / "combination.yaml"),
        packaged_weights=pd.read_csv(
            package / "forecasts" / "combination_weights.csv"
        ),
    )
    packaged = panels[panels["model_id"].str.startswith("NN-ENSEMBLE")].copy()
    keys = ["market", "model_id", "origin_date", "horizon"]
    reproduced["origin_date"] = reproduced["origin_date"].astype(str)
    packaged["origin_date"] = packaged["origin_date"].astype(str)
    aligned = reproduced.merge(
        packaged,
        on=keys,
        suffixes=("_reproduced", "_packaged"),
        validate="one_to_one",
    )
    assert len(aligned) == 21_686
    np.testing.assert_allclose(
        aligned["variance_forecast_reproduced"],
        aligned["variance_forecast_packaged"],
        rtol=0.0,
        atol=2e-18,
    )
