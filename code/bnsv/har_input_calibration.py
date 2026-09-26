"""The existing correction-scale calibration on four pre-2017 OC inputs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .forecast_features import fit_feature_transform, prepare_estimation_data
from .forecast_fit import NETWORK_WIDTH


CALIBRATION_SEED = 20260730
NEURAL_DRAWS = 500
LINEAR_DRAWS = 2000
TARGET_FUNCTION_SD = 0.10
REFERENCE_END = "2016-12-31"


def calibrate_har_input_scales(daily: pd.DataFrame) -> dict[str, object]:
    """Use the original prior-draw scheme, population SD and reference cutoff.

    The archived prior calibration used 500 neural and 2,000 linear draws,
    both seeded with 20260730. It excluded the first complete reference row.
    Scale homogeneity permits calibration from unit-scale functions without
    repeated root-finding; no forecast losses enter this calculation.
    """
    dates = pd.to_datetime(daily["date"], errors="raise")
    reference = daily.loc[dates <= pd.Timestamp(REFERENCE_END)].copy()
    transform = fit_feature_transform(
        reference, model_id="RV-NN-SV-HAR", training_end=REFERENCE_END
    )
    estimation = prepare_estimation_data(reference, transform)
    x = estimation.x[1:]
    if x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("HAR prior calibration requires complete pre-2017 input rows")
    dimension = x.shape[1]
    rng = np.random.default_rng(CALIBRATION_SEED)
    neural_sd = np.empty(NEURAL_DRAWS)
    for index in range(NEURAL_DRAWS):
        w1 = rng.normal(0.0, 1.0 / math.sqrt(dimension), (dimension, NETWORK_WIDTH))
        bias = rng.normal(0.0, 0.2, NETWORK_WIDTH)
        weights = np.sort(rng.lognormal(0.0, 0.2, NETWORK_WIDTH)) / math.sqrt(NETWORK_WIDTH)
        values = np.tanh(x @ w1 + bias) @ weights
        neural_sd[index] = np.std(values - values.mean(), ddof=0)

    rng = np.random.default_rng(CALIBRATION_SEED)
    beta = rng.normal(0.0, 1.0 / math.sqrt(dimension), (LINEAR_DRAWS, dimension))
    # Limit temporary matrices while retaining the original RNG draw order.
    centered = x - x.mean(axis=0)
    linear_sd = np.concatenate([
        np.std(centered @ batch.T, axis=0, ddof=0)
        for batch in np.array_split(beta, 20)
    ])
    medians = {"RV-NN-SV-HAR": float(np.median(neural_sd)),
               "RV-LIN-SV-HAR": float(np.median(linear_sd))}
    if any(not np.isfinite(value) or value <= 0 for value in medians.values()):
        raise ValueError("reference inputs do not identify a finite correction scale")
    return {
        "reference_end": REFERENCE_END,
        "first_reference_row": str(estimation.dates[1].date()),
        "last_reference_row": str(estimation.dates[-1].date()),
        "reference_rows": len(x),
        "feature_names": list(transform.feature_names),
        "seed": CALIBRATION_SEED,
        "neural_draws": NEURAL_DRAWS,
        "linear_draws": LINEAR_DRAWS,
        "ddof": 0,
        "target_function_sd": TARGET_FUNCTION_SD,
        "unit_median_sd": medians,
        "scales": {name: TARGET_FUNCTION_SD / value for name, value in medians.items()},
    }
