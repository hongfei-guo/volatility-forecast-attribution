"""Real-time feature construction for the state-space forecast models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    stan_file: str
    correction: str
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]


HAR_INPUT_MODELS = ("RV-NN-SV-HAR", "RV-LIN-SV-HAR")
HAR_FEATURE_NAMES = ("z_lag1", "log_rv_lag1", "log_rv_mean5", "log_rv_mean22")


MODEL_SPECS: dict[str, ModelSpec] = {
    "NN-SV": ModelSpec(
        stan_file="nn_sv.stan",
        correction="neural",
        feature_names=("z_lag1",),
        horizons=(1, 5, 10),
    ),
    "RV-NN-SV": ModelSpec(
        stan_file="nn_sv.stan",
        correction="neural",
        feature_names=("z_lag1", "log_rv_lag1"),
        horizons=(1, 5, 10),
    ),
    "RV-RA-NN-SV": ModelSpec(
        stan_file="rv_ra_nn_sv.stan",
        correction="neural_residual_asymmetry",
        feature_names=("z_lag1", "log_rv_lag1", "asymmetry_lag1"),
        horizons=(1, 5, 10),
    ),
    "RV-LIN-SV": ModelSpec(
        stan_file="rv_lin_sv.stan",
        correction="linear",
        feature_names=("z_lag1", "log_rv_lag1"),
        horizons=(1,),
    ),
    "SV": ModelSpec(
        stan_file="sv.stan",
        correction="zero",
        # Retain the lagged-return row alignment used by the matched models.
        # The SV likelihood itself receives no feature matrix.
        feature_names=("z_lag1",),
        horizons=(1, 5, 10),
    ),
    "RV-NN-SV-HAR": ModelSpec(
        stan_file="nn_sv.stan",
        correction="neural",
        feature_names=HAR_FEATURE_NAMES,
        horizons=(1,),
    ),
    "RV-LIN-SV-HAR": ModelSpec(
        stan_file="rv_lin_sv.stan",
        correction="linear",
        feature_names=HAR_FEATURE_NAMES,
        horizons=(1,),
    ),
}


@dataclass(frozen=True)
class ReturnScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ReturnScaler":
        sample = np.asarray(values, dtype=float)
        if sample.ndim != 1 or sample.size < 2 or np.any(~np.isfinite(sample)):
            raise ValueError("at least two finite returns are required")
        scale = float(np.std(sample, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("training returns must have positive variation")
        return cls(mean=float(np.mean(sample)), scale=scale)

    def standardize(self, values: np.ndarray | float) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale


@dataclass(frozen=True)
class FeatureTransform:
    model_id: str
    training_end: pd.Timestamp
    return_scaler: ReturnScaler
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    rv_floor: float | None

    def __post_init__(self) -> None:
        spec = model_spec(self.model_id)
        if self.feature_names != spec.feature_names:
            raise ValueError("feature names do not match the model")
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        if mean.shape != (len(self.feature_names),) or scale.shape != mean.shape:
            raise ValueError("feature moments do not match the model dimension")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("feature moments must be finite with positive scales")
        if "log_rv_lag1" in self.feature_names:
            if self.rv_floor is None or not np.isfinite(self.rv_floor) or self.rv_floor <= 0:
                raise ValueError("realised-variance models require a positive floor")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)


@dataclass(frozen=True)
class EstimationData:
    dates: pd.DatetimeIndex
    z: np.ndarray
    x: np.ndarray


def model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"unsupported forecast model: {model_id}") from exc


def base_model_id(model_id: str) -> str:
    """Identify the parent parameterization and deterministic initialization."""
    model_spec(model_id)
    return model_id.removesuffix("-HAR") if model_id in HAR_INPUT_MODELS else model_id


def _ordered_frame(
    frame: pd.DataFrame, *, model_id: str | None = None
) -> pd.DataFrame:
    required = {"date", "return_cc", "rv_oc"}
    if model_id == "RV-RA-NN-SV":
        required.add("asymmetry")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"daily data are missing columns: {sorted(missing)}")
    columns = ["date", "return_cc", "rv_oc"]
    if "asymmetry" in required:
        columns.append("asymmetry")
    result = frame.loc[:, columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["return_cc"] = pd.to_numeric(result["return_cc"], errors="raise")
    result["rv_oc"] = pd.to_numeric(result["rv_oc"], errors="raise")
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty or result["date"].duplicated().any():
        raise ValueError("daily dates must be nonempty and unique")
    numeric = ["return_cc", "rv_oc"]
    if "asymmetry" in result:
        result["asymmetry"] = pd.to_numeric(
            result["asymmetry"], errors="raise"
        )
        numeric.append("asymmetry")
    if np.any(~np.isfinite(result[numeric].to_numpy(float))):
        raise ValueError("daily feature values must be finite")
    if np.any(result["rv_oc"].to_numpy(float) < 0):
        raise ValueError("realised variance must be nonnegative")
    if "asymmetry" in result and np.any(
        np.abs(result["asymmetry"].to_numpy(float)) > 1
    ):
        raise ValueError("asymmetry must lie in [-1,1]")
    return result


def _raw_features(frame: pd.DataFrame, transform: FeatureTransform) -> pd.DataFrame:
    spec = model_spec(transform.model_id)
    result = pd.DataFrame({"date": frame["date"].to_numpy()})
    z = pd.Series(
        transform.return_scaler.standardize(frame["return_cc"].to_numpy(float)),
        index=frame.index,
    )
    result["z_lag1"] = z.shift(1)
    if "log_rv_lag1" in spec.feature_names:
        assert transform.rv_floor is not None
        log_rv = np.log(
            np.maximum(frame["rv_oc"].to_numpy(float), transform.rv_floor)
        )
        result["log_rv_lag1"] = pd.Series(log_rv, index=frame.index).shift(1)
        if transform.model_id in HAR_INPUT_MODELS:
            for window in (5, 22):
                result[f"log_rv_mean{window}"] = (
                    pd.Series(log_rv, index=frame.index)
                    .rolling(window, min_periods=window)
                    .mean()
                    .shift(1)
                )
    if "asymmetry_lag1" in spec.feature_names:
        result["asymmetry_lag1"] = pd.Series(
            frame["asymmetry"].to_numpy(float), index=frame.index
        ).shift(1)
    return result


def fit_feature_transform(
    frame: pd.DataFrame,
    *,
    model_id: str,
    training_end: str | pd.Timestamp,
    rv_floor_quantile: float = 0.001,
) -> FeatureTransform:
    """Fit return and feature moments using observations through ``training_end``."""
    spec = model_spec(model_id)
    if not 0 <= rv_floor_quantile < 1:
        raise ValueError("rv_floor_quantile must lie in [0,1)")
    daily = _ordered_frame(frame, model_id=model_id)
    end = pd.Timestamp(training_end).normalize()
    training = daily[daily["date"] <= end]
    if len(training) < 30:
        raise ValueError("the expanding estimation sample is too short")
    return_scaler = ReturnScaler.fit(training["return_cc"].to_numpy(float))

    rv_floor: float | None = None
    if "log_rv_lag1" in spec.feature_names:
        positive = training.loc[training["rv_oc"] > 0, "rv_oc"].to_numpy(float)
        if positive.size == 0:
            raise ValueError("the estimation sample contains no positive realised variance")
        rv_floor = max(
            float(np.quantile(positive, rv_floor_quantile)),
            np.finfo(float).tiny,
        )

    provisional = FeatureTransform(
        model_id=model_id,
        training_end=end,
        return_scaler=return_scaler,
        feature_names=spec.feature_names,
        feature_mean=np.zeros(len(spec.feature_names), dtype=float),
        feature_scale=np.ones(len(spec.feature_names), dtype=float),
        rv_floor=rv_floor,
    )
    raw = _raw_features(daily, provisional)
    rows = raw.loc[raw["date"] <= end, list(spec.feature_names)].dropna()
    if len(rows) < 2:
        raise ValueError("too few complete feature rows in the estimation sample")
    values = rows.to_numpy(float)
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.maximum(np.std(values, axis=0, ddof=1), 1e-12)
    return FeatureTransform(
        model_id=model_id,
        training_end=end,
        return_scaler=return_scaler,
        feature_names=spec.feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        rv_floor=rv_floor,
    )


def transform_features(
    frame: pd.DataFrame, transform: FeatureTransform
) -> pd.DataFrame:
    """Apply estimation-sample moments without using future observations."""
    daily = _ordered_frame(frame, model_id=transform.model_id)
    raw = _raw_features(daily, transform)
    columns = list(transform.feature_names)
    raw.loc[:, columns] = (
        raw.loc[:, columns].to_numpy(float) - transform.feature_mean
    ) / transform.feature_scale
    return raw


def prepare_estimation_data(
    frame: pd.DataFrame, transform: FeatureTransform
) -> EstimationData:
    """Return aligned response and feature arrays through the refit date."""
    daily = _ordered_frame(frame, model_id=transform.model_id)
    transformed = transform_features(daily, transform)
    merged = daily.loc[:, ["date", "return_cc"]].merge(
        transformed, on="date", how="left", validate="one_to_one"
    )
    selected = merged[merged["date"] <= transform.training_end]
    selected = selected.dropna(subset=list(transform.feature_names))
    x = selected.loc[:, list(transform.feature_names)].to_numpy(float)
    if len(selected) < 2:
        raise ValueError("too few complete observations for estimation")
    z = transform.return_scaler.standardize(
        selected["return_cc"].to_numpy(float)
    )
    return EstimationData(
        dates=pd.DatetimeIndex(selected["date"]),
        z=np.asarray(z, dtype=float),
        x=x,
    )
