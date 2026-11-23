from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config.run import CALIBRATION_METHOD, VALIDATION_BIAS_DAYS, validation_bias_days
from src.forecast.arrays import checked_raw_prediction
from src.forecast.intervals import estimate_interval_half_width
from src.metrics import compute_forecast_metrics


def estimate_validation_bias(
    predict_raw: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    past: pd.DataFrame, history_hours: int, future: pd.DataFrame, diurnal: pd.Series, alpha: float,
    *, prediction_hours: int,
) -> dict[str, Any]:
    """Average blended prediction errors by hourly offset across all held-out days."""
    validation = past.iloc[history_hours:]
    # Direct models need their full output window of known covariates at each
    # validation origin; only its first day contributes observed residuals.
    known = pd.concat([validation.drop(columns=["actual_kwh", "load"], errors="ignore"), future],
                      ignore_index=True)
    if prediction_hours < 24:
        raise ValueError("Validation bias requires a model that can forecast a complete day")
    audit = []
    residual_days = []
    for day, offset in enumerate(range(0, len(validation), 24), start=1):
        context = past.iloc[offset:offset + history_hours]
        actual_day = validation.iloc[offset:offset + 24]
        covariates = known.iloc[offset:offset + prediction_hours]
        raw = checked_raw_prediction(predict_raw(context, covariates).iloc[:24], actual_day["time"])
        blended = blend_with_diurnal(raw, diurnal_for_times(actual_day["time"], diurnal), alpha)
        actual = actual_day["actual_kwh"].to_numpy(dtype=np.float64)
        residual = blended - actual
        residual_days.append(residual)
        for hour, timestamp in enumerate(actual_day["time"]):
            audit.append({
                "time": timestamp.isoformat(), "validation_day": day, "horizon_offset": hour,
                "raw_predicted_kwh": float(raw[hour]), "blended_predicted_kwh": float(blended[hour]),
                "actual_kwh": float(actual[hour]), "residual_kwh": float(residual[hour]),
            })
    residuals = np.stack(residual_days)
    bias = residuals.mean(axis=0)
    half_width = estimate_interval_half_width((residuals - bias).ravel())
    return {
        "method": CALIBRATION_METHOD, "enabled": True, "validation_days": len(residual_days),
        "bias_by_hour": bias.tolist(), "samples_per_hour": [len(residual_days)] * 24,
        "bias_mean": float(bias.mean()), "bias_max_abs": float(np.max(np.abs(bias))),
        "interval_half_width": float(half_width) if np.isfinite(half_width) else None,
        "metrics": compute_forecast_metrics(pd.DataFrame(audit).rename(columns={"blended_predicted_kwh": "predicted_kwh"})),
        "validation_predictions": audit,
    }


def check_validation_bias_state(
    state: dict[str, Any], enabled: bool, validation_days: int = VALIDATION_BIAS_DAYS,
) -> None:
    days = validation_bias_days(enabled, validation_days)
    if state.get("validation_bias", False) is not enabled or ("calibration" in state) != enabled:
        raise ValueError("Reusable validation bias state differs from the forecast settings")
    if enabled:
        calibration = state["calibration"]
        if not isinstance(calibration, dict) or calibration.get("method") != CALIBRATION_METHOD or (
                calibration.get("validation_days") != days or calibration.get("enabled") is not True):
            raise ValueError(f"Reusable calibration must use the configured {days} validation days")
        bias = np.asarray(calibration.get("bias_by_hour"), dtype=float)
        if bias.shape != (24,) or not np.isfinite(bias).all():
            raise ValueError("Reusable validation bias must contain 24 finite hourly values")
        width = calibration.get("interval_half_width")
        if width is not None and (not isinstance(width, (int, float)) or not np.isfinite(width) or width < 0):
            raise ValueError("Reusable calibration interval half-width must be finite and non-negative")


def serialize_diurnal_profile(profile: pd.Series) -> dict[str, float]:
    return {
        str(hour): float(profile.loc[hour])
        for hour in range(24)
    }


def state_diurnal_profile(state: dict[str, Any]) -> pd.Series:
    raw = state.get("diurnal_by_hour")
    if not isinstance(raw, dict):
        raise ValueError("Reusable forecaster state is missing the diurnal profile.")
    values = {
        int(hour): float(value)
        for hour, value in raw.items()
    }
    missing = sorted(set(range(24)).difference(values))
    if missing:
        raise ValueError(
            "Reusable forecaster diurnal profile is incomplete; missing hour(s): "
            + ", ".join(str(hour) for hour in missing)
        )
    return pd.Series(values, dtype=np.float64).sort_index()


def rebuild_quantile_interval(
    point: np.ndarray,
    raw_q10: np.ndarray,
    raw_q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_arr = np.asarray(point, dtype=np.float64)
    q10_arr = np.asarray(raw_q10, dtype=np.float64)
    q90_arr = np.asarray(raw_q90, dtype=np.float64)
    raw_width = q90_arr - q10_arr
    valid = np.isfinite(point_arr) & np.isfinite(raw_width)
    half_width = np.where(valid, np.maximum(raw_width, 0.0) / 2.0, np.nan)
    rebuilt_q10 = np.where(valid, np.clip(point_arr - half_width, 0, None), np.nan)
    rebuilt_q90 = np.where(valid, point_arr + half_width, np.nan)
    rebuilt_q10 = np.where(valid, np.minimum(rebuilt_q10, point_arr), np.nan)
    rebuilt_q90 = np.where(valid, np.maximum(rebuilt_q90, point_arr), np.nan)
    return rebuilt_q10, rebuilt_q90


def build_diurnal_profile(history: pd.DataFrame) -> pd.Series:
    base = history.copy()
    base["hour"] = base["time"].dt.hour
    fallback = float(base["actual_kwh"].mean()) if not base.empty else 0.0
    return base.groupby("hour")["actual_kwh"].mean().reindex(range(24), fill_value=fallback)


def diurnal_for_times(times: pd.Series | pd.DatetimeIndex, diurnal_by_hour: pd.Series) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    return np.asarray([diurnal_by_hour.loc[int(ts.hour)] for ts in index], dtype=np.float64)


def blend_with_diurnal(fc: np.ndarray, diurnal: np.ndarray, alpha: float) -> np.ndarray:
    """Mix once with the frozen training mean curve, without rescaling that curve."""
    return np.clip((1.0 - alpha) * np.asarray(fc) + alpha * np.asarray(diurnal), 0, None)
