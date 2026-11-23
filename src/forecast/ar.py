from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.arrays import align_vector


ENERGY_PRICE_FALLBACK_ELASTICITY = 0.1
ENERGY_PRICE_ADJUSTMENT_MAX_RATIO = 0.3
ENERGY_PRICE_RESPONSE_MIN_OBS = 12


def estimate_energy_price_response(training_frame: pd.DataFrame) -> dict[str, Any]:
    if "actual_kwh" not in training_frame or "e_price" not in training_frame:
        return {"enabled": False, "reason": "missing_actual_or_energy_price"}

    data = training_frame.copy()
    if "time" in data:
        data["time"] = pd.to_datetime(data["time"])
        data["hour"] = data["time"].dt.hour
    elif "hour" not in data:
        data["hour"] = 0
    data["actual_kwh"] = pd.to_numeric(data["actual_kwh"], errors="coerce")
    data["e_price"] = pd.to_numeric(data["e_price"], errors="coerce")
    data = data[["actual_kwh", "e_price", "hour"]].dropna()
    if data.empty:
        return {"enabled": False, "reason": "no_valid_energy_price_training_rows"}

    mean_load = float(data["actual_kwh"].clip(lower=0).mean())
    mean_price = float(data["e_price"].replace(0, np.nan).mean())
    if not np.isfinite(mean_load) or mean_load <= 0 or not np.isfinite(mean_price) or mean_price <= 0:
        return {"enabled": False, "reason": "invalid_mean_load_or_energy_price"}

    fallback_slope = -ENERGY_PRICE_FALLBACK_ELASTICITY * mean_load / max(mean_price, 1e-9)
    max_abs_slope = ENERGY_PRICE_ADJUSTMENT_MAX_RATIO * mean_load / max(mean_price, 1e-9)
    source = "fallback_elasticity"
    slope = fallback_slope

    if len(data) >= ENERGY_PRICE_RESPONSE_MIN_OBS and data["e_price"].nunique(dropna=True) >= 2:
        load_baseline = data.groupby("hour")["actual_kwh"].transform("mean")
        price_baseline = data.groupby("hour")["e_price"].transform("mean")
        load_residual = data["actual_kwh"].to_numpy(dtype=np.float64) - load_baseline.to_numpy(dtype=np.float64)
        price_residual = data["e_price"].to_numpy(dtype=np.float64) - price_baseline.to_numpy(dtype=np.float64)
        if float(np.nanvar(price_residual)) <= 1e-12:
            price_residual = data["e_price"].to_numpy(dtype=np.float64) - mean_price
        denom = float(np.dot(price_residual, price_residual))
        if denom > 1e-12:
            empirical_slope = float(np.dot(price_residual, load_residual) / denom)
            if np.isfinite(empirical_slope) and empirical_slope < 0:
                slope = empirical_slope
                source = "empirical_hourly_residual"
            else:
                source = "fallback_nonnegative_empirical_response"
    else:
        source = "fallback_insufficient_energy_price_variation"

    slope = float(np.clip(slope, -max_abs_slope, 0.0))
    return {
        "enabled": True,
        "source": source,
        "slope_kwh_per_energy_price": round(slope, 6),
        "fallback_elasticity": ENERGY_PRICE_FALLBACK_ELASTICITY,
        "mean_training_load_kwh": round(mean_load, 4),
        "mean_training_energy_price": round(mean_price, 6),
        "max_adjustment_ratio": ENERGY_PRICE_ADJUSTMENT_MAX_RATIO,
        "training_rows": int(len(data)),
    }


def energy_price_reference_for_times(
    training_frame: pd.DataFrame,
    times: pd.Series | pd.DatetimeIndex,
    fallback_price: float,
) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    if "time" not in training_frame or "e_price" not in training_frame:
        return np.full(len(index), fallback_price, dtype=np.float64)
    data = training_frame[["time", "e_price"]].copy()
    data["time"] = pd.to_datetime(data["time"])
    data["e_price"] = pd.to_numeric(data["e_price"], errors="coerce")
    data = data.dropna()
    if data.empty:
        return np.full(len(index), fallback_price, dtype=np.float64)
    by_hour = data.groupby(data["time"].dt.hour)["e_price"].mean()
    return np.asarray([float(by_hour.get(int(ts.hour), fallback_price)) for ts in index], dtype=np.float64)


def apply_energy_price_response_adjustment(
    point: np.ndarray,
    horizon_frame: pd.DataFrame | None,
    training_frame: pd.DataFrame,
    *,
    fitted_response: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_arr = np.asarray(point, dtype=np.float64)
    zero_adjustment = np.zeros(len(point_arr), dtype=np.float64)
    response = dict(fitted_response) if fitted_response is not None else estimate_energy_price_response(training_frame)
    if not response.get("enabled"):
        return np.clip(point_arr, 0, None), zero_adjustment, response
    if horizon_frame is None or "e_price" not in horizon_frame or "time" not in horizon_frame:
        metadata = {**response, "enabled": False, "reason": "missing_horizon_energy_price"}
        return np.clip(point_arr, 0, None), zero_adjustment, metadata

    prices = pd.to_numeric(horizon_frame["e_price"], errors="coerce").ffill().bfill()
    fallback_price = float(response["mean_training_energy_price"])
    prices = prices.fillna(fallback_price).to_numpy(dtype=np.float64)
    prices = align_vector(prices, len(point_arr))
    reference = energy_price_reference_for_times(training_frame, horizon_frame["time"], fallback_price)
    reference = align_vector(reference, len(point_arr))

    slope = float(response["slope_kwh_per_energy_price"])
    raw_adjustment = slope * (prices - reference)
    floor = max(float(response["mean_training_load_kwh"]) * 0.05, 1e-9)
    limit = np.maximum(np.abs(point_arr) * ENERGY_PRICE_ADJUSTMENT_MAX_RATIO, floor)
    adjustment = np.clip(raw_adjustment, -limit, limit)
    adjusted = np.clip(point_arr + adjustment, 0, None)
    metadata = {
        **response,
        "enabled": True,
        "mean_horizon_energy_price": round(float(np.nanmean(prices)), 6),
        "mean_reference_energy_price": round(float(np.nanmean(reference)), 6),
        "mean_adjustment_kwh": round(float(np.nanmean(adjustment)), 6),
        "max_abs_adjustment_kwh": round(float(np.nanmax(np.abs(adjustment))), 6),
    }
    return adjusted, adjustment, metadata


def run_ar_prediction(
    history: pd.DataFrame, future: pd.DataFrame, *, training_history: pd.DataFrame,
    fitted_response: dict[str, Any],
) -> pd.DataFrame:
    hist = history.set_index("time")["actual_kwh"].astype(float).sort_index()
    target_index = pd.DatetimeIndex(future["time"])
    hourly_mean = hist.groupby(hist.index.hour).mean()
    autoregressive_values = hist.reindex(target_index - pd.Timedelta(days=7)).to_numpy()
    fallback = np.array([hourly_mean.loc[t.hour] for t in target_index], dtype=float)
    predicted = np.where(np.isnan(autoregressive_values), fallback, 0.72 * autoregressive_values + 0.28 * fallback)

    last_24 = hist.tail(24).sum()
    mean_daily = hist.resample("D").sum().mean()
    trend_scale = 1.0 if mean_daily == 0 or np.isnan(mean_daily) else np.clip(last_24 / mean_daily, 0.75, 1.25)
    predicted = np.maximum(predicted * trend_scale, 0.0)
    adjusted, energy_price_adjustment, adjustment_meta = apply_energy_price_response_adjustment(
        predicted,
        future,
        training_history,
        fitted_response=fitted_response,
    )
    result = pd.DataFrame({
        "time": target_index, "base_predicted_kwh": predicted,
        "energy_price_adjustment_kwh": energy_price_adjustment, "raw_predicted_kwh": adjusted,
    })
    result.attrs["energy_price_adjustment"] = adjustment_meta
    return result
