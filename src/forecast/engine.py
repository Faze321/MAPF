from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.run import CALIBRATION_METHOD, VALIDATION_BIAS_DAYS, validation_bias_days
from src.config.training import FORECAST_BACKENDS, LORA_BACKENDS, base_backend
from src.config.values import normalize_forecast_model_name
from src.forecast.arrays import checked_raw_prediction, require_fitted_backend
from src.forecast.backends import prepare_forecast_backend
from src.forecast.calibration import (
    blend_with_diurnal,
    build_diurnal_profile,
    check_validation_bias_state,
    diurnal_for_times,
    estimate_validation_bias,
    rebuild_quantile_interval,
    serialize_diurnal_profile,
    state_diurnal_profile,
)
from src.forecast.foundation import DEFAULT_TIMESFM_EXOG_COLS
from src.forecast.intervals import deterministic_interval
from src.metrics import compute_forecast_metrics


@dataclass(frozen=True)
class ForecastResult:
    hourly: pd.DataFrame
    summary: dict[str, Any]
    reusable_forecaster: dict[str, Any] | None = None


def forecast_zone(
    *,
    zone_id: str,
    category: str,
    load: pd.DataFrame,
    service_price: pd.DataFrame,
    energy_price: pd.DataFrame,
    occupancy: pd.DataFrame,
    weather: pd.DataFrame,
    profile: dict[str, Any],
    forecast_origin: pd.Timestamp | None,
    horizon_days: int = 1,
    history_days: int = 35,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    backend: str = "timesfm",
    diurnal_blend_alpha: float = 0.0,
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 840,
    timesfm_exog_cols: list[str] | None = None,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 840,
    chronos_exog_cols: list[str] | None = None,
    chronos_device: str = "auto",
    model_reference: dict[str, Any] | None = None,
    raw_prediction_cache: dict | None = None,
) -> ForecastResult:
    timesfm_exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if timesfm_exog_cols is None else timesfm_exog_cols)
    chronos_exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if chronos_exog_cols is None else chronos_exog_cols)
    for name, value in (("history_days", history_days), ("horizon_days", horizon_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    zone_id = str(zone_id)
    normalized_model = normalize_forecast_model_name(backend)
    model_family = base_backend(normalized_model)
    if normalized_model in LORA_BACKENDS:
        if model_family == "timesfm":
            timesfm_context_hours = history_days * 24
        else:
            chronos_context_hours = history_days * 24
    if model_reference is not None and (
        str(model_reference.get("zone_id")) != zone_id
        or model_reference.get("backend") != normalized_model
    ):
        raise ValueError("Model reference does not match the requested zone/backend")
    horizon_hours = horizon_days * 24
    if forecast_origin is None:
        forecast_origin = load["time"].max() - pd.Timedelta(hours=horizon_hours - 1)
    forecast_origin = pd.Timestamp(forecast_origin)
    forecast_end = forecast_origin + pd.Timedelta(hours=horizon_hours - 1)
    validation_hours = validation_bias_days(validation_bias, validation_days) * 24
    validation_start = forecast_origin - pd.Timedelta(hours=validation_hours)
    validation_end = forecast_origin - pd.Timedelta(hours=1)
    history_start = validation_start - pd.Timedelta(days=history_days)
    history_end = validation_start - pd.Timedelta(hours=1)

    zone_frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, zone_id)
    history = zone_frame[(zone_frame["time"] >= history_start) & (zone_frame["time"] <= history_end)]
    validation = zone_frame[zone_frame["time"].between(validation_start, validation_end)] if validation_bias else None
    forecast_features = zone_frame[(zone_frame["time"] >= forecast_origin) & (zone_frame["time"] <= forecast_end)].copy()
    if len(history) != history_days * 24:
        raise ValueError(f"Zone {zone_id} requires a complete history window of {history_days * 24} hours")
    future_times = pd.date_range(forecast_origin, periods=horizon_hours, freq="h")
    forecast_features = forecast_features.set_index("time").reindex(future_times).rename_axis("time").reset_index()

    hourly = forecast_load(
        history,
        forecast_features,
        forecast_origin,
        horizon_hours,
        backend=backend,
        validation_bias=validation_bias,
        validation_days=validation_days,
        validation=validation,
        diurnal_blend_alpha=diurnal_blend_alpha,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_exog_cols=timesfm_exog_cols,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_exog_cols=chronos_exog_cols,
        chronos_device=chronos_device,
        model_reference=model_reference,
        raw_prediction_cache=raw_prediction_cache,
    )
    forecast_attrs = dict(hourly.attrs)
    hourly_feature_cols = [
        col
        for col in [
            "time",
            "actual_kwh",
            "e_price",
            "s_price",
            "occupancy",
            "T",
            "U",
            "nRAIN",
            "hour",
            "is_weekend",
            "temp_price_idx",
        ]
        if col in forecast_features.columns
    ]
    hourly = hourly.merge(forecast_features[hourly_feature_cols], on="time", how="left")
    hourly = add_error_columns(hourly)
    metrics = compute_forecast_metrics(hourly)

    price_summary = {
        **summarize_price(service_price, zone_id, history_start, forecast_end, "service"),
        **summarize_price(energy_price, zone_id, history_start, forecast_end, "energy"),
    }
    weather_summary = summarize_weather(weather, forecast_origin, forecast_end)
    occupancy_summary = summarize_occupancy(occupancy, zone_id, history_start, forecast_end)
    capacity = float(profile.get("capacity_kw_proxy", 0.0) or 0.0)
    forecast_total = float(hourly["predicted_kwh"].sum())
    forecast_peak = float(hourly["predicted_kwh"].max())
    previous_window = history.tail(horizon_hours)
    previous_total = float(previous_window["actual_kwh"].sum()) if not previous_window.empty else 0.0
    predicted_change_pct = pct_change(forecast_total, previous_total)
    actual_total = float(hourly["actual_kwh"].sum()) if hourly["actual_kwh"].notna().any() else None
    actual_peak = float(hourly["actual_kwh"].max()) if hourly["actual_kwh"].notna().any() else None

    peak_capacity_ratio = forecast_peak / capacity if capacity > 0 else 0.0
    stress_level = str(profile.get("grid_stress_level") or "Low")
    summary = {
        "zone_id": zone_id,
        "category": category,
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "validation_bias": validation_bias,
        "forecast_origin": forecast_origin.isoformat(),
        "forecast_end": forecast_end.isoformat(),
        "forecast_horizon_days": int(horizon_days),
        "forecast_horizon_hours": int(horizon_hours),
        "backend": normalized_model,
        "diurnal_blend_alpha": float(diurnal_blend_alpha),
        "timesfm_repo": timesfm_repo if model_family == "timesfm" else None,
        "timesfm_context_hours": int(timesfm_context_hours) if model_family == "timesfm" else None,
        "timesfm_covariates": timesfm_exog_cols if model_family == "timesfm" else None,
        "chronos_repo": chronos_repo if normalized_model.startswith("chronos") else None,
        "chronos_context_hours": int(chronos_context_hours) if normalized_model.startswith("chronos") else None,
        "chronos_covariates": chronos_exog_cols
        if normalized_model.startswith("chronos")
        else None,
        "chronos_device": chronos_device if normalized_model.startswith("chronos") else None,
        "energy_price_adjustment": forecast_attrs.get("energy_price_adjustment"),
        "forecast_total_kwh": round(forecast_total, 2),
        "forecast_peak_kwh": round(forecast_peak, 2),
        "predicted_change_pct": round(predicted_change_pct, 2),
        "previous_comparable_total_kwh": previous_total,
        "actual_total_kwh": round(actual_total, 2) if actual_total is not None else None,
        "actual_peak_kwh": round(actual_peak, 2) if actual_peak is not None else None,
        "mae_kwh": metrics.get("MAE"),
        "rmse_kwh": metrics.get("RMSE"),
        "mape_pct": metrics.get("MAPE_pct"),
        "rae": metrics.get("RAE"),
        "wape_pct": metrics.get("WAPE_pct"),
        "capacity_kw_proxy": round(capacity, 2),
        "peak_capacity_ratio": round(peak_capacity_ratio, 3),
        "grid_stress_level": stress_level,
        "price": price_summary,
        "weather": weather_summary,
        "occupancy": occupancy_summary,
        "profile": compact_profile(profile),
        "daily_history_kwh": daily_totals(history, "actual_kwh"),
        "daily_forecast_kwh": daily_totals(hourly, "predicted_kwh"),
        "hourly_shape": hourly_shape(history),
        "metrics": metrics,
    }
    summary = {
        key: value for key, value in summary.items()
        if not key.startswith(("timesfm_", "chronos_", "lstm_", "xgboost_", "ar_"))
        or key.startswith(model_family.lower() + "_")
    }
    if validation_bias:
        summary.update(
            validation_days=validation_days, validation_start=validation_start.isoformat(),
            validation_end=validation_end.isoformat(), calibration_method=CALIBRATION_METHOD,
            calibration=forecast_attrs["calibration"],
        )
    if model_reference is not None:
        overlap = training_overlap_mask(hourly["time"], model_reference)
        hourly["model_id"] = model_reference["model_id"]
        hourly["training_period_overlap"] = overlap
        summary.update(
            model_id=model_reference["model_id"],
            model_training_start=model_reference["training_start"],
            model_training_end_exclusive=model_reference["training_end_exclusive"],
            model_parameters=deepcopy(model_reference["parameters"]),
            training_overlap_hours=int(overlap.sum()),
            evaluation_is_out_of_sample=bool(
                forecast_origin >= pd.Timestamp(model_reference["training_end_exclusive"])
            ),
        )
    known_future = forecast_features.drop(columns=["actual_kwh", "load"], errors="ignore").copy()
    reusable_forecaster = {
        "zone": zone_id,
        "backend": normalized_model,
        "forecast_origin": forecast_origin.isoformat(),
        "horizon_hours": int(horizon_hours),
        "history": history.copy(),
        "known_future": known_future,
        "forecast_parameters": {
            "timesfm_repo": timesfm_repo,
            "timesfm_context_hours": int(timesfm_context_hours),
            "timesfm_exog_cols": list(timesfm_exog_cols),
            "chronos_repo": chronos_repo,
            "chronos_context_hours": int(chronos_context_hours),
            "chronos_exog_cols": list(chronos_exog_cols),
            "chronos_device": chronos_device,
        },
        "fitted_state": forecast_attrs.get("reusable_forecaster_state"),
    }
    reusable_forecaster["forecast_parameters"] = {
        key: value for key, value in reusable_forecaster["forecast_parameters"].items()
        if key.startswith(model_family.lower() + "_")
    }
    reusable_forecaster["forecast_parameters"].update(
        diurnal_blend_alpha=float(diurnal_blend_alpha), validation_bias=validation_bias,
    )
    if validation_bias:
        reusable_forecaster["forecast_parameters"]["validation_days"] = validation_days
        reusable_forecaster["validation"] = validation.copy()
    return ForecastResult(
        hourly=hourly,
        summary=summary,
        reusable_forecaster=reusable_forecaster,
    )


def forecast_load(
    history: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    horizon_hours: int,
    *,
    backend: str,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    validation: pd.DataFrame | None = None,
    diurnal_blend_alpha: float = 0.0,
    timesfm_repo: str = 'google/timesfm-2.5-200m-pytorch',
    timesfm_context_hours: int = 840,
    timesfm_exog_cols: list[str] | None = None,
    chronos_repo: str = 'amazon/chronos-2',
    chronos_context_hours: int = 840,
    chronos_exog_cols: list[str] | None = None,
    chronos_device: str = 'auto',
    fitted_state: dict[str, Any] | None = None,
    model_reference: dict[str, Any] | None = None,
    raw_prediction_cache: dict | None = None,
) -> pd.DataFrame:
    """Forecast from frozen weights, optionally estimating hourly bias on held-out days."""
    parameters = {name: value for name, value in locals().copy().items()
                  if name.startswith(("timesfm_", "chronos_"))}
    backend = normalize_forecast_model_name(backend)
    if backend not in FORECAST_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, (int, np.integer)) or horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer")
    alpha = float(diurnal_blend_alpha)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("diurnal_blend_alpha must be finite and between 0 and 1")
    origin = pd.Timestamp(forecast_origin)
    if pd.isna(origin) or origin != origin.floor("h"):
        raise ValueError("forecast_origin must be a valid whole-hour timestamp")
    history = checked_hourly_history(history, "history")
    expected_hours = validation_bias_days(validation_bias, validation_days) * 24
    history_hours = len(history)
    if history_hours < 24:
        raise ValueError("Forecasting requires at least 24 historical hours")
    if validation_bias:
        if validation is None or len(validation) != expected_hours:
            raise ValueError(f"Validation bias requires {expected_hours} complete hourly observations")
        validation = checked_hourly_history(validation, "validation")
        if validation["time"].iloc[-1] != origin - pd.Timedelta(hours=1):
            raise ValueError("validation must end one hour before forecast_origin")
    elif validation is not None:
        raise ValueError("Validation observations require validation_bias=True")
    if history["time"].iloc[-1] != origin - pd.Timedelta(hours=expected_hours + 1):
        raise ValueError("history must end one hour before the validation window" if validation_bias
                         else "history must end one hour before forecast_origin")
    past = pd.concat([history, validation], ignore_index=True) if validation_bias else history
    future_times = pd.date_range(origin, periods=horizon_hours, freq="h")
    future_source = full_frame.copy()
    future_source["time"] = pd.to_datetime(future_source["time"])
    if future_source["time"].duplicated().any():
        raise ValueError("Forecast covariates contain duplicate timestamps")
    # Actuals may exist for evaluation but must never reach a backend's future features.
    future = (future_source.drop(columns=["actual_kwh", "load"], errors="ignore")
              .set_index("time").reindex(future_times).rename_axis("time").reset_index())

    if fitted_state is not None:
        require_fitted_backend(fitted_state, backend)
        if (pd.Timestamp(fitted_state.get("forecast_origin")) != origin
                or pd.Timestamp(fitted_state.get("history_start")) != history["time"].iloc[0]
                or fitted_state.get("diurnal_blend_alpha") != alpha):
            raise ValueError("Reusable forecaster state does not match the origin, history or alpha")
        check_validation_bias_state(fitted_state, validation_bias, validation_days)
        diurnal = state_diurnal_profile(fitted_state)
    else:
        diurnal = build_diurnal_profile(history)

    saved_reference = (fitted_state or {}).get("model_reference")
    if saved_reference is not None:
        if model_reference is not None and model_reference != saved_reference:
            raise ValueError("Reusable forecaster state cannot be paired with a different model reference")
        model_reference = saved_reference
    if model_reference is not None:
        dimensions = model_reference.get("parameters", {})
        if dimensions.get("input_hours") != history_hours or dimensions.get("output_hours") != horizon_hours:
            raise ValueError("Offline model input/output windows do not match history_days/horizon_days; prepare the matching model")
    predict_raw, model_state = prepare_forecast_backend(
        backend, history, parameters, max(24, int(horizon_hours)), fitted_state=fitted_state,
        model_reference=model_reference,
    )
    if model_reference is not None and raw_prediction_cache is not None:
        predict_raw = cache_raw_predictions(predict_raw, model_reference, raw_prediction_cache)
    calibration = None
    if validation_bias:
        calibration = fitted_state["calibration"] if fitted_state is not None else estimate_validation_bias(
            predict_raw, past, history_hours, future, diurnal, alpha,
            prediction_hours=horizon_hours if model_reference is not None else 24,
        )
    result = predict_raw(past.tail(history_hours), future).copy()
    raw = checked_raw_prediction(result, future_times)
    blended = blend_with_diurnal(raw, diurnal_for_times(future_times, diurnal), alpha)
    point = blended
    if calibration is not None:
        hourly_bias = np.asarray(calibration["bias_by_hour"])[np.arange(horizon_hours) % 24]
        point = np.clip(blended - hourly_bias, 0, None)
        result["bias_kwh"] = hourly_bias
        result["bias_corrected_kwh"] = point
        result.attrs["calibration"] = calibration
    if "q10_kwh" in result and "q90_kwh" in result:
        q10, q90 = rebuild_quantile_interval(point, result["q10_kwh"], result["q90_kwh"])
    else:
        q10 = q90 = np.full(len(result), np.nan)
    if calibration is not None and calibration["interval_half_width"] is not None:
        residual_q10, residual_q90 = deterministic_interval(point, calibration["interval_half_width"])
        available = np.isfinite(q10) & np.isfinite(q90)
        q10, q90 = np.where(available, q10, residual_q10), np.where(available, q90, residual_q90)
    result["raw_predicted_kwh"] = raw
    result["blended_predicted_kwh"] = blended
    result["predicted_kwh"] = point
    result["q10_kwh"], result["q50_kwh"], result["q90_kwh"] = q10, point, q90
    state = {
        **model_state, "backend": backend, "forecast_origin": origin.isoformat(),
        "history_start": history["time"].iloc[0].isoformat(), "diurnal_blend_alpha": alpha,
        "diurnal_by_hour": serialize_diurnal_profile(diurnal), "validation_bias": validation_bias,
    }
    if calibration is not None:
        state["calibration"] = calibration
    result.attrs["reusable_forecaster_state"] = state
    return result


def training_overlap_mask(times, reference: dict[str, Any]) -> np.ndarray:
    """Identify observations included in this model's full offline training history."""
    timestamps = pd.DatetimeIndex(times)
    overlap = np.zeros(len(timestamps), dtype=bool)
    for start, end in reference["training_intervals"]:
        overlap |= (timestamps >= pd.Timestamp(start)) & (timestamps < pd.Timestamp(end))
    return overlap


def cache_raw_predictions(predict, reference: dict[str, Any], cache: dict):
    """Share expensive unblended predictions across alpha experiments only."""
    def cached(context, future):
        digest = hashlib.sha256(reference["model_id"].encode())
        digest.update(reference["manifest_sha256"].encode())
        for frame in (context, future):
            digest.update(repr(tuple(frame.columns)).encode())
            digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        key = digest.hexdigest()
        if key not in cache:
            cache[key] = predict(context, future).copy(deep=True)
        return cache[key].copy(deep=True)
    return cached


def checked_hourly_history(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "time" not in frame or "actual_kwh" not in frame or frame.empty:
        raise ValueError(f"{label} must contain hourly time and actual_kwh values")
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"]).dt.as_unit("ns")
    result = result.sort_values("time").reset_index(drop=True)
    times = pd.DatetimeIndex(result["time"])
    if times.hasnans or times.has_duplicates or not times.equals(pd.date_range(times[0], periods=len(times), freq="h")):
        raise ValueError(f"{label} must contain complete consecutive hourly observations")
    result["actual_kwh"] = pd.to_numeric(result["actual_kwh"], errors="coerce")
    if not np.isfinite(result["actual_kwh"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} load must contain finite observations")
    if (result["actual_kwh"] < 0).any():
        raise ValueError(f"{label} load must contain non-negative observations")
    return result


def build_zone_model_frame(
    load: pd.DataFrame,
    service_price: pd.DataFrame,
    energy_price: pd.DataFrame,
    occupancy: pd.DataFrame,
    weather: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    frame = load[["time", zone_id]].rename(columns={zone_id: "actual_kwh"}).copy()
    frame = frame.merge(
        energy_price[["time", zone_id]].rename(columns={zone_id: "e_price"}),
        on="time",
        how="left",
    )
    frame = frame.merge(
        service_price[["time", zone_id]].rename(columns={zone_id: "s_price"}),
        on="time",
        how="left",
    )
    frame = frame.merge(
        occupancy[["time", zone_id]].rename(columns={zone_id: "occupancy"}),
        on="time",
        how="left",
    )
    frame = frame.merge(weather, on="time", how="left")
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["hour"] = frame["time"].dt.hour
    frame["is_weekend"] = frame["time"].dt.dayofweek.isin([5, 6]).astype(float)
    if "T" in frame and "e_price" in frame:
        frame["temp_price_idx"] = pd.to_numeric(frame["T"], errors="coerce") * pd.to_numeric(frame["e_price"], errors="coerce")
    else:
        frame["temp_price_idx"] = np.nan
    return frame


def add_error_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    if "actual_kwh" not in frame:
        frame["actual_kwh"] = np.nan
    frame["error_kwh"] = frame["actual_kwh"] - frame["predicted_kwh"]
    frame["abs_error_kwh"] = frame["error_kwh"].abs()
    frame["abs_pct_error"] = np.where(
        frame["actual_kwh"].abs() > 1e-9,
        frame["abs_error_kwh"] / frame["actual_kwh"].abs() * 100.0,
        np.nan,
    )
    return frame


def summarize_price(
    price: pd.DataFrame,
    zone_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_type: str,
) -> dict[str, float]:
    if price_type not in {"service", "energy"}:
        raise ValueError(f"Unsupported price_type: {price_type}")
    frame = price[(price["time"] >= start) & (price["time"] <= end)]
    values = frame[zone_id].replace(0, np.nan).astype(float)
    return {
        f"mean_{price_type}_price": finite_round(values.mean(skipna=True), 4),
        f"min_{price_type}_price": finite_round(values.min(skipna=True), 4),
        f"max_{price_type}_price": finite_round(values.max(skipna=True), 4),
    }


def summarize_occupancy(occupancy: pd.DataFrame, zone_id: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    frame = occupancy[(occupancy["time"] >= start) & (occupancy["time"] <= end)]
    values = frame[zone_id].astype(float)
    return {
        "mean": round(float(values.mean() or 0), 2),
        "peak": round(float(values.max() or 0), 2),
    }


def summarize_weather(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float | int | None]:
    frame = weather[(weather["time"] >= start) & (weather["time"] <= end)]
    result = {"mean_temp_c": None, "mean_humidity": None, "rain_hours": None, "total_rain": None}
    for column, key in (("T", "mean_temp_c"), ("U", "mean_humidity")):
        if column in frame and frame[column].notna().any():
            result[key] = round(float(frame[column].mean()), 2)
    if "nRAIN" in frame and frame["nRAIN"].notna().any():
        result["rain_hours"] = int((frame["nRAIN"] > 0).sum())
        result["total_rain"] = round(float(frame["nRAIN"].sum()), 2)
    return result


def daily_totals(frame: pd.DataFrame, value_col: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    values = frame.set_index("time")[value_col].astype(float).resample("D").sum()
    return [{"date": idx.date().isoformat(), "kwh": round(float(value), 2)} for idx, value in values.items()]


def hourly_shape(history: pd.DataFrame) -> dict[str, float]:
    hist = history.copy()
    hist["hour"] = hist["time"].dt.hour
    means = hist.groupby("hour")["actual_kwh"].mean()
    return {
        "morning_7_10": round(float(means.loc[7:10].mean()), 2),
        "noon_11_14": round(float(means.loc[11:14].mean()), 2),
        "evening_17_22": round(float(means.loc[17:22].mean()), 2),
        "night_20_6": round(float(pd.concat([means.loc[20:23], means.loc[0:6]]).mean()), 2),
    }


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
        "historical_min_service_price",
        "historical_max_service_price",
        "mean_energy_price",
    ]
    return {key: round_float(profile.get(key)) for key in keys}


def round_float(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def finite_round(value: Any, ndigits: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return round(number, ndigits)


def pct_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current / baseline - 1.0) * 100.0
