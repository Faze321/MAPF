from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import CALIBRATION_METHOD, normalize_forecast_model_name


_TIMESFM_MODEL_CACHE: dict[tuple[str, int, int], Any] = {}
_CHRONOS_MODEL_CACHE: dict[tuple[str, str], Any] = {}
DEFAULT_TIMESFM_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
KNOWN_FUTURE_COVARIATES = frozenset([*DEFAULT_TIMESFM_EXOG_COLS, "hour", "occupancy"])
ENERGY_PRICE_FALLBACK_ELASTICITY = 0.1
ENERGY_PRICE_ADJUSTMENT_MAX_RATIO = 0.3
ENERGY_PRICE_RESPONSE_MIN_OBS = 12


@dataclass(frozen=True)
class ForecastResult:
    hourly: pd.DataFrame
    summary: dict[str, Any]
    reusable_forecaster: dict[str, Any] | None = None


@dataclass(frozen=True)
class LSTMForecastBundle:
    model: Any
    load_mean: float
    load_std: float
    exog_mean: np.ndarray
    exog_std: np.ndarray
    context_hours: int
    device: str
    input_size: int
    hidden_size: int
    num_layers: int


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
    history_days: int = 30,
    validation_days: int = 5,
    backend: str = "timesfm",
    diurnal_blend_alpha: float = 0.0,
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 168,
    timesfm_exog_cols: list[str] | None = None,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 512,
    chronos_exog_cols: list[str] | None = None,
    chronos_device: str = "auto",
    lstm_context_hours: int = 24,
    lstm_exog_cols: list[str] | None = None,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    lstm_epochs: int = 50,
    lstm_learning_rate: float = 0.001,
    lstm_batch_size: int = 32,
    lstm_device: str = "auto",
    lstm_seed: int = 42,
    xgboost_lags: list[int] | None = None,
    xgboost_exog_cols: list[str] | None = None,
    xgboost_n_estimators: int = 300,
    xgboost_max_depth: int = 9,
    xgboost_learning_rate: float = 0.1,
    xgboost_colsample_bytree: float = 0.9,
    xgboost_tree_method: str = "approx",
    xgboost_n_jobs: int = 1,
    xgboost_seed: int = 42,
) -> ForecastResult:
    for name, value, minimum in (("history_days", history_days, 1), ("validation_days", validation_days, 0), ("horizon_days", horizon_days, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    zone_id = str(zone_id)
    normalized_model = normalize_forecast_model_name(backend)
    if normalized_model == "xgboost":
        from xgboost_forecaster import DEFAULT_XGBOOST_LAGS

        xgboost_lags = list(DEFAULT_XGBOOST_LAGS if xgboost_lags is None else xgboost_lags)
        xgboost_exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if xgboost_exog_cols is None else xgboost_exog_cols)
    horizon_hours = horizon_days * 24
    if forecast_origin is None:
        forecast_origin = load["time"].max() - pd.Timedelta(hours=horizon_hours - 1)
    forecast_origin = pd.Timestamp(forecast_origin)
    forecast_end = forecast_origin + pd.Timedelta(hours=horizon_hours - 1)
    validation_hours = max(0, int(validation_days) * 24)
    validation_start = forecast_origin - pd.Timedelta(hours=validation_hours) if validation_hours else forecast_origin
    validation_end = forecast_origin - pd.Timedelta(hours=1)
    history_start = validation_start - pd.Timedelta(days=history_days)
    history_end = validation_start - pd.Timedelta(hours=1)

    zone_frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, zone_id)
    history = zone_frame[(zone_frame["time"] >= history_start) & (zone_frame["time"] <= history_end)]
    validation = zone_frame[(zone_frame["time"] >= validation_start) & (zone_frame["time"] <= validation_end)]
    forecast_features = zone_frame[(zone_frame["time"] >= forecast_origin) & (zone_frame["time"] <= forecast_end)].copy()
    if len(history) != history_days * 24 or len(validation) != validation_hours:
        raise ValueError(f"Zone {zone_id} requires complete history and validation windows")
    future_times = pd.date_range(forecast_origin, periods=horizon_hours, freq="h")
    forecast_features = forecast_features.set_index("time").reindex(future_times).rename_axis("time").reset_index()

    hourly = forecast_load(
        history,
        validation,
        zone_frame,
        forecast_origin,
        horizon_hours,
        backend=backend,
        diurnal_blend_alpha=diurnal_blend_alpha,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_exog_cols=timesfm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_exog_cols=chronos_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        chronos_device=chronos_device,
        lstm_context_hours=lstm_context_hours,
        lstm_exog_cols=lstm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_epochs=lstm_epochs,
        lstm_learning_rate=lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
        lstm_device=lstm_device,
        lstm_seed=lstm_seed,
        xgboost_lags=xgboost_lags,
        xgboost_exog_cols=xgboost_exog_cols,
        xgboost_n_estimators=xgboost_n_estimators,
        xgboost_max_depth=xgboost_max_depth,
        xgboost_learning_rate=xgboost_learning_rate,
        xgboost_colsample_bytree=xgboost_colsample_bytree,
        xgboost_tree_method=xgboost_tree_method,
        xgboost_n_jobs=xgboost_n_jobs,
        xgboost_seed=xgboost_seed,
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
        "validation_start": validation_start.isoformat() if validation_hours else None,
        "validation_end": validation_end.isoformat() if validation_hours else None,
        "forecast_origin": forecast_origin.isoformat(),
        "forecast_end": forecast_end.isoformat(),
        "forecast_horizon_days": int(horizon_days),
        "forecast_horizon_hours": int(horizon_hours),
        "backend": normalized_model,
        "diurnal_blend_alpha": float(diurnal_blend_alpha),
        "calibration_method": CALIBRATION_METHOD,
        "timesfm_repo": timesfm_repo if normalized_model == "timesfm" else None,
        "timesfm_context_hours": int(timesfm_context_hours) if normalized_model == "timesfm" else None,
        "timesfm_covariates": (timesfm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS) if normalized_model == "timesfm" else None,
        "chronos_repo": chronos_repo if normalized_model.startswith("chronos") else None,
        "chronos_context_hours": int(chronos_context_hours) if normalized_model.startswith("chronos") else None,
        "chronos_covariates": (chronos_exog_cols or DEFAULT_TIMESFM_EXOG_COLS)
        if normalized_model.startswith("chronos")
        else None,
        "chronos_device": chronos_device if normalized_model.startswith("chronos") else None,
        "energy_price_adjustment": forecast_attrs.get("energy_price_adjustment"),
        "lstm_context_hours": int(lstm_context_hours) if normalized_model == "lstm" else None,
        "lstm_exog_cols": (lstm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS) if normalized_model == "lstm" else None,
        "lstm_hidden_size": int(lstm_hidden_size) if normalized_model == "lstm" else None,
        "lstm_num_layers": int(lstm_num_layers) if normalized_model == "lstm" else None,
        "lstm_epochs": int(lstm_epochs) if normalized_model == "lstm" else None,
        "lstm_learning_rate": float(lstm_learning_rate) if normalized_model == "lstm" else None,
        "lstm_batch_size": int(lstm_batch_size) if normalized_model == "lstm" else None,
        "lstm_device": lstm_device if normalized_model == "lstm" else None,
        "lstm_seed": int(lstm_seed) if normalized_model == "lstm" else None,
        "xgboost_lags": xgboost_lags if normalized_model == "xgboost" else None,
        "xgboost_exog_cols": xgboost_exog_cols if normalized_model == "xgboost" else None,
        "xgboost_n_estimators": xgboost_n_estimators if normalized_model == "xgboost" else None,
        "xgboost_max_depth": xgboost_max_depth if normalized_model == "xgboost" else None,
        "xgboost_learning_rate": xgboost_learning_rate if normalized_model == "xgboost" else None,
        "xgboost_colsample_bytree": xgboost_colsample_bytree if normalized_model == "xgboost" else None,
        "xgboost_tree_method": xgboost_tree_method if normalized_model == "xgboost" else None,
        "xgboost_n_jobs": xgboost_n_jobs if normalized_model == "xgboost" else None,
        "xgboost_seed": xgboost_seed if normalized_model == "xgboost" else None,
        "calibration": forecast_attrs.get("calibration"),
        "forecast_total_kwh": round(forecast_total, 2),
        "forecast_peak_kwh": round(forecast_peak, 2),
        "predicted_change_pct": round(predicted_change_pct, 2),
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
        or key.startswith(normalized_model.lower() + "_")
    }
    known_future = forecast_features.drop(columns=["actual_kwh", "load"], errors="ignore").copy()
    reusable_forecaster = {
        "zone": zone_id,
        "backend": normalized_model,
        "forecast_origin": forecast_origin.isoformat(),
        "horizon_hours": int(horizon_hours),
        "history": history.copy(),
        "validation": validation.copy(),
        "known_future": known_future,
        "forecast_parameters": {
            "timesfm_repo": timesfm_repo,
            "timesfm_context_hours": int(timesfm_context_hours),
            "timesfm_exog_cols": list(timesfm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS),
            "chronos_repo": chronos_repo,
            "chronos_context_hours": int(chronos_context_hours),
            "chronos_exog_cols": list(chronos_exog_cols or DEFAULT_TIMESFM_EXOG_COLS),
            "chronos_device": chronos_device,
            "lstm_context_hours": int(lstm_context_hours),
            "lstm_exog_cols": list(lstm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS),
            "lstm_hidden_size": int(lstm_hidden_size),
            "lstm_num_layers": int(lstm_num_layers),
            "lstm_epochs": int(lstm_epochs),
            "lstm_learning_rate": float(lstm_learning_rate),
            "lstm_batch_size": int(lstm_batch_size),
            "lstm_device": lstm_device,
            "lstm_seed": int(lstm_seed),
            "xgboost_lags": xgboost_lags,
            "xgboost_exog_cols": xgboost_exog_cols,
            "xgboost_n_estimators": xgboost_n_estimators,
            "xgboost_max_depth": xgboost_max_depth,
            "xgboost_learning_rate": xgboost_learning_rate,
            "xgboost_colsample_bytree": xgboost_colsample_bytree,
            "xgboost_tree_method": xgboost_tree_method,
            "xgboost_n_jobs": xgboost_n_jobs,
            "xgboost_seed": xgboost_seed,
        },
        "fitted_state": forecast_attrs.get("reusable_forecaster_state"),
    }
    reusable_forecaster["forecast_parameters"] = {
        key: value for key, value in reusable_forecaster["forecast_parameters"].items()
        if key.startswith(normalized_model.lower() + "_")
    }
    reusable_forecaster["forecast_parameters"].update(
        diurnal_blend_alpha=float(diurnal_blend_alpha), calibration_method=CALIBRATION_METHOD,
    )
    return ForecastResult(
        hourly=hourly,
        summary=summary,
        reusable_forecaster=reusable_forecaster,
    )


def forecast_load(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    horizon_hours: int,
    *,
    backend: str,
    diurnal_blend_alpha: float = 0.0,
    timesfm_repo: str = 'google/timesfm-2.5-200m-pytorch',
    timesfm_context_hours: int = 168,
    timesfm_exog_cols: list[str] | None = None,
    chronos_repo: str = 'amazon/chronos-2',
    chronos_context_hours: int = 512,
    chronos_exog_cols: list[str] | None = None,
    chronos_device: str = 'auto',
    lstm_context_hours: int = 24,
    lstm_exog_cols: list[str] | None = None,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    lstm_epochs: int = 50,
    lstm_learning_rate: float = 0.001,
    lstm_batch_size: int = 32,
    lstm_device: str = 'auto',
    lstm_seed: int = 42,
    xgboost_lags: list[int] | None = None,
    xgboost_exog_cols: list[str] | None = None,
    xgboost_n_estimators: int = 300,
    xgboost_max_depth: int = 9,
    xgboost_learning_rate: float = 0.1,
    xgboost_colsample_bytree: float = 0.9,
    xgboost_tree_method: str = 'approx',
    xgboost_n_jobs: int = 1,
    xgboost_seed: int = 42,
    fitted_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fit once, calibrate daily forecasts, then forecast without future observations."""
    parameters = {name: value for name, value in locals().copy().items()
                  if name.startswith(("timesfm_", "chronos_", "lstm_", "xgboost_"))}
    backend = normalize_forecast_model_name(backend)
    if backend not in {"timesfm", "chronos", "lstm", "xgboost", "AR"}:
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
    validation = checked_hourly_history(validation, "validation", allow_empty=True)
    history_hours = len(history)
    if history_hours < 24:
        raise ValueError("Forecasting requires at least 24 historical hours")
    if len(validation) % 24:
        raise ValueError("validation must contain complete 24-hour days")
    past = pd.concat([history, validation], ignore_index=True)
    expected_past = pd.date_range(end=origin - pd.Timedelta(hours=1), periods=len(past), freq="h")
    if not pd.DatetimeIndex(past["time"]).equals(expected_past):
        raise ValueError("history and validation must be consecutive hourly rows ending before forecast_origin")
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
        if (fitted_state.get("calibration", {}).get("method") != CALIBRATION_METHOD
                or pd.Timestamp(fitted_state.get("forecast_origin")) != origin
                or fitted_state.get("history_hours") != history_hours
                or fitted_state.get("diurnal_blend_alpha") != alpha):
            raise ValueError("Reusable forecaster state does not match the origin, history, alpha or calibration protocol")
        bias = state_vector(fitted_state, "bias_vec")
        if bias.shape != (24,) or not np.isfinite(bias).all():
            raise ValueError("Reusable bias_vec must contain 24 finite hourly values")
        diurnal = state_diurnal_profile(fitted_state)
        calibration = dict(fitted_state["calibration"])
    else:
        diurnal = build_diurnal_profile(history)
        bias = np.zeros(24, dtype=np.float64)

    predict_raw, model_state = prepare_forecast_backend(
        backend, history, parameters, max(24, int(horizon_hours)), fitted_state=fitted_state,
    )
    if fitted_state is None:
        validation_rows: list[dict[str, Any]] = []
        residual_days = []
        for day, offset in enumerate(range(0, len(validation), 24)):
            day_frame = validation.iloc[offset:offset + 24].copy()
            context = pd.concat([history, validation.iloc[:offset]], ignore_index=True).tail(history_hours)
            raw_frame = predict_raw(context, day_frame.drop(columns=["actual_kwh", "load"], errors="ignore"))
            raw = checked_raw_prediction(raw_frame, day_frame["time"])
            blended = blend_with_diurnal(raw, diurnal_for_times(day_frame["time"], diurnal), alpha)
            actual = day_frame["actual_kwh"].to_numpy(dtype=np.float64)
            residual = blended - actual
            residual_days.append(residual)
            for hour, timestamp in enumerate(day_frame["time"]):
                validation_rows.append({
                    "time": pd.Timestamp(timestamp).isoformat(), "validation_day": day + 1,
                    "horizon_offset": hour, "raw_predicted_kwh": float(raw[hour]),
                    "blended_predicted_kwh": float(blended[hour]), "actual_kwh": float(actual[hour]),
                    "residual_kwh": float(residual[hour]),
                })
        days = len(residual_days)
        half_width = np.nan
        metrics = None
        if days:
            residual_matrix = np.asarray(residual_days, dtype=np.float64)
            bias = residual_matrix.mean(axis=0)
            # All validation days contribute, after the same hourly mean correction.
            half_width = estimate_interval_half_width((residual_matrix - bias).ravel())
            comparison = pd.DataFrame(validation_rows).rename(columns={"blended_predicted_kwh": "predicted_kwh"})
            metrics = compute_forecast_metrics(comparison)
        calibration = {
            "method": CALIBRATION_METHOD, "enabled": bool(days), "validation_days": days,
            "bias_by_hour": bias.tolist(), "samples_per_hour": [days] * 24,
            "bias_mean": float(bias.mean()), "bias_max_abs": float(np.max(np.abs(bias))),
            "interval_half_width": float(half_width) if np.isfinite(half_width) else None,
            "metrics": metrics, "validation_predictions": validation_rows,
        }
        fitted_state = {
            **model_state, "backend": backend, "forecast_origin": origin.isoformat(),
            "history_hours": history_hours, "diurnal_blend_alpha": alpha,
            "diurnal_by_hour": serialize_diurnal_profile(diurnal),
            "bias_vec": bias.tolist(), "calibration": calibration,
        }

    result = predict_raw(past.tail(history_hours), future).copy()
    raw = checked_raw_prediction(result, future_times)
    blended = blend_with_diurnal(raw, diurnal_for_times(future_times, diurnal), alpha)
    hourly_bias = bias[np.arange(horizon_hours) % 24]
    point = np.clip(blended - hourly_bias, 0, None)
    width = calibration.get("interval_half_width")
    residual_q10, residual_q90 = deterministic_interval(point, np.nan if width is None else float(width))
    if "q10_kwh" in result and "q90_kwh" in result:
        q10, q90 = rebuild_quantile_interval(point, result["q10_kwh"], result["q90_kwh"])
        available = np.isfinite(q10) & np.isfinite(q90)
        q10, q90 = np.where(available, q10, residual_q10), np.where(available, q90, residual_q90)
    else:
        q10, q90 = residual_q10, residual_q90
    result["raw_predicted_kwh"] = raw
    result["blended_predicted_kwh"] = blended
    result["bias_kwh"] = hourly_bias
    result["bias_corrected_kwh"] = point
    result["predicted_kwh"] = point
    result["q10_kwh"], result["q50_kwh"], result["q90_kwh"] = q10, point, q90
    result.attrs["calibration"] = calibration
    result.attrs["reusable_forecaster_state"] = fitted_state
    return result


def checked_hourly_history(frame: pd.DataFrame, label: str, *, allow_empty: bool = False) -> pd.DataFrame:
    if frame.empty and allow_empty:
        return frame.copy()
    if "time" not in frame or "actual_kwh" not in frame or frame.empty:
        raise ValueError(f"{label} must contain hourly time and actual_kwh values")
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"])
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


def checked_raw_prediction(frame: pd.DataFrame, times) -> np.ndarray:
    if len(frame) != len(times) or not pd.DatetimeIndex(frame["time"]).equals(pd.DatetimeIndex(times)):
        raise ValueError("Backend prediction timestamps do not match the requested horizon")
    raw = frame["raw_predicted_kwh"].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Backend returned non-finite load predictions")
    return raw


def prepare_forecast_backend(
    backend: str, history: pd.DataFrame, parameters: dict[str, Any], max_horizon: int,
    *, fitted_state: dict[str, Any] | None = None,
) -> tuple[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], dict[str, Any]]:
    """Prepare one frozen model and return an uncalibrated inference callable."""
    p = parameters
    state = fitted_state or {}
    prefix = backend.lower() + "_"
    cols = p.get(prefix + "exog_cols")
    exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if cols is None else cols)
    if set(exog_cols) - KNOWN_FUTURE_COVARIATES:
        raise ValueError("Forecaster covariates must be known future features, never load or evaluation fields")

    if backend == "timesfm":
        context_hours = p["timesfm_context_hours"]
        model = load_timesfm_model(p["timesfm_repo"], context_hours, max(32, max_horizon))
        def predict(context, future):
            raw, q10, q90 = run_timesfm_prediction(
                model, context["actual_kwh"].to_numpy(dtype=float), build_exog_matrix(context, exog_cols),
                build_exog_matrix(future, exog_cols), len(future), context_hours, exog_cols,
            )
            return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw,
                                 "q10_kwh": q10, "q90_kwh": q90})
        return predict, {}
    if backend == "chronos":
        context_hours = p["chronos_context_hours"]
        model = load_chronos_model(p["chronos_repo"], p["chronos_device"])
        def predict(context, future):
            raw, q10, q90 = run_chronos_prediction(
                model, context["actual_kwh"].to_numpy(dtype=float), len(future), context_hours,
                exog_context=build_exog_matrix(context, exog_cols),
                exog_horizon=build_exog_matrix(future, exog_cols), exog_cols=exog_cols,
            )
            return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw,
                                 "q10_kwh": q10, "q90_kwh": q90})
        return predict, {}
    if backend == "lstm":
        if fitted_state is None:
            bundle = train_lstm_model(
                history, context_hours=p["lstm_context_hours"], exog_cols=exog_cols,
                hidden_size=p["lstm_hidden_size"], num_layers=p["lstm_num_layers"], epochs=p["lstm_epochs"],
                learning_rate=p["lstm_learning_rate"], batch_size=p["lstm_batch_size"],
                device=p["lstm_device"], seed=p["lstm_seed"],
            )
            model_state = {"lstm_bundle": serialize_lstm_bundle(bundle)}
        else:
            bundle = deserialize_lstm_bundle(state["lstm_bundle"], requested_device=p["lstm_device"])
            model_state = {"lstm_bundle": state["lstm_bundle"]}
        def predict(context, future):
            raw = run_lstm_prediction(bundle, context["actual_kwh"].to_numpy(dtype=float),
                                      build_lstm_exog_matrix(context, exog_cols),
                                      build_lstm_exog_matrix(future, exog_cols), len(future))
            return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw})
        return predict, model_state
    if backend == "xgboost":
        from xgboost_forecaster import prepare_xgboost_backend
        return prepare_xgboost_backend(history, p, fitted_state=fitted_state)
    if backend == "AR":
        response = state["energy_price_response"] if fitted_state is not None else estimate_energy_price_response(history)
        def predict(context, future):
            return run_ar_prediction(context, future, training_history=history, fitted_response=response)
        return predict, {"energy_price_response": response}
    raise ValueError(f"Unsupported backend: {backend}")


def load_timesfm_model(repo: str, context_hours: int, max_horizon: int):
    key = (repo, int(context_hours), int(max_horizon))
    if key in _TIMESFM_MODEL_CACHE:
        return _TIMESFM_MODEL_CACHE[key]

    try:
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timesfm is required for backend: timesfm. "
            "Install the official google-research/timesfm package and make sure "
            "this Python environment can import it."
        ) from exc

    patch_timesfm_hub_kwargs(TimesFM_2p5_200M_torch)
    model = TimesFM_2p5_200M_torch.from_pretrained(repo)
    model.compile(
        ForecastConfig(
            max_context=int(context_hours),
            max_horizon=int(max_horizon),
            normalize_inputs=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    _TIMESFM_MODEL_CACHE[key] = model
    return model


def patch_timesfm_hub_kwargs(model_cls) -> None:
    if getattr(model_cls, "_mapf_hub_kwargs_compat", False):
        return

    original = model_cls._from_pretrained

    @classmethod
    def _from_pretrained_compat(cls, *args, **kwargs):
        kwargs.pop("proxies", None)
        kwargs.pop("resume_download", None)
        return original(*args, **kwargs)

    model_cls._from_pretrained = _from_pretrained_compat
    model_cls._mapf_hub_kwargs_compat = True


def load_chronos_model(repo: str, device: str):
    resolved_device = resolve_torch_device(device)
    key = (repo, resolved_device)
    if key in _CHRONOS_MODEL_CACHE:
        return _CHRONOS_MODEL_CACHE[key]

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chronos-forecasting with Chronos2Pipeline is required for backend: chronos. "
            "Install it in this Python environment."
        ) from exc

    repo_lower = repo.lower()
    if "chronos-2" not in repo_lower and "chronos2" not in repo_lower:
        raise ValueError('This project only supports Chronos 2. Set chronos_repo: "amazon/chronos-2".')

    dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
    pipeline = Chronos2Pipeline.from_pretrained(
        repo,
        device_map=resolved_device,
        dtype=dtype,
    )
    _CHRONOS_MODEL_CACHE[key] = pipeline
    return pipeline


def resolve_torch_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized in {"gpu", "cuda"}:
        return "cuda"
    return normalized


def run_timesfm_prediction(
    model,
    context_load: np.ndarray,
    exog_context: np.ndarray | None,
    exog_horizon: np.ndarray | None,
    horizon: int,
    context_hours: int,
    exog_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = context_load[-context_hours:].astype(np.float64)
    if exog_context is not None and exog_horizon is not None and hasattr(model, "forecast_with_covariates"):
        exog_ctx = exog_context[-len(ctx) :].astype(np.float64)
        dyn_cov = {}
        for idx, col in enumerate(exog_cols):
            if idx < exog_ctx.shape[1] and idx < exog_horizon.shape[1]:
                combined = np.concatenate([exog_ctx[:, idx], exog_horizon[:, idx]])
                dyn_cov[col] = [combined.tolist()]
        try:
            result = model.forecast_with_covariates(
                inputs=[ctx.tolist()],
                dynamic_numerical_covariates=dyn_cov if dyn_cov else None,
                xreg_mode="xreg + timesfm",
                normalize_xreg_target_per_input=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM covariate forecasting requires the xreg dependencies. "
                "Install torch, jax, jaxlib, scikit-learn, and the official "
                "google-research/timesfm package."
            ) from exc
    else:
        result = model.forecast(horizon=int(horizon), inputs=[ctx])
    return parse_timesfm_result(result, horizon)


def run_chronos_prediction(
    pipeline,
    context_load: np.ndarray,
    horizon: int,
    context_hours: int,
    *,
    exog_context: np.ndarray | None = None,
    exog_horizon: np.ndarray | None = None,
    exog_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for backend: chronos.") from exc

    ctx = context_load[-context_hours:].astype(np.float32)
    inputs: list[Any]
    covariate_input = build_chronos_covariate_input(ctx, exog_context, exog_horizon, horizon, exog_cols or [])
    if covariate_input is not None:
        inputs = [covariate_input]
    else:
        inputs = [torch.tensor(ctx, dtype=torch.float32)]
    predict_kwargs: dict[str, Any] = {"limit_prediction_length": False}
    quantiles, _ = pipeline.predict_quantiles(
        inputs,
        prediction_length=int(horizon),
        quantile_levels=[0.1, 0.5, 0.9],
        **predict_kwargs,
    )
    if isinstance(quantiles, list):
        if not quantiles:
            raise RuntimeError("Chronos returned no quantile forecasts")
        q = np.asarray(quantiles[0].detach().cpu(), dtype=np.float64)
        if q.ndim == 3:
            q = q[0]
        q = q[np.newaxis, :, :]
    else:
        q = np.asarray(quantiles.detach().cpu(), dtype=np.float64)
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[-1] < 3:
        raise RuntimeError(f"Unexpected Chronos quantile shape: {q.shape}")
    q10 = q[0, :horizon, 0]
    q50 = q[0, :horizon, 1]
    q90 = q[0, :horizon, 2]
    return np.clip(q50, 0, None), np.clip(q10, 0, None), np.clip(q90, 0, None)


def build_chronos_covariate_input(
    context_load: np.ndarray,
    exog_context: np.ndarray | None,
    exog_horizon: np.ndarray | None,
    horizon: int,
    exog_cols: list[str],
) -> dict[str, Any] | None:
    if not exog_cols or exog_context is None or exog_horizon is None:
        return None
    ctx = np.asarray(context_load, dtype=np.float32)
    if len(ctx) == 0:
        return None
    context_arr = np.asarray(exog_context, dtype=np.float64)
    if len(context_arr) > len(ctx):
        context_matrix = context_arr[-len(ctx) :]
    else:
        context_matrix = align_exog(context_arr, len(ctx))
    horizon_matrix = align_exog(np.asarray(exog_horizon, dtype=np.float64), int(horizon))
    if context_matrix.shape[1] == 0 or horizon_matrix.shape[1] == 0:
        return None

    past_covariates: dict[str, np.ndarray] = {}
    future_covariates: dict[str, np.ndarray] = {}
    for idx, col in enumerate(exog_cols):
        if idx >= context_matrix.shape[1] or idx >= horizon_matrix.shape[1]:
            continue
        past_covariates[col] = context_matrix[:, idx].astype(np.float32)
        future_covariates[col] = horizon_matrix[:, idx].astype(np.float32)
    if not past_covariates or not future_covariates:
        return None
    return {
        "target": ctx,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def train_lstm_model(
    history: pd.DataFrame,
    *,
    context_hours: int,
    exog_cols: list[str],
    hidden_size: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    seed: int,
) -> LSTMForecastBundle:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for backend: lstm.") from exc

    load = history["actual_kwh"].astype(float).to_numpy(dtype=np.float64)
    exog = build_lstm_exog_matrix(history, exog_cols)
    if len(load) < 2:
        raise ValueError("LSTM requires at least two historical values")

    resolved_device = resolve_torch_device(device)
    torch.manual_seed(int(seed))
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    load_mean = float(np.nanmean(load))
    load_std = safe_std(load)
    exog_mean = np.nanmean(exog, axis=0) if exog.size else np.empty(0, dtype=np.float64)
    exog_std = np.nanstd(exog, axis=0) if exog.size else np.empty(0, dtype=np.float64)
    exog_std = np.where(exog_std > 1e-9, exog_std, 1.0)
    context = max(1, int(context_hours))

    x_train, y_train = build_lstm_training_arrays(
        load,
        exog,
        context,
        load_mean,
        load_std,
        exog_mean,
        exog_std,
    )
    input_size = int(x_train.shape[-1])
    model = create_lstm_regressor(
        input_size=input_size,
        hidden_size=max(1, int(hidden_size)),
        num_layers=max(1, int(num_layers)),
    ).to(resolved_device)

    features = torch.tensor(x_train, dtype=torch.float32, device=resolved_device)
    targets = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32, device=resolved_device)
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(max(1, int(batch_size)), len(dataset)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(max(1, int(epochs))):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return LSTMForecastBundle(
        model=model,
        load_mean=load_mean,
        load_std=load_std,
        exog_mean=exog_mean,
        exog_std=exog_std,
        context_hours=context,
        device=resolved_device,
        input_size=input_size,
        hidden_size=max(1, int(hidden_size)),
        num_layers=max(1, int(num_layers)),
    )


def serialize_lstm_bundle(bundle: LSTMForecastBundle) -> dict[str, Any]:
    return {
        "load_mean": float(bundle.load_mean),
        "load_std": float(bundle.load_std),
        "exog_mean": np.asarray(bundle.exog_mean, dtype=np.float64).tolist(),
        "exog_std": np.asarray(bundle.exog_std, dtype=np.float64).tolist(),
        "context_hours": int(bundle.context_hours),
        "input_size": int(bundle.input_size),
        "hidden_size": int(bundle.hidden_size),
        "num_layers": int(bundle.num_layers),
        "state_dict": {
            str(name): tensor.detach().cpu().tolist()
            for name, tensor in bundle.model.state_dict().items()
        },
    }


def deserialize_lstm_bundle(
    payload: dict[str, Any],
    *,
    requested_device: str,
) -> LSTMForecastBundle:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for backend: lstm.") from exc

    input_size = int(payload["input_size"])
    hidden_size = int(payload["hidden_size"])
    num_layers = int(payload["num_layers"])
    model = create_lstm_regressor(input_size, hidden_size, num_layers)
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("Reusable LSTM artifact has no trained state_dict.")
    state_dict = {
        str(name): torch.tensor(value, dtype=torch.float32)
        for name, value in raw_state.items()
    }
    model.load_state_dict(state_dict, strict=True)
    resolved_device = resolve_torch_device(requested_device)
    model = model.to(resolved_device)
    model.eval()
    return LSTMForecastBundle(
        model=model,
        load_mean=float(payload["load_mean"]),
        load_std=float(payload["load_std"]),
        exog_mean=np.asarray(payload.get("exog_mean") or [], dtype=np.float64),
        exog_std=np.asarray(payload.get("exog_std") or [], dtype=np.float64),
        context_hours=int(payload["context_hours"]),
        device=resolved_device,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


def create_lstm_regressor(input_size: int, hidden_size: int, num_layers: int):
    import torch

    class LSTMRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = torch.nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return LSTMRegressor()


def build_lstm_training_arrays(
    load: np.ndarray,
    exog: np.ndarray,
    context_hours: int,
    load_mean: float,
    load_std: float,
    exog_mean: np.ndarray,
    exog_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for target_idx in range(1, len(load)):
        x_rows.append(
            build_lstm_sequence(
                known_load=load,
                exog_until_target=exog,
                target_idx=target_idx,
                context_hours=context_hours,
                load_mean=load_mean,
                load_std=load_std,
                exog_mean=exog_mean,
                exog_std=exog_std,
            )
        )
        y_rows.append((float(load[target_idx]) - load_mean) / load_std)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def run_lstm_prediction(
    bundle: LSTMForecastBundle,
    rolling_load: np.ndarray,
    rolling_exog: np.ndarray,
    future_exog: np.ndarray,
    horizon: int,
) -> np.ndarray:
    import torch

    future_exog = align_exog(future_exog, int(horizon))
    predictions: list[float] = []
    exog_until_target = np.vstack([rolling_exog, future_exog])

    with torch.no_grad():
        for step_idx in range(int(horizon)):
            known_load = np.concatenate([rolling_load, np.asarray(predictions, dtype=np.float64)])
            target_idx = len(rolling_load) + step_idx
            seq = build_lstm_sequence(
                known_load=known_load,
                exog_until_target=exog_until_target,
                target_idx=target_idx,
                context_hours=bundle.context_hours,
                load_mean=bundle.load_mean,
                load_std=bundle.load_std,
                exog_mean=bundle.exog_mean,
                exog_std=bundle.exog_std,
            )
            tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32, device=bundle.device)
            pred_norm = float(bundle.model(tensor).detach().cpu().item())
            pred = pred_norm * bundle.load_std + bundle.load_mean
            predictions.append(float(max(pred, 0.0)))

    return np.asarray(predictions, dtype=np.float64)


def build_lstm_sequence(
    *,
    known_load: np.ndarray,
    exog_until_target: np.ndarray,
    target_idx: int,
    context_hours: int,
    load_mean: float,
    load_std: float,
    exog_mean: np.ndarray,
    exog_std: np.ndarray,
) -> np.ndarray:
    start = max(0, int(target_idx) - int(context_hours) + 1)
    rows = []
    for idx in range(start, int(target_idx) + 1):
        if len(known_load) == 0:
            prev_load = load_mean
        elif idx == 0:
            prev_load = float(known_load[0])
        else:
            prev_load = float(known_load[min(idx - 1, len(known_load) - 1)])
        load_feature = np.asarray([(prev_load - load_mean) / load_std], dtype=np.float64)
        exog_feature = standardize_exog(exog_until_target[idx], exog_mean, exog_std)
        rows.append(np.concatenate([load_feature, exog_feature]))

    sequence = np.asarray(rows, dtype=np.float32)
    if len(sequence) < context_hours:
        pad = np.repeat(sequence[[0]], context_hours - len(sequence), axis=0)
        sequence = np.vstack([pad, sequence])
    return sequence


def build_lstm_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    base = build_exog_matrix(frame, exog_cols)
    if "time" not in frame:
        time_features = np.zeros((len(frame), 4), dtype=np.float64)
    else:
        times = pd.DatetimeIndex(pd.to_datetime(frame["time"]))
        hour = times.hour.to_numpy(dtype=np.float64)
        dayofweek = times.dayofweek.to_numpy(dtype=np.float64)
        time_features = np.column_stack(
            [
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.sin(2.0 * np.pi * dayofweek / 7.0),
                np.cos(2.0 * np.pi * dayofweek / 7.0),
            ]
        )
    return np.hstack([base, time_features]) if base.size else time_features


def standardize_exog(exog_row: np.ndarray, exog_mean: np.ndarray, exog_std: np.ndarray) -> np.ndarray:
    if len(exog_mean) == 0:
        return np.empty(0, dtype=np.float64)
    row = np.asarray(exog_row, dtype=np.float64)
    return (row - exog_mean) / exog_std


def safe_std(values: np.ndarray) -> float:
    std = float(np.nanstd(values))
    return std if std > 1e-9 else 1.0


def estimate_interval_half_width(residual: np.ndarray) -> float:
    values = np.abs(np.asarray(residual, dtype=np.float64))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.nanquantile(values, 0.9))


def deterministic_interval(point: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    point_arr = np.asarray(point, dtype=np.float64)
    if not np.isfinite(half_width):
        return np.full(len(point_arr), np.nan), np.full(len(point_arr), np.nan)
    if half_width <= 0:
        return point_arr.copy(), point_arr.copy()
    q10 = np.clip(point_arr - half_width, 0, None)
    q90 = point_arr + half_width
    return np.minimum(q10, point_arr), np.maximum(q90, point_arr)


def parse_timesfm_result(result: Any, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, tuple):
        point_out, quantile_out = result
    else:
        arr = np.asarray(result)
        if arr.ndim == 3 and arr.shape[-1] > 5:
            point_out = arr[:, :, 5]
            quantile_out = arr
        else:
            point_out = arr
            quantile_out = None

    point = first_series(point_out, horizon)
    if quantile_out is None:
        q10 = np.full(horizon, np.nan, dtype=np.float64)
        q90 = np.full(horizon, np.nan, dtype=np.float64)
    else:
        quantiles = np.asarray(quantile_out, dtype=np.float64)
        if quantiles.ndim == 3:
            q10 = quantiles[0, :horizon, 0]
            q90_idx = min(8, quantiles.shape[-1] - 1)
            q90 = quantiles[0, :horizon, q90_idx]
        else:
            q10 = np.full(horizon, np.nan, dtype=np.float64)
            q90 = np.full(horizon, np.nan, dtype=np.float64)
    return np.clip(point, 0, None), np.clip(q10, 0, None), np.clip(q90, 0, None)


def first_series(values: Any, horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:horizon]
    return arr[0, :horizon]


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
        frame["temp_price_idx"] = frame["T"].astype(float) * frame["e_price"].astype(float)
    else:
        frame["temp_price_idx"] = 0.0
    return frame


def build_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    if not exog_cols:
        return np.empty((len(frame), 0), dtype=np.float64)
    exog = frame.reindex(columns=exog_cols).copy()
    exog = exog.apply(pd.to_numeric, errors="coerce")
    exog = exog.ffill().bfill()
    means = exog.mean(numeric_only=True)
    exog = exog.fillna(means).fillna(0.0)
    return exog.to_numpy(dtype=np.float64)


def align_exog(matrix: np.ndarray, target: int) -> np.ndarray:
    if len(matrix) == target:
        return matrix
    if len(matrix) > target:
        return matrix[:target]
    if len(matrix) == 0:
        return np.zeros((target, 0), dtype=np.float64)
    pad = np.tile(matrix[-1], (target - len(matrix), 1))
    return np.vstack([matrix, pad])


def align_vector(values: np.ndarray, target: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == target:
        return arr
    if len(arr) > target:
        return arr[:target]
    if len(arr) == 0:
        return np.zeros(target, dtype=np.float64)
    return np.concatenate([arr, np.repeat(arr[-1], target - len(arr))])


def disabled_calibration(**extra: Any) -> dict[str, Any]:
    return {
        "enabled": False,
        "bias_mean": 0.0,
        "bias_max_abs": 0.0,
        "metrics": None,
        **extra,
    }


def require_fitted_backend(state: dict[str, Any], expected_backend: str) -> None:
    actual = str(state.get("backend") or "").strip().lower().replace("_", "-")
    expected = str(expected_backend).strip().lower().replace("_", "-")
    if actual != expected:
        raise ValueError(
            f"Reusable forecaster state belongs to {actual or 'an unknown backend'}, "
            f"not {expected}."
        )


def state_vector(state: dict[str, Any], key: str) -> np.ndarray:
    if key not in state:
        raise ValueError(f"Reusable forecaster state is missing {key!r}.")
    values = np.asarray(state[key], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Reusable forecaster state {key!r} must be one-dimensional.")
    return values


def state_matrix(
    state: dict[str, Any],
    key: str,
    column_count: int,
    *,
    row_count: int | None = None,
) -> np.ndarray:
    if key not in state:
        raise ValueError(f"Reusable forecaster state is missing {key!r}.")
    values = np.asarray(state[key], dtype=np.float64)
    if values.size == 0:
        return np.empty((int(row_count or 0), int(column_count)), dtype=np.float64)
    if values.ndim == 1 and int(column_count) == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != int(column_count):
        raise ValueError(
            f"Reusable forecaster state {key!r} has shape {values.shape}; "
            f"expected (*, {int(column_count)})."
        )
    if row_count is not None and values.shape[0] != int(row_count):
        raise ValueError(
            f"Reusable forecaster state {key!r} has {values.shape[0]} rows; "
            f"expected {int(row_count)}."
        )
    return values


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


def compute_forecast_metrics(hourly: pd.DataFrame) -> dict[str, float | None]:
    valid = hourly[["actual_kwh", "predicted_kwh"]].dropna()
    if valid.empty:
        return {
            "MAE": None,
            "RMSE": None,
            "MAPE_pct": None,
            "RAE": None,
            "WAPE_pct": None,
            "n": 0,
        }

    actual = valid["actual_kwh"].astype(float).to_numpy()
    predicted = valid["predicted_kwh"].astype(float).to_numpy()
    error = actual - predicted
    abs_error = np.abs(error)
    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error * error)))
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    mape = float(np.nanmean(abs_error / denom) * 100.0)
    rae_denom = float(np.sum(np.abs(actual - actual.mean())))
    rae = float(np.sum(abs_error) / rae_denom) if rae_denom > 1e-9 else None
    actual_sum = float(np.sum(np.abs(actual)))
    wape = float(np.sum(abs_error) / actual_sum * 100.0) if actual_sum > 1e-9 else None

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_pct": round(mape, 4),
        "RAE": round(rae, 4) if rae is not None else None,
        "WAPE_pct": round(wape, 4) if wape is not None else None,
        "n": int(len(valid)),
    }


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
