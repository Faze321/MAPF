from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import normalize_forecast_model_name


_TIMESFM_MODEL_CACHE: dict[tuple[str, int, int], Any] = {}
_CHRONOS_MODEL_CACHE: dict[tuple[str, str], Any] = {}
DEFAULT_TIMESFM_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
ENERGY_PRICE_FALLBACK_ELASTICITY = 0.1
ENERGY_PRICE_ADJUSTMENT_MAX_RATIO = 0.3
ENERGY_PRICE_RESPONSE_MIN_OBS = 12


@dataclass(frozen=True)
class ForecastResult:
    hourly: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class LSTMForecastBundle:
    model: Any
    load_mean: float
    load_std: float
    exog_mean: np.ndarray
    exog_std: np.ndarray
    context_hours: int
    device: str


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
    forecast_start: pd.Timestamp | None,
    horizon_days: int,
    history_days: int,
    validation_days: int = 1,
    forecast_model: str = "timesfm",
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 168,
    timesfm_step_horizon: int = 24,
    timesfm_exog_cols: list[str] | None = None,
    timesfm_diurnal_blend_alpha: float = 1.0,
    timesfm_roll_actuals: bool = True,
    ar_diurnal_blend_alpha: float = 0.0,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 512,
    chronos_step_horizon: int = 24,
    chronos_exog_cols: list[str] | None = None,
    chronos_diurnal_blend_alpha: float = 0.0,
    chronos_device: str = "auto",
    chronos_roll_actuals: bool = True,
    lstm_context_hours: int = 24,
    lstm_step_horizon: int = 24,
    lstm_exog_cols: list[str] | None = None,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    lstm_epochs: int = 50,
    lstm_learning_rate: float = 0.001,
    lstm_batch_size: int = 32,
    lstm_diurnal_blend_alpha: float = 0.0,
    lstm_device: str = "auto",
    lstm_roll_actuals: bool = True,
    lstm_seed: int = 42,
) -> ForecastResult:
    zone_id = str(zone_id)
    normalized_model = normalize_forecast_model_name(forecast_model)
    horizon_hours = horizon_days * 24
    if forecast_start is None:
        forecast_start = load["time"].max() - pd.Timedelta(hours=horizon_hours - 1)
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)
    validation_hours = max(0, int(validation_days) * 24)
    validation_start = forecast_start - pd.Timedelta(hours=validation_hours) if validation_hours else forecast_start
    validation_end = forecast_start - pd.Timedelta(hours=1)
    history_start = validation_start - pd.Timedelta(days=history_days)
    history_end = validation_start - pd.Timedelta(hours=1)

    zone_frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, zone_id)
    history = zone_frame[(zone_frame["time"] >= history_start) & (zone_frame["time"] <= history_end)]
    validation = zone_frame[(zone_frame["time"] >= validation_start) & (zone_frame["time"] <= validation_end)]
    forecast_features = zone_frame[(zone_frame["time"] >= forecast_start) & (zone_frame["time"] <= forecast_end)].copy()
    if len(history) < 24:
        raise ValueError(f"Not enough history for zone {zone_id}: found {len(history)} hours")

    hourly = forecast_load(
        history,
        validation,
        zone_frame,
        forecast_start,
        horizon_hours,
        model_name=forecast_model,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_step_horizon=timesfm_step_horizon,
        timesfm_exog_cols=timesfm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
        timesfm_roll_actuals=timesfm_roll_actuals,
        ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_step_horizon=chronos_step_horizon,
        chronos_exog_cols=chronos_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
        chronos_device=chronos_device,
        chronos_roll_actuals=chronos_roll_actuals,
        lstm_context_hours=lstm_context_hours,
        lstm_step_horizon=lstm_step_horizon,
        lstm_exog_cols=lstm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_epochs=lstm_epochs,
        lstm_learning_rate=lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
        lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
        lstm_device=lstm_device,
        lstm_roll_actuals=lstm_roll_actuals,
        lstm_seed=lstm_seed,
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
    weather_summary = summarize_weather(weather, forecast_start, forecast_end)
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
    active_diurnal_blend_alpha = {
        "timesfm": timesfm_diurnal_blend_alpha,
        "AR": ar_diurnal_blend_alpha,
        "chronos": chronos_diurnal_blend_alpha,
        "lstm": lstm_diurnal_blend_alpha,
    }.get(normalized_model)

    summary = {
        "zone_id": zone_id,
        "category": category,
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "validation_start": validation_start.isoformat() if validation_hours else None,
        "validation_end": validation_end.isoformat() if validation_hours else None,
        "forecast_start": forecast_start.isoformat(),
        "forecast_end": forecast_end.isoformat(),
        "forecast_horizon_days": int(horizon_days),
        "forecast_horizon_hours": int(horizon_hours),
        "forecast_model": normalized_model,
        "diurnal_blend_alpha": round(float(active_diurnal_blend_alpha), 4)
        if active_diurnal_blend_alpha is not None
        else None,
        "timesfm_covariates": (timesfm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS) if normalized_model == "timesfm" else None,
        "timesfm_diurnal_blend_alpha": round(float(timesfm_diurnal_blend_alpha), 4) if normalized_model == "timesfm" else None,
        "timesfm_roll_actuals": bool(timesfm_roll_actuals) if normalized_model == "timesfm" else None,
        "ar_diurnal_blend_alpha": round(float(ar_diurnal_blend_alpha), 4) if normalized_model == "AR" else None,
        "chronos_repo": chronos_repo if normalized_model.startswith("chronos") else None,
        "chronos_context_hours": int(chronos_context_hours) if normalized_model.startswith("chronos") else None,
        "chronos_step_horizon": int(chronos_step_horizon) if normalized_model.startswith("chronos") else None,
        "chronos_covariates": (chronos_exog_cols or DEFAULT_TIMESFM_EXOG_COLS)
        if normalized_model.startswith("chronos")
        else None,
        "chronos_diurnal_blend_alpha": round(float(chronos_diurnal_blend_alpha), 4) if normalized_model.startswith("chronos") else None,
        "chronos_device": chronos_device if normalized_model.startswith("chronos") else None,
        "chronos_roll_actuals": bool(chronos_roll_actuals) if normalized_model.startswith("chronos") else None,
        "energy_price_adjustment": forecast_attrs.get("energy_price_adjustment"),
        "lstm_context_hours": int(lstm_context_hours) if normalized_model == "lstm" else None,
        "lstm_step_horizon": int(lstm_step_horizon) if normalized_model == "lstm" else None,
        "lstm_exog_cols": (lstm_exog_cols or DEFAULT_TIMESFM_EXOG_COLS) if normalized_model == "lstm" else None,
        "lstm_hidden_size": int(lstm_hidden_size) if normalized_model == "lstm" else None,
        "lstm_num_layers": int(lstm_num_layers) if normalized_model == "lstm" else None,
        "lstm_epochs": int(lstm_epochs) if normalized_model == "lstm" else None,
        "lstm_learning_rate": float(lstm_learning_rate) if normalized_model == "lstm" else None,
        "lstm_batch_size": int(lstm_batch_size) if normalized_model == "lstm" else None,
        "lstm_diurnal_blend_alpha": round(float(lstm_diurnal_blend_alpha), 4) if normalized_model == "lstm" else None,
        "lstm_device": lstm_device if normalized_model == "lstm" else None,
        "lstm_roll_actuals": bool(lstm_roll_actuals) if normalized_model == "lstm" else None,
        "lstm_seed": int(lstm_seed) if normalized_model == "lstm" else None,
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
    return ForecastResult(hourly=hourly, summary=summary)


def forecast_load(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    model_name: str,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_step_horizon: int,
    timesfm_exog_cols: list[str],
    timesfm_diurnal_blend_alpha: float,
    timesfm_roll_actuals: bool,
    ar_diurnal_blend_alpha: float,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_step_horizon: int,
    chronos_exog_cols: list[str],
    chronos_diurnal_blend_alpha: float,
    chronos_device: str,
    chronos_roll_actuals: bool,
    lstm_context_hours: int,
    lstm_step_horizon: int,
    lstm_exog_cols: list[str],
    lstm_hidden_size: int,
    lstm_num_layers: int,
    lstm_epochs: int,
    lstm_learning_rate: float,
    lstm_batch_size: int,
    lstm_diurnal_blend_alpha: float,
    lstm_device: str,
    lstm_roll_actuals: bool,
    lstm_seed: int,
) -> pd.DataFrame:
    normalized = normalize_forecast_model_name(model_name)
    if normalized == "AR":
        return ar_forecast(
            history,
            forecast_start,
            horizon_hours,
            validation=validation,
            full_frame=full_frame,
            diurnal_blend_alpha=ar_diurnal_blend_alpha,
        )
    if normalized == "timesfm":
        return timesfm_forecast(
            history,
            validation,
            full_frame,
            forecast_start,
            horizon_hours,
            repo=timesfm_repo,
            context_hours=timesfm_context_hours,
            step_horizon=timesfm_step_horizon,
            exog_cols=timesfm_exog_cols,
            diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
            roll_actuals=timesfm_roll_actuals,
        )
    if normalized in {"chronos"}:
        return chronos_forecast(
            history,
            validation,
            full_frame,
            forecast_start,
            horizon_hours,
            repo=chronos_repo,
            context_hours=chronos_context_hours,
            step_horizon=chronos_step_horizon,
            exog_cols=chronos_exog_cols,
            diurnal_blend_alpha=chronos_diurnal_blend_alpha,
            device=chronos_device,
            roll_actuals=chronos_roll_actuals,
        )
    if normalized == "lstm":
        return lstm_forecast(
            history,
            validation,
            full_frame,
            forecast_start,
            horizon_hours,
            context_hours=lstm_context_hours,
            step_horizon=lstm_step_horizon,
            exog_cols=lstm_exog_cols,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            epochs=lstm_epochs,
            learning_rate=lstm_learning_rate,
            batch_size=lstm_batch_size,
            diurnal_blend_alpha=lstm_diurnal_blend_alpha,
            device=lstm_device,
            roll_actuals=lstm_roll_actuals,
            seed=lstm_seed,
        )
    raise ValueError(f"Unsupported forecast_model: {model_name}")


def timesfm_forecast(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    repo: str,
    context_hours: int,
    step_horizon: int,
    exog_cols: list[str],
    diurnal_blend_alpha: float,
    roll_actuals: bool,
) -> pd.DataFrame:
    context_load = history["actual_kwh"].astype(float).to_numpy(dtype=np.float64)
    if len(context_load) == 0:
        raise ValueError("TimesFM requires at least one historical value")

    step = max(1, int(step_horizon))
    compile_horizon = max(step, 32)
    model = load_timesfm_model(repo, context_hours, compile_horizon)
    rolling_load = context_load.copy()
    rolling_exog = build_exog_matrix(history, exog_cols)
    diurnal_by_hour = build_diurnal_profile(history)

    calibration: dict[str, Any] = {
        "enabled": False,
        "bias_mean": 0.0,
        "bias_max_abs": 0.0,
        "metrics": None,
    }
    bias_vec = np.zeros(step, dtype=np.float64)
    if not validation.empty:
        val_horizon = min(step, len(validation))
        val_exog = align_exog(build_exog_matrix(validation, exog_cols), step)
        val_raw, _, _ = run_timesfm_prediction(
            model,
            rolling_load,
            rolling_exog,
            val_exog,
            step,
            context_hours,
            exog_cols,
        )
        val_times = pd.to_datetime(validation["time"].iloc[:val_horizon])
        val_blended = blend_with_diurnal(
            val_raw[:val_horizon],
            diurnal_for_times(val_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        val_actual = validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)[:val_horizon]
        bias_vec = align_vector(val_blended - val_actual, step)
        val_cmp = pd.DataFrame({"actual_kwh": val_actual, "predicted_kwh": val_blended})
        calibration = {
            "enabled": True,
            "bias_mean": round(float(np.nanmean(bias_vec)), 4),
            "bias_max_abs": round(float(np.nanmax(np.abs(bias_vec))), 4),
            "metrics": compute_forecast_metrics(val_cmp),
        }
        rolling_load = np.concatenate([rolling_load, validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)])
        rolling_exog = np.vstack([rolling_exog, build_exog_matrix(validation, exog_cols)])

    rows: list[pd.DataFrame] = []
    remaining = horizon_hours
    offset = 0
    while remaining > 0:
        chunk_horizon = min(step, remaining)
        chunk_start = forecast_start + pd.Timedelta(hours=offset)
        chunk_times = pd.date_range(chunk_start, periods=chunk_horizon, freq="h")
        chunk_frame = (
            full_frame.set_index("time")
            .reindex(chunk_times)
            .rename_axis("time")
            .reset_index()
        )
        chunk_exog = align_exog(build_exog_matrix(chunk_frame, exog_cols), chunk_horizon)
        raw, q10, q90 = run_timesfm_prediction(
            model,
            rolling_load,
            rolling_exog,
            chunk_exog,
            chunk_horizon,
            context_hours,
            exog_cols,
        )
        raw_point = raw[:chunk_horizon]
        raw_q10 = q10[:chunk_horizon]
        raw_q90 = q90[:chunk_horizon]
        bias = align_vector(bias_vec, chunk_horizon)
        bias_corrected = np.clip(raw_point - bias, 0, None)
        point = blend_with_diurnal(
            bias_corrected,
            diurnal_for_times(chunk_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        q10, q90 = rebuild_quantile_interval(point, raw_q10, raw_q90)

        rows.append(
            pd.DataFrame(
                {
                    "time": chunk_times,
                    "raw_predicted_kwh": raw_point,
                    "bias_corrected_kwh": bias_corrected,
                    "predicted_kwh": point,
                    "q10_kwh": q10,
                    "q50_kwh": point,
                    "q90_kwh": q90,
                }
            )
        )

        actual_chunk = chunk_frame.get("actual_kwh")
        if roll_actuals and actual_chunk is not None and actual_chunk.notna().all():
            roll_values = actual_chunk.astype(float).to_numpy(dtype=np.float64)
        else:
            roll_values = point
        rolling_load = np.concatenate([rolling_load, roll_values])
        rolling_exog = np.vstack([rolling_exog, chunk_exog])
        remaining -= chunk_horizon
        offset += chunk_horizon

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time", "predicted_kwh"])
    result.attrs["calibration"] = calibration
    return result


def chronos_forecast(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    repo: str,
    context_hours: int,
    step_horizon: int,
    diurnal_blend_alpha: float,
    device: str,
    roll_actuals: bool,
    exog_cols: list[str] | None = None,
) -> pd.DataFrame:
    context_load = history["actual_kwh"].astype(float).to_numpy(dtype=np.float64)
    if len(context_load) == 0:
        raise ValueError("Chronos requires at least one historical value")

    step = max(1, int(step_horizon))
    exog_cols = list(exog_cols or [])
    pipeline = load_chronos_model(repo, device)
    rolling_load = context_load.copy()
    rolling_exog = build_exog_matrix(history, exog_cols)
    diurnal_by_hour = build_diurnal_profile(history)

    calibration: dict[str, Any] = {
        "enabled": False,
        "bias_mean": 0.0,
        "bias_max_abs": 0.0,
        "metrics": None,
    }
    bias_vec = np.zeros(step, dtype=np.float64)
    if not validation.empty:
        val_horizon = min(step, len(validation))
        val_exog = align_exog(build_exog_matrix(validation.iloc[:val_horizon], exog_cols), val_horizon)
        val_raw, _, _ = run_chronos_prediction(
            pipeline,
            rolling_load,
            val_horizon,
            context_hours,
            exog_context=rolling_exog,
            exog_horizon=val_exog,
            exog_cols=exog_cols,
        )
        val_times = pd.to_datetime(validation["time"].iloc[:val_horizon])
        val_point = blend_with_diurnal(
            val_raw[:val_horizon],
            diurnal_for_times(val_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        val_actual = validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)[:val_horizon]
        bias_vec = align_vector(val_point - val_actual, step)
        val_cmp = pd.DataFrame({"actual_kwh": val_actual, "predicted_kwh": val_point})
        calibration = {
            "enabled": True,
            "bias_mean": round(float(np.nanmean(bias_vec)), 4),
            "bias_max_abs": round(float(np.nanmax(np.abs(bias_vec))), 4),
            "metrics": compute_forecast_metrics(val_cmp),
        }
        rolling_load = np.concatenate([rolling_load, validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)])
        rolling_exog = np.vstack([rolling_exog, build_exog_matrix(validation, exog_cols)])

    rows: list[pd.DataFrame] = []
    remaining = horizon_hours
    offset = 0
    while remaining > 0:
        chunk_horizon = min(step, remaining)
        chunk_start = forecast_start + pd.Timedelta(hours=offset)
        chunk_times = pd.date_range(chunk_start, periods=chunk_horizon, freq="h")
        chunk_frame = (
            full_frame.set_index("time")
            .reindex(chunk_times)
            .rename_axis("time")
            .reset_index()
        )
        chunk_exog = align_exog(build_exog_matrix(chunk_frame, exog_cols), chunk_horizon)
        raw, raw_q10, raw_q90 = run_chronos_prediction(
            pipeline,
            rolling_load,
            chunk_horizon,
            context_hours,
            exog_context=rolling_exog,
            exog_horizon=chunk_exog,
            exog_cols=exog_cols,
        )
        bias = align_vector(bias_vec, chunk_horizon)
        bias_corrected = np.clip(raw[:chunk_horizon] - bias, 0, None)
        point = blend_with_diurnal(
            bias_corrected,
            diurnal_for_times(chunk_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        q10, q90 = rebuild_quantile_interval(point, raw_q10[:chunk_horizon], raw_q90[:chunk_horizon])
        rows.append(
            pd.DataFrame(
                {
                    "time": chunk_times,
                    "raw_predicted_kwh": raw[:chunk_horizon],
                    "bias_corrected_kwh": bias_corrected,
                    "predicted_kwh": point,
                    "q10_kwh": q10,
                    "q50_kwh": point,
                    "q90_kwh": q90,
                }
            )
        )

        actual_chunk = chunk_frame.get("actual_kwh")
        if roll_actuals and actual_chunk is not None and actual_chunk.notna().all():
            roll_values = actual_chunk.astype(float).to_numpy(dtype=np.float64)
        else:
            roll_values = point
        rolling_load = np.concatenate([rolling_load, roll_values])
        rolling_exog = np.vstack([rolling_exog, chunk_exog])
        remaining -= chunk_horizon
        offset += chunk_horizon

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time", "predicted_kwh"])
    result.attrs["calibration"] = calibration
    result.attrs["chronos_covariates"] = exog_cols
    return result


def lstm_forecast(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    context_hours: int,
    step_horizon: int,
    exog_cols: list[str],
    hidden_size: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    diurnal_blend_alpha: float,
    device: str,
    roll_actuals: bool,
    seed: int,
) -> pd.DataFrame:
    context_load = history["actual_kwh"].astype(float).to_numpy(dtype=np.float64)
    if len(context_load) < 2:
        raise ValueError("LSTM requires at least two historical values")

    step = max(1, int(step_horizon))
    bundle = train_lstm_model(
        history,
        context_hours=context_hours,
        exog_cols=exog_cols,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    rolling_load = context_load.copy()
    rolling_exog = build_lstm_exog_matrix(history, exog_cols)
    diurnal_by_hour = build_diurnal_profile(history)

    calibration: dict[str, Any] = {
        "enabled": False,
        "bias_mean": 0.0,
        "bias_max_abs": 0.0,
        "interval_half_width": None,
        "metrics": None,
    }
    bias_vec = np.zeros(step, dtype=np.float64)
    interval_half_width = np.nan
    if not validation.empty:
        val_horizon = min(step, len(validation))
        val_exog = build_lstm_exog_matrix(validation.iloc[:val_horizon], exog_cols)
        val_raw = run_lstm_prediction(bundle, rolling_load, rolling_exog, val_exog, val_horizon)
        val_times = pd.to_datetime(validation["time"].iloc[:val_horizon])
        val_point = blend_with_diurnal(
            val_raw[:val_horizon],
            diurnal_for_times(val_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        val_actual = validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)[:val_horizon]
        residual = val_point - val_actual
        bias_vec = align_vector(residual, step)
        interval_half_width = estimate_interval_half_width(residual)
        val_cmp = pd.DataFrame({"actual_kwh": val_actual, "predicted_kwh": val_point})
        calibration = {
            "enabled": True,
            "bias_mean": round(float(np.nanmean(bias_vec)), 4),
            "bias_max_abs": round(float(np.nanmax(np.abs(bias_vec))), 4),
            "interval_half_width": finite_round(interval_half_width, 4),
            "metrics": compute_forecast_metrics(val_cmp),
        }
        rolling_load = np.concatenate([rolling_load, validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)])
        rolling_exog = np.vstack([rolling_exog, build_lstm_exog_matrix(validation, exog_cols)])

    rows: list[pd.DataFrame] = []
    remaining = horizon_hours
    offset = 0
    while remaining > 0:
        chunk_horizon = min(step, remaining)
        chunk_start = forecast_start + pd.Timedelta(hours=offset)
        chunk_times = pd.date_range(chunk_start, periods=chunk_horizon, freq="h")
        chunk_frame = (
            full_frame.set_index("time")
            .reindex(chunk_times)
            .rename_axis("time")
            .reset_index()
        )
        chunk_exog = build_lstm_exog_matrix(chunk_frame, exog_cols)
        raw = run_lstm_prediction(bundle, rolling_load, rolling_exog, chunk_exog, chunk_horizon)
        bias = align_vector(bias_vec, chunk_horizon)
        bias_corrected = np.clip(raw[:chunk_horizon] - bias, 0, None)
        point = blend_with_diurnal(
            bias_corrected,
            diurnal_for_times(chunk_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        q10, q90 = deterministic_interval(point, interval_half_width)

        rows.append(
            pd.DataFrame(
                {
                    "time": chunk_times,
                    "raw_predicted_kwh": raw[:chunk_horizon],
                    "bias_corrected_kwh": bias_corrected,
                    "predicted_kwh": point,
                    "q10_kwh": q10,
                    "q50_kwh": point,
                    "q90_kwh": q90,
                }
            )
        )

        actual_chunk = chunk_frame.get("actual_kwh")
        if roll_actuals and actual_chunk is not None and actual_chunk.notna().all():
            roll_values = actual_chunk.astype(float).to_numpy(dtype=np.float64)
        else:
            roll_values = point
        rolling_load = np.concatenate([rolling_load, roll_values])
        rolling_exog = np.vstack([rolling_exog, chunk_exog])
        remaining -= chunk_horizon
        offset += chunk_horizon

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time", "predicted_kwh"])
    result.attrs["calibration"] = calibration
    return result


def load_timesfm_model(repo: str, context_hours: int, max_horizon: int):
    key = (repo, int(context_hours), int(max_horizon))
    if key in _TIMESFM_MODEL_CACHE:
        return _TIMESFM_MODEL_CACHE[key]

    try:
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timesfm is required for forecast_model: timesfm. "
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
            "chronos-forecasting with Chronos2Pipeline is required for forecast_model: chronos. "
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
        raise RuntimeError("torch is required for forecast_model: chronos.") from exc

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
        raise RuntimeError("torch is required for forecast_model: lstm.") from exc

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
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0 or len(fc) == 0:
        return np.clip(fc, 0, None)
    fc_mean = float(np.nanmean(fc))
    diurnal_mean = float(np.nanmean(diurnal))
    scaled = diurnal * (fc_mean / diurnal_mean) if abs(diurnal_mean) > 1e-9 else diurnal
    return np.clip((1.0 - alpha) * fc + alpha * scaled, 0, None)


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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_arr = np.asarray(point, dtype=np.float64)
    zero_adjustment = np.zeros(len(point_arr), dtype=np.float64)
    response = estimate_energy_price_response(training_frame)
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


def ar_forecast(
    history: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    validation: pd.DataFrame | None = None,
    full_frame: pd.DataFrame | None = None,
    diurnal_blend_alpha: float = 0.0,
) -> pd.DataFrame:
    hist = history.set_index("time")["actual_kwh"].astype(float).sort_index()
    target_index = pd.date_range(forecast_start, periods=horizon_hours, freq="h")
    hourly_mean = hist.groupby(hist.index.hour).mean()
    autoregressive_values = hist.reindex(target_index - pd.Timedelta(days=7)).to_numpy()
    fallback = np.array([hourly_mean.loc[t.hour] for t in target_index], dtype=float)
    predicted = np.where(np.isnan(autoregressive_values), fallback, 0.72 * autoregressive_values + 0.28 * fallback)

    last_24 = hist.tail(24).sum()
    mean_daily = hist.resample("D").sum().mean()
    trend_scale = 1.0 if mean_daily == 0 or np.isnan(mean_daily) else np.clip(last_24 / mean_daily, 0.75, 1.25)
    predicted = np.maximum(predicted * trend_scale, 0.0)
    predicted = blend_with_diurnal(
        predicted,
        diurnal_for_times(target_index, build_diurnal_profile(history)),
        diurnal_blend_alpha,
    )
    horizon_frame = None
    if full_frame is not None and "time" in full_frame:
        horizon_frame = (
            full_frame.set_index("time")
            .reindex(target_index)
            .rename_axis("time")
            .reset_index()
        )
    training_parts = [history]
    if validation is not None and not validation.empty:
        training_parts.append(validation)
    training_frame = pd.concat(training_parts, ignore_index=True)
    adjusted, energy_price_adjustment, adjustment_meta = apply_energy_price_response_adjustment(
        predicted,
        horizon_frame,
        training_frame,
    )
    result = pd.DataFrame(
        {
            "time": target_index,
            "base_predicted_kwh": predicted,
            "energy_price_adjustment_kwh": energy_price_adjustment,
            "predicted_kwh": adjusted,
        }
    )
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


def summarize_weather(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float | int]:
    frame = weather[(weather["time"] >= start) & (weather["time"] <= end)]
    if frame.empty:
        return {"mean_temp_c": 0.0, "mean_humidity": 0.0, "rain_hours": 0, "total_rain": 0.0}
    return {
        "mean_temp_c": round(float(frame["T"].mean()), 2),
        "mean_humidity": round(float(frame["U"].mean()), 2),
        "rain_hours": int((frame["nRAIN"] > 0).sum()),
        "total_rain": round(float(frame["nRAIN"].sum()), 2),
    }


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
