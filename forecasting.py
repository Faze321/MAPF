from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import hashlib
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
    model_reference: dict[str, Any] | None = None,
    raw_prediction_cache: dict | None = None,
) -> ForecastResult:
    for name, value, minimum in (("history_days", history_days, 1), ("validation_days", validation_days, 0), ("horizon_days", horizon_days, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    zone_id = str(zone_id)
    normalized_model = normalize_forecast_model_name(backend)
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
    fitted_state: dict[str, Any] | None = None,
    model_reference: dict[str, Any] | None = None,
    raw_prediction_cache: dict | None = None,
) -> pd.DataFrame:
    """Load frozen weights, calibrate daily forecasts, then predict the full horizon."""
    parameters = {name: value for name, value in locals().copy().items()
                  if name.startswith(("timesfm_", "chronos_"))}
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

    saved_reference = (fitted_state or {}).get("model_reference")
    if saved_reference is not None:
        if model_reference is not None and model_reference != saved_reference:
            raise ValueError("Reusable calibration cannot be paired with a different model reference")
        model_reference = saved_reference
    if backend in {"lstm", "xgboost"} and model_reference is not None:
        dimensions = model_reference.get("parameters", {})
        if dimensions.get("input_hours") != history_hours or dimensions.get("output_hours") != horizon_hours:
            raise ValueError("Offline model input/output windows do not match history_days/horizon_days; prepare the matching model")
    predict_raw, model_state = prepare_forecast_backend(
        backend, history, parameters, max(24, int(horizon_hours)), fitted_state=fitted_state,
        model_reference=model_reference,
    )
    if model_reference is not None and raw_prediction_cache is not None:
        predict_raw = cache_raw_predictions(predict_raw, model_reference, raw_prediction_cache)
    if fitted_state is None:
        validation_rows: list[dict[str, Any]] = []
        residual_days = []
        for day, offset in enumerate(range(0, len(validation), 24)):
            day_frame = validation.iloc[offset:offset + 24].copy()
            context = pd.concat([history, validation.iloc[:offset]], ignore_index=True).tail(history_hours)
            day_covariates = day_frame.drop(columns=["actual_kwh", "load"], errors="ignore")
            if backend in {"lstm", "xgboost"} and model_reference is not None:
                # A direct model always receives and returns its entire trained
                # horizon. Daily bias uses only its first 24 offsets. Future
                # covariates may cross T, but no future actual load is included.
                known = pd.concat([history, validation, future_source], ignore_index=True)
                known = known.drop(columns=["actual_kwh", "load"], errors="ignore").drop_duplicates("time", keep="last")
                day_times = pd.date_range(day_frame["time"].iloc[0], periods=horizon_hours, freq="h")
                day_covariates = known.set_index("time").reindex(day_times).rename_axis("time").reset_index()
            raw_frame = predict_raw(context, day_covariates).iloc[:24].copy()
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


def checked_hourly_history(frame: pd.DataFrame, label: str, *, allow_empty: bool = False) -> pd.DataFrame:
    if frame.empty and allow_empty:
        return frame.copy()
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


def checked_raw_prediction(frame: pd.DataFrame, times) -> np.ndarray:
    if len(frame) != len(times) or not pd.DatetimeIndex(frame["time"]).as_unit("ns").equals(pd.DatetimeIndex(times).as_unit("ns")):
        raise ValueError("Backend prediction timestamps do not match the requested horizon")
    raw = frame["raw_predicted_kwh"].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Backend returned non-finite load predictions")
    return raw


def prepare_forecast_backend(
    backend: str, history: pd.DataFrame, parameters: dict[str, Any], max_horizon: int,
    *, fitted_state: dict[str, Any] | None = None,
    model_reference: dict[str, Any] | None = None,
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
    if backend in {"lstm", "xgboost"}:
        from zone_training import load_cached_predictor

        reference = model_reference or state.get("model_reference")
        if not isinstance(reference, dict) or reference.get("backend") != backend:
            raise ValueError(f"{backend} requires a prepared per-zone model_reference; run the train stage first")
        return load_cached_predictor(reference), {
            "model_reference": deepcopy(reference), "model_id": reference["model_id"],
        }
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
        frame["temp_price_idx"] = pd.to_numeric(frame["T"], errors="coerce") * pd.to_numeric(frame["e_price"], errors="coerce")
    else:
        frame["temp_price_idx"] = np.nan
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
