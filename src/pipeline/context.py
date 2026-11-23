from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.run import VALIDATION_BIAS_DAYS
from src.config.training import TRAINABLE_BACKENDS
from src.control.windows import (
    apply_load_policy_stress,
    build_pricing_windows_3h,
    clean_context_value,
    load_stress_thresholds,
)
from src.forecast.engine import ForecastResult, forecast_zone
from src.pipeline.errors import PipelineStageError


DECISION_SUMMARY_KEYS = {
    "zone_id",
    "category",
    "history_start",
    "history_end",
    "validation_start",
    "validation_end",
    "forecast_origin",
    "forecast_end",
    "forecast_horizon_days",
    "forecast_horizon_hours",
    "backend",
    "diurnal_blend_alpha",
    "forecast_total_kwh",
    "forecast_peak_kwh",
    "predicted_change_pct",
    "capacity_kw_proxy",
    "peak_capacity_ratio",
    "grid_stress_level",
    "grid_stress_basis",
    "grid_stress_load_kwh",
    "grid_stress_load_range_position_pct",
    "grid_stress_source_file",
    "grid_stress_window_hours",
    "grid_stress_historical_windows",
    "historical_min_load_3h_kwh",
    "historical_max_load_3h_kwh",
    "historical_load_range_3h_kwh",
    "extremely_low_low_threshold_pct",
    "low_medium_threshold_pct",
    "medium_high_threshold_pct",
    "high_extremely_high_threshold_pct",
    "load_3h_extremely_low_low_threshold_kwh",
    "load_3h_low_medium_threshold_kwh",
    "load_3h_medium_high_threshold_kwh",
    "load_3h_high_extremely_high_threshold_kwh",
    "zone_mean_energy_price",
    "price",
    "weather",
    "occupancy",
    "profile",
    "previous_comparable_total_kwh",
    "daily_history_kwh",
    "daily_forecast_kwh",
    "hourly_shape",
}


def build_contexts(
    *,
    pipeline_data,
    selected_zones: pd.DataFrame,
    forecast_origin: str | None,
    horizon_days: int,
    history_days: int,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    backend: str,
    diurnal_blend_alpha: float,
    zone_load_thresholds: pd.DataFrame,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_exog_cols: list[str] | None,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_device: str,
    zone_model_references: dict[str, dict[str, Any]] | None = None,
    raw_prediction_cache: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, ForecastResult]]:
    start = pd.Timestamp(forecast_origin) if forecast_origin else None
    contexts = []
    forecast_results = {}
    profiles_by_zone = pipeline_data.profiles.set_index("zone_id", drop=False)
    stress_thresholds_by_zone = zone_load_thresholds.set_index("zone_id", drop=False)
    for row in selected_zones.to_dict(orient="records"):
        zone_id = str(row["zone_id"])
        profile = profiles_by_zone.loc[zone_id].to_dict()
        zone_stress_thresholds = load_stress_thresholds(
            stress_thresholds_by_zone,
            zone_id,
            profile=profile,
        )
        try:
            model_reference = (zone_model_references or {}).get(zone_id)
            if backend in TRAINABLE_BACKENDS and model_reference is None:
                raise ValueError(f"No usable offline {backend} weights for zone {zone_id}; see the backend training index")
            if backend not in TRAINABLE_BACKENDS and model_reference is not None:
                raise ValueError(f"Pretrained forecaster {backend} cannot use offline weights; select the LoRA backend")
            zone_weather = pipeline_data.weather
            canonical_dataset = getattr(pipeline_data, "canonical_dataset", None)
            if model_reference is not None and canonical_dataset is not None:
                from src.training.windows import canonical_zone_frame
                # Match training covariates, preserving this site's missing weather values.
                zone_weather = canonical_zone_frame(canonical_dataset, zone_id)[["time", "T", "U", "nRAIN"]]
            raw_result = forecast_zone(
                zone_id=zone_id,
                category=row["category"],
                load=pipeline_data.load,
                service_price=pipeline_data.service_price,
                energy_price=pipeline_data.energy_price,
                occupancy=pipeline_data.occupancy,
                weather=zone_weather,
                profile=profile,
                forecast_origin=start,
                horizon_days=horizon_days,
                history_days=history_days,
                validation_bias=validation_bias,
                validation_days=validation_days,
                backend=backend,
                diurnal_blend_alpha=diurnal_blend_alpha,
                model_reference=model_reference,
                raw_prediction_cache=raw_prediction_cache,
                timesfm_repo=timesfm_repo,
                timesfm_context_hours=timesfm_context_hours,
                timesfm_exog_cols=timesfm_exog_cols,
                chronos_repo=chronos_repo,
                chronos_context_hours=chronos_context_hours,
                chronos_device=chronos_device,
            )
        except Exception as exc:
            raise PipelineStageError(
                stage="forecast",
                zone_id=zone_id,
                original=exc,
            ) from exc
        pricing_windows_3h = build_pricing_windows_3h(
            raw_result.hourly,
            stress_thresholds=zone_stress_thresholds,
            include_evaluation=False,
        )
        summary = apply_load_policy_stress(raw_result.summary, zone_stress_thresholds, pricing_windows_3h)
        result = ForecastResult(
            hourly=raw_result.hourly,
            summary=summary,
            reusable_forecaster=raw_result.reusable_forecaster,
        )
        forecast_results[zone_id] = result
        hourly_forecast = build_agent_hourly_data(result.hourly)
        hourly_averages = build_hourly_averages(result.hourly)
        horizon_text = format_horizon_days(summary.get("forecast_horizon_days", horizon_days))
        context = {
            **decision_summary(summary),
            "selection_reason": row["selection_reason"],
            "hourly_averages": hourly_averages,
            "hourly_forecast": hourly_forecast,
            "pricing_windows_3h": pricing_windows_3h,
            "instructions": {
                "forecast_task": f"Predict next {horizon_text} of EV charging load.",
                "behavior_task": "Explain demand using POI, weather, and temporal markers.",
                "pricing_task": "Suggest energy-price shifts for each 3-hour pricing window.",
            },
        }
        contexts.append(context)
    return contexts, forecast_results


def build_agent_hourly_data(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "time",
        "predicted_kwh",
        "q10_kwh",
        "q50_kwh",
        "q90_kwh",
        "s_price",
        "e_price",
        "occupancy",
        "T",
        "U",
        "nRAIN",
        "hour",
        "is_weekend",
    ]
    frame = hourly[[col for col in columns if col in hourly.columns]].copy()
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return [{key: clean_context_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def build_hourly_averages(hourly: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "predicted_kwh": "mean_predicted_kwh",
        "s_price": "mean_service_price",
        "e_price": "mean_energy_price",
        "occupancy": "mean_occupancy",
        "T": "mean_temp_c",
        "U": "mean_humidity",
        "nRAIN": "mean_rain",
    }
    averages: dict[str, Any] = {}
    for source_col, output_col in columns.items():
        if source_col in hourly:
            value = pd.to_numeric(hourly[source_col], errors="coerce").mean(skipna=True)
            averages[output_col] = clean_context_value(value)
    if "predicted_kwh" in hourly:
        predicted = pd.to_numeric(hourly["predicted_kwh"], errors="coerce")
        averages["peak_predicted_kwh"] = clean_context_value(predicted.max(skipna=True))
        averages["total_predicted_kwh"] = clean_context_value(predicted.sum(skipna=True))
    return averages


def decision_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the allowlisted forecast information that may reach an agent."""

    return {
        key: value
        for key, value in summary.items()
        if key in DECISION_SUMMARY_KEYS
    }


def format_horizon_days(value: Any) -> str:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if days == 1 else f"{days} days"
