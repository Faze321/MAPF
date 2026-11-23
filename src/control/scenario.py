from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.run import VALIDATION_BIAS_DAYS
from src.config.training import FORECAST_BACKENDS
from src.config.values import normalize_forecast_model_name
from src.control.pricing import energy_price_with_predicted_windows
from src.control.windows import build_pricing_windows_3h, load_stress_thresholds
from src.forecast.artifact import NativeForecasterArtifact
from src.pipeline.context import build_agent_hourly_data


def apply_price_conditioned_baseline_forecasts(
    *,
    reports: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    pipeline_data,
    zone_load_thresholds: pd.DataFrame,
    scenario_artifacts: dict[str, NativeForecasterArtifact],
    forecast_origin: str | None,
    horizon_days: int,
    history_days: int,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    backend: str,
    diurnal_blend_alpha: float,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_exog_cols: list[str] | None,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_device: str,
) -> list[dict[str, Any]]:
    normalized_model = normalize_forecast_model_name(backend)
    if normalized_model not in FORECAST_BACKENDS:
        return [
            mark_price_conditioned_baseline_unavailable(
                report,
                f"unsupported_forecast_model_{normalized_model}",
            )
            for report in reports
        ]

    contexts_by_zone = {str(context.get("zone_id")): context for context in contexts}
    profiles = pipeline_data.profiles.copy()
    profiles["zone_id"] = profiles["zone_id"].astype(str)
    profiles_by_zone = profiles.set_index("zone_id", drop=False)
    thresholds_by_zone = zone_load_thresholds.copy()
    thresholds_by_zone["zone_id"] = thresholds_by_zone["zone_id"].astype(str)
    thresholds_by_zone = thresholds_by_zone.set_index("zone_id", drop=False)

    price_conditioned_energy_price = pipeline_data.energy_price.copy()
    for report in reports:
        report_zone = str(report.get("zone_id"))
        price_conditioned_energy_price = energy_price_with_predicted_windows(
            price_conditioned_energy_price,
            report_zone,
            report.get("price_change_windows_3h") or [],
        )

    updated_reports: list[dict[str, Any]] = []
    for report in reports:
        zone_id = str(report.get("zone_id"))
        context = contexts_by_zone.get(zone_id)
        if context is None or zone_id not in profiles_by_zone.index:
            updated_reports.append(
                mark_price_conditioned_baseline_unavailable(
                    report,
                    "missing_context_or_profile",
                )
            )
            continue

        artifact = scenario_artifacts.get(zone_id)
        if artifact is None:
            updated_reports.append(
                mark_price_conditioned_baseline_unavailable(
                    report,
                    "missing_reusable_forecaster_artifact",
                )
            )
            continue
        batch = artifact.predict(price_conditioned_energy_price)
        conditioned_hourly = batch.hourly.rename(
            columns={
                "timestamp": "time",
                "predicted_load": "predicted_kwh",
                "energy_price": "e_price",
            }
        )
        conditioned_hourly["hour"] = pd.to_datetime(conditioned_hourly["time"]).dt.hour
        conditioned_hourly["is_weekend"] = pd.to_datetime(conditioned_hourly["time"]).dt.dayofweek >= 5
        for source, column in (
            (pipeline_data.service_price, "s_price"),
            (pipeline_data.occupancy, "occupancy"),
        ):
            if zone_id in source:
                conditioned_hourly = conditioned_hourly.merge(
                    source[["time", zone_id]].rename(columns={zone_id: column}),
                    on="time",
                    how="left",
                )
        weather_columns = [
            column for column in ("T", "U", "nRAIN") if column in pipeline_data.weather
        ]
        if weather_columns:
            conditioned_hourly = conditioned_hourly.merge(
                pipeline_data.weather[["time", *weather_columns]],
                on="time",
                how="left",
            )
        thresholds = load_stress_thresholds(
            thresholds_by_zone,
            zone_id,
            profile=profiles_by_zone.loc[zone_id].to_dict(),
        )
        conditioned_windows = build_pricing_windows_3h(
            conditioned_hourly,
            stress_thresholds=thresholds,
        )
        updated_report = attach_price_conditioned_baselines(
            report,
            conditioned_windows,
            backend=normalized_model,
            forecast_metadata=batch.metadata,
        )
        updated_report["_conditioned_hourly"] = build_agent_hourly_data(conditioned_hourly)
        updated_reports.append(updated_report)
    return updated_reports


def attach_price_conditioned_baselines(
    report: dict[str, Any],
    conditioned_windows: list[dict[str, Any]],
    *,
    backend: str,
    forecast_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conditioned_by_window = {
        (str(window.get("window_start")), str(window.get("window_end"))): window
        for window in conditioned_windows
    }
    updated_windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if not isinstance(window, dict):
            continue
        updated = dict(window)
        key = (str(window.get("window_start")), str(window.get("window_end")))
        conditioned = conditioned_by_window.get(key)
        if conditioned is None:
            updated = {k: v for k, v in updated.items() if not k.startswith("price_conditioned_")}
            updated["price_conditioned_baseline_source"] = "missing_price_conditioned_forecast_window"
        else:
            updated["price_conditioned_baseline_load_kwh"] = conditioned.get("sum_predicted_kwh")
            updated["price_conditioned_mean_predicted_kwh"] = conditioned.get("mean_predicted_kwh")
            updated["price_conditioned_peak_predicted_kwh"] = conditioned.get("peak_predicted_kwh")
            updated["price_conditioned_load_stress_level"] = conditioned.get("load_stress_level")
            updated["price_conditioned_load_range_position_pct"] = conditioned.get(
                "load_range_position_pct"
            )
            updated["price_conditioned_energy_price"] = conditioned.get("mean_energy_price")
            if updated.get("price_valid"):
                updated["predicted_energy_price"] = updated["proposed_energy_price"] = conditioned.get("mean_energy_price")
            updated["price_conditioned_baseline_source"] = (
                f"{backend}_native_backend_reforecast_with_proposed_energy_price"
            )
        updated_windows.append(updated)

    updated_report = dict(report)
    updated_report["price_change_windows_3h"] = updated_windows
    updated_report["price_conditioned_forecast_metadata"] = dict(
        forecast_metadata or {}
    )
    return updated_report


def mark_price_conditioned_baseline_unavailable(
    report: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    updated = dict(report)
    updated.pop("_conditioned_hourly", None)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if isinstance(window, dict):
            item = {k: v for k, v in window.items() if not k.startswith("price_conditioned_")}
            item["price_conditioned_baseline_source"] = reason
            windows.append(item)
    updated["price_change_windows_3h"] = windows
    return updated
