from __future__ import annotations

import json
from typing import Any


SYSTEM_MESSAGE = (
    "You are a concise EV charging demand analyst. Use only the provided UrbanEV context. Return valid JSON only, with no markdown."
)


def horizon_label(context: dict[str, Any]) -> str:
    days = context.get("forecast_horizon_days")
    try:
        value = int(days)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if value == 1 else f"{value} days"


def grid_prompt(context: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    return (
        f"""Phase A / Grid Analyst.

        Task: predict the next {horizon} of charging load for this zone. Use the baseline forecast as the numerical anchor unless the context strongly justifies a small adjustment.

        Return JSON with keys: forecast_total_kwh, forecast_peak_kwh, predicted_change_pct, grid_stress_level, forecast_summary.
        grid_stress_level must be exactly one of: Low, Medium, High, Extreme High.

        Context:{json.dumps(context, ensure_ascii=False)}"""
    )


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    return (
        f"""Phase B / Behavioural Agent.

        Task: explain why demand looks this way over the next {horizon} using POI mix, weather, temporal markers, hourly forecast data, 3-hour pricing windows, and the load shape. Also estimate a positive elasticity_factor for price response: higher means users are more willing to shift charging after a price increase. Rain and high occupancy should lower elasticity. Keep it specific and short.

        Return JSON with keys: agent_reasoning, demand_drivers, elasticity_factor, confidence.

        Context:{json.dumps(context, ensure_ascii=False)}
        Grid report:{json.dumps(grid_report, ensure_ascii=False)}"""
    )


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    horizon = horizon_label(context)
    economist_context = compact_economist_context(context)
    return (
        f"""Phase C / Market Economist.
        
        Task: prescribe service-fee shifts for the next {horizon} using only forecast-derived information, prior agent conclusions, category, service price, energy price, each 3-hour window's predicted load_stress_level, and the behavioural elasticity estimate. Residential users are more price-sensitive; CBD and hub users are less price-sensitive. Avoid extreme changes.

        Nash-equilibrium intent: each window should move toward a strategy where the expected load after price response is grid safe, users remain within tolerance, and the price recommendation is stable. The code will run the final mathematical equilibrium check after your JSON response.

        Do not use actual future load, actual future stress, forecast error, stress correctness, or any evaluation/ground-truth fields. Those fields are intentionally not provided to you.

        Return JSON with keys: suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.
        price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, action_label, price_rationale.

        Context:\n{json.dumps(economist_context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}"""
    )


def repair_economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    previous_report: dict[str, Any],
    validation_errors: list[str],
) -> str:
    economist_context = compact_economist_context(context)
    expected_count = len(economist_context.get("pricing_windows_3h", []))
    return (
        f"""Repair the Market Economist JSON response.

        The previous response failed schema validation. Return valid JSON only, with no markdown and no explanatory text.
        Validation errors: {json.dumps(validation_errors, ensure_ascii=False)}

        Required top-level keys:
        suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.

        price_change_windows_3h must contain exactly {expected_count} items, one for each context.pricing_windows_3h item in the same order.
        Each item must contain: window_start, window_end, suggested_price_shift_pct, action_label, price_rationale.
        Use the same window_start and window_end values from the context.

        Context:\n{json.dumps(economist_context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}
        Previous invalid response:\n{json.dumps(previous_report, ensure_ascii=False)}"""
    )


def compact_economist_context(context: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = [
        "category",
        "zone_id",
        "forecast_start",
        "forecast_end",
        "forecast_horizon_days",
        "forecast_horizon_hours",
        "forecast_total_kwh",
        "forecast_peak_kwh",
        "predicted_change_pct",
        "grid_stress_level",
        "capacity_kw_proxy",
        "grid_stress_q50_kwh",
        "grid_stress_q80_kwh",
        "grid_stress_q95_kwh",
    ]
    compact = {key: context.get(key) for key in scalar_keys if key in context}
    if "hourly_averages" in context:
        compact["hourly_averages"] = forecast_only_hourly_averages(context.get("hourly_averages"))
    if "pricing_windows_3h" in context:
        compact["pricing_windows_3h"] = forecast_only_pricing_windows(context.get("pricing_windows_3h"))
    return compact


def forecast_only_hourly_averages(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    blocked = {
        "mean_actual_kwh",
        "mean_abs_pct_error",
    }
    return {key: item for key, item in value.items() if key not in blocked}


def forecast_only_pricing_windows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "window_start",
        "window_end",
        "hours",
        "mean_predicted_kwh",
        "sum_predicted_kwh",
        "peak_predicted_kwh",
        "mean_service_price",
        "mean_energy_price",
        "mean_occupancy",
        "mean_temp_c",
        "mean_humidity",
        "total_rain",
        "mean_abs_pct_error",
        "load_stress_level",
        "grid_stress_level",
        "stress_load_3h_kwh",
        "stress_source_file",
        "stress_window_hours",
        "load_3h_q50_kwh",
        "load_3h_q80_kwh",
        "load_3h_q95_kwh",
    }
    blocked = {"mean_abs_pct_error"}
    sanitized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sanitized.append(
            {
                key: item.get(key)
                for key in allowed
                if key in item and key not in blocked
            }
        )
    return sanitized
