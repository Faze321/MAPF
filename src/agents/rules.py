from __future__ import annotations

from typing import Any

from src.agents.validation import has_text, model_response_failed, validate_economist_report
from src.control.numbers import as_float, optional_float
from src.control.policy import (
    EXTREMELY_HIGH_STRESS,
    EXTREMELY_LOW_STRESS,
    HIGH_STRESS,
    LOW_STRESS,
    MEDIUM_STRESS,
    STRESS_LEVELS,
    classify_load_percentage,
    load_range_position_pct,
    medium_load_bounds,
    target_medium_load,
)


GRID_STRESS_LEVELS = STRESS_LEVELS
MIN_ELASTICITY = 0.05
MAX_ELASTICITY = 0.7
GRID_STRESS_LEVEL_BY_KEY = {level.lower(): level for level in GRID_STRESS_LEVELS}


GRID_STRESS_LEVEL_BY_KEY.update(
    {
        "moderate": "Medium",
        "critical": EXTREMELY_HIGH_STRESS,
        "critica": EXTREMELY_HIGH_STRESS,
        "extrame high": EXTREMELY_HIGH_STRESS,
        "extreme high": EXTREMELY_HIGH_STRESS,
        "extreme_high": EXTREMELY_HIGH_STRESS,
        "extremely_high": EXTREMELY_HIGH_STRESS,
    }
)


def heuristic_grid(context: dict[str, Any]) -> dict[str, Any]:
    windows = context.get("pricing_windows_3h") or []
    return {
        "forecast_total_kwh": context["forecast_total_kwh"],
        "forecast_peak_kwh": context["forecast_peak_kwh"],
        "predicted_change_pct": context["predicted_change_pct"],
        "grid_stress_level": normalize_grid_stress_level(context.get("grid_stress_level"), "Low"),
        "forecast_summary": (
            f"{context['category']} zone forecast is anchored to the prior-week hourly "
            f"shape with a {context['predicted_change_pct']:+.1f}% change versus the "
            "recent comparable window."
        ),
        "reasoning_summary": (
            "The forecaster is the numerical source of truth; pricing is required "
            "for each window outside the Medium band."
        ),
        "adjustment_needed": any(
            normalize_grid_stress_level(window.get("load_stress_level"), LOW_STRESS)
            != MEDIUM_STRESS
            for window in windows
        ),
        "confidence": "medium",
        "window_assessments": [
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "predicted_load_kwh": window.get("sum_predicted_kwh"),
                "load_range_position_pct": window.get("load_range_position_pct"),
                "grid_stress_level": window.get("load_stress_level"),
                "adjustment_needed": window.get("load_stress_level") != MEDIUM_STRESS,
                "reasoning_summary": "Assessment uses the forecast-derived load position.",
            }
            for window in windows
        ],
    }


def heuristic_behavior(context: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    weather = context["weather"]
    shape = context["hourly_shape"]
    windows = context.get("pricing_windows_3h") or []
    horizon_days = context.get("forecast_horizon_days", "configured")
    forecast_origin = str(context.get("forecast_origin") or "the forecast start")
    forecast_end = str(context.get("forecast_end") or "the forecast end")
    change = as_float(context.get("predicted_change_pct"), 0)
    drivers = []
    high_windows = [
        window
        for window in windows
        if normalize_grid_stress_level(window.get("load_stress_level") or window.get("grid_stress_level"), "Low")
        in {HIGH_STRESS, EXTREMELY_HIGH_STRESS}
    ]
    peak_window = max(windows, key=lambda window: as_float(window.get("sum_predicted_kwh"), 0), default=None)
    if high_windows:
        drivers.append(f"{len(high_windows)} high-stress 3-hour windows")
    if peak_window:
        drivers.append(
            "peak window "
            f"{peak_window.get('window_start')} to {peak_window.get('window_end')} "
            f"at {as_float(peak_window.get('sum_predicted_kwh'), 0):.1f} kWh"
        )
    if abs(change) >= 5:
        drivers.append(f"{change:+.1f}% load change versus comparable history")
    else:
        drivers.append("stable total load versus comparable history")
    if (weather.get("rain_hours") or 0) > 0:
        drivers.append(f"{weather['rain_hours']} rainy hours")
    if shape["night_20_6"] >= shape["morning_7_10"]:
        drivers.append("night plateau")
    if shape["evening_17_22"] >= shape["morning_7_10"]:
        drivers.append("evening lift")
    profile = context.get("profile") or {}
    poi_total = profile.get("poi_total", 0)
    return {
        "agent_reasoning": (
            f"For the {horizon_days}-day window from {forecast_origin} to {forecast_end}, "
            f"{category} demand is best explained by {', '.join(drivers)} and the "
            f"local POI mix of {poi_total} assigned POIs."
        ),
        "reasoning_summary": (
            f"Demand is explained by {', '.join(drivers)} together with the local "
            "static and temporal context."
        ),
        "demand_drivers": drivers,
        "elasticity_factor": estimate_elasticity_factor(context, {}, peak_window or {}),
        "confidence": "medium",
    }


def heuristic_economist(context: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    stress = grid["grid_stress_level"]
    change = float(grid["predicted_change_pct"])
    shift, label = base_price_shift(stress, category, change)
    window_shifts = build_heuristic_price_windows(context, grid)
    return {
        "suggested_price_shift_pct": shift,
        "action_label": label,
        "reasoning_summary": (
            f"The schedule responds to {stress} forecast stress and the estimated "
            f"price sensitivity of the {category} zone."
        ),
        "price_rationale": (
            f"{stress} stress with {change:+.1f}% expected load change; "
            f"category elasticity proxy is {category}."
        ),
        "price_change_windows_3h": window_shifts,
    }


def merge_grid_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_grid(context)
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    for key in ("forecast_total_kwh", "forecast_peak_kwh", "predicted_change_pct", "grid_stress_level", "adjustment_needed"):
        merged[key] = fallback[key]
    notes = {(w.get("window_start"), w.get("window_end")): w for w in merged.get("window_assessments") or []
             if isinstance(w, dict)}
    # Preserve the model's reasoning, but window identity, numbers and bands belong to the forecaster.
    merged["window_assessments"] = [
        {**w, "reasoning_summary": notes.get((w["window_start"], w["window_end"]), {}).get("reasoning_summary")
                                   or w["reasoning_summary"]}
        for w in fallback["window_assessments"]
    ]
    return merged


def merge_behavior_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_behavior(context)
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


def merge_economist_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_economist(context, heuristic_grid(context))
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    if not isinstance(report.get("price_change_windows_3h"), list):
        merged["price_change_windows_3h"] = []
    merged["quote_errors"] = validate_economist_report(report, context.get("pricing_windows_3h"))
    merged.pop("action_label", None)
    if not has_text(report.get("price_rationale")):
        merged["price_rationale"] = model_response_failed("missing price_rationale")
    return merged


def base_price_shift(stress: str, category: str, change: float) -> tuple[int, str]:
    if stress == EXTREMELY_HIGH_STRESS and category == "Residential":
        return 15, "Raise energy price for peak deflection"
    if stress == EXTREMELY_HIGH_STRESS:
        return 15, "Raise energy price for congestion"
    if stress == HIGH_STRESS:
        return 8, "Raise energy price for high load"
    if stress == EXTREMELY_LOW_STRESS:
        return -15, "Reduce energy price for extremely low load"
    if stress == LOW_STRESS:
        return -8, "Reduce energy price for utilization"
    return 0, "Hold energy price"


def build_heuristic_price_windows(
    context: dict[str, Any],
    grid: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = context.get("pricing_windows_3h") or []
    averages = context.get("hourly_averages") or {}
    mean_load = as_float(averages.get("mean_predicted_kwh"), 0)
    mean_energy_price = as_float(averages.get("mean_energy_price"), 0)
    stress = normalize_grid_stress_level(grid.get("grid_stress_level"), context.get("grid_stress_level"))
    category = str(context.get("category") or "")
    change = as_float(grid.get("predicted_change_pct"), context.get("predicted_change_pct", 0))
    results = []
    for window in windows:
        window_stress = normalize_grid_stress_level(
            window.get("load_stress_level") or window.get("grid_stress_level"),
            stress,
        )
        window_load = as_float(window.get("mean_predicted_kwh"), mean_load)
        window_energy_price = as_float(window.get("mean_energy_price"), mean_energy_price)
        elasticity = estimate_elasticity_factor(context, {}, window)
        shift = round(price_shift_to_medium_band(context, window, elasticity), 2)
        if shift > 0:
            label = "Raise energy price"
        elif shift < 0:
            label = "Reduce energy price"
        else:
            label = "Hold window energy price"
        results.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                "mean_predicted_kwh": window.get("mean_predicted_kwh"),
                "mean_service_price": window.get("mean_service_price"),
                "mean_energy_price": window.get("mean_energy_price"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "historical_min_load_3h_kwh": window.get("historical_min_load_3h_kwh"),
                "historical_max_load_3h_kwh": window.get("historical_max_load_3h_kwh"),
                "historical_load_range_3h_kwh": window.get("historical_load_range_3h_kwh"),
                "load_range_position_pct": window.get("load_range_position_pct"),
                "extremely_low_low_threshold_pct": window.get("extremely_low_low_threshold_pct"),
                "low_medium_threshold_pct": window.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": window.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": window.get(
                    "high_extremely_high_threshold_pct"
                ),
                "load_3h_extremely_low_low_threshold_kwh": window.get(
                    "load_3h_extremely_low_low_threshold_kwh"
                ),
                "load_3h_low_medium_threshold_kwh": window.get(
                    "load_3h_low_medium_threshold_kwh"
                ),
                "load_3h_medium_high_threshold_kwh": window.get(
                    "load_3h_medium_high_threshold_kwh"
                ),
                "load_3h_high_extremely_high_threshold_kwh": window.get(
                    "load_3h_high_extremely_high_threshold_kwh"
                ),
                "zone_mean_energy_price": window.get("zone_mean_energy_price"),
                "suggested_price_shift_pct": shift,
                "action_label": label,
                "price_rationale": (
                    f"{window_stress} 3-hour load at "
                    f"{as_float(window.get('load_range_position_pct'), 0):.2f}% through the "
                    f"historical 3-hour load range; "
                    f"price shift targets the Medium band while keeping the resulting "
                    f"energy price non-negative from the {window_energy_price:.4f} baseline."
                ),
            }
        )
    return results


def window_predicted_load(window: dict[str, Any]) -> float:
    for key in ("price_conditioned_baseline_load_kwh", "sum_predicted_kwh"):
        load = optional_float(window.get(key))
        if load is not None:
            return max(0.0, load)
    mean_load = optional_float(window.get("mean_predicted_kwh")) or 0.0
    hours = optional_float(window.get("hours")) or 1.0
    return max(0.0, mean_load * hours)


def window_predicted_load_source(window: dict[str, Any]) -> str:
    for key, source in (
        ("price_conditioned_baseline_load_kwh", "price_conditioned_forecast_sum_predicted_kwh"),
        ("sum_predicted_kwh", "forecast_sum_predicted_kwh"),
    ):
        if optional_float(window.get(key)) is not None:
            return source
    return "forecast_mean_predicted_kwh_times_hours"


def historical_load_bounds_kwh(
    context: dict[str, Any],
    window: dict[str, Any],
) -> tuple[float, float, str]:
    minimum = first_nonnegative_float(
        window.get("historical_min_load_3h_kwh"),
        context.get("historical_min_load_3h_kwh"),
    )
    maximum = first_positive_float(
        window.get("historical_max_load_3h_kwh"),
        context.get("historical_max_load_3h_kwh"),
    )
    if minimum is not None and maximum is not None and maximum > minimum:
        return minimum, maximum, "zone_historical_3h_load_min_max_range"

    baseline = max(window_predicted_load(window), 0.0)
    return 0.0, max(baseline, 1.0), "fallback_zero_to_baseline_predicted_load"


def price_shift_to_medium_band(
    context: dict[str, Any],
    window: dict[str, Any],
    elasticity: float,
) -> float:
    baseline_load = window_predicted_load(window)
    historical_min_load, historical_max_load, _ = historical_load_bounds_kwh(
        context, window
    )
    if baseline_load <= 0 or historical_max_load <= historical_min_load or elasticity <= 0:
        return max(-100.0, as_float(window.get("suggested_price_shift_pct"), 0))

    load_position_pct = load_range_position_pct(
        baseline_load, historical_min_load, historical_max_load
    )
    stress = classify_load_percentage(load_position_pct)
    target_load = target_medium_load(
        baseline_load, historical_min_load, historical_max_load
    )
    raw_shift = as_float(window.get("suggested_price_shift_pct"), 0)

    if stress == MEDIUM_STRESS:
        candidate = clamp(
            raw_shift,
            -3.0,
            3.0,
        )
        expected_load = expected_load_after_price(baseline_load, candidate, elasticity)
        lower_load, upper_load = medium_load_bounds(
            historical_min_load, historical_max_load
        )
        if not lower_load <= expected_load < upper_load:
            candidate = 0.0
    else:
        candidate = ((1.0 - (target_load / baseline_load)) / elasticity) * 100.0

    return max(-100.0, candidate)


def estimate_elasticity_factor(
    context: dict[str, Any],
    behavior: dict[str, Any],
    window: dict[str, Any],
) -> float:
    window_value = optional_float(window.get("elasticity_factor"))
    if window_value is not None:
        return clamp(abs(window_value), MIN_ELASTICITY, MAX_ELASTICITY)

    reported = optional_float(behavior.get("elasticity_factor"))
    if reported is not None:
        return clamp(abs(reported), MIN_ELASTICITY, MAX_ELASTICITY)

    category = str(context.get("category") or "").lower()
    if "residential" in category:
        base = 0.45
    elif "commercial" in category or "mall" in category:
        base = 0.32
    elif "cbd" in category or "office" in category:
        base = 0.24
    elif "transport" in category or "hub" in category:
        base = 0.2
    elif "industrial" in category:
        base = 0.18
    else:
        base = 0.28

    occupancy = normalized_rate(window.get("mean_occupancy"))
    rain = optional_float(window.get("total_rain"))
    if rain is None:
        weather = context.get("weather") if isinstance(context.get("weather"), dict) else {}
        rain = optional_float(weather.get("rain_hours")) or 0.0
    occupancy_factor = 1.0 - 0.25 * occupancy
    rain_factor = 0.85 if rain > 0 else 1.0
    return clamp(base * occupancy_factor * rain_factor, MIN_ELASTICITY, MAX_ELASTICITY)


def expected_load_after_price(baseline_load: float, price_shift_pct: float, elasticity: float) -> float:
    response = elasticity * (price_shift_pct / 100.0)
    return max(0.0, baseline_load * (1.0 - response))


def normalized_rate(value: Any) -> float:
    number = optional_float(value)
    if number is None:
        return 0.0
    if number > 1.0:
        number /= 100.0
    return clamp(number, 0.0, 1.0)


def first_positive_float(*values: Any) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None and number > 0:
            return number
    return None


def first_nonnegative_float(*values: Any) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None and number >= 0:
            return number
    return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_grid_stress_level(value: Any, fallback: Any = "Low") -> str:
    key = str(value or "").strip().lower()
    if key in GRID_STRESS_LEVEL_BY_KEY:
        return GRID_STRESS_LEVEL_BY_KEY[key]
    fallback_key = str(fallback or "").strip().lower()
    return GRID_STRESS_LEVEL_BY_KEY.get(fallback_key, "Low")
