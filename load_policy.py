from __future__ import annotations

from typing import Any


LOW_STRESS = "Low"
MEDIUM_STRESS = "Medium"
HIGH_STRESS = "High"
EXTREMELY_HIGH_STRESS = "Extremely High"
LOAD_POLICY_ID = "historical_3h_min_max_range_35_80_90"

LOW_MAX_LOAD_PCT = 35.0
MEDIUM_MAX_LOAD_PCT = 80.0
HIGH_MAX_LOAD_PCT = 90.0
MEDIUM_LOWER_TARGET_PCT = 35.5
MEDIUM_UPPER_TARGET_PCT = 79.5


def load_range_position_pct(
    load_kwh: Any,
    historical_min_load_kwh: Any,
    historical_max_load_kwh: Any,
) -> float:
    load = _number(load_kwh)
    minimum = _number(historical_min_load_kwh)
    maximum = _number(historical_max_load_kwh)
    if load is None or minimum is None or maximum is None:
        return 0.0
    if maximum <= minimum:
        return 50.0
    return max(0.0, ((load - minimum) / (maximum - minimum)) * 100.0)


def classify_load_percentage(load_pct: Any) -> str:
    percentage = _number(load_pct) or 0.0
    if percentage < LOW_MAX_LOAD_PCT:
        return LOW_STRESS
    if percentage < MEDIUM_MAX_LOAD_PCT:
        return MEDIUM_STRESS
    if percentage < HIGH_MAX_LOAD_PCT:
        return HIGH_STRESS
    return EXTREMELY_HIGH_STRESS


def load_threshold_kwh(
    historical_min_load_kwh: Any,
    historical_max_load_kwh: Any,
    percentage: Any,
) -> float:
    minimum = max(0.0, _number(historical_min_load_kwh) or 0.0)
    maximum = max(minimum, _number(historical_max_load_kwh) or minimum)
    pct = _number(percentage) or 0.0
    return minimum + (maximum - minimum) * pct / 100.0


def medium_load_bounds(
    historical_min_load_kwh: Any,
    historical_max_load_kwh: Any,
) -> tuple[float, float]:
    return (
        load_threshold_kwh(
            historical_min_load_kwh,
            historical_max_load_kwh,
            LOW_MAX_LOAD_PCT,
        ),
        load_threshold_kwh(
            historical_min_load_kwh,
            historical_max_load_kwh,
            MEDIUM_MAX_LOAD_PCT,
        ),
    )


def target_medium_load(
    load_kwh: Any,
    historical_min_load_kwh: Any,
    historical_max_load_kwh: Any,
) -> float:
    load = max(0.0, _number(load_kwh) or 0.0)
    lower, upper = medium_load_bounds(historical_min_load_kwh, historical_max_load_kwh)
    if load < lower:
        return load_threshold_kwh(
            historical_min_load_kwh,
            historical_max_load_kwh,
            MEDIUM_LOWER_TARGET_PCT,
        )
    if load >= upper:
        return load_threshold_kwh(
            historical_min_load_kwh,
            historical_max_load_kwh,
            MEDIUM_UPPER_TARGET_PCT,
        )
    return load


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number
