from __future__ import annotations

from typing import Any


LOW_STRESS = "Low"
MEDIUM_STRESS = "Medium"
HIGH_STRESS = "High"
EXTREMELY_HIGH_STRESS = "Extremely High"

LOW_MAX_LOAD_PCT = 35.0
MEDIUM_MAX_LOAD_PCT = 80.0
HIGH_MAX_LOAD_PCT = 90.0
MEDIUM_LOWER_TARGET_PCT = 35.5
MEDIUM_UPPER_TARGET_PCT = 79.5


def load_percentage(load_kwh: Any, reference_load_kwh: Any) -> float:
    load = _number(load_kwh)
    reference = _number(reference_load_kwh)
    if load is None or reference is None or reference <= 0:
        return 0.0
    return max(0.0, (load / reference) * 100.0)


def classify_load_percentage(load_pct: Any) -> str:
    percentage = _number(load_pct) or 0.0
    if percentage < LOW_MAX_LOAD_PCT:
        return LOW_STRESS
    if percentage < MEDIUM_MAX_LOAD_PCT:
        return MEDIUM_STRESS
    if percentage < HIGH_MAX_LOAD_PCT:
        return HIGH_STRESS
    return EXTREMELY_HIGH_STRESS


def medium_load_bounds(reference_load_kwh: Any) -> tuple[float, float]:
    reference = max(0.0, _number(reference_load_kwh) or 0.0)
    return (
        reference * LOW_MAX_LOAD_PCT / 100.0,
        reference * MEDIUM_MAX_LOAD_PCT / 100.0,
    )


def target_medium_load(load_kwh: Any, reference_load_kwh: Any) -> float:
    load = max(0.0, _number(load_kwh) or 0.0)
    reference = max(0.0, _number(reference_load_kwh) or 0.0)
    lower, upper = medium_load_bounds(reference)
    if load < lower:
        return reference * MEDIUM_LOWER_TARGET_PCT / 100.0
    if load >= upper:
        return reference * MEDIUM_UPPER_TARGET_PCT / 100.0
    return load


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number
