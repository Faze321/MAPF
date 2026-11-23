from __future__ import annotations

import math
from typing import Any

from src.control.policy import MEDIUM_STRESS, STRESS_LEVELS, medium_gap_kwh
from src.control.windows import optional_number


def schedule_score(windows):
    """Lower is better. A partial or invalid scenario cannot win."""
    if not windows:
        return None
    medium_count = 0
    gap = movement = 0.0
    for w in windows:
        load, lower, upper, shift = (optional_number(w.get(key)) for key in (
            "price_conditioned_baseline_load_kwh", "load_3h_low_medium_threshold_kwh",
            "load_3h_medium_high_threshold_kwh", "suggested_price_shift_pct"))
        if (not w.get("price_valid") or w.get("price_conditioned_load_stress_level") not in STRESS_LEVELS
                or any(v is None or not math.isfinite(v) for v in (load, lower, upper, shift))):
            return None
        medium_count += int(w["price_conditioned_load_stress_level"] == MEDIUM_STRESS)
        gap += medium_gap_kwh(load, lower, upper)
        movement += abs(shift) * w["hours"]
    return -medium_count, gap, movement


def assess_control_report(report: dict[str, Any], *, attempt: int) -> dict[str, Any]:
    updated = dict(report)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        item = dict(window)
        stress = item.get("price_conditioned_load_stress_level")
        price_valid = bool(item.get("price_valid"))
        success = price_valid and stress == MEDIUM_STRESS
        item["control_attempt"] = attempt
        item["control_success"] = success
        if not price_valid:
            failure_reason = "invalid_non_negative_energy_price_constraint"
        elif stress is None:
            failure_reason = "price_conditioned_forecast_unavailable"
        elif stress != MEDIUM_STRESS:
            failure_reason = f"price_conditioned_load_is_{stress}"
        else:
            failure_reason = None
        item["control_failure_reason"] = failure_reason
        windows.append(item)
    updated["price_change_windows_3h"] = windows
    updated["control_status"] = (
        "success" if windows and all(item["control_success"] for item in windows) else "fail"
    )
    updated["failed_window_count"] = sum(not item["control_success"] for item in windows)
    updated["attempts_used"] = attempt
    return updated
