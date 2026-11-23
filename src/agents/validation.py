from __future__ import annotations

import math
from typing import Any


MODEL_RESPONSE_FAILED = "MODEL_RESPONSE_FAILED"


def validate_economist_report(report: dict[str, Any], expected_windows: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["response is not a JSON object"]

    if not has_text(report.get("reasoning_summary")):
        errors.append("missing reasoning_summary")
    if not has_text(report.get("price_rationale")):
        errors.append("missing price_rationale")

    expected = expected_windows if isinstance(expected_windows, list) else []
    actual = report.get("price_change_windows_3h")
    if not isinstance(actual, list):
        errors.append("missing price_change_windows_3h")
        return errors
    if len(actual) != len(expected):
        errors.append(f"price_change_windows_3h length {len(actual)} != expected {len(expected)}")

    for idx, expected_window in enumerate(expected):
        if idx >= len(actual) or not isinstance(actual[idx], dict):
            errors.append(f"price_change_windows_3h[{idx}] missing")
            continue
        item = actual[idx]
        for field in ("window_start", "window_end", "price_rationale"):
            if not has_text(item.get(field)):
                errors.append(f"price_change_windows_3h[{idx}] missing {field}")
        raw = item.get("suggested_price_shift_pct")
        if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw) or raw < -100:
            errors.append(f"price_change_windows_3h[{idx}] missing or invalid suggested_price_shift_pct")
        for field in ("window_start", "window_end"):
            expected_value = expected_window.get(field) if isinstance(expected_window, dict) else None
            if has_text(expected_value) and has_text(item.get(field)) and str(item.get(field)) != str(expected_value):
                errors.append(f"price_change_windows_3h[{idx}] {field} does not match context")
    return errors


def required_model_text(item: dict[str, Any], field: str, item_missing: bool) -> str:
    if item_missing:
        return model_response_failed("missing price_change_windows_3h item")
    value = item.get(field)
    return str(value).strip() if has_text(value) else model_response_failed(f"missing {field}")


def has_text(value: Any) -> bool:
    return bool(str(value).strip()) if value is not None else False


def model_response_failed(reason: str) -> str:
    return f"{MODEL_RESPONSE_FAILED}: {reason}"
