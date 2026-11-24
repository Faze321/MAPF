from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.control.numbers import optional_float
from src.control.windows import optional_number
from src.time_utils import parse_datetime_24h


def attach_price_change_diagnostics(
    report: dict[str, Any],
    reference: Any,
) -> dict[str, Any]:
    updated = dict(report)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        item = dict(window)
        shift = optional_number(item.get("suggested_price_shift_pct"))
        percentile = reference.percentile(shift) if reference is not None else None
        p95 = reference.p95_pct if reference is not None else None
        item["historical_price_change_percentile"] = percentile
        item["historical_price_change_p95_pct"] = p95
        item["exceeds_historical_p95"] = (
            abs(shift) > p95
            if shift is not None and p95 is not None
            else None
        )
        windows.append(item)
    updated["price_change_windows_3h"] = windows
    updated["historical_price_change_reference"] = (
        reference.to_dict(include_values=False) if reference is not None else None
    )
    return updated


def window_key(window: dict[str, Any]) -> tuple[str, str]:
    return str(window.get("window_start")), str(window.get("window_end"))


def shifted_energy_prices(prices: pd.Series, shift_pct: float) -> pd.Series:
    return prices * (1 + shift_pct / 100)


def energy_price_with_predicted_windows(
    energy_price: pd.DataFrame,
    zone_id: str,
    windows: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = energy_price.copy()
    if "time" not in frame or zone_id not in frame:
        raise ValueError(f"Incomplete original price schedule for zone {zone_id}")
    frame[zone_id] = frame[zone_id].astype(float)
    frame["time"] = pd.to_datetime(frame["time"])
    for window in windows:
        if not isinstance(window, dict):
            continue
        shift = optional_number(window.get("suggested_price_shift_pct"))
        if shift is None or not math.isfinite(shift) or shift < -100 or window.get("price_valid") is False or shift == 0:
            continue
        start = parse_datetime_24h(window.get("window_start"))
        end = parse_datetime_24h(window.get("window_end"))
        mask = (frame["time"] >= start) & (frame["time"] <= end)
        frame.loc[mask, zone_id] = shifted_energy_prices(energy_price.loc[mask, zone_id], shift)
    return frame


def finalize_price_window(window: dict[str, Any]) -> dict[str, Any]:
    updated = dict(window)
    base_price = optional_float(window.get("mean_energy_price"))
    shift = optional_float(window.get("suggested_price_shift_pct"))
    valid = shift is not None and math.isfinite(shift) and shift >= -100 and base_price is not None
    price = base_price * (1 + shift / 100) if valid else None
    updated["suggested_price_shift_pct"] = shift
    updated["action_label"] = final_price_action_label(shift) if valid else "Invalid quote"
    updated["price_valid"] = price is not None and math.isfinite(price) and price >= 0
    updated["proposed_energy_price"] = updated["predicted_energy_price"] = price if updated["price_valid"] else None
    updated["price_validation_error"] = (
        None if updated["price_valid"] else "proposed energy price must be non-negative"
    )
    return updated


def final_price_action_label(shift_pct: float) -> str:
    if shift_pct > 0:
        return "Raise energy price to reduce load toward Medium"
    if shift_pct < 0:
        return "Reduce energy price to increase load toward Medium"
    return "Hold original energy price"
