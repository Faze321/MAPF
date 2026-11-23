from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.control.policy import (
    EXTREMELY_LOW_MAX_LOAD_PCT,
    HIGH_MAX_LOAD_PCT,
    LOAD_POLICY_ID,
    LOW_MAX_LOAD_PCT,
    LOW_STRESS,
    MEDIUM_MAX_LOAD_PCT,
    STRESS_LEVELS,
    classify_load_percentage,
    load_range_position_pct,
    load_threshold_kwh,
)
from src.data.canonical import finite_float
from src.files import atomic_write_dataframe


def build_zone_3h_load_policy_thresholds(
    cache_dir: Path,
    *,
    force_cache: bool = False,
    source_file: str = "canonical_dataset",
    window_hours: int = 3,
    forecast_origin: pd.Timestamp | str | None = None,
    load_frame: pd.DataFrame,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        "historical_3h_load_policy.csv"
        if forecast_origin is not None
        else "zone_3h_load_quantiles.csv"
    )
    if cache_file.exists() and not force_cache:
        try:
            cached = pd.read_csv(cache_file, dtype={"zone_id": str})
        except (OSError, pd.errors.ParserError, ValueError):
            cached = pd.DataFrame()
        required_columns = {
            "load_policy_id",
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
        }
        source_ok = "stress_source_file" in cached.columns and (
            cached["stress_source_file"] == source_file
        ).all()
        window_ok = "stress_window_hours" in cached.columns and (
            cached["stress_window_hours"].astype(int) == int(window_hours)
        ).all()
        policy_ok = required_columns.issubset(cached.columns) and (
            (cached["load_policy_id"] == LOAD_POLICY_ID).all()
            and np.isclose(cached["extremely_low_low_threshold_pct"], EXTREMELY_LOW_MAX_LOAD_PCT).all()
            and np.isclose(cached["low_medium_threshold_pct"], LOW_MAX_LOAD_PCT).all()
            and np.isclose(cached["medium_high_threshold_pct"], MEDIUM_MAX_LOAD_PCT).all()
            and np.isclose(
                cached["high_extremely_high_threshold_pct"], HIGH_MAX_LOAD_PCT
            ).all()
        )
        cutoff_ok = True
        if forecast_origin is not None:
            cutoff_ok = "forecast_origin_cutoff" in cached.columns and (
                cached["forecast_origin_cutoff"] == pd.Timestamp(forecast_origin).isoformat()
            ).all()
        if source_ok and window_ok and policy_ok and cutoff_ok:
            return cached

    load = load_frame.copy()
    if forecast_origin is not None:
        load = load[load["time"] < pd.Timestamp(forecast_origin)].copy()
    zone_ids = [str(col) for col in load.columns if col != "time"]
    values = load.set_index("time")[zone_ids].apply(pd.to_numeric, errors="coerce").sort_index()
    grouped = values.resample(f"{int(window_hours)}h", origin="start_day").sum(min_count=1)

    rows = []
    for zone_id in zone_ids:
        zone_values = grouped[zone_id].dropna()
        historical_min = finite_float(zone_values.min())
        historical_max = finite_float(zone_values.max())
        rows.append(
            {
                "zone_id": zone_id,
                "stress_source_file": source_file,
                "stress_window_hours": int(window_hours),
                "forecast_origin_cutoff": (
                    pd.Timestamp(forecast_origin).isoformat()
                    if forecast_origin is not None
                    else None
                ),
                "historical_3h_windows": int(len(zone_values)),
                "load_policy_id": LOAD_POLICY_ID,
                "historical_min_load_3h_kwh": historical_min,
                "historical_max_load_3h_kwh": historical_max,
                "historical_load_range_3h_kwh": finite_float(
                    historical_max - historical_min
                ),
                "extremely_low_low_threshold_pct": EXTREMELY_LOW_MAX_LOAD_PCT,
                "low_medium_threshold_pct": LOW_MAX_LOAD_PCT,
                "medium_high_threshold_pct": MEDIUM_MAX_LOAD_PCT,
                "high_extremely_high_threshold_pct": HIGH_MAX_LOAD_PCT,
                "load_3h_extremely_low_low_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, EXTREMELY_LOW_MAX_LOAD_PCT)
                ),
                "load_3h_low_medium_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, LOW_MAX_LOAD_PCT)
                ),
                "load_3h_medium_high_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, MEDIUM_MAX_LOAD_PCT)
                ),
                "load_3h_high_extremely_high_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, HIGH_MAX_LOAD_PCT)
                ),
            }
        )
    thresholds = pd.DataFrame(rows)
    atomic_write_dataframe(thresholds, cache_file)
    return thresholds


STRESS_LEVEL_ORDER = {level: rank for rank, level in enumerate(STRESS_LEVELS)}


def optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def load_stress_thresholds(
    thresholds_by_zone: pd.DataFrame,
    zone_id: str,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or {}
    zone_mean_energy_price = safe_float(profile.get("mean_energy_price"))
    price_fields = {"zone_mean_energy_price": zone_mean_energy_price}
    if zone_id not in thresholds_by_zone.index:
        return {
            "available": False,
            "source_file": "volume.csv",
            "window_hours": 3,
            "historical_windows": 0,
            "historical_min_load_kwh": 0.0,
            "historical_max_load_kwh": 0.0,
            "historical_load_range_kwh": 0.0,
            "extremely_low_low_threshold_pct": EXTREMELY_LOW_MAX_LOAD_PCT,
            "low_medium_threshold_pct": LOW_MAX_LOAD_PCT,
            "medium_high_threshold_pct": MEDIUM_MAX_LOAD_PCT,
            "high_extremely_high_threshold_pct": HIGH_MAX_LOAD_PCT,
            "extremely_low_low_threshold_kwh": 0.0,
            "low_medium_threshold_kwh": 0.0,
            "medium_high_threshold_kwh": 0.0,
            "high_extremely_high_threshold_kwh": 0.0,
            **price_fields,
        }
    row = thresholds_by_zone.loc[zone_id]
    historical_min = safe_float(row.get("historical_min_load_3h_kwh"))
    historical_max = safe_float(row.get("historical_max_load_3h_kwh"))
    return {
        "available": True,
        "source_file": str(row.get("stress_source_file", "volume.csv")),
        "window_hours": int(row.get("stress_window_hours", 3) or 3),
        "historical_windows": int(row.get("historical_3h_windows", 0) or 0),
        "historical_min_load_kwh": historical_min,
        "historical_max_load_kwh": historical_max,
        "historical_load_range_kwh": safe_float(
            row.get("historical_load_range_3h_kwh", historical_max - historical_min)
        ),
        "extremely_low_low_threshold_pct": safe_float(
            row.get("extremely_low_low_threshold_pct", EXTREMELY_LOW_MAX_LOAD_PCT)
        ),
        "low_medium_threshold_pct": safe_float(
            row.get("low_medium_threshold_pct", LOW_MAX_LOAD_PCT)
        ),
        "medium_high_threshold_pct": safe_float(
            row.get("medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT)
        ),
        "high_extremely_high_threshold_pct": safe_float(
            row.get("high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT)
        ),
        "extremely_low_low_threshold_kwh": safe_float(
            row.get("load_3h_extremely_low_low_threshold_kwh")
        ),
        "low_medium_threshold_kwh": safe_float(
            row.get("load_3h_low_medium_threshold_kwh")
        ),
        "medium_high_threshold_kwh": safe_float(
            row.get("load_3h_medium_high_threshold_kwh")
        ),
        "high_extremely_high_threshold_kwh": safe_float(
            row.get("load_3h_high_extremely_high_threshold_kwh")
        ),
        **price_fields,
    }


def classify_load_stress(load_value: Any, thresholds: dict[str, Any]) -> str:
    if not thresholds.get("available", True):
        return LOW_STRESS
    position_pct = load_range_position_pct(
        load_value,
        thresholds.get("historical_min_load_kwh"),
        thresholds.get("historical_max_load_kwh"),
    )
    return classify_load_percentage(position_pct)


def apply_load_policy_stress(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    pricing_windows_3h: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(summary)
    stress_loads = [safe_float(window.get("sum_predicted_kwh")) for window in pricing_windows_3h]
    stress_load = max(stress_loads) if stress_loads else 0.0
    range_positions = [
        safe_float(window.get("load_range_position_pct")) for window in pricing_windows_3h
    ]
    window_levels = [window.get("load_stress_level") for window in pricing_windows_3h]
    updated["grid_stress_level"] = max_stress_level(window_levels) if window_levels else classify_load_stress(stress_load, thresholds)
    updated["grid_stress_basis"] = (
        "forecast_3h_sum_predicted_kwh_position_in_zone_historical_3h_min_max_range"
    )
    updated["grid_stress_load_kwh"] = stress_load
    updated["grid_stress_load_range_position_pct"] = (
        max(range_positions) if range_positions else 0.0
    )
    updated["grid_stress_source_file"] = thresholds.get("source_file", "volume.csv")
    updated["grid_stress_window_hours"] = thresholds.get("window_hours", 3)
    updated["grid_stress_historical_windows"] = thresholds.get("historical_windows", 0)
    updated["historical_min_load_3h_kwh"] = thresholds.get("historical_min_load_kwh", 0.0)
    updated["historical_max_load_3h_kwh"] = thresholds.get("historical_max_load_kwh", 0.0)
    updated["historical_load_range_3h_kwh"] = thresholds.get(
        "historical_load_range_kwh", 0.0
    )
    updated["extremely_low_low_threshold_pct"] = thresholds.get(
        "extremely_low_low_threshold_pct", EXTREMELY_LOW_MAX_LOAD_PCT
    )
    updated["low_medium_threshold_pct"] = thresholds.get(
        "low_medium_threshold_pct", LOW_MAX_LOAD_PCT
    )
    updated["medium_high_threshold_pct"] = thresholds.get(
        "medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT
    )
    updated["high_extremely_high_threshold_pct"] = thresholds.get(
        "high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT
    )
    updated["load_3h_extremely_low_low_threshold_kwh"] = thresholds.get(
        "extremely_low_low_threshold_kwh", 0.0
    )
    updated["load_3h_low_medium_threshold_kwh"] = thresholds.get(
        "low_medium_threshold_kwh", 0.0
    )
    updated["load_3h_medium_high_threshold_kwh"] = thresholds.get(
        "medium_high_threshold_kwh", 0.0
    )
    updated["load_3h_high_extremely_high_threshold_kwh"] = thresholds.get(
        "high_extremely_high_threshold_kwh", 0.0
    )
    updated["zone_mean_energy_price"] = thresholds.get("zone_mean_energy_price", 0.0)
    return updated


def build_pricing_windows_3h(
    hourly: pd.DataFrame,
    *,
    stress_thresholds: dict[str, Any] | None = None,
    include_evaluation: bool = False,
) -> list[dict[str, Any]]:
    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    windows: list[dict[str, Any]] = []
    for idx in range(0, len(frame), 3):
        chunk = frame.iloc[idx : idx + 3]
        if chunk.empty:
            continue
        window: dict[str, Any] = {
            "window_start": chunk["time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": chunk["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "hours": int(len(chunk)),
        }
        aggregations = {
            "predicted_kwh": ("mean_predicted_kwh", "sum_predicted_kwh", "peak_predicted_kwh"),
        }
        if include_evaluation:
            aggregations["actual_kwh"] = (
                "mean_actual_kwh",
                "sum_actual_kwh",
                "peak_actual_kwh",
            )
        for source_col, (mean_col, sum_col, max_col) in aggregations.items():
            if source_col in chunk:
                values = pd.to_numeric(chunk[source_col], errors="coerce")
                window[mean_col] = clean_context_value(values.mean(skipna=True))
                window[sum_col] = optional_number(values.sum(min_count=1))
                window[max_col] = clean_context_value(values.max(skipna=True))
        mean_columns = {
            "s_price": "mean_service_price",
            "e_price": "mean_energy_price",
            "occupancy": "mean_occupancy",
            "T": "mean_temp_c",
            "U": "mean_humidity",
        }
        for source_col, output_col in mean_columns.items():
            if source_col in chunk:
                mean = pd.to_numeric(chunk[source_col], errors="coerce").mean(skipna=True)
                window[output_col] = optional_number(mean) if source_col in {"s_price", "e_price"} else clean_context_value(mean)
        if "nRAIN" in chunk:
            window["total_rain"] = clean_context_value(pd.to_numeric(chunk["nRAIN"], errors="coerce").sum(skipna=True))
        if stress_thresholds is not None:
            stress_load = window.get("sum_predicted_kwh")
            stress_level = classify_load_stress(stress_load, stress_thresholds)
            actual_stress_load = window.get("sum_actual_kwh") if include_evaluation else None
            actual_stress_level = (
                classify_load_stress(actual_stress_load, stress_thresholds)
                if actual_stress_load is not None
                else None
            )
            window["load_stress_level"] = stress_level
            window["grid_stress_level"] = stress_level
            window["stress_load_3h_kwh"] = stress_load
            window["actual_load_stress_level"] = actual_stress_level
            window["actual_grid_stress_level"] = actual_stress_level
            window["actual_stress_load_3h_kwh"] = actual_stress_load
            predicted_rank = stress_rank(stress_level)
            actual_rank = stress_rank(actual_stress_level)
            window["stress_correct"] = predicted_rank == actual_rank if actual_rank is not None else None
            window["stress_missed"] = actual_rank > predicted_rank if actual_rank is not None and predicted_rank is not None else None
            window["stress_source_file"] = stress_thresholds.get("source_file", "volume.csv")
            window["stress_window_hours"] = stress_thresholds.get("window_hours", 3)
            historical_min_load = safe_float(
                stress_thresholds.get("historical_min_load_kwh")
            )
            historical_max_load = safe_float(
                stress_thresholds.get("historical_max_load_kwh")
            )
            historical_load_range = max(0.0, historical_max_load - historical_min_load)
            window["historical_min_load_3h_kwh"] = historical_min_load
            window["historical_max_load_3h_kwh"] = historical_max_load
            window["historical_load_range_3h_kwh"] = historical_load_range
            window["load_range_position_pct"] = load_range_position_pct(
                stress_load, historical_min_load, historical_max_load
            )
            window["actual_load_range_position_pct"] = (
                load_range_position_pct(actual_stress_load, historical_min_load, historical_max_load)
                if actual_stress_load is not None else None
            )
            window["extremely_low_low_threshold_pct"] = stress_thresholds.get(
                "extremely_low_low_threshold_pct", EXTREMELY_LOW_MAX_LOAD_PCT
            )
            window["low_medium_threshold_pct"] = stress_thresholds.get(
                "low_medium_threshold_pct", LOW_MAX_LOAD_PCT
            )
            window["medium_high_threshold_pct"] = stress_thresholds.get(
                "medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT
            )
            window["high_extremely_high_threshold_pct"] = stress_thresholds.get(
                "high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT
            )
            window["load_3h_extremely_low_low_threshold_kwh"] = stress_thresholds.get(
                "extremely_low_low_threshold_kwh",
                historical_min_load + historical_load_range * EXTREMELY_LOW_MAX_LOAD_PCT / 100,
            )
            window["load_3h_low_medium_threshold_kwh"] = stress_thresholds.get(
                "low_medium_threshold_kwh",
                historical_min_load + historical_load_range * LOW_MAX_LOAD_PCT / 100,
            )
            window["load_3h_medium_high_threshold_kwh"] = stress_thresholds.get(
                "medium_high_threshold_kwh",
                historical_min_load + historical_load_range * MEDIUM_MAX_LOAD_PCT / 100,
            )
            window["load_3h_high_extremely_high_threshold_kwh"] = stress_thresholds.get(
                "high_extremely_high_threshold_kwh",
                historical_min_load + historical_load_range * HIGH_MAX_LOAD_PCT / 100,
            )
            window["zone_mean_energy_price"] = stress_thresholds.get(
                "zone_mean_energy_price",
                0.0,
            )
        windows.append(window)
    return windows


def clean_context_value(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, ndigits)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return round(number, 4)


def max_stress_level(levels: list[Any]) -> str:
    normalized = [str(level) for level in levels if stress_rank(level) is not None]
    if not normalized:
        return "Low"
    return max(normalized, key=lambda level: STRESS_LEVEL_ORDER[level])


def stress_rank(level: Any) -> int | None:
    return STRESS_LEVEL_ORDER.get(str(level))
