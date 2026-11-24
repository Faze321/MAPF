from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.usage import cache_usage_fields
from src.config.agent import normalize_agent_mode
from src.config.values import normalize_forecast_model_name
from src.control.policy import LOAD_POLICY_ID
from src.control.windows import optional_number
from src.results.prices import price_comparison_fields
from src.time_utils import normalize_datetime_series_24h, parse_datetime_24h


def load_precomputed_window_data(path: Path | str | None) -> pd.DataFrame:
    if path in (None, ""):
        return pd.DataFrame()
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Precomputed window data file not found: {cache_path}")
    frame = pd.read_csv(cache_path, dtype={"zone_id": "string"})
    if "load_policy_id" not in frame or not frame["load_policy_id"].eq(LOAD_POLICY_ID).all():
        raise ValueError("Precomputed prices require the current load_policy_id; regenerate the window cache")
    required = {
        "zone_id",
        "window_start",
        "window_end",
        "predicted_energy_price",
        "price_conditioned_load_kwh",
        "zone_mean_energy_price",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Precomputed window data is missing required column(s): " + ", ".join(missing)
        )
    frame = frame.copy()
    frame["zone_id"] = frame["zone_id"].astype("string").str.strip()
    for column in ("window_start", "window_end"):
        parsed = normalize_datetime_series_24h(frame[column], errors="coerce")
        if parsed.isna().any():
            bad_rows = parsed[parsed.isna()].index.tolist()[:5]
            raise ValueError(f"Invalid {column} value in precomputed window data at row(s): {bad_rows}")
        frame[column] = parsed.map(lambda value: value.isoformat())
    return frame


def partition_precomputed_contexts(
    contexts: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    backend: str,
    agent_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[tuple[str, str, str], dict[str, Any]]]]:
    if frame.empty:
        return [], list(contexts), {}

    candidates = frame.copy()
    if "backend" in candidates:
        model_values = candidates["backend"].fillna("").astype(str).str.strip()
        model_matches = model_values.map(
            lambda value: not value or normalize_forecast_model_name(value) == backend
        )
        candidates = candidates[model_matches]
    if "agent_mode" in candidates:
        mode_values = candidates["agent_mode"].fillna("").astype(str).str.strip()

        def mode_matches(value: str) -> bool:
            if not value:
                return True
            try:
                return normalize_agent_mode(value) == agent_mode
            except ValueError:
                return False

        candidates = candidates[mode_values.map(mode_matches)]

    cache_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidates.to_dict(orient="records"):
        key = precomputed_window_key(row.get("zone_id"), row.get("window_start"), row.get("window_end"))
        if key in cache_index:
            raise ValueError(
                "Duplicate precomputed window data for "
                f"zone={key[0]}, window_start={key[1]}, window_end={key[2]}"
            )
        cache_index[key] = {name: clean_precomputed_value(value) for name, value in row.items()}

    cached_contexts: list[dict[str, Any]] = []
    live_contexts: list[dict[str, Any]] = []
    cached_windows_by_zone: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for context in contexts:
        zone_id = str(context.get("zone_id"))
        expected_windows = [
            window for window in context.get("pricing_windows_3h") or [] if isinstance(window, dict)
        ]
        zone_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        complete = bool(expected_windows)
        for window in expected_windows:
            key = precomputed_window_key(zone_id, window.get("window_start"), window.get("window_end"))
            row = cache_index.get(key)
            if (
                row is None
                or optional_number(row.get("predicted_energy_price")) is None
                or optional_number(row.get("price_conditioned_load_kwh")) is None
            ):
                complete = False
                break
            zone_rows[key] = row
        if complete:
            cached_contexts.append(context)
            cached_windows_by_zone[zone_id] = zone_rows
        else:
            live_contexts.append(context)
    return cached_contexts, live_contexts, cached_windows_by_zone


def apply_precomputed_window_data(
    reports: list[dict[str, Any]],
    cached_windows_by_zone: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    *,
    source_path: Path | str | None,
) -> list[dict[str, Any]]:
    source_name = Path(source_path).name if source_path not in (None, "") else "precomputed data"
    updated_reports: list[dict[str, Any]] = []
    passthrough_fields = (
        "load_stress_level",
        "load_range_position_pct",
        "price_conditioned_load_stress_level",
        "price_conditioned_load_range_position_pct",
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
        "target_peak_reduction_pct",
        "baseline_load_range_position_pct",
        "target_load_kwh",
        "target_load_range_position_pct",
        "medium_load_min_kwh",
        "medium_load_max_kwh",
        "elasticity_factor",
        "load_range_source",
        "expected_load_kwh",
        "expected_load_range_position_pct",
        "load_in_medium_band",
        "action_label",
        "price_rationale",
    )
    boolean_fields = {
        "load_in_medium_band",
    }

    for report in reports:
        zone_id = str(report.get("zone_id"))
        zone_rows = cached_windows_by_zone[zone_id]
        updated_windows: list[dict[str, Any]] = []
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            key = precomputed_window_key(zone_id, window.get("window_start"), window.get("window_end"))
            row = zone_rows[key]
            updated = dict(window)
            predicted_price = optional_number(row.get("predicted_energy_price"))
            conditioned_load = optional_number(row.get("price_conditioned_load_kwh"))
            baseline_price = optional_number(updated.get("mean_energy_price"))
            final_shift = optional_number(row.get("final_price_shift_pct"))
            if final_shift is None and baseline_price not in (None, 0) and predicted_price is not None:
                final_shift = ((predicted_price / baseline_price) - 1) * 100
            forecast_load = optional_number(row.get("forecast_load_kwh"))
            if forecast_load is not None:
                updated["sum_predicted_kwh"] = forecast_load
            updated["predicted_energy_price"] = predicted_price
            updated["proposed_energy_price"] = predicted_price
            updated["price_valid"] = predicted_price is not None and predicted_price >= 0
            updated["suggested_price_shift_pct"] = final_shift
            updated["price_conditioned_baseline_load_kwh"] = conditioned_load
            updated["price_conditioned_mean_predicted_kwh"] = optional_number(
                row.get("price_conditioned_mean_load_kwh")
            )
            updated["price_conditioned_peak_predicted_kwh"] = optional_number(
                row.get("price_conditioned_peak_load_kwh")
            )
            forecaster_price = optional_number(row.get("forecaster_input_predicted_energy_price"))
            updated["price_conditioned_energy_price"] = (
                forecaster_price if forecaster_price is not None else predicted_price
            )
            updated["price_conditioned_baseline_source"] = "precomputed_window_data"
            cached_baseline_load = optional_number(row.get("baseline_load_kwh"))
            updated["baseline_load_kwh"] = (
                cached_baseline_load if cached_baseline_load is not None else conditioned_load
            )
            updated["baseline_load_source"] = row.get("baseline_load_source") or "precomputed_window_data"
            for field in passthrough_fields:
                value = row.get(field)
                if value is not None:
                    updated[field] = normalize_precomputed_bool(value) if field in boolean_fields else value
            updated_windows.append(updated)

        updated_report = dict(report)
        updated_report["price_change_windows_3h"] = updated_windows
        shifts = [optional_number(window.get("suggested_price_shift_pct")) for window in updated_windows]
        shifts = [value for value in shifts if value is not None]
        forecast_loads = [optional_number(window.get("sum_predicted_kwh")) for window in updated_windows]
        if forecast_loads and all(value is not None for value in forecast_loads):
            updated_report["predicted_load_kwh"] = round(sum(value for value in forecast_loads if value is not None), 4)
        if shifts:
            updated_report["suggested_price_shift_pct"] = round(sum(shifts) / len(shifts), 2)
        updated_report["agent_reasoning"] = f"Reused window load and energy-price predictions from {source_name}."
        updated_report["action_label"] = "Reuse precomputed window predictions"
        updated_report["price_rationale"] = f"Loaded complete 3-hour window predictions from {source_name}."
        updated_report["agent_invoked"] = False
        updated_report["agent_call_count"] = 0
        updated_report["agent_prompt_tokens"] = 0
        updated_report["agent_completion_tokens"] = 0
        updated_report["agent_total_tokens"] = 0
        updated_report["agent_token_usage_complete"] = True
        updated_report.update({f"agent_{key}": value for key, value in
                               cache_usage_fields({"agent_call_count": 0}).items()})
        updated_report["agent_call_usage"] = []
        updated_report["source"] = "precomputed_window_data"
        updated_report["precomputed_window_data_source"] = str(source_path)
        updated_reports.append(updated_report)
    return updated_reports


def precomputed_window_key(zone_id: Any, window_start: Any, window_end: Any) -> tuple[str, str, str]:
    return (
        str(zone_id).strip(),
        parse_datetime_24h(window_start).isoformat(),
        parse_datetime_24h(window_end).isoformat(),
    )


def clean_precomputed_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def normalize_precomputed_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


WINDOW_LOAD_PRICE_CACHE_COLUMNS = [
    "load_policy_id",
    "zone_id",
    "category",
    "backend",
    "agent_mode",
    "source",
    "window_start",
    "window_end",
    "forecast_load_kwh",
    "price_conditioned_load_kwh",
    "price_conditioned_mean_load_kwh",
    "price_conditioned_peak_load_kwh",
    "predicted_energy_price",
    "forecaster_input_predicted_energy_price",
    "baseline_energy_price",
    "known_service_price",
    "final_price_shift_pct",
    "load_stress_level",
    "load_range_position_pct",
    "price_conditioned_load_stress_level",
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
    "baseline_load_kwh",
    "baseline_load_range_position_pct",
    "baseline_load_source",
    "target_load_kwh",
    "target_load_range_position_pct",
    "medium_load_min_kwh",
    "medium_load_max_kwh",
    "target_peak_reduction_pct",
    "elasticity_factor",
    "load_range_source",
    "expected_load_kwh",
    "expected_load_range_position_pct",
    "load_in_medium_band",
    "price_valid",
    "historical_price_change_percentile",
    "historical_price_change_p95_pct",
    "exceeds_historical_p95",
    "control_success",
    "control_failure_reason",
    "action_label",
    "price_rationale",
]


def write_window_load_price_cache(
    cache_csv: Path,
    reports: list[dict[str, Any]],
    *,
    backend: str | None = None,
    agent_mode: str | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            baseline_energy_price = window.get("mean_energy_price")
            comparison = price_comparison_fields(
                baseline_energy_price,
                window.get("suggested_price_shift_pct"),
                window.get("predicted_energy_price"),
            )
            rows.append(
                {
                    "zone_id": report.get("zone_id"),
                    "category": report.get("category"),
                    "backend": backend,
                    "agent_mode": agent_mode,
                    "source": report.get("source"),
                    "window_start": window.get("window_start"),
                    "window_end": window.get("window_end"),
                    "load_policy_id": LOAD_POLICY_ID,
                    "forecast_load_kwh": window.get("sum_predicted_kwh"),
                    "price_conditioned_load_kwh": window.get("price_conditioned_baseline_load_kwh"),
                    "price_conditioned_mean_load_kwh": window.get("price_conditioned_mean_predicted_kwh"),
                    "price_conditioned_peak_load_kwh": window.get("price_conditioned_peak_predicted_kwh"),
                    "predicted_energy_price": comparison["predicted_energy_price"],
                    "forecaster_input_predicted_energy_price": window.get("price_conditioned_energy_price"),
                    "baseline_energy_price": baseline_energy_price,
                    "known_service_price": window.get("mean_service_price"),
                    "final_price_shift_pct": window.get("suggested_price_shift_pct"),
                    "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                    "load_range_position_pct": window.get("load_range_position_pct"),
                    "price_conditioned_load_stress_level": window.get("price_conditioned_load_stress_level"),
                    "historical_min_load_3h_kwh": window.get(
                        "historical_min_load_3h_kwh"
                    ),
                    "historical_max_load_3h_kwh": window.get(
                        "historical_max_load_3h_kwh"
                    ),
                    "historical_load_range_3h_kwh": window.get(
                        "historical_load_range_3h_kwh"
                    ),
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
                    "baseline_load_kwh": window.get("baseline_load_kwh"),
                    "baseline_load_range_position_pct": window.get(
                        "baseline_load_range_position_pct"
                    ),
                    "baseline_load_source": window.get("baseline_load_source"),
                    "target_load_kwh": window.get("target_load_kwh"),
                    "target_load_range_position_pct": window.get(
                        "target_load_range_position_pct"
                    ),
                    "medium_load_min_kwh": window.get("medium_load_min_kwh"),
                    "medium_load_max_kwh": window.get("medium_load_max_kwh"),
                    "target_peak_reduction_pct": window.get("target_peak_reduction_pct"),
                    "elasticity_factor": window.get("elasticity_factor"),
                    "load_range_source": window.get("load_range_source"),
                    "expected_load_kwh": window.get("expected_load_kwh"),
                    "expected_load_range_position_pct": window.get(
                        "expected_load_range_position_pct"
                    ),
                    "load_in_medium_band": window.get("load_in_medium_band"),
                    "price_valid": window.get("price_valid"),
                    "historical_price_change_percentile": window.get(
                        "historical_price_change_percentile"
                    ),
                    "historical_price_change_p95_pct": window.get(
                        "historical_price_change_p95_pct"
                    ),
                    "exceeds_historical_p95": window.get("exceeds_historical_p95"),
                    "control_success": window.get("control_success"),
                    "control_failure_reason": window.get("control_failure_reason"),
                    "action_label": window.get("action_label"),
                    "price_rationale": window.get("price_rationale"),
                }
            )
    pd.DataFrame(rows, columns=WINDOW_LOAD_PRICE_CACHE_COLUMNS).to_csv(cache_csv, index=False)
