from __future__ import annotations

from typing import Any

from src.agents.rules import normalize_grid_stress_level
from src.agents.usage import cache_usage_fields, summarize_agent_call_usage
from src.agents.validation import required_model_text
from src.control.numbers import as_float, first_present, optional_float
from src.control.pricing import final_price_action_label, finalize_price_window


ECONOMIST_AGENT_OUTPUT_KEY = "_economist_agent_output"


def attach_economist_debug(report: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(report)
    tagged[ECONOMIST_AGENT_OUTPUT_KEY] = debug
    return tagged


def combine_reports(
    context: dict[str, Any],
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
    *,
    source: str,
    economist_debug: dict[str, Any] | None = None,
    agent_call_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    price_windows = normalize_price_windows(
        economist.get("price_change_windows_3h"),
        context.get("pricing_windows_3h", []),
    )
    price_windows = [finalize_price_window(window) for window in price_windows]
    final_shift = average_price_shift(price_windows, as_float(economist.get("suggested_price_shift_pct"), 0))
    if economist.get("quote_errors"):
        for window in price_windows:
            window.update(price_valid=False, price_validation_error="; ".join(economist["quote_errors"]))
    usage_summary = summarize_agent_call_usage(agent_call_usage or [])
    report = {
        "category": context["category"],
        "zone_id": context["zone_id"],
        "predicted_load_kwh": as_float(grid.get("forecast_total_kwh"), context["forecast_total_kwh"]),
        "predicted_peak_kwh": as_float(grid.get("forecast_peak_kwh"), context["forecast_peak_kwh"]),
        "predicted_change_pct": as_float(grid.get("predicted_change_pct"), context["predicted_change_pct"]),
        "grid_stress_level": normalize_grid_stress_level(
            grid.get("grid_stress_level"),
            context["grid_stress_level"],
        ),
        "agent_reasoning": str(
            behavior.get("reasoning_summary") or behavior.get("agent_reasoning") or ""
        ),
        "grid_reasoning_summary": str(
            grid.get("reasoning_summary") or grid.get("forecast_summary") or ""
        ),
        "behavior_reasoning_summary": str(
            behavior.get("reasoning_summary") or behavior.get("agent_reasoning") or ""
        ),
        "economist_reasoning_summary": str(
            economist.get("reasoning_summary") or economist.get("price_rationale") or ""
        ),
        "suggested_price_shift_pct": final_shift,
        "model_action_label": str(economist.get("action_label") or ""),
        "model_price_rationale": str(economist.get("price_rationale") or ""),
        "action_label": final_price_action_label(final_shift),
        "price_rationale": str(
            economist.get("reasoning_summary")
            or economist.get("price_rationale")
            or summarize_medium_target_schedule(price_windows, final_shift)
        ),
        "price_change_windows_3h": price_windows,
        "grid_output": grid,
        "behavior_output": behavior,
        "economist_output": {
            key: value
            for key, value in economist.items()
            if key != ECONOMIST_AGENT_OUTPUT_KEY
        },
        "agent_invoked": usage_summary["agent_invoked"],
        "agent_call_count": usage_summary["agent_call_count"],
        "agent_prompt_tokens": usage_summary["prompt_tokens"],
        "agent_completion_tokens": usage_summary["completion_tokens"],
        "agent_total_tokens": usage_summary["total_tokens"],
        "agent_token_usage_complete": usage_summary["token_usage_complete"],
        **{f"agent_{key}": value for key, value in cache_usage_fields(usage_summary).items()},
        "agent_call_usage": agent_call_usage or [],
        "source": source,
    }
    if economist_debug is not None:
        report[ECONOMIST_AGENT_OUTPUT_KEY] = economist_debug
    return report


def normalize_price_windows(value: Any, fallback_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = []
    normalized = []
    for idx, fallback in enumerate(fallback_windows):
        item_missing = idx >= len(value) or not isinstance(value[idx], dict)
        item = value[idx] if not item_missing else {}
        shift = optional_float(item.get("suggested_price_shift_pct"))
        normalized.append(
            {
                # Window identity is owned by the forecaster context. A model may
                # restate midnight as "24:00:00" or otherwise alter an identifier;
                # never let generated text change the time range being priced.
                "window_start": fallback.get("window_start"),
                "window_end": fallback.get("window_end"),
                "hours": fallback.get("hours"),
                "price_conditioned_baseline_load_kwh": first_present(
                    item.get("price_conditioned_baseline_load_kwh"),
                    fallback.get("price_conditioned_baseline_load_kwh"),
                ),
                "price_conditioned_baseline_source": first_present(
                    item.get("price_conditioned_baseline_source"),
                    fallback.get("price_conditioned_baseline_source"),
                ),
                "price_conditioned_mean_predicted_kwh": first_present(
                    item.get("price_conditioned_mean_predicted_kwh"),
                    fallback.get("price_conditioned_mean_predicted_kwh"),
                ),
                "price_conditioned_peak_predicted_kwh": first_present(
                    item.get("price_conditioned_peak_predicted_kwh"),
                    fallback.get("price_conditioned_peak_predicted_kwh"),
                ),
                "price_conditioned_load_stress_level": first_present(
                    item.get("price_conditioned_load_stress_level"),
                    fallback.get("price_conditioned_load_stress_level"),
                ),
                "price_conditioned_energy_price": first_present(
                    item.get("price_conditioned_energy_price"),
                    fallback.get("price_conditioned_energy_price"),
                ),
                "sum_predicted_kwh": fallback.get("sum_predicted_kwh"),
                "mean_predicted_kwh": fallback.get("mean_predicted_kwh"),
                "peak_predicted_kwh": fallback.get("peak_predicted_kwh"),
                "mean_service_price": fallback.get("mean_service_price"),
                "mean_energy_price": fallback.get("mean_energy_price"),
                "mean_occupancy": fallback.get("mean_occupancy"),
                "mean_temp_c": fallback.get("mean_temp_c"),
                "mean_humidity": fallback.get("mean_humidity"),
                "total_rain": fallback.get("total_rain"),
                "load_stress_level": fallback.get("load_stress_level") or fallback.get("grid_stress_level"),
                "stress_load_3h_kwh": fallback.get("stress_load_3h_kwh"),
                "stress_source_file": fallback.get("stress_source_file"),
                "stress_window_hours": fallback.get("stress_window_hours"),
                "historical_min_load_3h_kwh": fallback.get("historical_min_load_3h_kwh"),
                "historical_max_load_3h_kwh": fallback.get("historical_max_load_3h_kwh"),
                "historical_load_range_3h_kwh": fallback.get(
                    "historical_load_range_3h_kwh"
                ),
                "load_range_position_pct": fallback.get("load_range_position_pct"),
                "extremely_low_low_threshold_pct": fallback.get("extremely_low_low_threshold_pct"),
                "low_medium_threshold_pct": fallback.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": fallback.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": fallback.get(
                    "high_extremely_high_threshold_pct"
                ),
                "load_3h_extremely_low_low_threshold_kwh": fallback.get(
                    "load_3h_extremely_low_low_threshold_kwh"
                ),
                "load_3h_low_medium_threshold_kwh": fallback.get(
                    "load_3h_low_medium_threshold_kwh"
                ),
                "load_3h_medium_high_threshold_kwh": fallback.get(
                    "load_3h_medium_high_threshold_kwh"
                ),
                "load_3h_high_extremely_high_threshold_kwh": fallback.get(
                    "load_3h_high_extremely_high_threshold_kwh"
                ),
                "zone_mean_energy_price": fallback.get("zone_mean_energy_price"),
                "suggested_price_shift_pct": shift,

                "price_rationale": required_model_text(item, "price_rationale", item_missing),
            }
        )
    return normalized


def average_price_shift(windows: list[dict[str, Any]], fallback: float) -> float:
    if not windows:
        return fallback
    values = [as_float(window.get("suggested_price_shift_pct"), fallback) for window in windows]
    return round(sum(values) / len(values), 2)


def summarize_medium_target_schedule(
    windows: list[dict[str, Any]],
    average_shift_pct: float,
) -> str:
    return (
        f"Final {len(windows)}-window energy-price schedule applies an average "
        f"{average_shift_pct:+.2f}% change to move or keep load in the Medium "
        f"band while keeping every proposed energy price non-negative."
    )
