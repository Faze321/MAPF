from __future__ import annotations

import copy
import math
from typing import Any

from src.agents.usage import aggregate_usage, cache_usage_fields, summarize_agent_call_usage
from src.control.pricing import window_key
from src.control.windows import optional_number


def control_attempt_snapshot(
    report: dict[str, Any],
    attempt: int,
    *,
    previous_snapshot: dict[str, Any] | None = None,
    revised_keys: set[tuple[str, str]] | None = None,
    agent_call_usage: list[dict[str, Any]] | None = None,
    proposal_phase: str | None = None,
    triggered_by_attempt: int | None = None,
) -> dict[str, Any]:
    phase = proposal_phase or ("initial" if attempt == 1 else "frozen")
    revised = revised_keys or set()
    previous_windows = {
        window_key(window): window
        for window in (previous_snapshot or {}).get("windows", [])
        if isinstance(window, dict)
    }
    usage, usage_calls = build_zone_attempt_agent_usage(
        agent_call_usage or [],
        attempt=attempt,
        proposal_phase=phase,
        triggered_by_attempt=triggered_by_attempt,
    )
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if not isinstance(window, dict):
            continue
        key = window_key(window)
        previous = previous_windows.get(key, {})
        proposal_status = (
            "initial"
            if attempt == 1
            else "revised"
            if key in revised
            else "frozen"
        )
        previous_price = optional_number(
            window.get("mean_energy_price")
            if attempt == 1
            else previous.get("proposed_energy_price")
        )
        current_price = optional_number(window.get("proposed_energy_price"))
        previous_shift = (
            0.0
            if attempt == 1
            else optional_number(previous.get("suggested_price_shift_pct"))
        )
        current_shift = optional_number(window.get("suggested_price_shift_pct"))
        price_change_amount = numeric_difference(current_price, previous_price)
        price_change_pct = (
            round((price_change_amount / previous_price) * 100.0, 4)
            if price_change_amount is not None
            and previous_price is not None
            and previous_price > 0
            else None
        )
        shift_change = numeric_difference(current_shift, previous_shift)
        trigger_failure_reason = (
            previous.get("control_failure_reason")
            if proposal_status == "revised"
            else None
        )
        reforecast_load = window.get("price_conditioned_baseline_load_kwh")
        reforecast_position = window.get(
            "price_conditioned_load_range_position_pct"
        )
        reforecast_stress = window.get("price_conditioned_load_stress_level")
        windows.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "proposal_status": proposal_status,
                "price_valid": window.get("price_valid"),
                "mean_energy_price": window.get("mean_energy_price"),
                "previous_proposed_energy_price": previous_price,
                "proposed_energy_price": current_price,
                "price_change_amount": price_change_amount,
                "price_change_pct_vs_previous": price_change_pct,
                "previous_suggested_price_shift_pct": previous_shift,
                "suggested_price_shift_pct": current_shift,
                "shift_change_percentage_points": shift_change,
                "price_changed": numeric_value_changed(
                    previous_price,
                    current_price,
                ),
                "trigger_failure_reason": trigger_failure_reason,
                "direction_deviation": window.get("direction_deviation"),
                "historical_price_change_percentile": window.get(
                    "historical_price_change_percentile"
                ),
                "historical_price_change_p95_pct": window.get(
                    "historical_price_change_p95_pct"
                ),
                "exceeds_historical_p95": window.get(
                    "exceeds_historical_p95"
                ),
                "price_conditioned_baseline_load_kwh": reforecast_load,
                "price_conditioned_load_range_position_pct": reforecast_position,
                "price_conditioned_load_stress_level": reforecast_stress,
                "price_conditioned_baseline_source": window.get(
                    "price_conditioned_baseline_source"
                ),
                "reforecast_load_kwh": reforecast_load,
                "reforecast_load_range_position_pct": reforecast_position,
                "reforecast_load_stress_level": reforecast_stress,
                "control_success": window.get("control_success"),
                "control_failure_reason": window.get("control_failure_reason"),
                "price_rationale": window.get("price_rationale"),
            }
        )
    return {
        "attempt": attempt,
        "proposal_phase": phase,
        "triggered_by_attempt": triggered_by_attempt,
        "control_status": report.get("control_status"),
        "failed_window_count": report.get("failed_window_count"),
        "agent_usage": usage,
        "agent_call_usage": usage_calls,
        "agent_discussion_round_count": report.get(
            "agent_discussion_round_count",
            0,
        ),
        "agent_discussion_round_limit": report.get(
            "agent_discussion_round_limit",
            0,
        ),
        "agent_discussion_converged": bool(
            report.get("agent_discussion_converged", False)
        ),
        "agent_discussion_stop_reason": report.get(
            "agent_discussion_stop_reason"
        ),
        "agent_discussion_rounds": copy.deepcopy(
            report.get("agent_discussion_rounds") or []
        ),
        "grid_reasoning_summary": report.get("grid_reasoning_summary"),
        "behavior_reasoning_summary": report.get("behavior_reasoning_summary"),
        "economist_reasoning_summary": report.get("economist_reasoning_summary"),
        "price_rationale": report.get("price_rationale"),
        "price_conditioned_forecast_metadata": report.get(
            "price_conditioned_forecast_metadata"
        ),
        "windows": windows,
    }


def build_zone_attempt_agent_usage(
    records: list[dict[str, Any]],
    *,
    attempt: int,
    proposal_phase: str,
    triggered_by_attempt: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = copy.deepcopy(record)
        if item.get("token_usage_complete") is None:
            item["token_usage_complete"] = all(
                item.get(field) is not None
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        item["attempt"] = attempt
        item["proposal_phase"] = proposal_phase
        item["triggered_by_attempt"] = triggered_by_attempt
        calls.append(item)
    summarized = summarize_agent_call_usage(calls)
    usage = {
        "agent_invoked": summarized["agent_invoked"],
        "agent_call_count": summarized["agent_call_count"],
        "prompt_tokens": summarized["prompt_tokens"],
        "completion_tokens": summarized["completion_tokens"],
        "total_tokens": summarized["total_tokens"],
        "token_usage_complete": summarized["token_usage_complete"],
        **cache_usage_fields(summarized),
        "proposal_phase": proposal_phase,
        "triggered_by_attempt": triggered_by_attempt,
    }
    return usage, calls


def build_global_agent_round_usage(
    *,
    attempt: int,
    zone_usage: list[dict[str, Any]],
    proposal_phase: str,
    triggered_by_attempt: int | None,
) -> dict[str, Any]:
    invoked = [item for item in zone_usage if item.get("agent_invoked")]
    return {
        "attempt": attempt,
        "proposal_phase": proposal_phase,
        "triggered_by_attempt": triggered_by_attempt,
        "agent_invoked": bool(invoked),
        "agent_invoked_zone_count": len(invoked),
        **aggregate_usage(zone_usage),
        "invoked_zone_ids": [item.get("zone_id") for item in invoked],
    }


def cumulative_zone_agent_usage(
    attempts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usage_rows = [
        attempt.get("agent_usage")
        for attempt in attempts
        if isinstance(attempt, dict) and isinstance(attempt.get("agent_usage"), dict)
    ]
    calls = [
        copy.deepcopy(call)
        for attempt in attempts
        if isinstance(attempt, dict)
        for call in (attempt.get("agent_call_usage") or [])
        if isinstance(call, dict)
    ]
    summary = summarize_agent_call_usage(calls)
    return (
        {
            "agent_attempt_count": len(attempts),
            "agent_invoked_attempt_count": sum(
                bool(item.get("agent_invoked")) for item in usage_rows
            ),
            "agent_invoked": summary["agent_invoked"],
            "agent_call_count": summary["agent_call_count"],
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"],
            "total_tokens": summary["total_tokens"],
            **cache_usage_fields(summary),
            "token_usage_complete": all(
                bool(item.get("token_usage_complete")) for item in usage_rows
            ),
        },
        calls,
    )


def retry_proposal_phase(agent_mode: str) -> str:
    return {
        "multi_agent_economist_retry": "economist_retry",
        "multi_agent_full_retry": "full_multi_agent_retry",
        "multi_agent_discussion_3rounds": "three_round_agent_discussion_retry",
        "single_agent_price_retry": "single_agent_price_retry",
        "single_agent_full_retry": "single_agent_full_retry",
    }.get(agent_mode, "price_retry")


def numeric_difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return round(current - previous, 4)


def numeric_value_changed(previous: float | None, current: float | None) -> bool:
    if previous is None or current is None:
        return previous is not None or current is not None
    return not math.isclose(previous, current, rel_tol=0.0, abs_tol=1e-9)
