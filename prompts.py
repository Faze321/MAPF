from __future__ import annotations

import json
from typing import Any


SYSTEM_MESSAGE = (
    "You are a concise load-management analyst. Use only the provided forecast context. "
    "Return valid JSON only, with no markdown. Provide structured reasoning summaries, not hidden chain-of-thought."
)

AGENT_CONTEXT_KEYS = {
    "category",
    "zone_id",
    "selection_reason",
    "history_start",
    "history_end",
    "validation_start",
    "validation_end",
    "forecast_start",
    "forecast_end",
    "forecast_horizon_days",
    "forecast_horizon_hours",
    "forecast_model",
    "forecast_total_kwh",
    "forecast_peak_kwh",
    "predicted_change_pct",
    "capacity_kw_proxy",
    "peak_capacity_ratio",
    "grid_stress_level",
    "grid_stress_basis",
    "grid_stress_load_kwh",
    "grid_stress_load_range_position_pct",
    "grid_stress_window_hours",
    "historical_min_load_3h_kwh",
    "historical_max_load_3h_kwh",
    "historical_load_range_3h_kwh",
    "low_medium_threshold_pct",
    "medium_high_threshold_pct",
    "high_extremely_high_threshold_pct",
    "load_3h_low_medium_threshold_kwh",
    "load_3h_medium_high_threshold_kwh",
    "load_3h_high_extremely_high_threshold_kwh",
    "zone_mean_energy_price",
    "calibration",
    "price",
    "weather",
    "occupancy",
    "profile",
    "daily_history_kwh",
    "daily_forecast_kwh",
    "hourly_shape",
    "hourly_averages",
    "hourly_forecast",
    "pricing_windows_3h",
    "retry_feedback",
    "instructions",
}


def horizon_label(context: dict[str, Any]) -> str:
    days = context.get("forecast_horizon_days")
    try:
        value = int(days)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if value == 1 else f"{value} days"


def grid_prompt(context: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    safe_context = agent_safe_context(context)
    return (
        f"""Phase A / Grid Analyst.

        Task: predict the next {horizon} of charging load for this zone. Use the baseline forecast as the numerical anchor unless the context strongly justifies a small adjustment.

        The forecaster values are authoritative: assess them but do not replace them.
        Return JSON with keys: reasoning_summary, adjustment_needed, confidence, window_assessments.
        window_assessments must contain window_start, window_end, predicted_load_kwh, load_range_position_pct, grid_stress_level, adjustment_needed, reasoning_summary.
        Classify each 3-hour load by its position in that zone's full historical 3-hour load range: position = (load - historical minimum) / (historical maximum - historical minimum) * 100. Below 35% is Low, 35% to below 80% is Medium, 80% to below 90% is High, and 90% or above is Extremely High.
        grid_stress_level must be exactly one of: Low, Medium, High, Extremely High.

        Never use forecast-period actual load, forecast-period errors, or evaluation labels.

        Context:{json.dumps(safe_context, ensure_ascii=False)}"""
    )


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    safe_context = agent_safe_context(context)
    return (
        f"""Phase B / Behavioural Agent.

        Task: explain why demand looks this way over the next {horizon} using POI mix, weather, temporal markers, hourly forecast data, 3-hour pricing windows, and the load shape. Also estimate a positive elasticity_factor for price response: higher means users are more willing to shift charging after a price increase. Rain and high occupancy should lower elasticity. Keep it specific and short.

        Return JSON with keys: reasoning_summary, demand_drivers, elasticity_factor, window_elasticities, confidence.

        Context:{json.dumps(safe_context, ensure_ascii=False)}
        Grid report:{json.dumps(grid_report, ensure_ascii=False)}"""
    )


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    horizon = horizon_label(context)
    economist_context = compact_economist_context(context)
    return (
        f"""Phase C / Market Economist.
        
        Task: prescribe energy-price shifts for the next {horizon} using only forecast-derived information, prior agent conclusions, category, known service price, baseline energy price, each 3-hour window's predicted load_stress_level, and the behavioural elasticity estimate. Residential users are more price-sensitive; CBD and hub users are less price-sensitive. Adjust only energy price; service price remains unchanged.

        Pricing policy: keep expected load in the Medium band, defined as 35% to below 80% through the zone's full historical 3-hour min-max load range. For Low load, reduce energy price to increase load toward Medium. For Medium load, hold energy price or use only a small change that keeps load inside Medium. For High load, increase energy price to reduce load into Medium. For Extremely High load, use a larger energy-price increase than for High when needed. Every recommended energy price must be non-negative. Historical price-change percentiles are diagnostic only and are intentionally not supplied to you.

        Do not use actual future load, actual future stress, forecast error, stress correctness, or any evaluation/ground-truth fields. Those fields are intentionally not provided to you.

        Return JSON with keys: reasoning_summary, suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.
        price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, proposed_energy_price, action_label, price_rationale.

        Context:\n{json.dumps(economist_context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}"""
    )


def discussion_grid_prompt(
    context: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return (
        grid_prompt(context)
        + collaborative_round_instruction(
            role="Grid Analyst",
            discussion_round=discussion_round,
            previous_exchange=previous_exchange,
            task=(
                "Review the previous Behavioural Agent and Market Economist conclusions, "
                "identify agreements or disagreements that affect grid stress, and revise "
                "your structured assessment when justified. Do not change the forecaster's "
                "numerical load values. Also return agreements, disagreements, "
                "revisions_from_prior_round, and message_to_other_agents."
            ),
        )
    )


def discussion_behavior_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return (
        behavior_prompt(context, grid_report)
        + collaborative_round_instruction(
            role="Behavioural Agent",
            discussion_round=discussion_round,
            previous_exchange=previous_exchange,
            task=(
                "Review the previous Grid and Economist conclusions together with the "
                "current Grid report. Reconcile demand drivers and elasticity estimates, "
                "and explain any revision. Also return agreements, disagreements, "
                "revisions_from_prior_round, and message_to_other_agents."
            ),
        )
    )


def discussion_economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return (
        economist_prompt(context, grid_report, behavior_report)
        + collaborative_round_instruction(
            role="Market Economist",
            discussion_round=discussion_round,
            previous_exchange=previous_exchange,
            task=(
                "Review the previous round and the current Grid and Behavioural reports. "
                "Resolve disagreements explicitly in the structured summary and revise "
                "the price schedule when the shared evidence supports it. Preserve all "
                "required economist pricing keys. Also return agreements, disagreements, "
                "revisions_from_prior_round, and message_to_other_agents."
            ),
        )
    )


def collaborative_round_instruction(
    *,
    role: str,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
    task: str,
) -> str:
    previous = previous_exchange or {
        "status": "No previous exchange; establish the round-1 position."
    }
    return (
        f"""

        Collaborative discussion round {discussion_round} of 3 / {role}.
        {task}
        Treat other agents' outputs as advice, not ground truth. Use only the supplied,
        no-leakage forecast context. Return structured reasoning summaries only, never
        hidden chain-of-thought.

        Previous round exchange:\n{json.dumps(previous, ensure_ascii=False)}"""
    )


def single_agent_prompt(context: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    economist_context = compact_economist_context(context)
    return (
        f"""Single pricing agent.

        Task: replace the separate Grid Analyst, Behavioural Agent, and Market Economist stages for the next {horizon}. Use only forecast-derived information, category, known service price, baseline energy price, predicted 3-hour load_stress_level, weather, occupancy, and load shape. Estimate price-response elasticity and prescribe energy-price shifts for each 3-hour pricing window. Adjust only energy price; service price remains unchanged.

        Pricing policy: reduce energy price for Low load, keep changes small for Medium, raise energy price for High, and raise it more strongly for Extremely High. Target the Medium load band. Every resulting energy price must be non-negative.

        Do not use actual future load, actual future stress, forecast error, stress correctness, or any evaluation/ground-truth fields. Those fields are intentionally not provided to you.

        Return JSON with keys: reasoning_summary, window_assessments, demand_drivers, elasticity_factor, window_elasticities, confidence, suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.
        Classify each 3-hour load by its position in that zone's full historical 3-hour min-max load range: below 35% is Low, 35% to below 80% is Medium, 80% to below 90% is High, and 90% or above is Extremely High.
        grid_stress_level must be exactly one of: Low, Medium, High, Extremely High.
        price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, proposed_energy_price, action_label, price_rationale.

        Context:\n{json.dumps(economist_context, ensure_ascii=False)}"""
    )


def repair_economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    previous_report: dict[str, Any],
    validation_errors: list[str],
) -> str:
    economist_context = compact_economist_context(context)
    expected_count = len(economist_context.get("pricing_windows_3h", []))
    return (
        f"""Repair the Market Economist JSON response.

        The previous response failed schema validation. Return valid JSON only, with no markdown and no explanatory text.
        Validation errors: {json.dumps(validation_errors, ensure_ascii=False)}

        Required top-level keys:
        reasoning_summary, suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.

        price_change_windows_3h must contain exactly {expected_count} items, one for each context.pricing_windows_3h item in the same order.
        Each item must contain: window_start, window_end, suggested_price_shift_pct, proposed_energy_price, action_label, price_rationale.
        Use the same window_start and window_end values from the context.

        Context:\n{json.dumps(economist_context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}
        Previous invalid response:\n{json.dumps(previous_report, ensure_ascii=False)}"""
    )


def price_retry_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    previous_report: dict[str, Any],
    *,
    single_agent: bool = False,
) -> str:
    safe_context = agent_safe_context(context)
    role = "Single pricing agent" if single_agent else "Market Economist"
    return (
        f"""{role} retry.

        Revise prices only for the failed windows listed in retry_feedback. Keep every
        non-failed window's previous price unchanged. Use the previous proposed price,
        price-conditioned forecast load, load-range position, stress, and failure reason.
        Do not use actual future load or evaluation information. Final energy prices must
        be non-negative.

        Return JSON with keys: reasoning_summary, suggested_price_shift_pct,
        action_label, price_rationale, price_change_windows_3h. Return every pricing
        window in original order. Each window requires window_start, window_end,
        suggested_price_shift_pct, proposed_energy_price, action_label, price_rationale.

        Context:\n{json.dumps(safe_context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}
        Previous report:\n{json.dumps(previous_report, ensure_ascii=False)}"""
    )


def compact_economist_context(context: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = [
        "category",
        "zone_id",
        "forecast_start",
        "forecast_end",
        "forecast_horizon_days",
        "forecast_horizon_hours",
        "forecast_total_kwh",
        "forecast_peak_kwh",
        "predicted_change_pct",
        "grid_stress_level",
        "capacity_kw_proxy",
        "historical_min_load_3h_kwh",
        "historical_max_load_3h_kwh",
        "historical_load_range_3h_kwh",
        "grid_stress_load_range_position_pct",
        "low_medium_threshold_pct",
        "medium_high_threshold_pct",
        "high_extremely_high_threshold_pct",
        "load_3h_low_medium_threshold_kwh",
        "load_3h_medium_high_threshold_kwh",
        "load_3h_high_extremely_high_threshold_kwh",
        "zone_mean_energy_price",
    ]
    compact = {key: context.get(key) for key in scalar_keys if key in context}
    if "hourly_averages" in context:
        compact["hourly_averages"] = forecast_only_hourly_averages(context.get("hourly_averages"))
    if "pricing_windows_3h" in context:
        compact["pricing_windows_3h"] = forecast_only_pricing_windows(context.get("pricing_windows_3h"))
    if "retry_feedback" in context:
        compact["retry_feedback"] = forecast_only_retry_feedback(context.get("retry_feedback"))
    return compact


def agent_safe_context(context: dict[str, Any]) -> dict[str, Any]:
    """Construct prompts from an allowlist instead of trying to block leakage names."""

    safe = {key: context.get(key) for key in AGENT_CONTEXT_KEYS if key in context}
    if "hourly_averages" in safe:
        safe["hourly_averages"] = forecast_only_hourly_averages(safe["hourly_averages"])
    if "pricing_windows_3h" in safe:
        safe["pricing_windows_3h"] = forecast_only_pricing_windows(safe["pricing_windows_3h"])
    if "hourly_forecast" in safe:
        safe["hourly_forecast"] = forecast_only_hourly_rows(safe["hourly_forecast"])
    if "retry_feedback" in safe:
        safe["retry_feedback"] = forecast_only_retry_feedback(safe["retry_feedback"])
    return safe


def forecast_only_hourly_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "time",
        "predicted_kwh",
        "q10_kwh",
        "q50_kwh",
        "q90_kwh",
        "s_price",
        "e_price",
        "occupancy",
        "T",
        "U",
        "nRAIN",
        "hour",
        "is_weekend",
    }
    return [
        {key: item.get(key) for key in allowed if key in item}
        for item in value
        if isinstance(item, dict)
    ]


def forecast_only_retry_feedback(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "attempt",
        "window_start",
        "window_end",
        "previous_price_shift_pct",
        "previous_energy_price",
        "reforecast_load_kwh",
        "load_range_position_pct",
        "load_stress_level",
        "failure_reason",
    }
    return [
        {key: item.get(key) for key in allowed if key in item}
        for item in value
        if isinstance(item, dict)
    ]


def forecast_only_hourly_averages(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    blocked = {
        "mean_actual_kwh",
        "mean_abs_pct_error",
    }
    return {key: item for key, item in value.items() if key not in blocked}


def forecast_only_pricing_windows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "window_start",
        "window_end",
        "hours",
        "mean_predicted_kwh",
        "sum_predicted_kwh",
        "peak_predicted_kwh",
        "mean_service_price",
        "mean_energy_price",
        "mean_occupancy",
        "mean_temp_c",
        "mean_humidity",
        "total_rain",
        "load_stress_level",
        "grid_stress_level",
        "stress_load_3h_kwh",
        "stress_source_file",
        "stress_window_hours",
        "historical_min_load_3h_kwh",
        "historical_max_load_3h_kwh",
        "historical_load_range_3h_kwh",
        "load_range_position_pct",
        "low_medium_threshold_pct",
        "medium_high_threshold_pct",
        "high_extremely_high_threshold_pct",
        "load_3h_low_medium_threshold_kwh",
        "load_3h_medium_high_threshold_kwh",
        "load_3h_high_extremely_high_threshold_kwh",
        "zone_mean_energy_price",
    }
    sanitized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sanitized.append(
            {
                key: item.get(key)
                for key in allowed
                if key in item
            }
        )
    return sanitized
