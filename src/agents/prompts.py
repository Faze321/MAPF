from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from src.control.policy import STRESS_GUIDE


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
    "forecast_origin",
    "forecast_end",
    "forecast_horizon_days",
    "forecast_horizon_hours",
    "backend",
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
    "extremely_low_low_threshold_pct",
    "low_medium_threshold_pct",
    "medium_high_threshold_pct",
    "high_extremely_high_threshold_pct",
    "load_3h_extremely_low_low_threshold_kwh",
    "load_3h_low_medium_threshold_kwh",
    "load_3h_medium_high_threshold_kwh",
    "load_3h_high_extremely_high_threshold_kwh",
    "zone_mean_energy_price",
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


def prompt_json(value: Any) -> str:
    """Canonical object keys across workers, retaining array order and every value."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def context_json(context: dict[str, Any]) -> str:
    """Keep changing retry feedback last without changing its context field path."""
    stable = {key: value for key, value in context.items() if key != "retry_feedback"}
    serialized = prompt_json(stable)
    if "retry_feedback" not in context:
        return serialized
    separator = "," if stable else ""
    return serialized[:-1] + separator + '"retry_feedback":' + prompt_json(context["retry_feedback"]) + "}"


def _json_section(label: str, value: Any) -> str:
    return f"\n\n{label}:\n{prompt_json(value)}"


def _context_prompt(
    instructions: str,
    context: dict[str, Any],
    *reports: tuple[str, Any],
) -> str:
    # Static instructions, reusable forecast context, then changing Agent reports.
    return (
        instructions + "\n\nContext:\n" + context_json(context)
        + "".join(_json_section(label, value) for label, value in reports)
    )


QUOTE_POLICY = (
    "Only Medium is successful. Reduce energy prices for Low or Extremely Low; "
    "consider a deeper cut for Extremely Low. Prefer hold for Medium; increase for High "
    "or Extremely High, considering a larger increase for Extremely High. When retry_feedback is present, "
    "revise only its windows: copy accepted_shift_pct unchanged for every other window. "
    "Rejected shifts and loads describe discarded trials, not the starting schedule. "
    "Use only a finite percentage >= -100 relative to the ORIGINAL "
    "hourly energy prices of this window, never relative to a previous quote or the zone mean. "
    "Zero preserves all original hours. Do not quote absolute prices. Service price is fixed. "
    + STRESS_GUIDE
)

GRID_INSTRUCTIONS = dedent(f"""\
    Phase A / Grid Analyst.

    Task: assess the supplied charging-load forecast over the configured forecast horizon in Context.

    The forecaster values are authoritative: assess them but do not replace them.
    Return JSON with keys: reasoning_summary, adjustment_needed, confidence, window_assessments.
    window_assessments must contain window_start, window_end, predicted_load_kwh, load_range_position_pct, grid_stress_level, adjustment_needed, reasoning_summary.
    Use the supplied forecaster classification. {STRESS_GUIDE}

    Never use forecast-period actual load, forecast-period errors, or evaluation labels.
""").strip()

BEHAVIOR_INSTRUCTIONS = dedent("""\
    Phase B / Behavioural Agent.

    Task: explain why demand looks this way over the configured forecast horizon in Context using POI mix, weather, temporal markers, hourly forecast data, 3-hour pricing windows, and the load shape. Also estimate a positive elasticity_factor for price response: higher means users are more willing to shift charging after a price increase. Rain and high occupancy should lower elasticity. Keep it specific and short.

    Return JSON with keys: reasoning_summary, demand_drivers, elasticity_factor, window_elasticities, confidence.
""").strip()

ECONOMIST_INSTRUCTIONS = dedent(f"""\
    Phase C / Market Economist.

    Task: prescribe energy-price shifts for the configured forecast horizon in Context using only forecast-derived information, prior agent conclusions, category, known service price, baseline energy price, each 3-hour window's predicted load_stress_level, and the behavioural elasticity estimate. Residential users are more price-sensitive; CBD and hub users are less price-sensitive. Adjust only energy price; service price remains unchanged.

    {QUOTE_POLICY}

    Do not use actual future load, actual future stress, forecast error, stress correctness, or any evaluation/ground-truth fields. Those fields are intentionally not provided to you.

    Return JSON with keys: reasoning_summary, price_rationale, price_change_windows_3h.
    price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, price_rationale.
""").strip()

SINGLE_AGENT_INSTRUCTIONS = dedent(f"""\
    Single pricing agent.

    Task: replace the separate Grid Analyst, Behavioural Agent, and Market Economist stages for the configured forecast horizon in Context. Use only forecast-derived information, category, known service price, baseline energy price, predicted 3-hour load_stress_level, weather, occupancy, and load shape. Estimate price-response elasticity and prescribe energy-price shifts for each 3-hour pricing window. Adjust only energy price; service price remains unchanged.

    {QUOTE_POLICY}

    Do not use actual future load, actual future stress, forecast error, stress correctness, or any evaluation/ground-truth fields. Those fields are intentionally not provided to you.

    Return JSON with keys: reasoning_summary, window_assessments, demand_drivers, elasticity_factor, window_elasticities, confidence, price_rationale, price_change_windows_3h.
    Use the supplied forecaster classification.
    price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, price_rationale.
""").strip()

REPAIR_INSTRUCTIONS = dedent(f"""\
    Repair the Market Economist JSON response. {QUOTE_POLICY}

    The previous response failed schema validation. Return valid JSON only, with no markdown and no explanatory text. Correct the Validation errors listed below.

    Required top-level keys:
    reasoning_summary, price_rationale, price_change_windows_3h.

    price_change_windows_3h must contain exactly as many items as context.pricing_windows_3h, one for each item in the same order. The Expected window count is supplied below.
    Each item must contain: window_start, window_end, suggested_price_shift_pct, price_rationale.
    Use the same window_start and window_end values from the context.
""").strip()

PRICE_RETRY_INSTRUCTIONS = dedent(f"""\
    {QUOTE_POLICY}
    Revise prices only for the failed windows listed in context.retry_feedback. Keep every
    non-failed window's previous price unchanged. Use the previous proposed price,
    price-conditioned forecast load, load-range position, stress, and failure reason.
    Do not use actual future load or evaluation information. Final energy prices must
    be non-negative.

    Return JSON with keys: reasoning_summary, price_rationale, price_change_windows_3h. Return every pricing
    window in original order. Each window requires window_start, window_end,
    suggested_price_shift_pct, price_rationale.
""").strip()


def grid_prompt(context: dict[str, Any]) -> str:
    return _context_prompt(GRID_INSTRUCTIONS, agent_safe_context(context))


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    return _context_prompt(BEHAVIOR_INSTRUCTIONS, agent_safe_context(context), ("Grid report", grid_report))


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    return _context_prompt(
        ECONOMIST_INSTRUCTIONS, compact_economist_context(context),
        ("Grid report", grid_report), ("Behaviour report", behavior_report),
    )


def _discussion_prompt(
    instructions: str,
    task: str,
    context: dict[str, Any],
    *reports: tuple[str, Any],
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    # The round number must not interrupt the reusable instructions/context prefix.
    collaboration = dedent("""\
        Collaborative discussion, up to 3 rounds.
        Treat other agents' outputs as advice, not ground truth. Use only the supplied,
        no-leakage forecast context. Return structured reasoning summaries only, never
        hidden chain-of-thought.
        Also return agreements, disagreements, revisions_from_prior_round, and message_to_other_agents.

        Previous round compact handoff contains only conclusion_summary,
        disagreements, and key_decisions. Full prior Agent outputs are intentionally
        omitted; do not ask for or reconstruct them.
    """).strip()
    previous = previous_exchange or {"status": "No previous exchange; establish the round-1 position."}
    return _context_prompt(
        instructions + "\n\n" + collaboration + "\n" + task, context, *reports,
        ("Discussion round", discussion_round), ("Previous round compact handoff", previous),
    )


def discussion_grid_prompt(
    context: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return _discussion_prompt(
        GRID_INSTRUCTIONS,
        "Review the previous Behavioural Agent and Market Economist conclusions, "
        "identify agreements or disagreements that affect grid stress, and revise "
        "your structured assessment when justified. Do not change the forecaster's numerical load values.",
        agent_safe_context(context), discussion_round=discussion_round, previous_exchange=previous_exchange,
    )


def discussion_behavior_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return _discussion_prompt(
        BEHAVIOR_INSTRUCTIONS,
        "Review the previous Grid and Economist conclusions together with the current Grid report. "
        "Reconcile demand drivers and elasticity estimates, and explain any revision.",
        agent_safe_context(context), ("Grid report", grid_report),
        discussion_round=discussion_round, previous_exchange=previous_exchange,
    )


def discussion_economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    *,
    discussion_round: int,
    previous_exchange: dict[str, Any] | None,
) -> str:
    return _discussion_prompt(
        ECONOMIST_INSTRUCTIONS,
        "Review the previous round and the current Grid and Behavioural reports. "
        "Resolve disagreements explicitly in the structured summary and revise the price schedule "
        "when the shared evidence supports it. Preserve all required economist pricing keys.",
        compact_economist_context(context), ("Grid report", grid_report), ("Behaviour report", behavior_report),
        discussion_round=discussion_round, previous_exchange=previous_exchange,
    )


def single_agent_prompt(context: dict[str, Any]) -> str:
    return _context_prompt(SINGLE_AGENT_INSTRUCTIONS, compact_economist_context(context))


def repair_economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    previous_report: dict[str, Any],
    validation_errors: list[str],
) -> str:
    economist_context = compact_economist_context(context)
    expected_count = len(economist_context.get("pricing_windows_3h", []))
    return _context_prompt(
        REPAIR_INSTRUCTIONS, economist_context,
        ("Grid report", grid_report), ("Behaviour report", behavior_report),
        ("Expected window count", expected_count),
        ("Previous invalid response", previous_report), ("Validation errors", validation_errors),
    )


def price_retry_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
    previous_report: dict[str, Any],
    *,
    single_agent: bool = False,
) -> str:
    role = "Single pricing agent" if single_agent else "Market Economist"
    return _context_prompt(
        role + " retry.\n\n" + PRICE_RETRY_INSTRUCTIONS, agent_safe_context(context),
        ("Grid report", grid_report), ("Behaviour report", behavior_report), ("Previous report", previous_report),
    )


def compact_economist_context(context: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = [
        "category",
        "zone_id",
        "forecast_origin",
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
        "extremely_low_low_threshold_pct",
        "low_medium_threshold_pct",
        "medium_high_threshold_pct",
        "high_extremely_high_threshold_pct",
        "load_3h_extremely_low_low_threshold_kwh",
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
        "accepted_attempt", "rejected_attempt", "rejected_mean_energy_price", "rejected_shift_pct", "rejected_load_kwh", "rejected_stress",
        "rejection_reason", "rejected_lost_medium_count", "rejected_gained_medium_count",
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
        "accepted_shift_pct",
        "accepted_mean_energy_price",
        "rejected_attempt", "rejected_shift_pct", "rejected_mean_energy_price", "rejected_load_kwh", "rejected_stress",
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
        "extremely_low_low_threshold_pct",
        "low_medium_threshold_pct",
        "medium_high_threshold_pct",
        "high_extremely_high_threshold_pct",
        "load_3h_extremely_low_low_threshold_kwh",
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
