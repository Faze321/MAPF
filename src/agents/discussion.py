from __future__ import annotations

import json
from typing import Any

from src.agents.client import ChatClient, complete_agent_json
from src.agents.prompts import (
    discussion_behavior_prompt,
    discussion_economist_prompt,
    discussion_grid_prompt,
)
from src.agents.repair import complete_validated_economist_report
from src.agents.reports import ECONOMIST_AGENT_OUTPUT_KEY, combine_reports
from src.agents.rules import (
    heuristic_behavior,
    heuristic_economist,
    heuristic_grid,
    merge_behavior_fallback,
    merge_economist_fallback,
    merge_grid_fallback,
)
from src.agents.usage import summarize_agent_call_usage


MULTI_AGENT_DISCUSSION_ROUNDS = 3


async def run_multi_agent_discussion_zone_chain(
    context: dict[str, Any],
    *,
    client: ChatClient,
    temperature: float,
) -> dict[str, Any]:
    previous_exchange: dict[str, Any] | None = None
    discussion_rounds: list[dict[str, Any]] = []
    agent_call_usage: list[dict[str, Any]] = []
    final_grid: dict[str, Any] = {}
    final_behavior: dict[str, Any] = {}
    final_economist: dict[str, Any] = {}
    previous_decision_signature: str | None = None
    discussion_converged = False
    discussion_stop_reason = "max_rounds_reached"

    for discussion_round in range(1, MULTI_AGENT_DISCUSSION_ROUNDS + 1):
        grid_result = await complete_agent_json(
            client,
            discussion_grid_prompt(
                context,
                discussion_round=discussion_round,
                previous_exchange=previous_exchange,
            ),
            context=context,
            temperature=temperature,
            stage=f"agent.discussion.round_{discussion_round}.grid",
            agent=f"Grid Analyst Discussion Round {discussion_round}",
        )
        grid = merge_grid_fallback(grid_result.content, context)

        behavior_result = await complete_agent_json(
            client,
            discussion_behavior_prompt(
                context,
                grid,
                discussion_round=discussion_round,
                previous_exchange=previous_exchange,
            ),
            context=context,
            temperature=temperature,
            stage=f"agent.discussion.round_{discussion_round}.behavior",
            agent=f"Behavioural Agent Discussion Round {discussion_round}",
        )
        behavior = merge_behavior_fallback(behavior_result.content, context)

        economist_report, economist_usage = await complete_validated_economist_report(
            client,
            context,
            grid,
            behavior,
            temperature=temperature,
            prompt_override=discussion_economist_prompt(
                context,
                grid,
                behavior,
                discussion_round=discussion_round,
                previous_exchange=previous_exchange,
            ),
            stage=f"agent.discussion.round_{discussion_round}.economist",
            agent=f"Market Economist Discussion Round {discussion_round}",
            repair_stage=(
                f"agent.discussion.round_{discussion_round}.economist.schema_repair"
            ),
            repair_agent=(
                f"Market Economist Discussion Round {discussion_round} Schema Repair"
            ),
        )
        economist = merge_economist_fallback(economist_report, context)
        economist_debug = economist_report.get(ECONOMIST_AGENT_OUTPUT_KEY)
        round_usage = [grid_result.usage, behavior_result.usage, *economist_usage]
        agent_call_usage.extend(round_usage)

        exchange = discussion_round_record(
            discussion_round=discussion_round,
            grid=grid,
            behavior=behavior,
            economist=economist,
            economist_debug=economist_debug,
            agent_call_usage=round_usage,
        )
        decision_signature = discussion_decision_signature(
            grid,
            behavior,
            economist,
        )
        matches_previous_round = (
            previous_decision_signature is not None
            and decision_signature == previous_decision_signature
        )
        exchange["matches_previous_round"] = matches_previous_round
        discussion_rounds.append(exchange)
        previous_exchange = discussion_exchange_for_prompt(exchange)
        exchange["next_round_handoff"] = previous_exchange
        previous_decision_signature = decision_signature
        final_grid = grid
        final_behavior = behavior
        final_economist = economist
        if discussion_round >= 2 and matches_previous_round:
            discussion_converged = True
            discussion_stop_reason = (
                f"round_{discussion_round}_matches_round_{discussion_round - 1}"
            )
            break

    actual_round_count = len(discussion_rounds)
    debug = {
        "zone_id": context.get("zone_id"),
        "discussion_round_count": actual_round_count,
        "discussion_round_limit": MULTI_AGENT_DISCUSSION_ROUNDS,
        "discussion_converged": discussion_converged,
        "discussion_stop_reason": discussion_stop_reason,
        "discussion_rounds": discussion_rounds,
        "agent_call_usage": agent_call_usage,
        "agent_usage_summary": summarize_agent_call_usage(agent_call_usage),
    }
    report = combine_reports(
        context,
        final_grid,
        final_behavior,
        final_economist,
        source="multi_agent_discussion_3rounds",
        economist_debug=debug,
        agent_call_usage=agent_call_usage,
    )
    report["agent_discussion_round_count"] = actual_round_count
    report["agent_discussion_round_limit"] = MULTI_AGENT_DISCUSSION_ROUNDS
    report["agent_discussion_converged"] = discussion_converged
    report["agent_discussion_stop_reason"] = discussion_stop_reason
    report["agent_discussion_rounds"] = discussion_rounds
    return report


def discussion_round_record(
    *,
    discussion_round: int,
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
    economist_debug: dict[str, Any] | None,
    agent_call_usage: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "discussion_round": discussion_round,
        "grid_output": grid,
        "behavior_output": behavior,
        "economist_output": {
            key: value
            for key, value in economist.items()
            if key != ECONOMIST_AGENT_OUTPUT_KEY
        },
        "communication_summary": {
            "grid_message": grid.get("message_to_other_agents")
            or grid.get("reasoning_summary"),
            "behavior_message": behavior.get("message_to_other_agents")
            or behavior.get("reasoning_summary"),
            "economist_message": economist.get("message_to_other_agents")
            or economist.get("reasoning_summary")
            or economist.get("price_rationale"),
        },
        "economist_validation": economist_debug,
        "agent_usage": summarize_agent_call_usage(agent_call_usage),
        "agent_call_usage": [dict(item) for item in agent_call_usage],
    }


def discussion_exchange_for_prompt(exchange: dict[str, Any]) -> dict[str, Any]:
    grid = exchange.get("grid_output")
    grid = grid if isinstance(grid, dict) else {}
    behavior = exchange.get("behavior_output")
    behavior = behavior if isinstance(behavior, dict) else {}
    economist = exchange.get("economist_output")
    economist = economist if isinstance(economist, dict) else {}
    communication = exchange.get("communication_summary")
    communication = communication if isinstance(communication, dict) else {}
    return {
        "conclusion_summary": {
            "grid": grid.get("reasoning_summary")
            or grid.get("forecast_summary")
            or communication.get("grid_message"),
            "behavior": behavior.get("reasoning_summary")
            or communication.get("behavior_message"),
            "economist": economist.get("reasoning_summary")
            or economist.get("price_rationale")
            or communication.get("economist_message"),
        },
        "disagreements": {
            "grid": compact_discussion_list(grid.get("disagreements")),
            "behavior": compact_discussion_list(behavior.get("disagreements")),
            "economist": compact_discussion_list(
                economist.get("disagreements")
            ),
        },
        "key_decisions": discussion_key_decisions(grid, behavior, economist),
    }


def discussion_key_decisions(
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
) -> dict[str, Any]:
    return {
        "grid": {
            "adjustment_needed": grid.get("adjustment_needed"),
            "grid_stress_level": grid.get("grid_stress_level"),
            "confidence": grid.get("confidence"),
            "window_assessments": [
                {
                    key: item.get(key)
                    for key in (
                        "window_start",
                        "window_end",
                        "predicted_load_kwh",
                        "load_range_position_pct",
                        "grid_stress_level",
                        "adjustment_needed",
                    )
                    if key in item
                }
                for item in grid.get("window_assessments") or []
                if isinstance(item, dict)
            ],
        },
        "behavior": {
            "demand_drivers": behavior.get("demand_drivers"),
            "elasticity_factor": behavior.get("elasticity_factor"),
            "window_elasticities": behavior.get("window_elasticities"),
            "confidence": behavior.get("confidence"),
        },
        "economist": {
            "suggested_price_shift_pct": economist.get(
                "suggested_price_shift_pct"
            ),
            "action_label": economist.get("action_label"),
            "price_change_windows_3h": [
                {
                    key: item.get(key)
                    for key in (
                        "window_start",
                        "window_end",
                        "suggested_price_shift_pct",
                        "proposed_energy_price",
                        "action_label",
                    )
                    if key in item
                }
                for item in economist.get("price_change_windows_3h") or []
                if isinstance(item, dict)
            ],
        },
    }


def compact_discussion_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    return value if isinstance(value, list) else [value]


def discussion_decision_signature(
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
) -> str:
    """Compare substantive decisions while ignoring free-form explanation text."""

    key_decisions = discussion_key_decisions(grid, behavior, economist)
    # Confidence and descriptive demand-driver labels do not change the control
    # result; convergence is based on stress, elasticity, and price decisions.
    key_decisions["grid"].pop("confidence", None)
    key_decisions["behavior"].pop("confidence", None)
    key_decisions["behavior"].pop("demand_drivers", None)
    return json.dumps(
        canonical_discussion_decision(key_decisions),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_discussion_decision(value: Any) -> Any:
    if isinstance(value, dict):
        explanation_keys = {
            "reasoning_summary",
            "forecast_summary",
            "price_rationale",
            "message_to_other_agents",
            "agreements",
            "disagreements",
            "revisions_from_prior_round",
        }
        return {
            str(key): canonical_discussion_decision(item)
            for key, item in value.items()
            if str(key).strip().lower() not in explanation_keys
        }
    if isinstance(value, list):
        return [canonical_discussion_decision(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        text = value.strip()
        try:
            return round(float(text), 6)
        except ValueError:
            return text
    return value


def heuristic_discussion_zone_chain(
    context: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    discussion_rounds: list[dict[str, Any]] = []
    final_grid: dict[str, Any] = {}
    final_behavior: dict[str, Any] = {}
    final_economist: dict[str, Any] = {}
    previous_decision_signature: str | None = None
    discussion_converged = False
    discussion_stop_reason = "max_rounds_reached"
    for discussion_round in range(1, MULTI_AGENT_DISCUSSION_ROUNDS + 1):
        grid = heuristic_grid(context)
        behavior = heuristic_behavior(context)
        economist = heuristic_economist(context, grid)
        grid["message_to_other_agents"] = (
            f"Dry-run Grid summary for discussion round {discussion_round}."
        )
        behavior["message_to_other_agents"] = (
            f"Dry-run Behaviour summary for discussion round {discussion_round}."
        )
        economist["message_to_other_agents"] = (
            f"Dry-run Economist summary for discussion round {discussion_round}."
        )
        exchange = discussion_round_record(
            discussion_round=discussion_round,
            grid=grid,
            behavior=behavior,
            economist=economist,
            economist_debug=None,
            agent_call_usage=[],
        )
        decision_signature = discussion_decision_signature(grid, behavior, economist)
        matches_previous_round = (
            previous_decision_signature is not None
            and decision_signature == previous_decision_signature
        )
        exchange["matches_previous_round"] = matches_previous_round
        discussion_rounds.append(exchange)
        previous_decision_signature = decision_signature
        final_grid = grid
        final_behavior = behavior
        final_economist = economist
        previous_exchange = discussion_exchange_for_prompt(exchange)
        exchange["next_round_handoff"] = previous_exchange
        if discussion_round >= 2 and matches_previous_round:
            discussion_converged = True
            discussion_stop_reason = (
                f"round_{discussion_round}_matches_round_{discussion_round - 1}"
            )
            break
    report = combine_reports(
        context,
        final_grid,
        final_behavior,
        final_economist,
        source=source,
        agent_call_usage=[],
    )
    report["agent_discussion_round_count"] = len(discussion_rounds)
    report["agent_discussion_round_limit"] = MULTI_AGENT_DISCUSSION_ROUNDS
    report["agent_discussion_converged"] = discussion_converged
    report["agent_discussion_stop_reason"] = discussion_stop_reason
    report["agent_discussion_rounds"] = discussion_rounds
    return report
