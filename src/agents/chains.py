from __future__ import annotations

import asyncio
from typing import Any

from src.agents.client import ChatClient, complete_agent_json
from src.agents.discussion import heuristic_discussion_zone_chain, run_multi_agent_discussion_zone_chain
from src.agents.prompts import (
    behavior_prompt,
    grid_prompt,
    price_retry_prompt,
    single_agent_prompt,
)
from src.agents.repair import complete_validated_economist_report, review_quote
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


async def run_zone_chain(
    context: dict[str, Any],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
    heuristic_source: str = "dry-run",
    chain_mode: str = "multi_agent",
) -> dict[str, Any]:
    if client is None:
        if chain_mode == "multi_agent_discussion":
            return heuristic_discussion_zone_chain(
                context,
                source=f"{heuristic_source}_multi_agent_discussion_3rounds",
            )
        return heuristic_zone_chain(context, source=heuristic_source)

    if chain_mode == "single_agent":
        return await run_single_agent_zone_chain(
            context,
            client=client,
            temperature=temperature,
        )
    if chain_mode == "multi_agent_discussion":
        return await run_multi_agent_discussion_zone_chain(
            context,
            client=client,
            temperature=temperature,
        )

    grid_result = await complete_agent_json(
        client,
        grid_prompt(context),
        context=context,
        temperature=temperature,
        stage="agent.grid",
        agent="Grid Analyst",
    )
    grid = merge_grid_fallback(
        grid_result.content,
        context,
    )
    behavior_result = await complete_agent_json(
        client,
        behavior_prompt(context, grid),
        context=context,
        temperature=temperature,
        stage="agent.behavior",
        agent="Behavioural Agent",
    )
    behavior = merge_behavior_fallback(
        behavior_result.content,
        context,
    )
    economist_report, economist_call_usage = await complete_validated_economist_report(
        client,
        context,
        grid,
        behavior,
        temperature=temperature,
    )
    economist_debug = economist_report.get(ECONOMIST_AGENT_OUTPUT_KEY)
    economist = merge_economist_fallback(
        economist_report,
        context,
    )
    agent_call_usage = [grid_result.usage, behavior_result.usage, *economist_call_usage]
    return combine_reports(
        context,
        grid,
        behavior,
        economist,
        source="multi_agent",
        economist_debug=economist_debug,
        agent_call_usage=agent_call_usage,
    )


async def run_single_agent_zone_chain(
    context: dict[str, Any],
    *,
    client: ChatClient,
    temperature: float,
) -> dict[str, Any]:
    result = await complete_agent_json(
        client,
        single_agent_prompt(context),
        context=context,
        temperature=temperature,
        stage="agent.single_agent",
        agent="Single Pricing Agent",
    )
    response = result.content
    grid = merge_grid_fallback(response, context)
    behavior = merge_behavior_fallback(response, context)
    review = await review_quote(result, client, context, grid, behavior, temperature=temperature,
                                stage="agent.single_agent.schema_repair", agent="Single Pricing Agent Schema Repair")
    selected, usage = review.selected, review.usage
    if selected is review.repaired:
        grid = merge_grid_fallback(selected, context)
        behavior = merge_behavior_fallback(selected, context)
    economist = merge_economist_fallback(selected, context)
    debug = {
        "zone_id": context.get("zone_id"),
        "category": context.get("category"),
        "forecast_origin": context.get("forecast_origin"),
        "forecast_end": context.get("forecast_end"),
        "expected_price_window_count": len(context.get("pricing_windows_3h", []))
        if isinstance(context.get("pricing_windows_3h"), list)
        else 0,
        "single_agent_response": response,
        "single_agent_validation_errors": review.initial_errors,
        "repair_response": review.repaired,
        "repair_validation_errors": review.repair_errors,
        "selected_response_source": "repair" if selected is review.repaired else "single_agent",
        "agent_call_usage": usage,
        "agent_usage_summary": summarize_agent_call_usage(usage),
    }
    return combine_reports(
        context,
        grid,
        behavior,
        economist,
        source="single_agent",
        economist_debug=debug,
        agent_call_usage=usage,
    )


async def run_all_zone_chains(
    contexts: list[dict[str, Any]],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
    heuristic_source: str = "dry-run",
    chain_mode: str = "multi_agent",
) -> list[dict[str, Any]]:
    tasks = [
        run_zone_chain(
            context,
            client=client,
            temperature=temperature,
            heuristic_source=heuristic_source,
            chain_mode=chain_mode,
        )
        for context in contexts
    ]
    return await asyncio.gather(*tasks)


async def retry_zone_prices(
    context: dict[str, Any],
    previous_report: dict[str, Any],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
    single_agent: bool = False,
    heuristic_source: str = "dry-run",
) -> dict[str, Any]:
    """Revise only the price decision while preserving prior analysis outputs."""

    grid = previous_report.get("grid_output")
    if not isinstance(grid, dict):
        grid = heuristic_grid(context)
    behavior = previous_report.get("behavior_output")
    if not isinstance(behavior, dict):
        behavior = heuristic_behavior(context)
    if previous_report.get("selected_attempt", 0) > 0:
        # Price-only retries do not rerun either analyst. Never resend their old load estimates.
        grid = heuristic_grid(context)
        grid["forecast_summary"] = "Numerical assessment of the currently accepted price schedule."
        behavior = {key: behavior[key] for key in ("elasticity_factor", "window_elasticities", "confidence")
                    if key in behavior}

    if client is None:
        economist = heuristic_economist(context, grid)
        return combine_reports(
            context,
            grid,
            behavior,
            economist,
            source=f"{heuristic_source}_price_retry",
        )

    stage = "agent.single_agent_price_retry" if single_agent else "agent.economist_retry"
    agent_name = "Single Pricing Agent Retry" if single_agent else "Market Economist Retry"
    result = await complete_agent_json(
        client,
        price_retry_prompt(
            context,
            grid,
            behavior,
            {"reasoning_summary": previous_report.get("price_rationale"),
             "price_change_windows_3h": [{k: w.get(k) for k in ("window_start", "window_end",
                 "suggested_price_shift_pct", "proposed_energy_price", "price_rationale")}
                 for w in previous_report.get("price_change_windows_3h", [])]},
            single_agent=single_agent,
        ),
        context=context,
        temperature=temperature,
        stage=stage,
        agent=agent_name,
    )
    review = await review_quote(result, client, context, grid, behavior, temperature=temperature,
                                stage=f"{stage}.schema_repair", agent=f"{agent_name} Schema Repair")
    selected, usage = review.selected, review.usage
    economist = merge_economist_fallback(selected, context)
    debug = {
        "zone_id": context.get("zone_id"),
        "retry": True,
        "single_agent": single_agent,
        "response": result.content,
        "validation_errors": review.initial_errors,
        "repair_response": review.repaired,
        "repair_validation_errors": review.repair_errors,
        "agent_call_usage": usage,
    }
    return combine_reports(
        context,
        grid,
        behavior,
        economist,
        source="single_agent_price_retry" if single_agent else "economist_retry",
        economist_debug=debug,
        agent_call_usage=usage,
    )


def heuristic_zone_chain(
    context: dict[str, Any],
    *,
    source: str = "dry-run",
) -> dict[str, Any]:
    grid = heuristic_grid(context)
    behavior = heuristic_behavior(context)
    economist = heuristic_economist(context, grid)
    return combine_reports(context, grid, behavior, economist, source=source)
