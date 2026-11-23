from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.client import AgentCallResult, ChatClient, complete_agent_json
from src.agents.prompts import economist_prompt, repair_economist_prompt
from src.agents.reports import attach_economist_debug
from src.agents.usage import summarize_agent_call_usage
from src.agents.validation import validate_economist_report


@dataclass
class QuoteReview:
    selected: dict
    initial_errors: list[str]
    usage: list[dict]
    repaired: dict | None = None
    repair_errors: list[str] | None = None


async def review_quote(initial: AgentCallResult, client: ChatClient, context: dict, grid: dict,
                       behavior: dict, *, temperature: float, stage: str, agent: str,
                       prefer_fewer_errors: bool = False) -> QuoteReview:
    windows = context.get("pricing_windows_3h", [])
    errors = validate_economist_report(initial.content, windows)
    review = QuoteReview(initial.content, errors, [initial.usage])
    if errors:
        result = await complete_agent_json(
            client, repair_economist_prompt(context, grid, behavior, initial.content, errors),
            context=context, temperature=min(temperature, 0.1), stage=stage, agent=agent,
        )
        review.usage.append(result.usage)
        review.repaired = result.content
        review.repair_errors = validate_economist_report(result.content, windows)
        if not review.repair_errors or (prefer_fewer_errors and len(review.repair_errors) < len(errors)):
            review.selected = result.content
    return review


async def complete_validated_economist_report(
    client: ChatClient,
    context: dict[str, Any],
    grid: dict[str, Any],
    behavior: dict[str, Any],
    *,
    temperature: float,
    prompt_override: str | None = None,
    stage: str = "agent.economist",
    agent: str = "Market Economist",
    repair_stage: str = "agent.economist_repair",
    repair_agent: str = "Market Economist Repair",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_windows = context.get("pricing_windows_3h", [])
    initial_result = await complete_agent_json(
        client,
        prompt_override or economist_prompt(context, grid, behavior),
        context=context,
        temperature=temperature,
        stage=stage,
        agent=agent,
    )
    review = await review_quote(initial_result, client, context, grid, behavior, temperature=temperature,
                                stage=repair_stage, agent=repair_agent, prefer_fewer_errors=True)
    repaired = review.repaired is not None and review.selected is review.repaired
    debug: dict[str, Any] = {
        "zone_id": context.get("zone_id"),
        "category": context.get("category"),
        "forecast_origin": context.get("forecast_origin"),
        "forecast_end": context.get("forecast_end"),
        "expected_price_window_count": len(expected_windows) if isinstance(expected_windows, list) else 0,
        "initial_response": initial_result.content,
        "initial_validation_errors": review.initial_errors,
        "repair_attempted": review.repaired is not None,
        "repair_response": review.repaired,
        "repair_validation_errors": review.repair_errors,
        "selected_response_source": "repair" if repaired else "initial",
        "selected_validation_errors": review.repair_errors if repaired else review.initial_errors,
        "agent_call_usage": review.usage,
        "agent_usage_summary": summarize_agent_call_usage(review.usage),
    }
    return attach_economist_debug(review.selected, debug), review.usage
