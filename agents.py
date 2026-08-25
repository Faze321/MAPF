from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from config import AgentConfig
from load_policy import (
    EXTREMELY_HIGH_STRESS,
    HIGH_STRESS,
    LOW_STRESS,
    MEDIUM_STRESS,
    classify_load_percentage,
    load_range_position_pct,
    medium_load_bounds,
    target_medium_load,
)
from prompts import (
    SYSTEM_MESSAGE,
    behavior_prompt,
    discussion_behavior_prompt,
    discussion_economist_prompt,
    discussion_grid_prompt,
    economist_prompt,
    grid_prompt,
    price_retry_prompt,
    repair_economist_prompt,
    single_agent_prompt,
)


GRID_STRESS_LEVELS = (LOW_STRESS, MEDIUM_STRESS, HIGH_STRESS, EXTREMELY_HIGH_STRESS)
MODEL_RESPONSE_FAILED = "MODEL_RESPONSE_FAILED"
ECONOMIST_AGENT_OUTPUT_KEY = "_economist_agent_output"
AGENT_COMPLETION_USAGE_KEY = "_agent_completion_usage"
MIN_ELASTICITY = 0.05
MAX_ELASTICITY = 0.7
MULTI_AGENT_DISCUSSION_ROUNDS = 3
GRID_STRESS_LEVEL_BY_KEY = {level.lower(): level for level in GRID_STRESS_LEVELS}
GRID_STRESS_LEVEL_BY_KEY.update(
    {
        "moderate": "Medium",
        "critical": EXTREMELY_HIGH_STRESS,
        "critica": EXTREMELY_HIGH_STRESS,
        "extrame high": EXTREMELY_HIGH_STRESS,
        "extreme high": EXTREMELY_HIGH_STRESS,
        "extreme_high": EXTREMELY_HIGH_STRESS,
        "extremely_high": EXTREMELY_HIGH_STRESS,
    }
)


class AgentStageError(RuntimeError):
    def __init__(self, *, stage: str, zone_id: Any, agent: str, original: Exception) -> None:
        self.stage = stage
        self.zone_id = zone_id
        self.agent = agent
        self.original = original
        message = (
            f"{agent} failed for zone {zone_id} at {stage}: "
            f"{type(original).__name__}: {original}"
        )
        super().__init__(message)


class ProviderResponseDecodeError(RuntimeError):
    def __init__(
        self,
        *,
        model: str,
        attempts: int,
        original: json.JSONDecodeError,
    ) -> None:
        self.model = model
        self.attempts = attempts
        self.original = original
        super().__init__(
            f"Provider returned malformed or truncated HTTP JSON for model {model} "
            f"after {attempts} attempts. This occurred before model completion content "
            f"parsing; last_decode_error={original.msg} at line {original.lineno}, "
            f"column {original.colno}, char {original.pos}"
        )


class ChatClient(Protocol):
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AgentCallResult:
    content: dict[str, Any]
    usage: dict[str, Any]


@dataclass
class AgentChatClient:
    config: AgentConfig

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise ValueError(
                f"agent.{self.config.profile}.api_key is required when dry-run is disabled"
            )
        if self.config.max_concurrent_requests < 1:
            raise ValueError(
                f"agent.{self.config.profile}.max_concurrent_requests must be at least 1"
            )
        if self.config.provider_json_retries < 0:
            raise ValueError(
                f"agent.{self.config.profile}.provider_json_retries cannot be negative"
            )
        if self.config.provider_json_retry_backoff_seconds < 0:
            raise ValueError(
                f"agent.{self.config.profile}.provider_json_retry_backoff_seconds "
                "cannot be negative"
            )
        from openai import AsyncOpenAI

        headers = {}
        # if self.config.http_referer:
        #     headers["HTTP-Referer"] = self.config.http_referer
        # if self.config.title:
        #     headers["X-Title"] = self.config.title
        #     headers["X-OpenRouter-Title"] = self.config.title
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=headers or None,
            timeout=self.config.timeout_seconds,
        )
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }
        extra_body = reasoning_extra_body(self.config)
        if extra_body is not None:
            request["extra_body"] = extra_body
        response, provider_attempt_count = await self._create_completion_with_retry(request)
        content = completion_message_content(response, requested_model=self.config.model)
        payload = extract_json_object(content)
        token_usage = response_token_usage(response)
        token_usage["provider_attempt_count"] = provider_attempt_count
        token_usage["token_usage_complete"] = (
            provider_attempt_count == 1
            and all(
                token_usage.get(field) is not None
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        )
        payload[AGENT_COMPLETION_USAGE_KEY] = token_usage
        return payload

    async def aclose(self) -> None:
        """Close the underlying HTTP client on the event loop that used it."""

        client = getattr(self, "_client", None)
        if client is None or getattr(self, "_closed", False):
            return
        self._closed = True
        await client.close()

    async def __aenter__(self) -> "AgentChatClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    async def _create_completion_with_retry(
        self,
        request: dict[str, Any],
    ) -> tuple[Any, int]:
        retry_count = self.config.provider_json_retries
        semaphore = getattr(self, "_request_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            self._request_semaphore = semaphore

        for attempt_index in range(retry_count + 1):
            try:
                async with semaphore:
                    response = await self._client.chat.completions.create(**request)
                    return response, attempt_index + 1
            except json.JSONDecodeError as exc:
                if attempt_index >= retry_count:
                    raise ProviderResponseDecodeError(
                        model=self.config.model,
                        attempts=attempt_index + 1,
                        original=exc,
                    ) from exc
                delay = self.config.provider_json_retry_backoff_seconds * (2**attempt_index)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise AssertionError("provider response retry loop exited unexpectedly")


@dataclass
class DryRunChatClient:
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        raise RuntimeError("DryRunChatClient should not receive raw prompts")


def reasoning_extra_body(config: AgentConfig) -> dict[str, Any] | None:
    effort = str(config.reasoning_effort or "").strip()
    if not effort or config.model.strip().lower().startswith("meta-llama/"):
        return None
    return {"reasoning": {"effort": effort}}


def completion_message_content(response: Any, *, requested_model: str) -> str:
    choices = response_field(response, "choices")
    if not choices:
        details = provider_response_error_details(response)
        suffix = f"; {details}" if details else ""
        raise RuntimeError(
            f"Model provider returned no completion choices for model {requested_model}{suffix}"
        )
    choice = choices[0]
    message = response_field(choice, "message")
    if message is None:
        raise RuntimeError(
            f"Model provider returned a completion choice without a message for model {requested_model}"
        )
    content = response_field(message, "content")
    if not isinstance(content, str) or not content.strip():
        refusal = response_field(message, "refusal")
        refusal_text = f"; refusal={refusal}" if refusal else ""
        raise RuntimeError(
            f"Model provider returned empty completion content for model {requested_model}{refusal_text}"
        )
    return content


def provider_response_error_details(response: Any) -> str:
    error = response_field(response, "error")
    if error is None:
        model_extra = response_field(response, "model_extra")
        if isinstance(model_extra, dict):
            error = model_extra.get("error")
    if error is None:
        return ""
    code = response_field(error, "code")
    message = response_field(error, "message")
    if isinstance(error, str):
        message = error
    parts = []
    if code not in (None, ""):
        parts.append(f"provider_error_code={code}")
    if message not in (None, ""):
        parts.append(f"provider_error_message={str(message)[:500]}")
    return "; ".join(parts)


def response_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


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
        discussion_rounds.append(exchange)
        previous_exchange = discussion_exchange_for_prompt(exchange)
        final_grid = grid
        final_behavior = behavior
        final_economist = economist

    debug = {
        "zone_id": context.get("zone_id"),
        "discussion_round_count": MULTI_AGENT_DISCUSSION_ROUNDS,
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
    report["agent_discussion_round_count"] = MULTI_AGENT_DISCUSSION_ROUNDS
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
    }


def discussion_exchange_for_prompt(exchange: dict[str, Any]) -> dict[str, Any]:
    return {
        "discussion_round": exchange.get("discussion_round"),
        "grid_output": exchange.get("grid_output"),
        "behavior_output": exchange.get("behavior_output"),
        "economist_output": exchange.get("economist_output"),
        "communication_summary": exchange.get("communication_summary"),
    }


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
    validation_errors = validate_economist_report(response, context.get("pricing_windows_3h", []))
    usage = [result.usage]
    selected = response
    repair_response = None
    repair_errors = None
    if validation_errors:
        repaired = await complete_agent_json(
            client,
            repair_economist_prompt(
                context,
                grid,
                behavior,
                response,
                validation_errors,
            ),
            context=context,
            temperature=min(temperature, 0.1),
            stage="agent.single_agent.schema_repair",
            agent="Single Pricing Agent Schema Repair",
        )
        usage.append(repaired.usage)
        repair_response = repaired.content
        repair_errors = validate_economist_report(
            repair_response,
            context.get("pricing_windows_3h", []),
        )
        if not repair_errors:
            selected = repair_response
            grid = merge_grid_fallback(selected, context)
            behavior = merge_behavior_fallback(selected, context)
    economist = merge_economist_fallback(selected, context)
    debug = {
        "zone_id": context.get("zone_id"),
        "category": context.get("category"),
        "forecast_start": context.get("forecast_start"),
        "forecast_end": context.get("forecast_end"),
        "expected_price_window_count": len(context.get("pricing_windows_3h", []))
        if isinstance(context.get("pricing_windows_3h"), list)
        else 0,
        "single_agent_response": response,
        "single_agent_validation_errors": validation_errors,
        "repair_response": repair_response,
        "repair_validation_errors": repair_errors,
        "selected_response_source": "repair" if selected is repair_response else "single_agent",
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
            previous_report.get("economist_output") or previous_report,
            single_agent=single_agent,
        ),
        context=context,
        temperature=temperature,
        stage=stage,
        agent=agent_name,
    )
    selected = result.content
    usage = [result.usage]
    validation_errors = validate_economist_report(
        selected,
        context.get("pricing_windows_3h", []),
    )
    repair_response = None
    repair_errors = None
    if validation_errors:
        repaired = await complete_agent_json(
            client,
            repair_economist_prompt(
                context,
                grid,
                behavior,
                selected,
                validation_errors,
            ),
            context=context,
            temperature=min(temperature, 0.1),
            stage=f"{stage}.schema_repair",
            agent=f"{agent_name} Schema Repair",
        )
        usage.append(repaired.usage)
        repair_response = repaired.content
        repair_errors = validate_economist_report(
            repair_response,
            context.get("pricing_windows_3h", []),
        )
        if not repair_errors:
            selected = repair_response
    economist = merge_economist_fallback(selected, context)
    debug = {
        "zone_id": context.get("zone_id"),
        "retry": True,
        "single_agent": single_agent,
        "response": result.content,
        "validation_errors": validation_errors,
        "repair_response": repair_response,
        "repair_validation_errors": repair_errors,
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


async def complete_agent_json(
    client: ChatClient,
    prompt: str,
    *,
    context: dict[str, Any],
    temperature: float,
    stage: str,
    agent: str,
) -> AgentCallResult:
    started = time.perf_counter()
    try:
        response = await client.complete_json(prompt, temperature=temperature)
    except AgentStageError:
        raise
    except Exception as exc:
        raise AgentStageError(
            stage=stage,
            zone_id=context.get("zone_id"),
            agent=agent,
            original=exc,
        ) from exc
    elapsed = round(time.perf_counter() - started, 4)
    token_usage = response.pop(AGENT_COMPLETION_USAGE_KEY, {})
    response = sanitize_structured_agent_output(response)
    return AgentCallResult(
        content=response,
        usage=agent_call_usage_record(
            stage=stage,
            agent=agent,
            zone_id=context.get("zone_id"),
            elapsed_seconds=elapsed,
            token_usage=token_usage,
        ),
    )


def sanitize_structured_agent_output(value: Any) -> Any:
    """Persist decision fields and summaries, never model-supplied hidden reasoning."""

    blocked = {"chain_of_thought", "chain-of-thought", "thoughts", "analysis", "reasoning"}
    if isinstance(value, dict):
        return {
            key: sanitize_structured_agent_output(item)
            for key, item in value.items()
            if str(key).strip().lower() not in blocked
        }
    if isinstance(value, list):
        return [sanitize_structured_agent_output(item) for item in value]
    return value


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
    report = initial_result.content
    call_usage = [initial_result.usage]
    errors = validate_economist_report(report, expected_windows)
    debug: dict[str, Any] = {
        "zone_id": context.get("zone_id"),
        "category": context.get("category"),
        "forecast_start": context.get("forecast_start"),
        "forecast_end": context.get("forecast_end"),
        "expected_price_window_count": len(expected_windows) if isinstance(expected_windows, list) else 0,
        "initial_response": report,
        "initial_validation_errors": errors,
        "repair_attempted": False,
        "repair_response": None,
        "repair_validation_errors": None,
        "selected_response_source": "initial",
        "selected_validation_errors": errors,
    }
    if not errors:
        debug["agent_call_usage"] = call_usage
        debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
        return attach_economist_debug(report, debug), call_usage

    repair_result = await complete_agent_json(
        client,
        repair_economist_prompt(context, grid, behavior, report, errors),
        context=context,
        temperature=min(temperature, 0.1),
        stage=repair_stage,
        agent=repair_agent,
    )
    repaired = repair_result.content
    call_usage.append(repair_result.usage)
    repair_errors = validate_economist_report(repaired, expected_windows)
    debug.update(
        {
            "repair_attempted": True,
            "repair_response": repaired,
            "repair_validation_errors": repair_errors,
        }
    )
    if not repair_errors or len(repair_errors) < len(errors):
        debug["selected_response_source"] = "repair"
        debug["selected_validation_errors"] = repair_errors
        debug["agent_call_usage"] = call_usage
        debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
        return attach_economist_debug(repaired, debug), call_usage
    debug["agent_call_usage"] = call_usage
    debug["agent_usage_summary"] = summarize_agent_call_usage(call_usage)
    return attach_economist_debug(report, debug), call_usage


def attach_economist_debug(report: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(report)
    tagged[ECONOMIST_AGENT_OUTPUT_KEY] = debug
    return tagged


def response_token_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = usage_value(usage, "prompt_tokens", "input_tokens")
    completion_tokens = usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = usage_value(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def usage_value(usage: Any, *keys: str) -> int | None:
    if usage is None:
        return None
    for key in keys:
        value = None
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def agent_call_usage_record(
    *,
    stage: str,
    agent: str,
    zone_id: Any,
    elapsed_seconds: float,
    token_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    usage = token_usage if isinstance(token_usage, dict) else {}
    prompt_tokens = optional_int_usage(usage.get("prompt_tokens"))
    completion_tokens = optional_int_usage(usage.get("completion_tokens"))
    total_tokens = optional_int_usage(usage.get("total_tokens"))
    provider_attempt_count = optional_int_usage(usage.get("provider_attempt_count")) or 1
    explicit_complete = usage.get("token_usage_complete")
    token_usage_complete = (
        bool(explicit_complete)
        if explicit_complete is not None
        else all(
            value is not None
            for value in (prompt_tokens, completion_tokens, total_tokens)
        )
    )
    return {
        "stage": stage,
        "agent": agent,
        "zone_id": zone_id,
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_complete": token_usage_complete,
        "provider_attempt_count": provider_attempt_count,
    }


def summarize_agent_call_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "agent_invoked": bool(records),
        "agent_call_count": len(records),
        "elapsed_seconds": round(sum(optional_float_usage(record.get("elapsed_seconds")) for record in records), 4),
        "prompt_tokens": sum_token_usage(records, "prompt_tokens"),
        "completion_tokens": sum_token_usage(records, "completion_tokens"),
        "total_tokens": sum_token_usage(records, "total_tokens"),
        "token_usage_complete": all(
            call_token_usage_complete(record)
            for record in records
        ),
    }


def call_token_usage_complete(record: dict[str, Any]) -> bool:
    explicit = record.get("token_usage_complete")
    if explicit is not None:
        return bool(explicit)
    return all(
        optional_int_usage(record.get(field)) is not None
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def sum_token_usage(records: list[dict[str, Any]], field: str) -> int:
    total = 0
    for record in records:
        value = optional_int_usage(record.get(field))
        if value is not None:
            total += value
    return total


def optional_int_usage(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float_usage(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def heuristic_zone_chain(
    context: dict[str, Any],
    *,
    source: str = "dry-run",
) -> dict[str, Any]:
    grid = heuristic_grid(context)
    behavior = heuristic_behavior(context)
    economist = heuristic_economist(context, grid)
    return combine_reports(context, grid, behavior, economist, source=source)


def heuristic_discussion_zone_chain(
    context: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    discussion_rounds: list[dict[str, Any]] = []
    final_grid: dict[str, Any] = {}
    final_behavior: dict[str, Any] = {}
    final_economist: dict[str, Any] = {}
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
        discussion_rounds.append(
            discussion_round_record(
                discussion_round=discussion_round,
                grid=grid,
                behavior=behavior,
                economist=economist,
                economist_debug=None,
                agent_call_usage=[],
            )
        )
        final_grid = grid
        final_behavior = behavior
        final_economist = economist
    report = combine_reports(
        context,
        final_grid,
        final_behavior,
        final_economist,
        source=source,
        agent_call_usage=[],
    )
    report["agent_discussion_round_count"] = MULTI_AGENT_DISCUSSION_ROUNDS
    report["agent_discussion_rounds"] = discussion_rounds
    return report


def heuristic_grid(context: dict[str, Any]) -> dict[str, Any]:
    windows = context.get("pricing_windows_3h") or []
    return {
        "forecast_total_kwh": context["forecast_total_kwh"],
        "forecast_peak_kwh": context["forecast_peak_kwh"],
        "predicted_change_pct": context["predicted_change_pct"],
        "grid_stress_level": normalize_grid_stress_level(context.get("grid_stress_level"), "Low"),
        "forecast_summary": (
            f"{context['category']} zone forecast is anchored to the prior-week hourly "
            f"shape with a {context['predicted_change_pct']:+.1f}% change versus the "
            "recent comparable window."
        ),
        "reasoning_summary": (
            "The forecaster is the numerical source of truth; pricing is required "
            "for each window outside the Medium band."
        ),
        "adjustment_needed": any(
            normalize_grid_stress_level(window.get("load_stress_level"), LOW_STRESS)
            != MEDIUM_STRESS
            for window in windows
        ),
        "confidence": "medium",
        "window_assessments": [
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "predicted_load_kwh": window.get("sum_predicted_kwh"),
                "load_range_position_pct": window.get("load_range_position_pct"),
                "grid_stress_level": window.get("load_stress_level"),
                "adjustment_needed": window.get("load_stress_level") != MEDIUM_STRESS,
                "reasoning_summary": "Assessment uses the forecast-derived load position.",
            }
            for window in windows
        ],
    }


def heuristic_behavior(context: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    weather = context["weather"]
    shape = context["hourly_shape"]
    windows = context.get("pricing_windows_3h") or []
    horizon_days = context.get("forecast_horizon_days", "configured")
    forecast_start = str(context.get("forecast_start") or "the forecast start")
    forecast_end = str(context.get("forecast_end") or "the forecast end")
    change = as_float(context.get("predicted_change_pct"), 0)
    drivers = []
    high_windows = [
        window
        for window in windows
        if normalize_grid_stress_level(window.get("load_stress_level") or window.get("grid_stress_level"), "Low")
        in {HIGH_STRESS, EXTREMELY_HIGH_STRESS}
    ]
    peak_window = max(windows, key=lambda window: as_float(window.get("sum_predicted_kwh"), 0), default=None)
    if high_windows:
        drivers.append(f"{len(high_windows)} high-stress 3-hour windows")
    if peak_window:
        drivers.append(
            "peak window "
            f"{peak_window.get('window_start')} to {peak_window.get('window_end')} "
            f"at {as_float(peak_window.get('sum_predicted_kwh'), 0):.1f} kWh"
        )
    if abs(change) >= 5:
        drivers.append(f"{change:+.1f}% load change versus comparable history")
    else:
        drivers.append("stable total load versus comparable history")
    if weather["rain_hours"] > 0:
        drivers.append(f"{weather['rain_hours']} rainy hours")
    if shape["night_20_6"] >= shape["morning_7_10"]:
        drivers.append("night plateau")
    if shape["evening_17_22"] >= shape["morning_7_10"]:
        drivers.append("evening lift")
    profile = context.get("profile") or {}
    poi_total = profile.get("poi_total", 0)
    return {
        "agent_reasoning": (
            f"For the {horizon_days}-day window from {forecast_start} to {forecast_end}, "
            f"{category} demand is best explained by {', '.join(drivers)} and the "
            f"local POI mix of {poi_total} assigned POIs."
        ),
        "reasoning_summary": (
            f"Demand is explained by {', '.join(drivers)} together with the local "
            "static and temporal context."
        ),
        "demand_drivers": drivers,
        "elasticity_factor": estimate_elasticity_factor(context, {}, peak_window or {}),
        "confidence": "medium",
    }


def heuristic_economist(context: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    stress = grid["grid_stress_level"]
    change = float(grid["predicted_change_pct"])
    shift, label = base_price_shift(stress, category, change)
    window_shifts = build_heuristic_price_windows(context, grid)
    return {
        "suggested_price_shift_pct": shift,
        "action_label": label,
        "reasoning_summary": (
            f"The schedule responds to {stress} forecast stress and the estimated "
            f"price sensitivity of the {category} zone."
        ),
        "price_rationale": (
            f"{stress} stress with {change:+.1f}% expected load change; "
            f"category elasticity proxy is {category}."
        ),
        "price_change_windows_3h": window_shifts,
    }


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
        "action_label": str(
            economist.get("action_label") or final_price_action_label(final_shift)
        ),
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
        "agent_time_cost_seconds": usage_summary["elapsed_seconds"],
        "agent_invoked": usage_summary["agent_invoked"],
        "agent_call_count": usage_summary["agent_call_count"],
        "agent_prompt_tokens": usage_summary["prompt_tokens"],
        "agent_completion_tokens": usage_summary["completion_tokens"],
        "agent_total_tokens": usage_summary["total_tokens"],
        "agent_token_usage_complete": usage_summary["token_usage_complete"],
        "agent_call_usage": agent_call_usage or [],
        "source": source,
    }
    if economist_debug is not None:
        report[ECONOMIST_AGENT_OUTPUT_KEY] = economist_debug
    return report


def finalize_price_window(window: dict[str, Any]) -> dict[str, Any]:
    updated = dict(window)
    base_price = optional_float(window.get("mean_energy_price"))
    shift = optional_float(window.get("suggested_price_shift_pct"))
    proposed = optional_float(window.get("proposed_energy_price"))
    if proposed is None and base_price is not None and shift is not None:
        proposed = base_price * (1.0 + shift / 100.0)
    if shift is None and proposed is not None and base_price not in (None, 0):
        shift = ((proposed / base_price) - 1.0) * 100.0
    updated["suggested_price_shift_pct"] = round(float(shift or 0.0), 4)
    updated["predicted_energy_price"] = round(proposed, 4) if proposed is not None else None
    updated["proposed_energy_price"] = updated["predicted_energy_price"]
    updated["price_valid"] = proposed is not None and proposed >= 0
    updated["price_validation_error"] = (
        None if updated["price_valid"] else "proposed energy price must be non-negative"
    )
    return updated


def merge_grid_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_grid(context)
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    merged["grid_stress_level"] = normalize_grid_stress_level(
        merged.get("grid_stress_level"),
        fallback["grid_stress_level"],
    )
    return merged


def merge_behavior_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_behavior(context)
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


def merge_economist_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_economist(context, heuristic_grid(context))
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    if not isinstance(report.get("price_change_windows_3h"), list):
        merged["price_change_windows_3h"] = []
    if not has_text(report.get("action_label")):
        merged["action_label"] = model_response_failed("missing action_label")
    if not has_text(report.get("price_rationale")):
        merged["price_rationale"] = model_response_failed("missing price_rationale")
    return merged


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def validate_economist_report(report: dict[str, Any], expected_windows: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["response is not a JSON object"]

    if not is_number_like(report.get("suggested_price_shift_pct")):
        errors.append("missing or invalid suggested_price_shift_pct")
    if not has_text(report.get("reasoning_summary")):
        errors.append("missing reasoning_summary")
    if not has_text(report.get("action_label")):
        errors.append("missing action_label")
    if not has_text(report.get("price_rationale")):
        errors.append("missing price_rationale")

    expected = expected_windows if isinstance(expected_windows, list) else []
    actual = report.get("price_change_windows_3h")
    if not isinstance(actual, list):
        errors.append("missing price_change_windows_3h")
        return errors
    if len(actual) != len(expected):
        errors.append(f"price_change_windows_3h length {len(actual)} != expected {len(expected)}")

    for idx, expected_window in enumerate(expected):
        if idx >= len(actual) or not isinstance(actual[idx], dict):
            errors.append(f"price_change_windows_3h[{idx}] missing")
            continue
        item = actual[idx]
        for field in ("window_start", "window_end", "action_label", "price_rationale"):
            if not has_text(item.get(field)):
                errors.append(f"price_change_windows_3h[{idx}] missing {field}")
        if not is_number_like(item.get("suggested_price_shift_pct")):
            errors.append(f"price_change_windows_3h[{idx}] missing or invalid suggested_price_shift_pct")
        if not is_number_like(item.get("proposed_energy_price")):
            errors.append(f"price_change_windows_3h[{idx}] missing or invalid proposed_energy_price")
        for field in ("window_start", "window_end"):
            expected_value = expected_window.get(field) if isinstance(expected_window, dict) else None
            if has_text(expected_value) and has_text(item.get(field)) and str(item.get(field)) != str(expected_value):
                errors.append(f"price_change_windows_3h[{idx}] {field} does not match context")
    return errors


def is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def as_float(value: Any, fallback: float) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return round(float(fallback), 2)


def base_price_shift(stress: str, category: str, change: float) -> tuple[int, str]:
    if stress == EXTREMELY_HIGH_STRESS and category == "Residential":
        return 15, "Raise energy price for peak deflection"
    if stress == EXTREMELY_HIGH_STRESS:
        return 15, "Raise energy price for congestion"
    if stress == HIGH_STRESS:
        return 8, "Raise energy price for high load"
    if stress == LOW_STRESS:
        return -8, "Reduce energy price for utilization"
    return 0, "Hold energy price"


def build_heuristic_price_windows(
    context: dict[str, Any],
    grid: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = context.get("pricing_windows_3h") or []
    averages = context.get("hourly_averages") or {}
    mean_load = as_float(averages.get("mean_predicted_kwh"), 0)
    mean_energy_price = as_float(averages.get("mean_energy_price"), 0)
    stress = normalize_grid_stress_level(grid.get("grid_stress_level"), context.get("grid_stress_level"))
    category = str(context.get("category") or "")
    change = as_float(grid.get("predicted_change_pct"), context.get("predicted_change_pct", 0))
    results = []
    for window in windows:
        window_stress = normalize_grid_stress_level(
            window.get("load_stress_level") or window.get("grid_stress_level"),
            stress,
        )
        window_load = as_float(window.get("mean_predicted_kwh"), mean_load)
        window_energy_price = as_float(window.get("mean_energy_price"), mean_energy_price)
        elasticity = estimate_elasticity_factor(context, {}, window)
        shift = round(price_shift_to_medium_band(context, window, elasticity), 2)
        if shift > 0:
            label = "Raise energy price"
        elif shift < 0:
            label = "Reduce energy price"
        else:
            label = "Hold window energy price"
        results.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                "mean_predicted_kwh": window.get("mean_predicted_kwh"),
                "mean_service_price": window.get("mean_service_price"),
                "mean_energy_price": window.get("mean_energy_price"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "historical_min_load_3h_kwh": window.get("historical_min_load_3h_kwh"),
                "historical_max_load_3h_kwh": window.get("historical_max_load_3h_kwh"),
                "historical_load_range_3h_kwh": window.get("historical_load_range_3h_kwh"),
                "load_range_position_pct": window.get("load_range_position_pct"),
                "low_medium_threshold_pct": window.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": window.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": window.get(
                    "high_extremely_high_threshold_pct"
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
                "suggested_price_shift_pct": shift,
                "action_label": label,
                "price_rationale": (
                    f"{window_stress} 3-hour load at "
                    f"{as_float(window.get('load_range_position_pct'), 0):.2f}% through the "
                    f"historical 3-hour load range; "
                    f"price shift targets the Medium band while keeping the resulting "
                    f"energy price non-negative from the {window_energy_price:.4f} baseline."
                ),
            }
        )
    return results


def normalize_price_windows(value: Any, fallback_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = []
    normalized = []
    for idx, fallback in enumerate(fallback_windows):
        item_missing = idx >= len(value) or not isinstance(value[idx], dict)
        item = value[idx] if not item_missing else {}
        shift = as_float(item.get("suggested_price_shift_pct"), 0)
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
                "low_medium_threshold_pct": fallback.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": fallback.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": fallback.get(
                    "high_extremely_high_threshold_pct"
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
                "proposed_energy_price": item.get("proposed_energy_price"),
                "action_label": required_model_text(item, "action_label", item_missing),
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
        f"{average_shift_pct:+.2f}% change to move or keep load in the 35%-80% Medium "
        f"band while keeping every proposed energy price non-negative."
    )


def window_predicted_load(window: dict[str, Any]) -> float:
    for key in ("price_conditioned_baseline_load_kwh", "sum_predicted_kwh"):
        load = optional_float(window.get(key))
        if load is not None:
            return max(0.0, load)
    mean_load = optional_float(window.get("mean_predicted_kwh")) or 0.0
    hours = optional_float(window.get("hours")) or 1.0
    return max(0.0, mean_load * hours)


def window_predicted_load_source(window: dict[str, Any]) -> str:
    for key, source in (
        ("price_conditioned_baseline_load_kwh", "price_conditioned_forecast_sum_predicted_kwh"),
        ("sum_predicted_kwh", "forecast_sum_predicted_kwh"),
    ):
        if optional_float(window.get(key)) is not None:
            return source
    return "forecast_mean_predicted_kwh_times_hours"


def historical_load_bounds_kwh(
    context: dict[str, Any],
    window: dict[str, Any],
) -> tuple[float, float, str]:
    minimum = first_nonnegative_float(
        window.get("historical_min_load_3h_kwh"),
        context.get("historical_min_load_3h_kwh"),
    )
    maximum = first_positive_float(
        window.get("historical_max_load_3h_kwh"),
        context.get("historical_max_load_3h_kwh"),
    )
    if minimum is not None and maximum is not None and maximum > minimum:
        return minimum, maximum, "zone_historical_3h_load_min_max_range"

    legacy_q95 = first_positive_float(
        window.get("load_3h_q95_reference_kwh"),
        window.get("load_3h_q95_kwh"),
        context.get("grid_stress_q95_reference_kwh"),
        context.get("grid_stress_q95_kwh"),
    )
    if legacy_q95 is not None:
        return 0.0, legacy_q95, "legacy_zero_to_q95_fallback"

    baseline = max(window_predicted_load(window), 0.0)
    return 0.0, max(baseline, 1.0), "fallback_zero_to_baseline_predicted_load"


def price_shift_to_medium_band(
    context: dict[str, Any],
    window: dict[str, Any],
    elasticity: float,
) -> float:
    baseline_load = window_predicted_load(window)
    historical_min_load, historical_max_load, _ = historical_load_bounds_kwh(
        context, window
    )
    if baseline_load <= 0 or historical_max_load <= historical_min_load or elasticity <= 0:
        return max(-100.0, as_float(window.get("suggested_price_shift_pct"), 0))

    load_position_pct = load_range_position_pct(
        baseline_load, historical_min_load, historical_max_load
    )
    stress = classify_load_percentage(load_position_pct)
    target_load = target_medium_load(
        baseline_load, historical_min_load, historical_max_load
    )
    raw_shift = as_float(window.get("suggested_price_shift_pct"), 0)

    if stress == MEDIUM_STRESS:
        candidate = clamp(
            raw_shift,
            -3.0,
            3.0,
        )
        expected_load = expected_load_after_price(baseline_load, candidate, elasticity)
        lower_load, upper_load = medium_load_bounds(
            historical_min_load, historical_max_load
        )
        if not lower_load <= expected_load < upper_load:
            candidate = 0.0
    else:
        candidate = ((1.0 - (target_load / baseline_load)) / elasticity) * 100.0

    return max(-100.0, candidate)


def final_price_action_label(shift_pct: float) -> str:
    if shift_pct > 0:
        return "Raise energy price to reduce load toward Medium"
    if shift_pct < 0:
        return "Reduce energy price to increase load toward Medium"
    return "Hold energy price in Medium load band"


def estimate_elasticity_factor(
    context: dict[str, Any],
    behavior: dict[str, Any],
    window: dict[str, Any],
) -> float:
    window_value = optional_float(window.get("elasticity_factor"))
    if window_value is not None:
        return clamp(abs(window_value), MIN_ELASTICITY, MAX_ELASTICITY)

    reported = optional_float(behavior.get("elasticity_factor"))
    if reported is not None:
        return clamp(abs(reported), MIN_ELASTICITY, MAX_ELASTICITY)

    category = str(context.get("category") or "").lower()
    if "residential" in category:
        base = 0.45
    elif "commercial" in category or "mall" in category:
        base = 0.32
    elif "cbd" in category or "office" in category:
        base = 0.24
    elif "transport" in category or "hub" in category:
        base = 0.2
    elif "industrial" in category:
        base = 0.18
    else:
        base = 0.28

    occupancy = normalized_rate(window.get("mean_occupancy"))
    rain = optional_float(window.get("total_rain"))
    if rain is None:
        weather = context.get("weather") if isinstance(context.get("weather"), dict) else {}
        rain = optional_float(weather.get("rain_hours")) or 0.0
    occupancy_factor = 1.0 - 0.25 * occupancy
    rain_factor = 0.85 if rain > 0 else 1.0
    return clamp(base * occupancy_factor * rain_factor, MIN_ELASTICITY, MAX_ELASTICITY)


def expected_load_after_price(baseline_load: float, price_shift_pct: float, elasticity: float) -> float:
    response = elasticity * (price_shift_pct / 100.0)
    return max(0.0, baseline_load * (1.0 - response))


def normalized_rate(value: Any) -> float:
    number = optional_float(value)
    if number is None:
        return 0.0
    if number > 1.0:
        number /= 100.0
    return clamp(number, 0.0, 1.0)


def first_positive_float(*values: Any) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None and number > 0:
            return number
    return None


def first_nonnegative_float(*values: Any) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None and number >= 0:
            return number
    return None


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def required_model_text(item: dict[str, Any], field: str, item_missing: bool) -> str:
    if item_missing:
        return model_response_failed("missing price_change_windows_3h item")
    value = item.get(field)
    return str(value).strip() if has_text(value) else model_response_failed(f"missing {field}")


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def has_text(value: Any) -> bool:
    return bool(str(value).strip()) if value is not None else False


def model_response_failed(reason: str) -> str:
    return f"{MODEL_RESPONSE_FAILED}: {reason}"


def normalize_grid_stress_level(value: Any, fallback: Any = "Low") -> str:
    key = str(value or "").strip().lower()
    if key in GRID_STRESS_LEVEL_BY_KEY:
        return GRID_STRESS_LEVEL_BY_KEY[key]
    fallback_key = str(fallback or "").strip().lower()
    return GRID_STRESS_LEVEL_BY_KEY.get(fallback_key, "Low")
