"""Native Anthropic Messages adapter; no OpenAI request or response assumptions."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from config import AgentConfig, DEFAULT_ANTHROPIC_MAX_TOKENS


def anthropic_base_url(base_url: str) -> str:
    """The SDK appends /v1/messages, including behind a gateway path prefix."""
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    if path.endswith("/messages"):
        raise ValueError("Anthropic base_url must be a service root, not a /messages endpoint")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def create_anthropic_client(config: AgentConfig) -> Any:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=config.client_api_key,
        base_url=anthropic_base_url(config.base_url),
        timeout=config.timeout_seconds,
        # All attempts pass through the shared limiter and are counted by our
        # retry loop. SDK-internal retries would hide consumed/unknown tokens.
        max_retries=0,
    )


def anthropic_output_format(
    response_format: dict[str, Any], *, output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate JSON mode using a concrete Agent contract, never an empty object.

    Anthropic has no arbitrary-key json_object mode. The caller supplies its
    stage's schema, or an explicit OpenAI-style json_schema response format.
    """
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be a mapping")
    kind = response_format.get("type")
    if kind == "json_object":
        schema = output_schema
        if not isinstance(schema, dict) or not schema.get("properties"):
            raise ValueError(
                "Anthropic json_object requires a concrete output_schema; "
                "use complete_agent_json for automatic Agent stage schemas, "
                "or supply a json_schema response_format"
            )
    elif kind == "json_schema":
        specification = response_format.get("json_schema")
        schema = specification.get("schema") if isinstance(specification, dict) else None
    else:
        raise ValueError("Anthropic response_format.type must be json_object or json_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Anthropic Agent output schema must describe a JSON object")
    from anthropic import transform_schema

    try:
        json.dumps(schema, allow_nan=False)
        transformed = transform_schema(deepcopy(schema))
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError) as exc:
        raise ValueError(f"Invalid Anthropic output schema: {exc}") from exc
    return {"type": "json_schema", "schema": transformed}


def anthropic_request(
    config: AgentConfig, prompt: str, *, system: str,
    response_format: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    effort = config.reasoning_effort
    if config.anthropic_thinking is not None:
        request["thinking"] = deepcopy(config.anthropic_thinking)
    elif effort is not None:
        request["thinking"] = {"type": "disabled" if effort == "none" else "adaptive"}
    if effort is not None and effort != "none":
        request["output_config"] = {"effort": effort}
    if response_format is not None:
        request.setdefault("output_config", {})["format"] = anthropic_output_format(
            response_format, output_schema=output_schema,
        )
    if config.anthropic_cache_control is not None:
        request["cache_control"] = deepcopy(config.anthropic_cache_control)
    # SDK 2.x has no temperature/json_object arguments. JSON mode uses native
    # output_config.format; existing validators still check window identities,
    # counts and pricing rules that cannot be guaranteed by a static schema.
    return request


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def anthropic_message_content(response: Any, *, requested_model: str) -> str:
    stop = _field(response, "stop_reason")
    if stop == "max_tokens":
        raise RuntimeError(
            f"Anthropic completion for model {requested_model} was truncated by max_tokens; "
            "increase agent.max_tokens before retrying"
        )
    if stop in {"refusal", "tool_use", "pause_turn"}:
        raise RuntimeError(f"Anthropic returned stop_reason={stop} for model {requested_model}; expected final JSON text")
    text = "".join(
        _field(block, "text")
        for block in (_field(response, "content") or [])
        if _field(block, "type") == "text" and isinstance(_field(block, "text"), str)
    )
    if not text.strip():
        raise RuntimeError(f"Anthropic returned no final text content for model {requested_model}")
    return text


def is_retryable_anthropic_error(error: Exception) -> bool:
    from anthropic import APIConnectionError, APIStatusError

    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        # Match the SDK's retry contract, including an explicit server override.
        directive = error.response.headers.get("x-should-retry")
        if directive in {"true", "false"}:
            return directive == "true"
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    return False
