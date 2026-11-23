"""Provider request adapters, output budgets, and native response handling."""

from __future__ import annotations

import copy
import json
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.agents.fields import response_field
from src.config.agent import AgentConfig, DEFAULT_ANTHROPIC_MAX_TOKENS


def openai_request(
    config: AgentConfig, prompt: str, *, system: str,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Omit sampling/vendor extensions; leave model support checks to the API."""
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "response_format": deepcopy(response_format) if response_format is not None else {"type": "json_object"},
        **output_token_limit_params(config),
    }
    effort = config.reasoning_effort
    if isinstance(effort, str):
        effort = effort.strip()
    if effort is not None and effort != "":
        request["reasoning_effort"] = effort
    return request


def output_token_limit_params(config: AgentConfig) -> dict[str, int]:
    """Keep one config budget and send exactly the selected backend's field.

    Model names do not identify a gateway's wire protocol. Generic compatible
    endpoints retain max_tokens; select openai for gateways requiring the newer
    max_completion_tokens parameter. Anthropic requires an explicit budget.
    """
    backend = config.resolved_backend
    limit = config.max_tokens
    if limit is None:
        if backend != "anthropic":
            return {}
        limit = DEFAULT_ANTHROPIC_MAX_TOKENS
    parameter = "max_completion_tokens" if backend == "openai" else "max_tokens"
    return {parameter: limit}


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
        **output_token_limit_params(config),
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


def anthropic_message_content(response: Any, *, requested_model: str) -> str:
    stop = response_field(response, "stop_reason")
    if stop == "max_tokens":
        raise RuntimeError(
            f"Anthropic completion for model {requested_model} was truncated by max_tokens; "
            "increase agent.max_tokens before retrying"
        )
    if stop in {"refusal", "tool_use", "pause_turn"}:
        raise RuntimeError(f"Anthropic returned stop_reason={stop} for model {requested_model}; expected final JSON text")
    text = "".join(
        response_field(block, "text")
        for block in (response_field(response, "content") or [])
        if response_field(block, "type") == "text" and isinstance(response_field(block, "text"), str)
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


def reasoning_extra_body(config: AgentConfig) -> dict[str, Any] | None:
    if config.resolved_backend != "openrouter":
        return None
    effort = str(config.reasoning_effort or "").strip()
    if not effort or config.model.strip().lower().startswith("meta-llama/"):
        return None
    return {"reasoning": {"effort": effort}}


def request_extra_body(config: AgentConfig) -> dict[str, Any] | None:
    """Merge provider defaults with explicit extensions without mutating config."""
    if config.resolved_backend in {"openai", "anthropic"}:
        return None
    body = reasoning_extra_body(config) or {}
    body.update(copy.deepcopy(config.extra_body))
    template_kwargs = body.get("chat_template_kwargs", copy.deepcopy(config.chat_template_kwargs or {}))
    if config.resolved_backend in {"vllm", "sglang"}:
        effort = str(body.get("reasoning_effort", config.reasoning_effort) or "").strip().lower()
        if effort:
            # The SDK merges extra_body into the top-level HTTP JSON. Local
            # servers expose this field directly, unlike OpenRouter's envelope.
            body.setdefault("reasoning_effort", effort)
            # Templates such as Qwen3.5 use an on/off switch. An explicitly
            # supplied template option takes precedence over this default.
            if isinstance(template_kwargs, dict):
                template_kwargs.setdefault("enable_thinking", effort != "none")
    if template_kwargs or "chat_template_kwargs" in body:
        body["chat_template_kwargs"] = template_kwargs
    return body or None


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
