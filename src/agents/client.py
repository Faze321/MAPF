from __future__ import annotations

import asyncio
import copy
import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agents.limits import SharedRequestLimiter
from src.agents.monitoring import SGLangMetricsCollector
from src.agents.prompts import SYSTEM_MESSAGE
from src.agents.providers import (
    ProviderResponseDecodeError,
    anthropic_base_url,
    anthropic_message_content,
    anthropic_request,
    completion_message_content,
    create_anthropic_client,
    is_retryable_anthropic_error,
    openai_request,
    output_token_limit_params,
    request_extra_body,
)
from src.agents.usage import (
    agent_call_usage_record,
    anthropic_response_token_usage,
    cache_usage_fields,
    response_token_usage,
)
from src.config.agent import AgentConfig


AGENT_COMPLETION_USAGE_KEY = "_agent_completion_usage"


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
        api_key = self.config.client_api_key
        if not api_key:
            raise ValueError(
                f"agent.{self.config.profile}.api_key is required when dry-run is disabled"
            )
        self._uses_anthropic = self.config.resolved_backend == "anthropic"
        limiter_url = self.config.base_url
        if self._uses_anthropic:
            self._client = create_anthropic_client(self.config)
            limiter_url = anthropic_base_url(self.config.base_url)
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                default_headers=None,
                timeout=self.config.timeout_seconds,
            )
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._shared_request_limiter = (
            SharedRequestLimiter(limiter_url, self.config.max_concurrent_requests_total)
            if self.config.max_concurrent_requests_total is not None
            else None
        )

    async def start_metrics(self, output_dir: Path, *, context: dict[str, Any] | None = None) -> None:
        """Collect server metrics only for an explicitly selected SGLang run."""
        if (self.config.backend != "sglang" or not self.config.sglang_metrics.get("enabled", False)
                or getattr(self, "_metrics_collector", None) is not None or getattr(self, "_closed", False)):
            return
        try:
            self._metrics_collector = SGLangMetricsCollector(
                base_url=self.config.base_url, api_key=self.config.api_key,
                output_dir=output_dir / "sglang_metrics", settings=self.config.sglang_metrics,
                context={**(context or {}), "agent_model": self.config.model, "agent_profile": self.config.profile},
            )
            await self._metrics_collector.start()
        except Exception as exc:
            # Never include exception text: URLs or credentials may occur in it.
            print(f"SGLang metrics could not start ({type(exc).__name__}); continuing experiment.", flush=True)

    @property
    def metrics_outputs(self) -> dict[str, Path]:
        collector = getattr(self, "_metrics_collector", None)
        return dict(collector.outputs) if collector is not None else {}

    async def complete_json(
        self, prompt: str, *, temperature: float,
        response_format: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._uses_anthropic:
            request = anthropic_request(
                self.config, prompt, system=SYSTEM_MESSAGE,
                response_format=response_format if response_format is not None else {"type": "json_object"},
                output_schema=output_schema,
            )
        elif self.config.resolved_backend == "openai":
            request = openai_request(
                self.config, prompt, system=SYSTEM_MESSAGE,
                response_format=response_format,
            )
        else:
            request = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                "response_format": copy.deepcopy(response_format) if response_format is not None else {"type": "json_object"},
                "temperature": temperature,
            }
            request.update(output_token_limit_params(self.config))
            extra_body = request_extra_body(self.config)
            if extra_body is not None:
                request["extra_body"] = extra_body
        response, provider_attempt_count = await self._create_completion_with_retry(request)
        if self._uses_anthropic:
            content = anthropic_message_content(response, requested_model=self.config.model)
            token_usage = anthropic_response_token_usage(response)
        else:
            content = completion_message_content(response, requested_model=self.config.model)
            token_usage = response_token_usage(response)
        payload = extract_json_object(content)
        token_usage["provider_attempt_count"] = provider_attempt_count
        if provider_attempt_count > 1:
            # Failed provider responses may have consumed unreported cache/input tokens.
            token_usage["cache_usage_complete"] = False
            token_usage.update(cache_usage_fields(token_usage))
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
        try:
            collector = getattr(self, "_metrics_collector", None)
            if collector is not None:
                try:
                    await collector.stop()
                except Exception as exc:
                    print(f"SGLang metrics could not finish ({type(exc).__name__}); continuing experiment.", flush=True)
        finally:
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
                    async with AsyncExitStack() as slots:
                        limiter = getattr(self, "_shared_request_limiter", None)
                        if limiter is not None:
                            await slots.enter_async_context(limiter.slot())
                        if self._uses_anthropic:
                            response = await self._client.messages.create(**request)
                        else:
                            response = await self._client.chat.completions.create(**request)
                        return response, attempt_index + 1
            except Exception as exc:
                is_decode_error = isinstance(exc, json.JSONDecodeError)
                retryable = is_decode_error
                if self._uses_anthropic and not retryable:
                    retryable = is_retryable_anthropic_error(exc)
                if not retryable:
                    raise
                if attempt_index >= retry_count:
                    if is_decode_error:
                        raise ProviderResponseDecodeError(
                            model=self.config.model,
                            attempts=attempt_index + 1,
                            original=exc,
                        ) from exc
                    raise
                delay = self.config.provider_json_retry_backoff_seconds * (2**attempt_index)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise AssertionError("provider response retry loop exited unexpectedly")


async def complete_agent_json(
    client: ChatClient,
    prompt: str,
    *,
    context: dict[str, Any],
    temperature: float,
    stage: str,
    agent: str,
) -> AgentCallResult:
    try:
        if isinstance(client, AgentChatClient) and client.config.resolved_backend == "anthropic":
            from src.agents.schemas import agent_output_schema

            response = await client.complete_json(
                prompt, temperature=temperature, output_schema=agent_output_schema(stage),
            )
        else:
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
    token_usage = response.pop(AGENT_COMPLETION_USAGE_KEY, {})
    response = sanitize_structured_agent_output(response)
    return AgentCallResult(
        content=response,
        usage=agent_call_usage_record(
            stage=stage,
            agent=agent,
            zone_id=context.get("zone_id"),
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
