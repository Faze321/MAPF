from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.values import (
    _expand_env_vars,
    normalize_typed_list,
    optional_str,
    positive_integer,
    read_config_mapping,
)


AGENT_BACKENDS = {"openai", "openrouter", "openai_compatible", "vllm", "sglang", "anthropic"}
LOCAL_AGENT_BACKENDS = {"vllm", "sglang"}
LOCAL_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
ANTHROPIC_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192


def validate_agent_backend(backend: Any) -> str:
    """Require an explicit provider protocol; never infer it from a URL."""
    if not isinstance(backend, str) or backend not in AGENT_BACKENDS:
        raise ValueError(f"agent.backend must be explicitly set to one of: {', '.join(sorted(AGENT_BACKENDS))}")
    return backend


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    base_url: str
    model: str
    profile: str = "multi_agent"
    single_agent_model: str | None = None
    reasoning_effort: str | None = "none"
    http_referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 90.0
    max_concurrent_requests: int = 4
    provider_json_retries: int = 2
    provider_json_retry_backoff_seconds: float = 1.0
    backend: str = field(kw_only=True)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    max_concurrent_requests_total: int | None = None
    anthropic_thinking: dict[str, Any] | None = None
    anthropic_cache_control: dict[str, Any] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    sglang_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_agent_backend(self.backend)
        if self.backend in LOCAL_AGENT_BACKENDS or self.backend == "anthropic":
            provider = "anthropic" if self.backend == "anthropic" else "vllm/sglang"
            efforts = ANTHROPIC_REASONING_EFFORTS if self.backend == "anthropic" else LOCAL_REASONING_EFFORTS
            effort = self.reasoning_effort
            if effort is not None and not isinstance(effort, str):
                raise ValueError(f"agent.reasoning_effort must be a string or null for {provider}")
            if effort is not None:
                effort = effort.strip().lower() or None
            if effort is not None and effort not in efforts:
                raise ValueError(
                    f"agent.reasoning_effort for {provider} must be null or one of: "
                    + ", ".join(sorted(efforts))
                )
            object.__setattr__(self, "reasoning_effort", effort)
        kwargs = self.chat_template_kwargs
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError("agent.chat_template_kwargs must contain a mapping")

        def validate_keys(value: Any) -> None:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("agent.chat_template_kwargs keys must be strings")
                for item in value.values():
                    validate_keys(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    validate_keys(item)

        try:
            validate_keys(kwargs)
            json.dumps(kwargs, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                "agent.chat_template_kwargs must be JSON serializable with string keys and finite numbers"
            ) from exc
        if "enable_thinking" in kwargs and not isinstance(kwargs["enable_thinking"], bool):
            raise ValueError("agent.chat_template_kwargs.enable_thinking must be a boolean")
        if self.backend == "anthropic" and kwargs:
            raise ValueError("agent.chat_template_kwargs is unsupported for anthropic; use agent.anthropic_thinking")
        object.__setattr__(self, "chat_template_kwargs", deepcopy(kwargs))
        extra_body = self.extra_body if self.extra_body is not None else {}
        if not isinstance(extra_body, dict):
            raise ValueError("agent.extra_body must contain a mapping or null")
        object.__setattr__(self, "extra_body", deepcopy(extra_body))
        metrics: dict[str, Any] = {}
        if self.backend == "sglang":
            options = self.sglang_metrics if self.sglang_metrics is not None else {}
            if not isinstance(options, dict):
                raise ValueError("agent.sglang_metrics must contain a mapping or null")
            metrics = {"enabled": True, "url": None, "interval_seconds": 5.0, "timeout_seconds": 3.0, **options}
            if not isinstance(metrics["enabled"], bool):
                raise ValueError("agent.sglang_metrics.enabled must be a boolean")
            if metrics["enabled"]:
                if metrics["url"] is not None and not isinstance(metrics["url"], str):
                    raise ValueError("agent.sglang_metrics.url must be a string or null")
                for name in ("interval_seconds", "timeout_seconds"):
                    value = metrics[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"agent.sglang_metrics.{name} must be a finite positive number")
        object.__setattr__(self, "sglang_metrics", deepcopy(metrics))
        for name in ("max_tokens", "max_concurrent_requests_total"):
            value = getattr(self, name)
            if value is not None:
                positive_integer(value, f"agent.{name}")
        thinking = self.anthropic_thinking
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise ValueError("agent.anthropic_thinking must be a mapping or null")
            mode = thinking.get("type")
            if not isinstance(mode, str) or mode not in {"disabled", "adaptive", "enabled"}:
                raise ValueError("agent.anthropic_thinking.type must be disabled, adaptive or enabled")
            keys = {"type", "budget_tokens"} if mode == "enabled" else {"type"}
            if set(thinking) != keys:
                raise ValueError(f"agent.anthropic_thinking with type {mode} requires exactly: {', '.join(sorted(keys))}")
            if mode == "enabled":
                budget = positive_integer(thinking["budget_tokens"], "agent.anthropic_thinking.budget_tokens")
                if budget < 1024:
                    raise ValueError("agent.anthropic_thinking.budget_tokens must be at least 1024")
                if self.backend == "anthropic" and budget >= (self.max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS):
                    raise ValueError("agent.anthropic_thinking.budget_tokens must be smaller than max_tokens (8192 when unset)")
            object.__setattr__(self, "anthropic_thinking", deepcopy(thinking))
        cache_control = self.anthropic_cache_control
        if cache_control is not None:
            if (not isinstance(cache_control, dict)
                    or cache_control.get("type") != "ephemeral"
                    or not set(cache_control).issubset({"type", "ttl"})):
                raise ValueError("agent.anthropic_cache_control must contain type: ephemeral and optionally ttl")
            if "ttl" in cache_control and cache_control["ttl"] not in ("5m", "1h"):
                raise ValueError("agent.anthropic_cache_control.ttl must be 5m or 1h")
            object.__setattr__(self, "anthropic_cache_control", deepcopy(cache_control))
        positive_integer(self.max_concurrent_requests, "agent.max_concurrent_requests")
        if (
            isinstance(self.provider_json_retries, bool)
            or not isinstance(self.provider_json_retries, int)
            or self.provider_json_retries < 0
        ):
            raise ValueError("agent.provider_json_retries must be a non-negative integer")
        for name, allow_zero in (
            ("timeout_seconds", False),
            ("provider_json_retry_backoff_seconds", True),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if allow_zero else value <= 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"agent.{name} must be a finite {qualifier} number")

    @property
    def resolved_backend(self) -> str:
        return self.backend

    @property
    def client_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.backend in LOCAL_AGENT_BACKENDS:
            return "EMPTY"
        return None

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        model: str | None = None,
        required: bool = True,
        agent_mode: str | None = None,
    ) -> "AgentConfig":
        profile = agent_config_profile(agent_mode)
        if not path.exists():
            # Keep the required argument for existing callers, but an Agent
            # cannot be configured without an explicitly selected backend.
            raise FileNotFoundError(
                f"Config file not found: {path}. Copy config.example.yaml to config.yaml and set agent.backend."
            )

        # Expand only the selected profile. This lets either provider use its own
        # environment variables without requiring credentials for the inactive one.
        raw = read_config_mapping(path, expand_env=False)
        settings = raw.get("agent")
        if not isinstance(settings, dict):
            raise ValueError('Config key "agent" must contain a mapping')

        profile_settings = settings.get(profile, {})
        if profile_settings is None:
            profile_settings = {}
        if not isinstance(profile_settings, dict):
            raise ValueError(f'Config key "agent.{profile}" must contain a mapping')

        setting_names = (
            "api_key",
            "base_url",
            "model",
            "reasoning_effort",
            "http_referer",
            "title",
            "timeout_seconds",
            "max_concurrent_requests",
            "provider_json_retries",
            "provider_json_retry_backoff_seconds",
            "backend",
            "chat_template_kwargs",
            "extra_body",
            "sglang_metrics",
            "max_tokens",
            "max_concurrent_requests_total",
            "anthropic_thinking",
            "anthropic_cache_control",
        )
        def extension_settings(layer: dict[str, Any]) -> dict[str, Any]:
            values = dict(layer)
            # Treat both spellings as one profile-level setting. New extra_body
            # replaces legacy templates, including when explicitly cleared.
            if "extra_body" in values:
                values["chat_template_kwargs"] = None
            elif "chat_template_kwargs" in values:
                values["extra_body"] = None
            return values

        resolved = extension_settings({name: settings[name] for name in setting_names if name in settings})
        resolved.update(extension_settings(profile_settings))

        # A nested model wins over the legacy single_agent_model, while --model
        # overrides only whichever profile is active for this run.
        profile_model = profile_settings.get("model")
        legacy_single_model = (
            settings.get("single_agent_model") if profile == "single_agent" else None
        )
        flat_model = settings.get("model")
        resolved["model"] = (
            model
            or (profile_model if optional_str(profile_model) else None)
            or (legacy_single_model if optional_str(legacy_single_model) else None)
            or (flat_model if optional_str(flat_model) else None)
        )
        backend = validate_agent_backend(_expand_env_vars(resolved.get("backend")))
        if backend != "sglang":
            # Inactive monitoring must not require SGLang-only environment values.
            resolved.pop("sglang_metrics", None)
        resolved = _expand_env_vars(resolved)

        # Keep legacy numeric strings (including environment variables), but do
        # not silently turn booleans or fractional concurrency counts into ints.
        def legacy_number(name: str, default: int | float, convert: type) -> int | float:
            value = resolved.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"agent.{name} must be a number, not a boolean")
            try:
                converted = convert(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"agent.{name} must be a valid {convert.__name__}") from exc
            if convert is int and not isinstance(value, str) and converted != value:
                raise ValueError(f"agent.{name} must be an integer")
            return converted

        native_anthropic = backend == "anthropic"
        if backend == "openai" and not optional_str(resolved.get("model")):
            raise ValueError("agent.model is required for backend openai; specify the endpoint's model ID")
        default_base_url = (
            "https://api.anthropic.com" if native_anthropic else
            "https://api.openai.com/v1" if backend == "openai" else
            "https://openrouter.ai/api/v1"
        )
        base_url = optional_str(resolved.get("base_url")) or default_base_url
        reasoning_effort = resolved.get("reasoning_effort", None if backend == "openai" else "none")
        if backend not in LOCAL_AGENT_BACKENDS | {"anthropic", "openai"}:
            # Preserve other cloud providers' legacy default; OpenAI and native/
            # local backends retain null so their model or server chooses effort.
            reasoning_effort = optional_str(reasoning_effort) or "none"

        return cls(
            api_key=optional_str(resolved.get("api_key")),
            base_url=base_url,
            model=optional_str(resolved.get("model")) or (
                "claude-sonnet-4-6" if native_anthropic else "meta-llama/llama-3.1-8b-instruct"),
            profile=profile,
            single_agent_model=optional_str(settings.get("single_agent_model")),
            reasoning_effort=reasoning_effort,
            http_referer=optional_str(resolved.get("http_referer")),
            title=optional_str(resolved.get("title")) or "MAPF UrbanEV",
            timeout_seconds=legacy_number("timeout_seconds", 90.0, float),
            max_concurrent_requests=legacy_number("max_concurrent_requests", 4, int),
            provider_json_retries=legacy_number("provider_json_retries", 2, int),
            provider_json_retry_backoff_seconds=legacy_number(
                "provider_json_retry_backoff_seconds", 1.0, float
            ),
            backend=backend,
            chat_template_kwargs=resolved.get("chat_template_kwargs"),
            extra_body=resolved.get("extra_body"),
            sglang_metrics=resolved.get("sglang_metrics"),
            max_tokens=resolved.get("max_tokens"),
            max_concurrent_requests_total=resolved.get("max_concurrent_requests_total"),
            anthropic_thinking=resolved.get("anthropic_thinking"),
            anthropic_cache_control=resolved.get("anthropic_cache_control"),
        )

    @classmethod
    def default(
        cls,
        *,
        backend: str,
        model: str | None = None,
        profile: str = "multi_agent",
    ) -> "AgentConfig":
        return cls(
            api_key=None,
            backend=backend,
            base_url="https://openrouter.ai/api/v1",
            model=model or "meta-llama/llama-3.1-8b-instruct",
            profile=profile,
            title="MAPF UrbanEV",
        )


def normalize_agent_mode(value: str | None) -> str:
    normalized = (value or "multi_agent_economist_retry").strip().lower().replace("-", "_")
    supported = {
        "multi_agent_economist_retry": "multi_agent_economist_retry",
        "multi_agent_full_retry": "multi_agent_full_retry",
        "multi_agent_discussion_3rounds": "multi_agent_discussion_3rounds",
        "multi_agent_discussion": "multi_agent_discussion_3rounds",
        "multi_agent_debate_3rounds": "multi_agent_discussion_3rounds",
        "multi_agent_communication_3rounds": "multi_agent_discussion_3rounds",
        "single_agent_price_retry": "single_agent_price_retry",
        "single_agent_full_retry": "single_agent_full_retry",
    }
    if normalized in supported:
        return supported[normalized]
    raise ValueError(f"Unsupported agent_mode: {value}")


def agent_config_profile(agent_mode: str | None) -> str:
    """Return the independent provider profile used by a control mode."""
    normalized = normalize_agent_mode(agent_mode)
    return "single_agent" if normalized.startswith("single_agent_") else "multi_agent"


def normalize_agent_mode_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_agent_mode)
