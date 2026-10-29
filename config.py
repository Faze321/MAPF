from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit
import json
import math
import re
import os
import yaml

from training_config import TrainingConfig, TRAINING_PARAMETER_NAMES

BACKEND_PARAMETER_NAMES = (
    "timesfm_repo", "timesfm_context_hours", "timesfm_exog_cols",
    "chronos_repo", "chronos_context_hours", "chronos_device",
)

FORECAST_BACKENDS = ("timesfm", "chronos", "lstm", "xgboost", "AR")
BACKEND_BLEND_PARAMETER_NAMES = tuple(f"{backend.lower()}_diurnal_blend_alphas" for backend in FORECAST_BACKENDS)
CALIBRATION_METHOD = "daily_hour_mean_v1"


def reject_retired_forecast_settings(settings: dict[str, Any]) -> None:
    replacements = {
        "forecast_start": "forecast_starts",
        "forecast_model": "forecast_models",
        "diurnal_blend_alpha": "diurnal_blend_alphas",
        **{f"{backend.lower()}_diurnal_blend_alpha": f"{backend.lower()}_diurnal_blend_alphas"
           for backend in FORECAST_BACKENDS},
    }
    retired = [key for key in settings if key in replacements
               or key in {"roll_actuals", "step_horizon"}
               or key.endswith(("_roll_actuals", "_step_horizon"))]
    if retired:
        details = [f"run.{key} was removed" + (f"; use run.{replacements[key]}" if key in replacements
                   else "; forecasts now use fixed daily blocks and never roll forecast-period actuals") for key in retired]
        raise ValueError("Deprecated forecast configuration: " + ". ".join(details))


def resolve_backend_blend_alphas(backend: str, global_values: Any, overrides: dict[str, Any]) -> list[float]:
    name = f"{normalize_forecast_model_name(backend).lower()}_diurnal_blend_alphas"
    local = normalize_float_list(overrides.get(name))
    global_alphas = normalize_float_list(global_values)
    return local if local is not None else global_alphas if global_alphas is not None else [0.0]


PIPELINE_STAGES = {"train", "full", "forecaster", "agent"}
AGENT_BACKENDS = {"auto", "openai", "openrouter", "openai_compatible", "vllm", "sglang", "anthropic"}
LOCAL_AGENT_BACKENDS = {"vllm", "sglang"}
LOCAL_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
ANTHROPIC_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192


def positive_integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{key} must be a positive integer')
    return value


def dataset_path_key(path: str | Path) -> str:
    """Match dataset folders consistently for relative paths and batch child processes."""
    return os.path.normcase(str(Path(path).resolve()))


def normalize_data_dirs(value: Any) -> str | list[str]:
    if value is None:
        return "data/UrbanEV"
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError('run.data_dir must be a folder path or a non-empty list of folder paths')
    paths = [path.strip() for path in paths]
    resolved = [dataset_path_key(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('run.data_dir contains duplicate dataset folders')
    return paths if isinstance(value, list) else paths[0]


def normalize_experiment_zone_counts(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("run.experiment_zone_counts must map dataset folder paths to positive integers")
    counts = {}
    for folder, count in value.items():
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError("run.experiment_zone_counts keys must be non-empty dataset folder paths")
        key = dataset_path_key(folder.strip())
        if key in counts:
            raise ValueError(f"run.experiment_zone_counts contains duplicate dataset folder: {folder}")
        counts[key] = positive_integer(count, f"run.experiment_zone_counts[{folder}]")
    return counts

_ENV_REF_RE = re.compile(r"""
                         \$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\} 
                         |\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*) 
                         |%(?P<windows>[A-Za-z_][A-Za-z0-9_]*)%
                         """, re.VERBOSE)


@dataclass(frozen=True)
class DataConfig:
    adapter: str = "auto"
    cache_dir: str | None = None
    timeseries_file: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    static_file: str | None = None
    static_mapping: dict[str, str] = field(default_factory=dict)
    unit_conversions: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DataConfig":
        settings = raw or {}
        column_mapping = settings.get("column_mapping") or {}
        static_mapping = settings.get("static_mapping") or {}
        unit_conversions = settings.get("unit_conversions") or {}
        if not isinstance(column_mapping, dict):
            raise ValueError('Config key "data.column_mapping" must contain a mapping')
        if not isinstance(static_mapping, dict):
            raise ValueError('Config key "data.static_mapping" must contain a mapping')
        if not isinstance(unit_conversions, dict):
            raise ValueError('Config key "data.unit_conversions" must contain a mapping')
        return cls(
            adapter=optional_str(settings.get("adapter")) or "auto",
            cache_dir=optional_str(settings.get("cache_dir")),
            timeseries_file=optional_str(settings.get("timeseries_file")),
            column_mapping={str(key): str(value) for key, value in column_mapping.items()},
            static_file=optional_str(settings.get("static_file")),
            static_mapping={str(key): str(value) for key, value in static_mapping.items()},
            unit_conversions={str(key): float(value) for key, value in unit_conversions.items()},
        )


@dataclass(frozen=True)
class RunConfig:
    data_dir: str | list[str] = "data/UrbanEV"
    max_parallel_datasets: int = 2
    max_dataset_errors: int = 3
    output_dir: str = "output"
    experiment_name: str | None = None
    weather_file: str = "weather_airport.csv"
    dry_run: bool = False
    force_cache: bool = False
    force_train: bool = False
    pipeline_stage: str = "full"
    forecaster_output_dir: str | None = None
    agent_output_dir: str | None = None
    precomputed_window_data: str | None = None
    max_poi_rows: int | None = None
    horizon_days: int = 1
    history_days: int = 30
    validation_days: int = 5
    zone_ids: list[str] | None = None
    experiment_zone_count: int = 12
    experiment_zone_counts: dict[str, int] = field(default_factory=dict)
    experiment_zone_selection: str = "representative"
    experiment_zone_seed: int = 42
    forecast_starts: list[str] | None = None
    forecast_start_count: int = 1
    forecast_models: list[str] | None = None
    experiment_seeds: list[int] | None = None
    agent_mode: str = "multi_agent_economist_retry"
    agent_modes: list[str] | None = None
    diurnal_blend_alphas: list[float] | None = None
    timesfm_diurnal_blend_alphas: list[float] | None = None
    chronos_diurnal_blend_alphas: list[float] | None = None
    lstm_diurnal_blend_alphas: list[float] | None = None
    xgboost_diurnal_blend_alphas: list[float] | None = None
    ar_diurnal_blend_alphas: list[float] | None = None
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_context_hours: int = 168
    timesfm_exog_cols: list[str] | None = None
    chronos_repo: str = "amazon/chronos-2"
    chronos_context_hours: int = 512
    chronos_device: str = "auto"
    temperature: float = 0.2

    @property
    def output_folder(self) -> str:
        """Canonical output root; output_dir remains as a compatibility alias."""

        return self.output_dir

    def zone_count_for(self, data_dir: str | Path) -> int:
        return self.experiment_zone_counts.get(dataset_path_key(data_dir), self.experiment_zone_count)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RunConfig":
        settings = raw or {}
        reject_retired_forecast_settings(settings)
        moved = [key for key in settings if key.startswith(("lstm_", "xgboost_"))
                 and key not in ("lstm_diurnal_blend_alphas", "xgboost_diurnal_blend_alphas")]
        if moved:
            raise ValueError("Training settings moved out of run; use training.defaults: " + ", ".join(moved))
        zone_ids = settings.get("zone_ids", settings.get("zones"))
        forecast_starts = normalize_string_list(settings.get("forecast_starts"))
        forecast_models = normalize_forecast_model_list(settings.get("forecast_models"))
        experiment_seeds = normalize_int_list(settings.get("experiment_seeds"))
        agent_modes = normalize_agent_mode_list(settings.get("agent_modes"))
        diurnal_blend_alphas = normalize_float_list(settings.get("diurnal_blend_alphas"))
        defaults = cls()
        numeric_settings = {}
        for convert, names in (
            (optional_int, (
                "horizon_days",
                "history_days",
                "validation_days",
                "experiment_zone_count",
                "experiment_zone_seed",
                "timesfm_context_hours",
                "chronos_context_hours",
            )),
            (optional_float, (
                "temperature",
            )),
        ):
            for name in names:
                value = convert(settings.get(name))
                numeric_settings[name] = getattr(defaults, name) if value is None else value
        return cls(
            **numeric_settings,
            data_dir=normalize_data_dirs(settings.get("data_dir")),
            experiment_zone_counts=normalize_experiment_zone_counts(settings.get("experiment_zone_counts")),
            max_parallel_datasets=positive_integer(settings.get("max_parallel_datasets", 2), "run.max_parallel_datasets"),
            max_dataset_errors=positive_integer(
                settings.get("max_dataset_errors", defaults.max_dataset_errors), "run.max_dataset_errors"
            ),
            output_dir=(
                optional_str(settings.get("output_folder"))
                or optional_str(settings.get("output_dir"))
                or "output"
            ),
            experiment_name=optional_str(settings.get("experiment_name")),
            weather_file=optional_str(settings.get("weather_file")) or "weather_airport.csv",
            dry_run=optional_bool(settings.get("dry_run"), False),
            force_cache=optional_bool(settings.get("force_cache"), False),
            force_train=optional_bool(settings.get("force_train"), False),
            pipeline_stage=normalize_pipeline_stage(settings.get("pipeline_stage")),
            forecaster_output_dir=optional_str(settings.get("forecaster_output_dir")),
            agent_output_dir=optional_str(settings.get("agent_output_dir")),
            precomputed_window_data=optional_str(settings.get("precomputed_window_data")),
            max_poi_rows=optional_int(settings.get("max_poi_rows")),
            zone_ids=normalize_zone_id_list(zone_ids),
            experiment_zone_selection=normalize_zone_selection(settings.get("experiment_zone_selection")),
            forecast_starts=forecast_starts,
            forecast_start_count=positive_integer(settings.get("forecast_start_count", 1), "run.forecast_start_count"),
            forecast_models=forecast_models,
            experiment_seeds=experiment_seeds,
            agent_mode=normalize_agent_mode(optional_str(settings.get("agent_mode"))),
            agent_modes=agent_modes,
            diurnal_blend_alphas=diurnal_blend_alphas,
            **{name: normalize_float_list(settings.get(name)) for name in BACKEND_BLEND_PARAMETER_NAMES},
            timesfm_repo=optional_str(settings.get("timesfm_repo"))
            or "google/timesfm-2.5-200m-pytorch",
            timesfm_exog_cols=normalize_zone_id_list(settings.get("timesfm_exog_cols")),
            chronos_repo=optional_str(settings.get("chronos_repo")) or "amazon/chronos-2",
            chronos_device="auto",
        )


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
    backend: str = "auto"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    max_concurrent_requests_total: int | None = None
    anthropic_thinking: dict[str, Any] | None = None
    anthropic_cache_control: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in AGENT_BACKENDS:
            raise ValueError(f"agent.backend must be one of: {', '.join(sorted(AGENT_BACKENDS))}")
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
        if self.backend != "auto":
            return self.backend
        hostname = (urlsplit(self.base_url).hostname or "").lower().rstrip(".")
        if hostname == "api.openai.com":
            return "openai"
        if hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai"):
            return "openrouter"
        return "openai_compatible"

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
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
                )
            return cls.default(model=model, profile=profile)

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
            "max_tokens",
            "max_concurrent_requests_total",
            "anthropic_thinking",
            "anthropic_cache_control",
        )
        resolved = {name: settings[name] for name in setting_names if name in settings}
        resolved.update(profile_settings)

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

        backend = resolved.get("backend", "auto")
        reasoning_effort = resolved.get("reasoning_effort", "none")
        if not isinstance(backend, str) or backend not in LOCAL_AGENT_BACKENDS | {"anthropic"}:
            # Preserve the legacy cloud default. Local null explicitly delegates
            # reasoning settings to the server/model instead of disabling them.
            reasoning_effort = optional_str(reasoning_effort) or "none"
        native_anthropic = backend == "anthropic"
        if backend == "openai" and not optional_str(resolved.get("model")):
            raise ValueError("agent.model is required for backend openai; specify the endpoint's model ID")
        default_base_url = (
            "https://api.anthropic.com" if native_anthropic else
            "https://api.openai.com/v1" if backend == "openai" else
            "https://openrouter.ai/api/v1"
        )

        return cls(
            api_key=optional_str(resolved.get("api_key")),
            base_url=optional_str(resolved.get("base_url")) or default_base_url,
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
            max_tokens=resolved.get("max_tokens"),
            max_concurrent_requests_total=resolved.get("max_concurrent_requests_total"),
            anthropic_thinking=resolved.get("anthropic_thinking"),
            anthropic_cache_control=resolved.get("anthropic_cache_control"),
        )

    @classmethod
    def default(
        cls,
        *,
        model: str | None = None,
        profile: str = "multi_agent",
    ) -> "AgentConfig":
        return cls(
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            model=model or "meta-llama/llama-3.1-8b-instruct",
            profile=profile,
            title="MAPF UrbanEV",
        )


def normalize_zone_selection(value: Any) -> str:
    selection = str(value or "representative").strip().lower()
    if selection not in {"representative", "random"}:
        raise ValueError("run.experiment_zone_selection must be representative or random")
    return selection


def normalize_pipeline_stage(value: Any) -> str:
    stage = str(value or "full").strip().lower().replace("-", "_")
    aliases = {
        "forecast": "forecaster",
        "forecast_only": "forecaster",
        "forecaster_only": "forecaster",
        "agent_only": "agent",
        "all": "full",
    }
    stage = aliases.get(stage, stage)
    if stage not in PIPELINE_STAGES:
        raise ValueError(
            f"Unsupported pipeline stage: {value}. Expected one of: train, agent, forecaster, full."
        )
    return stage

@dataclass(frozen=True)
class AppConfig:
    agent: AgentConfig
    run: RunConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        required: bool = False,
        load_agent: bool = True,
    ) -> "AppConfig":
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
                )
            return cls(agent=AgentConfig.default(), run=RunConfig(), data=DataConfig())

        raw = read_config_mapping(path, expand_env=False)
        agent_settings = raw.get("agent")
        if load_agent and not isinstance(agent_settings, dict):
            raise ValueError('Config key "agent" must contain a mapping')
        run_settings = raw.get("run", {})
        if run_settings is None:
            run_settings = {}
        if not isinstance(run_settings, dict):
            raise ValueError('Config key "run" must contain a mapping')
        data_settings = raw.get("data", {})
        if data_settings is None:
            data_settings = {}
        if not isinstance(data_settings, dict):
            raise ValueError('Config key "data" must contain a mapping')
        run_settings = _expand_env_vars(run_settings)
        data_settings = _expand_env_vars(data_settings)
        return cls(
            agent=(
                AgentConfig.from_file(path, required=required)
                if load_agent
                else AgentConfig.default()
            ),
            run=RunConfig.from_mapping(run_settings),
            data=DataConfig.from_mapping(data_settings),
            training=TrainingConfig.from_mapping(_expand_env_vars(raw.get("training"))),
        )

def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        missing_env = []
        for match in _ENV_REF_RE.finditer(obj):
            name = match.group("braced") or match.group("plain") or match.group("windows")
            if name not in os.environ:
                missing_env.append(name)
        if missing_env:
            raise ValueError(f"Missing environment variable(s): {', '.join(missing_env)}")
        return os.path.expandvars(obj)
    elif isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    else:
        return obj

def read_config_mapping(path: Path, *, expand_env: bool = True) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _expand_env_vars(value) if expand_env else value


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_forecast_model_name(value: str | None) -> str:
    normalized = (value or "timesfm").strip().lower().replace("-", "_")
    if normalized in {"xgb", "xg_boost"}:
        return "xgboost"
    return "AR" if normalized == "ar" else normalized


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


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean config value: {value!r}")


def normalize_zone_id_list(value: Any) -> list[str] | None:
    return normalize_string_list(value)


def normalize_string_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    raw_values = value if isinstance(value, list) else [value]
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            value_text = part.strip()
            if value_text and value_text not in seen:
                values.append(value_text)
                seen.add(value_text)
    return values or None


_ListItem = TypeVar("_ListItem")


def normalize_typed_list(value: Any, convert: Callable[[str], _ListItem]) -> list[_ListItem] | None:
    """Parse once, then deduplicate converted values while keeping input order."""
    return list(dict.fromkeys(convert(item) for item in normalize_string_list(value) or [])) or None


def normalize_int_list(value: Any) -> list[int] | None:
    return normalize_typed_list(value, int)


def normalize_float_list(value: Any) -> list[float] | None:
    """Normalize an alpha candidate list; null inherits, an empty list is invalid."""
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, bool) or (isinstance(value, list) and any(isinstance(item, bool) for item in value)):
        raise ValueError("Diurnal blend alpha must be a finite number between 0 and 1")
    def convert(item: str) -> float:
        number = float(item)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"Diurnal blend alpha must be a finite number between 0 and 1: {item}")
        return round(number, 6)

    values = normalize_typed_list(value, convert)
    if not values:
        raise ValueError("Diurnal blend alpha lists cannot be empty; use null to inherit")
    return values


def normalize_agent_mode_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_agent_mode)


def normalize_forecast_model_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_forecast_model_name)
