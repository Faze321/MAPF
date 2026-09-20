from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar
import re
import os
import yaml

BACKEND_PARAMETER_NAMES = (
    "timesfm_repo",
    "timesfm_context_hours",
    "timesfm_step_horizon",
    "timesfm_exog_cols",
    "timesfm_diurnal_blend_alpha",
    "timesfm_roll_actuals",
    "ar_diurnal_blend_alpha",
    "chronos_repo",
    "chronos_context_hours",
    "chronos_step_horizon",
    "chronos_diurnal_blend_alpha",
    "chronos_device",
    "chronos_roll_actuals",
    "lstm_context_hours",
    "lstm_step_horizon",
    "lstm_exog_cols",
    "lstm_hidden_size",
    "lstm_num_layers",
    "lstm_epochs",
    "lstm_learning_rate",
    "lstm_batch_size",
    "lstm_diurnal_blend_alpha",
    "lstm_device",
    "lstm_roll_actuals",
    "lstm_seed",
)

PIPELINE_STAGES = {"full", "forecaster", "agent"}


def positive_integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{key} must be a positive integer')
    return value


def normalize_data_dirs(value: Any) -> str | list[str]:
    if value is None:
        return "data/UrbanEV"
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError('run.data_dir must be a folder path or a non-empty list of folder paths')
    paths = [path.strip() for path in paths]
    resolved = [os.path.normcase(str(Path(path).resolve())) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('run.data_dir contains duplicate dataset folders')
    return paths if isinstance(value, list) else paths[0]

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
    output_dir: str = "output"
    experiment_name: str | None = None
    weather_file: str = "weather_airport.csv"
    dry_run: bool = False
    force_cache: bool = False
    pipeline_stage: str = "full"
    forecaster_output_dir: str | None = None
    agent_output_dir: str | None = None
    precomputed_window_data: str | None = None
    max_poi_rows: int | None = None
    forecast_start: str | None = None
    horizon_days: int = 4
    history_days: int = 7
    validation_days: int = 1
    zone_ids: list[str] | None = None
    experiment_zone_count: int = 12
    experiment_zone_selection: str = "representative"
    experiment_zone_seed: int = 42
    forecast_model: str = "timesfm"
    forecast_starts: list[str] | None = None
    forecast_start_count: int = 1
    forecast_models: list[str] | None = None
    experiment_seeds: list[int] | None = None
    agent_mode: str = "multi_agent_economist_retry"
    agent_modes: list[str] | None = None
    diurnal_blend_alpha: float | None = None
    diurnal_blend_alphas: list[float] | None = None
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_context_hours: int = 168
    timesfm_step_horizon: int = 24
    timesfm_exog_cols: list[str] | None = None
    timesfm_diurnal_blend_alpha: float = 1.0
    timesfm_roll_actuals: bool = False
    ar_diurnal_blend_alpha: float = 0.0
    chronos_repo: str = "amazon/chronos-2"
    chronos_context_hours: int = 512
    chronos_step_horizon: int = 24
    chronos_diurnal_blend_alpha: float = 0.0
    chronos_device: str = "auto"
    chronos_roll_actuals: bool = False
    lstm_context_hours: int = 24
    lstm_step_horizon: int = 24
    lstm_exog_cols: list[str] | None = None
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_epochs: int = 50
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 32
    lstm_diurnal_blend_alpha: float = 0.0
    lstm_device: str = "auto"
    lstm_roll_actuals: bool = False
    lstm_seed: int = 42
    temperature: float = 0.2

    @property
    def output_folder(self) -> str:
        """Canonical output root; output_dir remains as a compatibility alias."""

        return self.output_dir

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RunConfig":
        settings = raw or {}
        zone_ids = settings.get("zone_ids", settings.get("zones"))
        forecast_starts = normalize_string_list(settings.get("forecast_starts"))
        forecast_models = normalize_forecast_model_list(settings.get("forecast_models"))
        experiment_seeds = normalize_int_list(settings.get("experiment_seeds"))
        agent_modes = normalize_agent_mode_list(settings.get("agent_modes"))
        diurnal_blend_alphas = normalize_float_list(settings.get("diurnal_blend_alphas"))
        diurnal_blend_alpha = optional_float(settings.get("diurnal_blend_alpha"))
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
                "timesfm_step_horizon",
                "chronos_context_hours",
                "chronos_step_horizon",
                "lstm_context_hours",
                "lstm_step_horizon",
                "lstm_hidden_size",
                "lstm_num_layers",
                "lstm_epochs",
                "lstm_batch_size",
                "lstm_seed",
            )),
            (optional_float, (
                "timesfm_diurnal_blend_alpha",
                "ar_diurnal_blend_alpha",
                "chronos_diurnal_blend_alpha",
                "lstm_learning_rate",
                "lstm_diurnal_blend_alpha",
                "temperature",
            )),
        ):
            for name in names:
                value = convert(settings.get(name))
                numeric_settings[name] = getattr(defaults, name) if value is None else value
        return cls(
            **numeric_settings,
            data_dir=normalize_data_dirs(settings.get("data_dir")),
            max_parallel_datasets=positive_integer(settings.get("max_parallel_datasets", 2), "run.max_parallel_datasets"),
            output_dir=(
                optional_str(settings.get("output_folder"))
                or optional_str(settings.get("output_dir"))
                or "output"
            ),
            experiment_name=optional_str(settings.get("experiment_name")),
            weather_file=optional_str(settings.get("weather_file")) or "weather_airport.csv",
            dry_run=optional_bool(settings.get("dry_run"), False),
            force_cache=optional_bool(settings.get("force_cache"), False),
            pipeline_stage=normalize_pipeline_stage(settings.get("pipeline_stage")),
            forecaster_output_dir=optional_str(settings.get("forecaster_output_dir")),
            agent_output_dir=optional_str(settings.get("agent_output_dir")),
            precomputed_window_data=optional_str(settings.get("precomputed_window_data")),
            max_poi_rows=optional_int(settings.get("max_poi_rows")),
            forecast_start=optional_str(settings.get("forecast_start")),
            zone_ids=normalize_zone_id_list(zone_ids),
            experiment_zone_selection=normalize_zone_selection(settings.get("experiment_zone_selection")),
            forecast_model=normalize_forecast_model_name(optional_str(settings.get("forecast_model"))),
            forecast_starts=forecast_starts,
            forecast_start_count=positive_integer(settings.get("forecast_start_count", 1), "run.forecast_start_count"),
            forecast_models=forecast_models,
            experiment_seeds=experiment_seeds,
            agent_mode=normalize_agent_mode(optional_str(settings.get("agent_mode"))),
            agent_modes=agent_modes,
            diurnal_blend_alpha=diurnal_blend_alpha,
            diurnal_blend_alphas=diurnal_blend_alphas,
            timesfm_repo=optional_str(settings.get("timesfm_repo"))
            or "google/timesfm-2.5-200m-pytorch",
            timesfm_exog_cols=normalize_zone_id_list(settings.get("timesfm_exog_cols")),
            timesfm_roll_actuals=False,
            chronos_repo=optional_str(settings.get("chronos_repo")) or "amazon/chronos-2",
            chronos_device="auto",
            chronos_roll_actuals=False,
            lstm_exog_cols=normalize_zone_id_list(settings.get("lstm_exog_cols")),
            lstm_device="auto",
            lstm_roll_actuals=False,
        )


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    base_url: str
    model: str
    profile: str = "multi_agent"
    single_agent_model: str | None = None
    reasoning_effort: str = "none"
    http_referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 90.0
    max_concurrent_requests: int = 4
    provider_json_retries: int = 2
    provider_json_retry_backoff_seconds: float = 1.0

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
            or "meta-llama/llama-3.1-8b-instruct"
        )
        resolved = _expand_env_vars(resolved)

        return cls(
            api_key=optional_str(resolved.get("api_key")),
            base_url=optional_str(resolved.get("base_url")) or "https://openrouter.ai/api/v1",
            model=optional_str(resolved.get("model")) or "meta-llama/llama-3.1-8b-instruct",
            profile=profile,
            single_agent_model=optional_str(settings.get("single_agent_model")),
            reasoning_effort=optional_str(resolved.get("reasoning_effort")) or "none",
            http_referer=optional_str(resolved.get("http_referer")),
            title=optional_str(resolved.get("title")) or "MAPF UrbanEV",
            timeout_seconds=float(resolved.get("timeout_seconds", 90)),
            max_concurrent_requests=int(resolved.get("max_concurrent_requests", 4)),
            provider_json_retries=int(resolved.get("provider_json_retries", 2)),
            provider_json_retry_backoff_seconds=float(
                resolved.get("provider_json_retry_backoff_seconds", 1.0)
            ),
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
            f"Unsupported pipeline stage: {value}. Expected one of: agent, forecaster, full."
        )
    return stage

@dataclass(frozen=True)
class AppConfig:
    agent: AgentConfig
    run: RunConfig
    data: DataConfig = field(default_factory=DataConfig)

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
        if not isinstance(agent_settings, dict):
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
    def convert(item: str) -> float:
        number = float(item)
        if number < 0.0 or number > 1.0:
            raise ValueError(f"Diurnal blend alpha must be between 0 and 1: {item}")
        return round(number, 6)

    return normalize_typed_list(value, convert)


def normalize_agent_mode_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_agent_mode)


def normalize_forecast_model_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_forecast_model_name)
