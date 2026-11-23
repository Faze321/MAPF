from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.agent import normalize_agent_mode, normalize_agent_mode_list
from src.config.training import FORECAST_BACKENDS, RETIRED_LORA_SWITCHES, TRAINING_PARAMETER_NAMES
from src.config.values import (
    dataset_path_key,
    normalize_data_dirs,
    normalize_dataset_overrides,
    normalize_experiment_zone_counts,
    normalize_float_list,
    normalize_forecast_model_name,
    normalize_int_list,
    normalize_string_list,
    normalize_typed_list,
    normalize_zone_id_list,
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    positive_integer,
    strict_bool,
)


BACKEND_PARAMETER_NAMES = (
    "timesfm_repo", "timesfm_context_hours", "timesfm_exog_cols",
    "chronos_repo", "chronos_context_hours", "chronos_device",
)

BACKEND_BLEND_PARAMETER_NAMES = tuple(f"{backend.lower()}_diurnal_blend_alphas" for backend in FORECAST_BACKENDS)
CALIBRATION_METHOD = "daily_hour_mean_v1"
VALIDATION_BIAS_DAYS = 7


def reject_retired_forecast_settings(settings: dict[str, Any]) -> None:
    for key, backend in RETIRED_LORA_SWITCHES.items():
        if key in settings:
            raise ValueError(f"run.{key} was removed; select {backend} in run.forecast_models to train and use LoRA")
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


def validation_bias_days(enabled: bool, validation_days: int = VALIDATION_BIAS_DAYS) -> int:
    """Return the historical holdout required by the forecast bias switch."""
    days = positive_integer(validation_days, "validation_days")
    return days if strict_bool(enabled, "validation_bias") else 0


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
    history_days: int = 35
    validation_bias: bool = False
    validation_days: int = VALIDATION_BIAS_DAYS
    validation_bias_by_dataset: dict[str, bool] = field(default_factory=dict)
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
    timesfm_lora_diurnal_blend_alphas: list[float] | None = None
    chronos_diurnal_blend_alphas: list[float] | None = None
    chronos_lora_diurnal_blend_alphas: list[float] | None = None
    lstm_diurnal_blend_alphas: list[float] | None = None
    xgboost_diurnal_blend_alphas: list[float] | None = None
    ar_diurnal_blend_alphas: list[float] | None = None
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_context_hours: int = 840
    timesfm_exog_cols: list[str] | None = None
    chronos_repo: str = "amazon/chronos-2"
    chronos_context_hours: int = 840
    chronos_device: str = "auto"
    temperature: float = 0.2

    @property
    def output_folder(self) -> str:
        """Canonical output root; output_dir remains as a compatibility alias."""

        return self.output_dir

    def zone_count_for(self, data_dir: str | Path) -> int:
        return self.experiment_zone_counts.get(dataset_path_key(data_dir), self.experiment_zone_count)

    def validation_bias_for(self, data_dir: str | Path) -> bool:
        return self.validation_bias_by_dataset.get(dataset_path_key(data_dir), self.validation_bias)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RunConfig":
        settings = raw or {}
        reject_retired_forecast_settings(settings)
        moved = [key for key in settings if key in TRAINING_PARAMETER_NAMES or (
            key.startswith(("lstm_", "xgboost_"))
            and key not in ("lstm_diurnal_blend_alphas", "xgboost_diurnal_blend_alphas"))]
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
            validation_bias=strict_bool(settings.get("validation_bias", defaults.validation_bias), "run.validation_bias"),
            validation_days=positive_integer(settings.get("validation_days", defaults.validation_days), "run.validation_days"),
            validation_bias_by_dataset=normalize_dataset_overrides(
                settings.get("validation_bias_by_dataset"), "validation_bias_by_dataset", strict_bool,
            ),
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


def normalize_forecast_model_list(value: Any) -> list[str] | None:
    return normalize_typed_list(value, normalize_forecast_model_name)
