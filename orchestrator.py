from __future__ import annotations

import asyncio
import copy
import json
import math
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from agents import (
    AgentChatClient,
    AgentStageError,
    retry_zone_prices,
    run_zone_chain,
    run_all_zone_chains,
    summarize_agent_call_usage,
)
from config import (
    AgentConfig,
    agent_config_profile,
    normalize_agent_mode,
    normalize_forecast_model_name,
    normalize_pipeline_stage,
)
from data_loader import (
    build_zone_3h_load_policy_thresholds,
    build_zone_profiles_from_canonical,
    load_pipeline_data,
)
from dataset_adapter import DatasetSpec, load_canonical_dataset, split_cache_key
from forecasting import ForecastResult, forecast_zone
from global_forecaster import NATIVE_ARTIFACT_SCHEMA_VERSION, NativeForecasterArtifact
from load_policy import (
    EXTREMELY_HIGH_STRESS,
    HIGH_MAX_LOAD_PCT,
    HIGH_STRESS,
    LOW_MAX_LOAD_PCT,
    LOW_STRESS,
    MEDIUM_MAX_LOAD_PCT,
    MEDIUM_STRESS,
    classify_load_percentage,
    load_range_position_pct,
)
from reporting import safe_filename, write_agent_outputs, write_forecaster_outputs
from time_utils import normalize_datetime_series_24h, parse_datetime_24h
from zone_selection import select_zone_categories


STRESS_LEVEL_ORDER = {
    LOW_STRESS: 0,
    MEDIUM_STRESS: 1,
    HIGH_STRESS: 2,
    EXTREMELY_HIGH_STRESS: 3,
}
DECISION_SUMMARY_KEYS = {
    "zone_id",
    "category",
    "history_start",
    "history_end",
    "validation_start",
    "validation_end",
    "forecast_start",
    "forecast_end",
    "forecast_horizon_days",
    "forecast_horizon_hours",
    "forecast_model",
    "diurnal_blend_alpha",
    "forecast_total_kwh",
    "forecast_peak_kwh",
    "predicted_change_pct",
    "capacity_kw_proxy",
    "peak_capacity_ratio",
    "grid_stress_level",
    "grid_stress_basis",
    "grid_stress_load_kwh",
    "grid_stress_load_range_position_pct",
    "grid_stress_source_file",
    "grid_stress_window_hours",
    "grid_stress_historical_windows",
    "historical_min_load_3h_kwh",
    "historical_max_load_3h_kwh",
    "historical_load_range_3h_kwh",
    "low_medium_threshold_pct",
    "medium_high_threshold_pct",
    "high_extremely_high_threshold_pct",
    "load_3h_low_medium_threshold_kwh",
    "load_3h_medium_high_threshold_kwh",
    "load_3h_high_extremely_high_threshold_kwh",
    "zone_mean_energy_price",
    "calibration",
    "price",
    "weather",
    "occupancy",
    "profile",
    "daily_history_kwh",
    "daily_forecast_kwh",
    "hourly_shape",
}
ERROR_CODE_BY_STAGE = {
    "zone_selection": "MAPF-E010",
    "load_data": "MAPF-E020",
    "forecast": "MAPF-E100",
    "build_contexts": "MAPF-E110",
    "load_forecaster_output": "MAPF-E115",
    "precomputed_window_data": "MAPF-E120",
    "agent_chain": "MAPF-E200",
    "agent.grid": "MAPF-E210",
    "agent.behavior": "MAPF-E220",
    "agent.economist": "MAPF-E230",
    "agent.economist_repair": "MAPF-E231",
    "price_conditioned_forecast": "MAPF-E240",
    "write_outputs": "MAPF-E300",
    "unexpected": "MAPF-E900",
}
MAX_CONTROL_ATTEMPTS = 3
CONTROL_AGENT_USAGE_KEY = "_control_agent_usage"
FORECASTER_MANIFEST_SCHEMA_VERSION = 3
FORECAST_PARAMETER_NAMES = (
    "forecast_start",
    "horizon_days",
    "history_days",
    "validation_days",
    "forecast_model",
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
PRICE_CONDITIONED_PARAMETER_NAMES = tuple(
    name
    for name in FORECAST_PARAMETER_NAMES
    if name not in {"timesfm_roll_actuals", "chronos_roll_actuals", "lstm_roll_actuals"}
)


class PipelineStageError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        original: Exception,
        zone_id: Any = None,
        agent: str | None = None,
    ) -> None:
        self.stage = stage
        self.zone_id = zone_id
        self.agent = agent
        self.original = original
        location = f" zone={zone_id}" if zone_id is not None else ""
        agent_text = f" agent={agent}" if agent else ""
        super().__init__(
            f"{stage}{location}{agent_text}: {type(original).__name__}: {original}"
        )


@dataclass(frozen=True)
class ForecasterBundle:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    selected_zones: pd.DataFrame
    contexts: list[dict[str, Any]]
    forecast_parameters: dict[str, Any]
    outputs: dict[str, Path]
    scenario_artifacts: dict[str, NativeForecasterArtifact]


def run_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path | None = None,
    config_path: Path = Path("config.yaml"),
    model: str | None = None,
    weather_file: str = "weather_airport.csv",
    dry_run: bool = False,
    force_cache: bool = False,
    max_poi_rows: int | None = None,
    forecast_start: str | None = None,
    horizon_days: int = 4,
    history_days: int = 7,
    validation_days: int = 1,
    zone_ids: str | Iterable[str] | None = None,
    experiment_zone_count: int = 12,
    agent_mode: str = "multi_agent_economist_retry",
    forecast_model: str = "timesfm",
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 168,
    timesfm_step_horizon: int = 24,
    timesfm_exog_cols: list[str] | None = None,
    timesfm_diurnal_blend_alpha: float = 1.0,
    timesfm_roll_actuals: bool = False,
    ar_diurnal_blend_alpha: float = 0.0,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 512,
    chronos_step_horizon: int = 24,
    chronos_diurnal_blend_alpha: float = 0.0,
    chronos_device: str = "auto",
    chronos_roll_actuals: bool = False,
    lstm_context_hours: int = 24,
    lstm_step_horizon: int = 24,
    lstm_exog_cols: list[str] | None = None,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    lstm_epochs: int = 50,
    lstm_learning_rate: float = 0.001,
    lstm_batch_size: int = 32,
    lstm_diurnal_blend_alpha: float = 0.0,
    lstm_device: str = "auto",
    lstm_roll_actuals: bool = False,
    lstm_seed: int = 42,
    temperature: float = 0.2,
    precomputed_window_data: Path | str | None = None,
    pipeline_stage: str = "full",
    forecaster_output_dir: Path | str | None = None,
    agent_output_dir: Path | str | None = None,
    dataset_spec: DatasetSpec | None = None,
) -> dict[str, Path]:
    forecast_model = normalize_forecast_model_name(forecast_model)
    pipeline_stage = normalize_pipeline_stage(pipeline_stage)
    agent_mode = normalize_agent_mode(agent_mode)
    chain_mode = (
        "single_agent"
        if agent_mode.startswith("single_agent")
        else "multi_agent_discussion"
        if agent_mode == "multi_agent_discussion_3rounds"
        else "multi_agent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_spec = dataset_spec or DatasetSpec(
        path=data_dir,
        weather_file=weather_file,
        cache_dir=cache_dir,
    )
    shared_cache_dir = cache_dir or dataset_spec.resolved_cache_dir
    run_output_dir = forecast_output_dir(output_dir, forecast_model)
    forecaster_dir = resolve_forecaster_output_dir(
        run_output_dir,
        forecaster_output_dir if pipeline_stage == "agent" else None,
    )
    resolved_agent_output_dir = (
        Path(agent_output_dir)
        if agent_output_dir not in (None, "")
        else forecaster_dir.parent / "agent"
    )
    forecaster_outputs: dict[str, Path] = {}
    bundle: ForecasterBundle | None = None
    forecast_parameters = make_forecast_parameters(
        forecast_start=forecast_start,
        horizon_days=horizon_days,
        history_days=history_days,
        validation_days=validation_days,
        forecast_model=forecast_model,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_step_horizon=timesfm_step_horizon,
        timesfm_exog_cols=timesfm_exog_cols,
        timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
        timesfm_roll_actuals=timesfm_roll_actuals,
        ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_step_horizon=chronos_step_horizon,
        chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
        chronos_device=chronos_device,
        chronos_roll_actuals=chronos_roll_actuals,
        lstm_context_hours=lstm_context_hours,
        lstm_step_horizon=lstm_step_horizon,
        lstm_exog_cols=lstm_exog_cols,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_epochs=lstm_epochs,
        lstm_learning_rate=lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
        lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
        lstm_device=lstm_device,
        lstm_roll_actuals=lstm_roll_actuals,
        lstm_seed=lstm_seed,
    )

    if pipeline_stage == "agent":
        try:
            bundle = load_forecaster_bundle(forecaster_dir)
        except Exception as exc:
            raise PipelineStageError(stage="load_forecaster_output", original=exc) from exc
        selected_zones = bundle.selected_zones
        contexts = bundle.contexts
        forecast_parameters = bundle.forecast_parameters
        forecast_model = str(forecast_parameters["forecast_model"])
        source = bundle.manifest.get("data_source") or {}
        recorded_data_dir = source.get("data_dir")
        if recorded_data_dir:
            data_dir = Path(str(recorded_data_dir))
        weather_file = str(source.get("weather_file") or weather_file)
        dataset_spec = replace(
            dataset_spec,
            path=data_dir,
            weather_file=weather_file,
        )
        if cache_dir is None and dataset_spec.cache_dir is None:
            shared_cache_dir = dataset_spec.resolved_cache_dir
        forecaster_outputs = bundle.outputs
        scenario_artifacts = bundle.scenario_artifacts

    canonical_dataset = load_canonical_dataset(dataset_spec, force_cache=force_cache)
    if bundle is not None:
        recorded_fingerprint = (bundle.manifest.get("data_source") or {}).get(
            "dataset_fingerprint"
        )
        if recorded_fingerprint and recorded_fingerprint != canonical_dataset.dataset_fingerprint:
            raise PipelineStageError(
                stage="load_forecaster_output",
                original=ValueError(
                    "Dataset fingerprint changed after the forecaster artifact was created; "
                    "rerun the forecaster stage before running the control stage."
                ),
            )
    effective_forecast_start = (
        pd.Timestamp(forecast_parameters.get("forecast_start"))
        if forecast_parameters.get("forecast_start")
        else pd.Timestamp(canonical_dataset.timeseries["timestamp"].max())
        - pd.Timedelta(hours=int(forecast_parameters["horizon_days"]) * 24 - 1)
    )
    split_key = split_cache_key(
        canonical_dataset.dataset_fingerprint,
        effective_forecast_start,
        window_hours=3,
        policy_id="historical_3h_min_max_range_35_80_90",
    )
    split_cache_dir = (
        shared_cache_dir
        / "datasets"
        / canonical_dataset.dataset_fingerprint
        / "splits"
        / split_key
    )
    profiles = build_zone_profiles_from_canonical(
        canonical_dataset,
        split_cache_dir,
        forecast_start=effective_forecast_start,
        force_cache=force_cache,
    )
    zone_load_thresholds = build_zone_3h_load_policy_thresholds(
        data_dir,
        split_cache_dir,
        force_cache=force_cache,
        source_file="canonical_dataset",
        forecast_start=effective_forecast_start,
        load_frame=canonical_dataset.wide_frame(
            "load_kwh",
            profiles["zone_id"].astype(str).tolist(),
        ),
    )
    if pipeline_stage != "agent":
        requested_zone_ids = normalize_zone_ids(zone_ids)
        try:
            selected_zones = (
                select_requested_zones(profiles, requested_zone_ids)
                if requested_zone_ids
                else select_zone_categories(profiles)
            )
        except Exception as exc:
            raise PipelineStageError(stage="zone_selection", original=exc) from exc
    selected_zone_ids = selected_zones["zone_id"].astype(str).tolist()
    try:
        pipeline_data = load_pipeline_data(
            data_dir,
            profiles,
            selected_zone_ids,
            weather_file=weather_file,
            canonical_dataset=canonical_dataset,
        )
        pipeline_data = replace(
            pipeline_data,
            cache_keys={
                "dataset": canonical_dataset.dataset_fingerprint,
                "split": split_key,
            },
        )
    except Exception as exc:
        raise PipelineStageError(stage="load_data", original=exc) from exc
    if pipeline_stage != "agent":
        try:
            contexts, forecast_results = build_contexts(
                pipeline_data=pipeline_data,
                selected_zones=selected_zones,
                zone_load_thresholds=zone_load_thresholds,
                **forecast_parameters,
            )
            scenario_artifacts = {
                str(zone_id): NativeForecasterArtifact.from_forecast_result(result)
                for zone_id, result in forecast_results.items()
            }
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError(stage="build_contexts", original=exc) from exc
        try:
            forecaster_outputs = write_forecaster_outputs(
                output_dir=forecaster_dir,
                selected_zones=selected_zones,
                contexts=contexts,
                forecast_results=forecast_results,
                manifest=build_forecaster_manifest(
                    data_dir=data_dir,
                    weather_file=weather_file,
                    selected_zone_ids=selected_zone_ids,
                    forecast_parameters=forecast_parameters,
                    dataset_fingerprint=canonical_dataset.dataset_fingerprint,
                    cache_keys=pipeline_data.cache_keys or {},
                    feature_manifest=pipeline_data.feature_manifest or {},
                    dataset_adapter=dataset_spec.adapter,
                ),
            )
        except Exception as exc:
            raise PipelineStageError(stage="write_outputs", original=exc) from exc
        if pipeline_stage == "forecaster":
            return forecaster_outputs

    try:
        precomputed_frame = load_precomputed_window_data(precomputed_window_data)
        cached_contexts, live_contexts, cached_windows_by_zone = partition_precomputed_contexts(
            contexts,
            precomputed_frame,
            forecast_model=forecast_model,
            agent_mode=agent_mode,
        )
    except Exception as exc:
        raise PipelineStageError(stage="precomputed_window_data", original=exc) from exc

    async def execute_agent_control_stage() -> list[dict[str, Any]]:
        reports_by_zone: dict[str, dict[str, Any]] = {}
        client: AgentChatClient | None = None
        heuristic_source = "dry-run"
        try:
            try:
                if cached_contexts:
                    cached_reports = await run_all_zone_chains(
                        cached_contexts,
                        client=None,
                        temperature=temperature,
                        heuristic_source="precomputed_window_data",
                        chain_mode=chain_mode,
                    )
                    cached_reports = apply_precomputed_window_data(
                        cached_reports,
                        cached_windows_by_zone,
                        source_path=precomputed_window_data,
                    )
                    reports_by_zone.update(
                        {str(report.get("zone_id")): report for report in cached_reports}
                    )

                if live_contexts:
                    if dry_run:
                        heuristic_source = f"dry-run_{agent_mode}"
                    else:
                        config = AgentConfig.from_file(
                            config_path,
                            model=model,
                            required=True,
                            agent_mode=agent_mode,
                        )
                        if not config.api_key:
                            raise RuntimeError(
                                f"agent.{config.profile}.api_key is required in config.yaml, "
                                "or pass --dry-run"
                            )
                        config = select_agent_config_for_mode(
                            config,
                            agent_mode=agent_mode,
                            cli_model=model,
                        )
                        client = AgentChatClient(config)
                    live_reports = await run_all_zone_chains(
                        live_contexts,
                        client=client,
                        temperature=temperature,
                        heuristic_source=heuristic_source,
                        chain_mode=chain_mode,
                    )
                    reports_by_zone.update(
                        {str(report.get("zone_id")): report for report in live_reports}
                    )
                reports = [
                    reports_by_zone[str(context.get("zone_id"))] for context in contexts
                ]
            except AgentStageError:
                raise
            except Exception as exc:
                raise PipelineStageError(stage="agent_chain", original=exc) from exc

            try:
                live_zone_ids = {str(context.get("zone_id")) for context in live_contexts}
                live_reports = [
                    report
                    for report in reports
                    if str(report.get("zone_id")) in live_zone_ids
                ]
                live_reports = await run_control_loop(
                    reports=live_reports,
                    contexts=live_contexts,
                    client=client,
                    heuristic_source=heuristic_source,
                    agent_mode=agent_mode,
                    temperature=temperature,
                    pipeline_data=pipeline_data,
                    zone_load_thresholds=zone_load_thresholds,
                    price_change_reference=pipeline_data.price_change_reference,
                    forecast_parameters=forecast_parameters,
                    scenario_artifacts=scenario_artifacts,
                )
                reports_by_zone.update(
                    {str(report.get("zone_id")): report for report in live_reports}
                )
                return [
                    reports_by_zone[str(context.get("zone_id"))] for context in contexts
                ]
            except Exception as exc:
                if isinstance(exc, PipelineStageError):
                    raise
                raise PipelineStageError(
                    stage="price_conditioned_forecast",
                    original=exc,
                ) from exc
        finally:
            if client is not None:
                await client.aclose()

    reports = asyncio.run(execute_agent_control_stage())
    try:
        agent_outputs = write_agent_outputs(
            output_dir=resolved_agent_output_dir,
            reports=reports,
            forecast_model=forecast_model,
            agent_mode=agent_mode,
            manifest=build_agent_manifest(
                agent_mode=agent_mode,
                forecast_model=forecast_model,
                forecaster_dir=forecaster_dir,
                selected_zone_ids=selected_zone_ids,
                forecast_origin=forecast_parameters.get("forecast_start"),
                dataset_fingerprint=canonical_dataset.dataset_fingerprint,
                cache_keys=pipeline_data.cache_keys or {},
                feature_manifest=pipeline_data.feature_manifest or {},
            ),
        )
        return {**forecaster_outputs, **agent_outputs}
    except Exception as exc:
        raise PipelineStageError(stage="write_outputs", original=exc) from exc


def make_forecast_parameters(**values: Any) -> dict[str, Any]:
    missing = [name for name in FORECAST_PARAMETER_NAMES if name not in values]
    if missing:
        raise ValueError("Missing forecast parameter(s): " + ", ".join(missing))
    parameters = {name: values[name] for name in FORECAST_PARAMETER_NAMES}
    parameters["forecast_model"] = normalize_forecast_model_name(
        str(parameters["forecast_model"])
    )
    return parameters


def price_conditioned_parameters(forecast_parameters: dict[str, Any]) -> dict[str, Any]:
    missing = [
        name for name in PRICE_CONDITIONED_PARAMETER_NAMES if name not in forecast_parameters
    ]
    if missing:
        raise ValueError(
            "Forecaster output is missing price-conditioned forecast parameter(s): "
            + ", ".join(missing)
        )
    return {name: forecast_parameters[name] for name in PRICE_CONDITIONED_PARAMETER_NAMES}


def resolve_forecaster_output_dir(
    run_output_dir: Path,
    value: Path | str | None,
) -> Path:
    if value in (None, ""):
        return run_output_dir / "forecaster"
    path = Path(value)
    if path.name.lower() == "forecaster_manifest.json":
        return path.parent
    return path


def build_forecaster_manifest(
    *,
    data_dir: Path,
    weather_file: str,
    selected_zone_ids: list[str],
    forecast_parameters: dict[str, Any],
    dataset_fingerprint: str | None = None,
    cache_keys: dict[str, str] | None = None,
    feature_manifest: dict[str, Any] | None = None,
    dataset_adapter: str = "urbanev",
) -> dict[str, Any]:
    return {
        "schema_version": FORECASTER_MANIFEST_SCHEMA_VERSION,
        "stage": "forecaster",
        "created_at": pd.Timestamp.now().isoformat(),
        "data_source": {
            "data_dir": str(data_dir.resolve()),
            "weather_file": weather_file,
            "adapter": dataset_adapter,
            "dataset_fingerprint": dataset_fingerprint,
            "cache_keys": cache_keys or {},
        },
        "feature_manifest": feature_manifest or {},
        "zone_ids": list(selected_zone_ids),
        "forecast_parameters": dict(forecast_parameters),
    }


def build_agent_manifest(
    *,
    agent_mode: str,
    forecast_model: str,
    forecaster_dir: Path,
    selected_zone_ids: list[str],
    forecast_origin: Any = None,
    dataset_fingerprint: str | None = None,
    cache_keys: dict[str, str] | None = None,
    feature_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "agent",
        "completed_at": pd.Timestamp.now().isoformat(),
        "agent_mode": agent_mode,
        "forecast_model": forecast_model,
        "forecast_origin": pd.Timestamp(forecast_origin).isoformat()
        if forecast_origin is not None
        else None,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_keys": cache_keys or {},
        "feature_manifest": feature_manifest or {},
        "forecaster_output_dir": str(forecaster_dir.resolve()),
        "zone_ids": list(selected_zone_ids),
    }


def load_forecaster_bundle(path: Path | str) -> ForecasterBundle:
    output_dir = resolve_forecaster_output_dir(Path("."), path)
    manifest_path = output_dir / "forecaster_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Forecaster manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Forecaster manifest must contain a JSON object: {manifest_path}")
    if manifest.get("schema_version") != FORECASTER_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported forecaster manifest schema_version: "
            f"{manifest.get('schema_version')}; expected {FORECASTER_MANIFEST_SCHEMA_VERSION}. "
            "Run the forecaster stage again so closed-loop validation can reuse "
            "the original backend instead of a legacy approximation artifact."
        )
    if manifest.get("stage") != "forecaster":
        raise ValueError(f"Invalid forecaster manifest stage: {manifest.get('stage')}")

    raw_parameters = manifest.get("forecast_parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("Forecaster manifest is missing forecast_parameters")
    forecast_parameters = make_forecast_parameters(**raw_parameters)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Forecaster manifest is missing artifacts")
    selected_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("selected_zones"),
        "selected_zones",
    )
    contexts_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("context_snippets"),
        "context_snippets",
    )
    model_artifact_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("forecaster_artifact_json"),
        "forecaster_artifact_json",
    )
    selected_zones = pd.read_csv(selected_path, dtype={"zone_id": "string"})
    if "zone_id" not in selected_zones:
        raise ValueError(f"Forecaster selected_zones is missing zone_id: {selected_path}")
    selected_zones["zone_id"] = selected_zones["zone_id"].astype("string").str.strip()
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if not isinstance(contexts, list) or not all(isinstance(item, dict) for item in contexts):
        raise ValueError(f"Forecaster context_snippets must contain a JSON list: {contexts_path}")
    model_payload = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    if (
        not isinstance(model_payload, dict)
        or model_payload.get("schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION
        or model_payload.get("artifact_type") != "native_backend_reforecast"
    ):
        raise ValueError(
            "Forecaster output uses a legacy approximation artifact. Run the "
            "forecaster stage again to create native backend reforecast state."
        )
    raw_scenario_artifacts = model_payload.get("artifacts") if isinstance(model_payload, dict) else None
    if not isinstance(raw_scenario_artifacts, dict):
        raise ValueError("Forecaster artifact is missing per-zone artifacts")
    scenario_artifacts = {
        str(zone_id): NativeForecasterArtifact.from_dict(payload)
        for zone_id, payload in raw_scenario_artifacts.items()
        if isinstance(payload, dict)
    }

    selected_ids = selected_zones["zone_id"].astype(str).tolist()
    context_ids = [str(context.get("zone_id")) for context in contexts]
    manifest_ids = [str(value) for value in manifest.get("zone_ids") or []]
    if selected_ids != context_ids or selected_ids != manifest_ids:
        raise ValueError(
            "Forecaster output zone order mismatch between manifest, selected_zones, "
            "and context_snippets"
        )
    if set(scenario_artifacts) != set(selected_ids):
        raise ValueError(
            "Forecaster artifact Zone set does not match the no-leakage handoff."
        )

    outputs = {
        "forecaster_output_dir": output_dir,
        "forecaster_manifest_json": manifest_path,
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "forecaster_artifact_json": model_artifact_path,
    }
    for name in (
        "forecast_metrics_csv",
        "forecast_metrics_md",
        "forecast_details_dir",
    ):
        value = artifacts.get(name)
        if value:
            outputs[name] = output_dir / str(value)
    return ForecasterBundle(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        selected_zones=selected_zones,
        contexts=contexts,
        forecast_parameters=forecast_parameters,
        outputs=outputs,
        scenario_artifacts=scenario_artifacts,
    )


def forecaster_artifact_path(output_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Forecaster manifest is missing artifact path: {name}")
    path = output_dir / value
    if not path.is_file():
        raise FileNotFoundError(f"Forecaster artifact not found: {path}")
    return path


def select_agent_config_for_mode(
    config: AgentConfig,
    *,
    agent_mode: str,
    cli_model: str | None,
) -> AgentConfig:
    expected_profile = agent_config_profile(agent_mode)
    if config.profile == expected_profile:
        return replace(config, model=cli_model) if cli_model else config
    if expected_profile == "single_agent" and cli_model is None and config.single_agent_model:
        return replace(
            config,
            model=config.single_agent_model,
            profile="single_agent",
        )
    return config


def run_experiment_matrix(
    *,
    data_dir: Path,
    output_dir: Path,
    forecast_starts: Iterable[str],
    forecast_models: Iterable[str],
    zone_ids: str | Iterable[str] | None = None,
    experiment_zone_count: int = 12,
    experiment_seeds: Iterable[int] | None = None,
    agent_modes: Iterable[str] | None = None,
    diurnal_blend_alphas: Iterable[float] | None = None,
    experiment_name: str | None = None,
    **pipeline_kwargs: Any,
) -> dict[str, Path]:
    starts = normalize_forecast_starts(forecast_starts)
    models = normalize_forecast_models(forecast_models)
    seeds = normalize_experiment_seeds(experiment_seeds)
    matrix_stage = normalize_pipeline_stage(pipeline_kwargs.get("pipeline_stage", "full"))
    requested_modes = normalize_agent_modes(
        agent_modes,
        pipeline_kwargs.get("agent_mode", "multi_agent_economist_retry"),
    )
    modes = requested_modes
    if matrix_stage == "forecaster":
        modes = modes[:1]
    blend_alphas = normalize_blend_alphas(diurnal_blend_alphas)
    selected_zone_ids = normalize_zone_ids(zone_ids)
    shared_cache_dir = Path(pipeline_kwargs.pop("cache_dir", data_dir / "cache"))
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    base_force_cache = bool(pipeline_kwargs.pop("force_cache", False))
    if not selected_zone_ids:
        matrix_dataset_spec = pipeline_kwargs.get("dataset_spec") or DatasetSpec(
            path=data_dir,
            cache_dir=shared_cache_dir,
            weather_file=str(pipeline_kwargs.get("weather_file") or "weather_airport.csv"),
        )
        canonical_dataset = load_canonical_dataset(
            matrix_dataset_spec,
            force_cache=base_force_cache,
        )
        selection_start = pd.Timestamp(min(starts))
        selection_split_key = split_cache_key(
            canonical_dataset.dataset_fingerprint,
            selection_start,
            window_hours=3,
            policy_id="historical_3h_min_max_range_35_80_90",
        )
        profiles = build_zone_profiles_from_canonical(
            canonical_dataset,
            shared_cache_dir
            / "datasets"
            / canonical_dataset.dataset_fingerprint
            / "splits"
            / selection_split_key,
            forecast_start=selection_start,
            force_cache=base_force_cache,
        )
        selected_zone_ids = select_representative_zone_ids(
            profiles,
            count=experiment_zone_count,
        )
    resolved_experiment_name = normalize_experiment_name(
        experiment_name,
        default=experiment_slug(
            zone_count=len(selected_zone_ids),
            time_count=len(starts),
            mode_count=len(requested_modes),
            blend_count=len(blend_alphas),
        ),
    )
    experiment_dir = output_dir / resolved_experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    runs_path = experiment_dir / "experiment_runs.csv"
    metrics_path = experiment_dir / "experiment_forecast_metrics.csv"
    price_path = experiment_dir / "experiment_price_comparison_summary.csv"
    rationale_path = experiment_dir / "experiment_rationale_trace.csv"
    explainability_path = experiment_dir / "experiment_explainability_review_packet.csv"
    agent_attempt_usage_path = experiment_dir / "experiment_agent_attempt_usage.csv"
    agent_step_token_usage_path = experiment_dir / "experiment_agent_step_token_usage.csv"
    control_attempt_trace_path = experiment_dir / "experiment_control_attempt_trace.csv"
    forecast_summary_path = experiment_dir / "experiment_forecast_summary.csv"
    price_summary_path = experiment_dir / "experiment_price_summary.csv"
    rationale_summary_path = experiment_dir / "experiment_rationale_summary.csv"
    decision_summary_path = experiment_dir / "experiment_decision_quality_summary.csv"

    run_records: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    rationale_frames: list[pd.DataFrame] = []
    explainability_frames: list[pd.DataFrame] = []
    agent_attempt_usage_frames: list[pd.DataFrame] = []
    agent_step_token_usage_frames: list[pd.DataFrame] = []
    control_attempt_trace_frames: list[pd.DataFrame] = []
    cache_attempted = False
    add_seed_folder = len(seeds) > 1
    add_mode_folder = len(modes) > 1
    add_blend_folder = len(blend_alphas) > 1

    for forecast_start in starts:
        time_output_dir = experiment_dir / forecast_start_slug(forecast_start)
        for forecast_model in models:
            for seed in seeds:
                for agent_mode in modes:
                    for blend_alpha in blend_alphas:
                        run_base_dir = time_output_dir
                        if add_seed_folder and seed is not None:
                            run_base_dir = run_base_dir / f"seed_{seed}"
                        if add_mode_folder:
                            run_base_dir = run_base_dir / f"agent_{safe_filename(agent_mode)}"
                        if add_blend_folder and blend_alpha is not None:
                            alpha_text = safe_filename(f"{float(blend_alpha):.3f}".rstrip("0").rstrip("."))
                            run_base_dir = run_base_dir / f"blend_{alpha_text}"
                        run_output_dir = forecast_output_dir(run_base_dir, forecast_model)
                        run_kwargs = dict(pipeline_kwargs)
                        run_kwargs["agent_mode"] = agent_mode
                        if seed is not None:
                            run_kwargs["lstm_seed"] = seed
                        if blend_alpha is not None:
                            alpha_value = float(blend_alpha)
                            run_kwargs["timesfm_diurnal_blend_alpha"] = alpha_value
                            run_kwargs["ar_diurnal_blend_alpha"] = alpha_value
                            run_kwargs["chronos_diurnal_blend_alpha"] = alpha_value
                            run_kwargs["lstm_diurnal_blend_alpha"] = alpha_value
                        if matrix_stage == "agent":
                            shared_run_base_dir = time_output_dir
                            if add_seed_folder and seed is not None:
                                shared_run_base_dir = shared_run_base_dir / f"seed_{seed}"
                            if add_blend_folder and blend_alpha is not None:
                                shared_run_base_dir = shared_run_base_dir / f"blend_{alpha_text}"
                            if not run_kwargs.get("forecaster_output_dir"):
                                run_kwargs["forecaster_output_dir"] = (
                                    forecast_output_dir(shared_run_base_dir, forecast_model)
                                    / "forecaster"
                                )
                            if not run_kwargs.get("agent_output_dir"):
                                run_kwargs["agent_output_dir"] = run_output_dir / "agent"
                        record: dict[str, Any] = {
                            "forecast_start": forecast_start,
                            "forecast_model": forecast_model,
                            "experiment_seed": seed,
                            "agent_mode": agent_mode,
                            "diurnal_blend_alpha": blend_alpha,
                            "zone_count": len(selected_zone_ids),
                            "zone_ids": ",".join(selected_zone_ids),
                            "run_output_dir": str(run_output_dir),
                            "status": "running",
                            "error": "",
                            "error_code": "",
                            "error_stage": "",
                            "error_zone_id": "",
                            "error_agent": "",
                            "error_type": "",
                            "error_message": "",
                            "traceback_file": "",
                            "completed_at": "",
                        }
                        try:
                            outputs = run_pipeline(
                                data_dir=data_dir,
                                output_dir=run_base_dir,
                                cache_dir=shared_cache_dir,
                                force_cache=base_force_cache and not cache_attempted,
                                forecast_start=forecast_start,
                                forecast_model=forecast_model,
                                zone_ids=selected_zone_ids,
                                **run_kwargs,
                            )
                            record["status"] = "success"
                            record["completed_at"] = pd.Timestamp.now().isoformat()
                            metadata = {
                                "forecast_start": forecast_start,
                                "forecast_model": forecast_model,
                                "experiment_seed": seed,
                                "agent_mode": agent_mode,
                                "diurnal_blend_alpha": blend_alpha,
                                "run_output_dir": str(run_output_dir),
                            }
                            append_experiment_frame(
                                metrics_frames,
                                outputs.get("forecast_metrics_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                price_frames,
                                outputs.get("price_comparison_summary_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                rationale_frames,
                                outputs.get("rationale_trace_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                explainability_frames,
                                outputs.get("explainability_review_packet_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                agent_attempt_usage_frames,
                                outputs.get("agent_attempt_usage_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                agent_step_token_usage_frames,
                                outputs.get("agent_step_token_usage_csv"),
                                metadata,
                            )
                            append_experiment_frame(
                                control_attempt_trace_frames,
                                outputs.get("control_attempt_trace_csv"),
                                metadata,
                            )
                        except Exception as exc:
                            record["status"] = "failed"
                            record["completed_at"] = pd.Timestamp.now().isoformat()
                            record.update(
                                experiment_error_record(
                                    exc,
                                    experiment_dir=experiment_dir,
                                    record=record,
                                    run_index=len(run_records) + 1,
                                )
                            )
                            raise
                        finally:
                            cache_attempted = True
                            run_records.append(record)
                            pd.DataFrame(run_records).to_csv(runs_path, index=False)

    metrics = write_experiment_summary(metrics_frames, metrics_path)
    prices = write_experiment_summary(price_frames, price_path)
    rationales = write_experiment_summary(rationale_frames, rationale_path)
    write_experiment_summary(explainability_frames, explainability_path)
    write_experiment_summary(agent_attempt_usage_frames, agent_attempt_usage_path)
    write_experiment_summary(
        agent_step_token_usage_frames,
        agent_step_token_usage_path,
    )
    write_experiment_summary(control_attempt_trace_frames, control_attempt_trace_path)
    write_numeric_summary(
        metrics,
        forecast_summary_path,
        metric_columns=["MAE", "RMSE", "MAPE_pct", "RAE", "WAPE_pct", "stress_accuracy", "miss_stress_rate"],
    )
    write_numeric_summary(
        prices,
        price_summary_path,
        metric_columns=[
            "price_accuracy",
            "avg_predicted_minus_baseline_energy_price",
            "avg_predicted_vs_baseline_pct",
        ],
    )
    write_numeric_summary(
        rationales,
        rationale_summary_path,
        metric_columns=["stress_accuracy", "miss_stress_rate", "final_price_shift_pct"],
    )
    write_decision_quality_summary(prices, rationales, decision_summary_path)
    return {
        "experiment_dir": experiment_dir,
        "experiment_runs_csv": runs_path,
        "experiment_forecast_metrics_csv": metrics_path,
        "experiment_price_comparison_summary_csv": price_path,
        "experiment_rationale_trace_csv": rationale_path,
        "experiment_explainability_review_packet_csv": explainability_path,
        "experiment_agent_attempt_usage_csv": agent_attempt_usage_path,
        "experiment_agent_step_token_usage_csv": agent_step_token_usage_path,
        "experiment_control_attempt_trace_csv": control_attempt_trace_path,
        "experiment_forecast_summary_csv": forecast_summary_path,
        "experiment_price_summary_csv": price_summary_path,
        "experiment_rationale_summary_csv": rationale_summary_path,
        "experiment_decision_quality_summary_csv": decision_summary_path,
    }


def normalize_forecast_starts(forecast_starts: Iterable[str]) -> list[str]:
    starts: list[str] = []
    seen: set[str] = set()
    for value in forecast_starts:
        if value in (None, ""):
            continue
        timestamp = pd.Timestamp(str(value).strip())
        start = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if start not in seen:
            starts.append(start)
            seen.add(start)
    if not starts:
        raise ValueError("At least one forecast start is required for an experiment matrix.")
    return starts


def normalize_forecast_models(forecast_models: Iterable[str]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for value in forecast_models:
        if value in (None, ""):
            continue
        model_name = normalize_forecast_model_name(str(value))
        if model_name not in seen:
            models.append(model_name)
            seen.add(model_name)
    if not models:
        raise ValueError("At least one forecast model is required for an experiment matrix.")
    return models


def normalize_experiment_seeds(experiment_seeds: Iterable[int] | None) -> list[int | None]:
    if experiment_seeds is None:
        return [None]
    seeds: list[int | None] = []
    seen: set[int] = set()
    for value in experiment_seeds:
        seed = int(value)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return seeds or [None]


def normalize_agent_modes(agent_modes: Iterable[str] | None, default: str) -> list[str]:
    raw_modes = list(agent_modes) if agent_modes is not None else [default]
    modes: list[str] = []
    seen: set[str] = set()
    for value in raw_modes:
        mode = normalize_agent_mode(str(value))
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    return modes or [normalize_agent_mode(default)]


def normalize_blend_alphas(diurnal_blend_alphas: Iterable[float] | None) -> list[float | None]:
    if diurnal_blend_alphas is None:
        return [None]
    values: list[float | None] = []
    seen: set[float] = set()
    for value in diurnal_blend_alphas:
        alpha = round(float(value), 6)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Diurnal blend alpha must be between 0 and 1: {value}")
        if alpha not in seen:
            values.append(alpha)
            seen.add(alpha)
    return values or [None]


def experiment_slug(
    *,
    zone_count: int,
    time_count: int,
    mode_count: int,
    blend_count: int,
) -> str:
    return f"{int(zone_count)}zonesx{int(time_count)}timesx{int(mode_count)}modes_{int(blend_count)}blends"


def forecast_start_slug(forecast_start: str) -> str:
    return pd.Timestamp(forecast_start).strftime("%Y-%m-%d_%H%M%S")


def append_experiment_frame(
    frames: list[pd.DataFrame],
    path: Path | None,
    metadata: dict[str, Any],
) -> None:
    if path is None or not path.exists():
        return
    frame = pd.read_csv(path)
    insert_at = 0
    for column, value in metadata.items():
        if column in frame.columns:
            frame[column] = value
        else:
            frame.insert(insert_at, column, value)
            insert_at += 1
    frames.append(frame)


def write_experiment_summary(frames: list[pd.DataFrame], path: Path) -> pd.DataFrame:
    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False)
    return frame


def experiment_error_record(
    exc: Exception,
    *,
    experiment_dir: Path,
    record: dict[str, Any],
    run_index: int,
) -> dict[str, Any]:
    details = classify_experiment_error(exc)
    error = (
        f"{details['error_code']} {details['error_stage']}: "
        f"{details['error_type']}: {details['error_message']}"
    )
    failure_record = {
        "error": error,
        "error_code": details["error_code"],
        "error_stage": details["error_stage"],
        "error_zone_id": details["error_zone_id"],
        "error_agent": details["error_agent"],
        "error_type": details["error_type"],
        "error_message": details["error_message"],
    }
    traceback_path = write_error_traceback(
        exc,
        experiment_dir=experiment_dir,
        record={**record, **failure_record},
        run_index=run_index,
        error_code=details["error_code"],
    )
    return {
        **failure_record,
        "traceback_file": str(traceback_path),
    }


def classify_experiment_error(exc: Exception) -> dict[str, str]:
    source = exc
    stage = "unexpected"
    zone_id = ""
    agent = ""

    if isinstance(exc, AgentStageError):
        source = exc.original
        stage = exc.stage
        zone_id = "" if exc.zone_id is None else str(exc.zone_id)
        agent = exc.agent
    elif isinstance(exc, PipelineStageError):
        source = exc.original
        stage = exc.stage
        zone_id = "" if exc.zone_id is None else str(exc.zone_id)
        agent = exc.agent or ""
        if isinstance(source, AgentStageError):
            stage = source.stage
            zone_id = "" if source.zone_id is None else str(source.zone_id)
            agent = source.agent
            source = source.original

    return {
        "error_code": error_code_for_stage(stage),
        "error_stage": stage,
        "error_zone_id": zone_id,
        "error_agent": agent,
        "error_type": type(source).__name__,
        "error_message": str(source),
    }


def error_code_for_stage(stage: str) -> str:
    if stage in ERROR_CODE_BY_STAGE:
        return ERROR_CODE_BY_STAGE[stage]
    if stage.startswith("agent.discussion."):
        if stage.endswith(".grid"):
            return ERROR_CODE_BY_STAGE["agent.grid"]
        if stage.endswith(".behavior"):
            return ERROR_CODE_BY_STAGE["agent.behavior"]
        if stage.endswith(".economist.schema_repair"):
            return ERROR_CODE_BY_STAGE["agent.economist_repair"]
        if stage.endswith(".economist"):
            return ERROR_CODE_BY_STAGE["agent.economist"]
    return ERROR_CODE_BY_STAGE["unexpected"]


def format_failure_message(exc: Exception) -> str:
    details = classify_experiment_error(exc)
    lines = [
        "Execution stopped because an error occurred.",
        f"error_code: {details['error_code']}",
        f"stage: {details['error_stage']}",
    ]
    if details["error_zone_id"]:
        lines.append(f"zone_id: {details['error_zone_id']}")
    if details["error_agent"]:
        lines.append(f"agent: {details['error_agent']}")
    lines.extend(
        [
            f"error_type: {details['error_type']}",
            f"reason: {details['error_message']}",
        ]
    )
    return "\n".join(lines)


def write_error_traceback(
    exc: Exception,
    *,
    experiment_dir: Path,
    record: dict[str, Any],
    run_index: int,
    error_code: str,
) -> Path:
    errors_dir = experiment_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    blend_value = record.get("diurnal_blend_alpha")
    blend_text = "blend" if blend_value in (None, "") else str(blend_value)
    name_parts = [
        f"{run_index:04d}",
        error_code,
        forecast_start_slug(str(record.get("forecast_start") or "unknown")),
        safe_filename(record.get("forecast_model") or "model"),
        safe_filename(record.get("agent_mode") or "mode"),
        safe_filename(blend_text),
    ]
    path = errors_dir / ("_".join(name_parts) + ".txt")
    traceback_record = {**record, "traceback_file": str(path)}
    metadata = "\n".join(f"{key}: {value}" for key, value in traceback_record.items())
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path.write_text(f"{metadata}\n\nTRACEBACK:\n{trace}", encoding="utf-8")
    return path


def write_numeric_summary(
    frame: pd.DataFrame,
    path: Path,
    *,
    metric_columns: list[str],
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = group_columns or ["forecast_model", "agent_mode", "diurnal_blend_alpha"]
    output_columns = [
        *group_columns,
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
    ]
    if frame.empty:
        empty = pd.DataFrame(columns=output_columns)
        empty.to_csv(path, index=False)
        return empty

    groups = [column for column in group_columns if column in frame.columns]
    rows = []
    grouped = frame.groupby(groups, dropna=False) if groups else [((), frame)]
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        group_meta = dict(zip(groups, key_values))
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    **group_meta,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": round(float(values.mean()), 6),
                    "std": round(std, 6),
                    "sem": round(std / (len(values) ** 0.5), 6) if len(values) > 1 else 0.0,
                    "min": round(float(values.min()), 6),
                    "max": round(float(values.max()), 6),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=output_columns)
    summary.to_csv(path, index=False)
    return summary


def write_decision_quality_summary(
    prices: pd.DataFrame,
    rationales: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not prices.empty:
        frames.append(
            long_metric_frame(
                prices,
                metrics=[
                    "price_accuracy",
                    "avg_predicted_minus_baseline_energy_price",
                    "avg_predicted_vs_baseline_pct",
                ],
                metric_family="price_decision",
            )
        )
    if not rationales.empty:
        frames.append(
            long_metric_frame(
                rationales,
                metrics=["stress_accuracy", "miss_stress_rate", "final_price_shift_pct"],
                metric_family="stress_and_price_trace",
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    columns = [
        "forecast_model",
        "agent_mode",
        "diurnal_blend_alpha",
        "metric_family",
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
    ]
    if combined.empty:
        summary = pd.DataFrame(columns=columns)
        summary.to_csv(path, index=False)
        return summary

    groups = [
        column
        for column in ["forecast_model", "agent_mode", "diurnal_blend_alpha", "metric_family", "metric"]
        if column in combined.columns
    ]
    rows = []
    for key, group in combined.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        group_meta = dict(zip(groups, key_values))
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                **group_meta,
                "n": int(len(values)),
                "mean": round(float(values.mean()), 6),
                "std": round(std, 6),
                "sem": round(std / (len(values) ** 0.5), 6) if len(values) > 1 else 0.0,
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(columns=columns)
    summary.to_csv(path, index=False)
    return summary


def long_metric_frame(frame: pd.DataFrame, *, metrics: list[str], metric_family: str) -> pd.DataFrame:
    group_cols = [col for col in ["forecast_model", "agent_mode", "diurnal_blend_alpha"] if col in frame.columns]
    rows = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        for _, row in frame.iterrows():
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    **{col: row.get(col) for col in group_cols},
                    "metric_family": metric_family,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def select_representative_zone_ids(profiles: pd.DataFrame, *, count: int) -> list[str]:
    frame = profiles.copy()
    if frame.empty:
        raise ValueError("Cannot select representative zones from an empty profile table.")
    frame["zone_id"] = frame["zone_id"].astype(str)
    target = max(1, min(int(count), len(frame)))
    selected: list[str] = []
    seen: set[str] = set()

    for zone_id in select_zone_categories(frame)["zone_id"].astype(str).tolist():
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            return selected

    ranked = frame.sort_values(["mean_load_kwh", "load_cv", "peak_capacity_ratio", "zone_id"]).reset_index(drop=True)
    if target == 1:
        candidate_indices = [len(ranked) // 2]
    else:
        candidate_indices = [
            round(idx * (len(ranked) - 1) / (target - 1))
            for idx in range(target)
        ]
    for idx in candidate_indices:
        zone_id = str(ranked.iloc[int(idx)]["zone_id"])
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            return selected

    for zone_id in ranked["zone_id"].astype(str).tolist():
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            break
    return selected


def forecast_output_dir(output_dir: Path, forecast_model: str) -> Path:
    normalized = normalize_forecast_model_name(forecast_model)
    return output_dir / safe_filename(normalized or "forecast")


def normalize_experiment_name(value: Any, *, default: str) -> str:
    raw = str(value).strip() if value not in (None, "") else str(default).strip()
    name = safe_filename(raw)
    if not name or name in {".", ".."}:
        raise ValueError("experiment_name must contain at least one letter or number.")
    return name


def load_precomputed_window_data(path: Path | str | None) -> pd.DataFrame:
    if path in (None, ""):
        return pd.DataFrame()
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Precomputed window data file not found: {cache_path}")
    frame = pd.read_csv(cache_path, dtype={"zone_id": "string"})
    if "price_conditioned_load_kwh" not in frame and "predicted_load_kwh" in frame:
        frame["price_conditioned_load_kwh"] = frame["predicted_load_kwh"]
    if "forecast_load_kwh" not in frame and "sum_predicted_kwh" in frame:
        frame["forecast_load_kwh"] = frame["sum_predicted_kwh"]
    required = {
        "zone_id",
        "window_start",
        "window_end",
        "predicted_energy_price",
        "price_conditioned_load_kwh",
        "zone_mean_energy_price",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Precomputed window data is missing required column(s): " + ", ".join(missing)
        )
    frame = frame.copy()
    frame["zone_id"] = frame["zone_id"].astype("string").str.strip()
    for column in ("window_start", "window_end"):
        parsed = normalize_datetime_series_24h(frame[column], errors="coerce")
        if parsed.isna().any():
            bad_rows = parsed[parsed.isna()].index.tolist()[:5]
            raise ValueError(f"Invalid {column} value in precomputed window data at row(s): {bad_rows}")
        frame[column] = parsed.map(lambda value: value.isoformat())
    return frame


def partition_precomputed_contexts(
    contexts: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    forecast_model: str,
    agent_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[tuple[str, str, str], dict[str, Any]]]]:
    if frame.empty:
        return [], list(contexts), {}

    candidates = frame.copy()
    if "forecast_model" in candidates:
        model_values = candidates["forecast_model"].fillna("").astype(str).str.strip()
        model_matches = model_values.map(
            lambda value: not value or normalize_forecast_model_name(value) == forecast_model
        )
        candidates = candidates[model_matches]
    if "agent_mode" in candidates:
        mode_values = candidates["agent_mode"].fillna("").astype(str).str.strip()

        def mode_matches(value: str) -> bool:
            if not value:
                return True
            try:
                return normalize_agent_mode(value) == agent_mode
            except ValueError:
                return False

        candidates = candidates[mode_values.map(mode_matches)]

    cache_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidates.to_dict(orient="records"):
        key = precomputed_window_key(row.get("zone_id"), row.get("window_start"), row.get("window_end"))
        if key in cache_index:
            raise ValueError(
                "Duplicate precomputed window data for "
                f"zone={key[0]}, window_start={key[1]}, window_end={key[2]}"
            )
        cache_index[key] = {name: clean_precomputed_value(value) for name, value in row.items()}

    cached_contexts: list[dict[str, Any]] = []
    live_contexts: list[dict[str, Any]] = []
    cached_windows_by_zone: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for context in contexts:
        zone_id = str(context.get("zone_id"))
        expected_windows = [
            window for window in context.get("pricing_windows_3h") or [] if isinstance(window, dict)
        ]
        zone_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        complete = bool(expected_windows)
        for window in expected_windows:
            key = precomputed_window_key(zone_id, window.get("window_start"), window.get("window_end"))
            row = cache_index.get(key)
            if (
                row is None
                or optional_number(row.get("predicted_energy_price")) is None
                or optional_number(row.get("price_conditioned_load_kwh")) is None
            ):
                complete = False
                break
            zone_rows[key] = row
        if complete:
            cached_contexts.append(context)
            cached_windows_by_zone[zone_id] = zone_rows
        else:
            live_contexts.append(context)
    return cached_contexts, live_contexts, cached_windows_by_zone


def apply_precomputed_window_data(
    reports: list[dict[str, Any]],
    cached_windows_by_zone: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    *,
    source_path: Path | str | None,
) -> list[dict[str, Any]]:
    source_name = Path(source_path).name if source_path not in (None, "") else "precomputed data"
    updated_reports: list[dict[str, Any]] = []
    passthrough_fields = (
        "load_stress_level",
        "load_range_position_pct",
        "price_conditioned_load_stress_level",
        "price_conditioned_load_range_position_pct",
        "historical_min_load_3h_kwh",
        "historical_max_load_3h_kwh",
        "historical_load_range_3h_kwh",
        "low_medium_threshold_pct",
        "medium_high_threshold_pct",
        "high_extremely_high_threshold_pct",
        "load_3h_low_medium_threshold_kwh",
        "load_3h_medium_high_threshold_kwh",
        "load_3h_high_extremely_high_threshold_kwh",
        "zone_mean_energy_price",
        "target_peak_reduction_pct",
        "baseline_load_range_position_pct",
        "target_load_kwh",
        "target_load_range_position_pct",
        "medium_load_min_kwh",
        "medium_load_max_kwh",
        "elasticity_factor",
        "load_range_source",
        "expected_load_kwh",
        "expected_load_range_position_pct",
        "load_in_medium_band",
        "action_label",
        "price_rationale",
    )
    boolean_fields = {
        "load_in_medium_band",
    }

    for report in reports:
        zone_id = str(report.get("zone_id"))
        zone_rows = cached_windows_by_zone[zone_id]
        updated_windows: list[dict[str, Any]] = []
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            key = precomputed_window_key(zone_id, window.get("window_start"), window.get("window_end"))
            row = zone_rows[key]
            updated = dict(window)
            predicted_price = optional_number(row.get("predicted_energy_price"))
            conditioned_load = optional_number(row.get("price_conditioned_load_kwh"))
            baseline_price = optional_number(updated.get("mean_energy_price"))
            final_shift = optional_number(row.get("final_price_shift_pct"))
            if final_shift is None and baseline_price not in (None, 0) and predicted_price is not None:
                final_shift = ((predicted_price / baseline_price) - 1) * 100
            forecast_load = optional_number(row.get("forecast_load_kwh"))
            if forecast_load is not None:
                updated["sum_predicted_kwh"] = forecast_load
            updated["predicted_energy_price"] = predicted_price
            updated["proposed_energy_price"] = predicted_price
            updated["price_valid"] = predicted_price is not None and predicted_price >= 0
            updated["suggested_price_shift_pct"] = round(final_shift, 4) if final_shift is not None else None
            updated["price_conditioned_baseline_load_kwh"] = conditioned_load
            updated["price_conditioned_mean_predicted_kwh"] = optional_number(
                row.get("price_conditioned_mean_load_kwh")
            )
            updated["price_conditioned_peak_predicted_kwh"] = optional_number(
                row.get("price_conditioned_peak_load_kwh")
            )
            forecaster_price = optional_number(row.get("forecaster_input_predicted_energy_price"))
            updated["price_conditioned_energy_price"] = (
                forecaster_price if forecaster_price is not None else predicted_price
            )
            updated["price_conditioned_baseline_source"] = "precomputed_window_data"
            cached_baseline_load = optional_number(row.get("baseline_load_kwh"))
            updated["baseline_load_kwh"] = (
                cached_baseline_load if cached_baseline_load is not None else conditioned_load
            )
            updated["baseline_load_source"] = row.get("baseline_load_source") or "precomputed_window_data"
            for field in passthrough_fields:
                value = row.get(field)
                if value is not None:
                    updated[field] = normalize_precomputed_bool(value) if field in boolean_fields else value
            updated_windows.append(updated)

        updated_report = dict(report)
        updated_report["price_change_windows_3h"] = updated_windows
        shifts = [optional_number(window.get("suggested_price_shift_pct")) for window in updated_windows]
        shifts = [value for value in shifts if value is not None]
        forecast_loads = [optional_number(window.get("sum_predicted_kwh")) for window in updated_windows]
        if forecast_loads and all(value is not None for value in forecast_loads):
            updated_report["predicted_load_kwh"] = round(sum(value for value in forecast_loads if value is not None), 4)
        if shifts:
            updated_report["suggested_price_shift_pct"] = round(sum(shifts) / len(shifts), 2)
        updated_report["agent_reasoning"] = f"Reused window load and energy-price predictions from {source_name}."
        updated_report["action_label"] = "Reuse precomputed window predictions"
        updated_report["price_rationale"] = f"Loaded complete 3-hour window predictions from {source_name}."
        updated_report["agent_invoked"] = False
        updated_report["agent_call_count"] = 0
        updated_report["agent_prompt_tokens"] = 0
        updated_report["agent_completion_tokens"] = 0
        updated_report["agent_total_tokens"] = 0
        updated_report["agent_token_usage_complete"] = True
        updated_report["agent_call_usage"] = []
        updated_report["source"] = "precomputed_window_data"
        updated_report["precomputed_window_data_source"] = str(source_path)
        updated_reports.append(updated_report)
    return updated_reports


def precomputed_window_key(zone_id: Any, window_start: Any, window_end: Any) -> tuple[str, str, str]:
    return (
        str(zone_id).strip(),
        parse_datetime_24h(window_start).isoformat(),
        parse_datetime_24h(window_end).isoformat(),
    )


def clean_precomputed_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def normalize_precomputed_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def apply_price_conditioned_baseline_forecasts(
    *,
    reports: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    pipeline_data,
    zone_load_thresholds: pd.DataFrame,
    scenario_artifacts: dict[str, NativeForecasterArtifact],
    forecast_start: str | None,
    horizon_days: int,
    history_days: int,
    validation_days: int,
    forecast_model: str,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_step_horizon: int,
    timesfm_exog_cols: list[str] | None,
    timesfm_diurnal_blend_alpha: float,
    ar_diurnal_blend_alpha: float,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_step_horizon: int,
    chronos_diurnal_blend_alpha: float,
    chronos_device: str,
    lstm_context_hours: int,
    lstm_step_horizon: int,
    lstm_exog_cols: list[str] | None,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    lstm_epochs: int,
    lstm_learning_rate: float,
    lstm_batch_size: int,
    lstm_diurnal_blend_alpha: float,
    lstm_device: str,
    lstm_seed: int,
) -> list[dict[str, Any]]:
    normalized_model = normalize_forecast_model_name(forecast_model)
    if normalized_model not in {"timesfm", "lstm", "chronos", "AR"}:
        return [
            mark_price_conditioned_baseline_unavailable(
                report,
                f"unsupported_forecast_model_{normalized_model}",
            )
            for report in reports
        ]

    contexts_by_zone = {str(context.get("zone_id")): context for context in contexts}
    profiles = pipeline_data.profiles.copy()
    profiles["zone_id"] = profiles["zone_id"].astype(str)
    profiles_by_zone = profiles.set_index("zone_id", drop=False)
    thresholds_by_zone = zone_load_thresholds.copy()
    thresholds_by_zone["zone_id"] = thresholds_by_zone["zone_id"].astype(str)
    thresholds_by_zone = thresholds_by_zone.set_index("zone_id", drop=False)

    price_conditioned_energy_price = pipeline_data.energy_price.copy()
    for report in reports:
        report_zone = str(report.get("zone_id"))
        price_conditioned_energy_price = energy_price_with_predicted_windows(
            price_conditioned_energy_price,
            report_zone,
            report.get("price_change_windows_3h") or [],
        )

    updated_reports: list[dict[str, Any]] = []
    for report in reports:
        zone_id = str(report.get("zone_id"))
        context = contexts_by_zone.get(zone_id)
        if context is None or zone_id not in profiles_by_zone.index:
            updated_reports.append(
                mark_price_conditioned_baseline_unavailable(
                    report,
                    "missing_context_or_profile",
                )
            )
            continue

        artifact = scenario_artifacts.get(zone_id)
        if artifact is None:
            updated_reports.append(
                mark_price_conditioned_baseline_unavailable(
                    report,
                    "missing_reusable_forecaster_artifact",
                )
            )
            continue
        batch = artifact.predict(price_conditioned_energy_price)
        conditioned_hourly = batch.hourly.rename(
            columns={
                "timestamp": "time",
                "predicted_load": "predicted_kwh",
                "energy_price": "e_price",
            }
        )
        for source, column in (
            (pipeline_data.service_price, "s_price"),
            (pipeline_data.occupancy, "occupancy"),
        ):
            if zone_id in source:
                conditioned_hourly = conditioned_hourly.merge(
                    source[["time", zone_id]].rename(columns={zone_id: column}),
                    on="time",
                    how="left",
                )
        weather_columns = [
            column for column in ("T", "U", "nRAIN") if column in pipeline_data.weather
        ]
        if weather_columns:
            conditioned_hourly = conditioned_hourly.merge(
                pipeline_data.weather[["time", *weather_columns]],
                on="time",
                how="left",
            )
        thresholds = load_stress_thresholds(
            thresholds_by_zone,
            zone_id,
            profile=profiles_by_zone.loc[zone_id].to_dict(),
        )
        conditioned_windows = build_pricing_windows_3h(
            conditioned_hourly,
            stress_thresholds=thresholds,
        )
        updated_report = attach_price_conditioned_baselines(
            report,
            conditioned_windows,
            forecast_model=normalized_model,
            forecast_metadata=batch.metadata,
        )
        updated_reports.append(updated_report)
    return updated_reports


async def run_control_loop(
    *,
    reports: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    client: Any,
    heuristic_source: str,
    agent_mode: str,
    temperature: float,
    pipeline_data: Any,
    zone_load_thresholds: pd.DataFrame,
    price_change_reference: Any,
    forecast_parameters: dict[str, Any],
    scenario_artifacts: dict[str, NativeForecasterArtifact],
) -> list[dict[str, Any]]:
    """Run at most three forecast-validated price proposals."""

    current = [copy.deepcopy(report) for report in reports]
    contexts_by_zone = {str(context.get("zone_id")): context for context in contexts}
    traces: dict[str, list[dict[str, Any]]] = {
        str(report.get("zone_id")): [] for report in current
    }
    pending_usage_by_zone: dict[str, list[dict[str, Any]]] = {
        str(report.get("zone_id")): copy.deepcopy(
            report.get("agent_call_usage")
            if isinstance(report.get("agent_call_usage"), list)
            else []
        )
        for report in current
    }
    pending_revised_keys_by_zone: dict[str, set[tuple[str, str]]] = {
        str(report.get("zone_id")): set() for report in current
    }
    pending_phase_by_zone: dict[str, str] = {
        str(report.get("zone_id")): "initial" for report in current
    }
    agent_round_usage: list[dict[str, Any]] = []

    for attempt in range(1, MAX_CONTROL_ATTEMPTS + 1):
        current = [
            attach_price_change_diagnostics(report, price_change_reference)
            for report in current
        ]
        conditioned = apply_price_conditioned_baseline_forecasts(
            reports=current,
            contexts=[contexts_by_zone[str(report.get("zone_id"))] for report in current],
            pipeline_data=pipeline_data,
            zone_load_thresholds=zone_load_thresholds,
            scenario_artifacts=scenario_artifacts,
            **price_conditioned_parameters(forecast_parameters),
        )
        conditioned_by_zone = {
            str(report.get("zone_id")): report for report in conditioned
        }
        current = [
            assess_control_report(
                conditioned_by_zone[str(report.get("zone_id"))],
                attempt=attempt,
            )
            for report in current
        ]

        zone_usage_for_round: list[dict[str, Any]] = []
        for report in current:
            zone_id = str(report.get("zone_id"))
            previous_snapshot = traces[zone_id][-1] if traces[zone_id] else None
            snapshot = control_attempt_snapshot(
                report,
                attempt,
                previous_snapshot=previous_snapshot,
                revised_keys=pending_revised_keys_by_zone.get(zone_id, set()),
                agent_call_usage=pending_usage_by_zone.get(zone_id, []),
                proposal_phase=pending_phase_by_zone.get(zone_id, "frozen"),
                triggered_by_attempt=attempt - 1 if attempt > 1 else None,
            )
            traces[zone_id].append(snapshot)
            zone_usage_for_round.append(
                {
                    "zone_id": zone_id,
                    **snapshot["agent_usage"],
                }
            )

        agent_round_usage.append(
            build_global_agent_round_usage(
                attempt=attempt,
                zone_usage=zone_usage_for_round,
                proposal_phase=(
                    "initial" if attempt == 1 else retry_proposal_phase(agent_mode)
                ),
                triggered_by_attempt=attempt - 1 if attempt > 1 else None,
            )
        )

        if all(report.get("control_status") == "success" for report in current):
            break
        if attempt >= MAX_CONTROL_ATTEMPTS:
            break

        retry_jobs: list[tuple[dict[str, Any], dict[str, Any], set[tuple[str, str]]]] = []
        for report in current:
            if report.get("control_status") == "success":
                continue
            zone_id = str(report.get("zone_id"))
            failed_keys = {
                window_key(window)
                for window in report.get("price_change_windows_3h") or []
                if not window.get("control_success")
            }
            retry_context = build_retry_context(
                contexts_by_zone[zone_id],
                report,
                attempt=attempt,
                replace_forecast=agent_mode
                in {
                    "multi_agent_full_retry",
                    "multi_agent_discussion_3rounds",
                    "single_agent_full_retry",
                },
            )
            retry_jobs.append((retry_context, report, failed_keys))

        revisions = await revise_control_reports(
            retry_jobs,
            client=client,
            temperature=temperature,
            heuristic_source=heuristic_source,
            agent_mode=agent_mode,
        )
        revisions_by_zone = {
            str(report.get("zone_id")): (report, failed_keys)
            for report, failed_keys in revisions
        }
        next_reports: list[dict[str, Any]] = []
        next_usage_by_zone: dict[str, list[dict[str, Any]]] = {}
        next_revised_keys_by_zone: dict[str, set[tuple[str, str]]] = {}
        next_phase_by_zone: dict[str, str] = {}
        for report in current:
            zone_id = str(report.get("zone_id"))
            if zone_id not in revisions_by_zone:
                next_reports.append(report)
                next_usage_by_zone[zone_id] = []
                next_revised_keys_by_zone[zone_id] = set()
                next_phase_by_zone[zone_id] = "frozen"
                continue
            revision, failed_keys = revisions_by_zone[zone_id]
            next_reports.append(
                merge_failed_window_revision(report, revision, failed_keys)
            )
            revision_usage = revision.get("agent_call_usage")
            next_usage_by_zone[zone_id] = copy.deepcopy(
                revision_usage if isinstance(revision_usage, list) else []
            )
            next_revised_keys_by_zone[zone_id] = set(failed_keys)
            next_phase_by_zone[zone_id] = retry_proposal_phase(agent_mode)
        current = next_reports
        pending_usage_by_zone = next_usage_by_zone
        pending_revised_keys_by_zone = next_revised_keys_by_zone
        pending_phase_by_zone = next_phase_by_zone

    global_cumulative_usage = cumulative_global_agent_usage(agent_round_usage)
    finalized: list[dict[str, Any]] = []
    for report in current:
        zone_id = str(report.get("zone_id"))
        updated = dict(report)
        updated["control_attempt_trace"] = traces[zone_id]
        updated["attempts_used"] = len(traces[zone_id])
        final_windows = updated.get("price_change_windows_3h") or []
        updated["control_status"] = (
            "success"
            if final_windows
            and all(
                window.get("control_success")
                for window in final_windows
            )
            else "fail"
        )
        cumulative_usage, cumulative_calls = cumulative_zone_agent_usage(
            traces[zone_id]
        )
        updated["agent_cumulative_usage"] = cumulative_usage
        updated["agent_call_usage"] = cumulative_calls
        updated["agent_invoked"] = cumulative_usage["agent_invoked"]
        updated["agent_call_count"] = cumulative_usage["agent_call_count"]
        updated["agent_prompt_tokens"] = cumulative_usage["prompt_tokens"]
        updated["agent_completion_tokens"] = cumulative_usage[
            "completion_tokens"
        ]
        updated["agent_total_tokens"] = cumulative_usage["total_tokens"]
        updated["agent_token_usage_complete"] = cumulative_usage[
            "token_usage_complete"
        ]
        updated["agent_token_totals"] = token_total_summary(cumulative_usage)
        updated[CONTROL_AGENT_USAGE_KEY] = {
            "agent_round_usage": agent_round_usage,
            "agent_cumulative_usage": global_cumulative_usage,
            "agent_token_totals": token_total_summary(global_cumulative_usage),
        }
        finalized.append(updated)
    return finalized


async def revise_control_reports(
    jobs: list[tuple[dict[str, Any], dict[str, Any], set[tuple[str, str]]]],
    *,
    client: Any,
    temperature: float,
    heuristic_source: str,
    agent_mode: str,
) -> list[tuple[dict[str, Any], set[tuple[str, str]]]]:
    async def revise(
        context: dict[str, Any],
        previous: dict[str, Any],
        failed_keys: set[tuple[str, str]],
    ) -> tuple[dict[str, Any], set[tuple[str, str]]]:
        if agent_mode in {
            "multi_agent_full_retry",
            "multi_agent_discussion_3rounds",
            "single_agent_full_retry",
        }:
            revision = await run_zone_chain(
                context,
                client=client,
                temperature=temperature,
                heuristic_source=heuristic_source,
                chain_mode=(
                    "single_agent"
                    if agent_mode == "single_agent_full_retry"
                    else "multi_agent_discussion"
                    if agent_mode == "multi_agent_discussion_3rounds"
                    else "multi_agent"
                ),
            )
        else:
            revision = await retry_zone_prices(
                context,
                previous,
                client=client,
                temperature=temperature,
                single_agent=agent_mode == "single_agent_price_retry",
                heuristic_source=heuristic_source,
            )
        return revision, failed_keys

    return await asyncio.gather(
        *(revise(context, previous, failed_keys) for context, previous, failed_keys in jobs)
    )


def attach_price_change_diagnostics(
    report: dict[str, Any],
    reference: Any,
) -> dict[str, Any]:
    updated = dict(report)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        item = dict(window)
        shift = optional_number(item.get("suggested_price_shift_pct"))
        percentile = reference.percentile(shift) if reference is not None else None
        p95 = reference.p95_pct if reference is not None else None
        item["historical_price_change_percentile"] = percentile
        item["historical_price_change_p95_pct"] = p95
        item["exceeds_historical_p95"] = (
            abs(shift) > p95
            if shift is not None and p95 is not None
            else None
        )
        windows.append(item)
    updated["price_change_windows_3h"] = windows
    updated["historical_price_change_reference"] = (
        reference.to_dict(include_values=False) if reference is not None else None
    )
    return updated


def assess_control_report(report: dict[str, Any], *, attempt: int) -> dict[str, Any]:
    updated = dict(report)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        item = dict(window)
        stress = item.get("price_conditioned_load_stress_level")
        price_valid = bool(item.get("price_valid"))
        success = price_valid and stress == MEDIUM_STRESS
        item["control_attempt"] = attempt
        item["control_success"] = success
        if not price_valid:
            failure_reason = "invalid_non_negative_energy_price_constraint"
        elif stress is None:
            failure_reason = "price_conditioned_forecast_unavailable"
        elif stress != MEDIUM_STRESS:
            failure_reason = f"price_conditioned_load_is_{stress}"
        else:
            failure_reason = None
        item["control_failure_reason"] = failure_reason
        windows.append(item)
    updated["price_change_windows_3h"] = windows
    updated["control_status"] = (
        "success" if windows and all(item["control_success"] for item in windows) else "fail"
    )
    updated["failed_window_count"] = sum(not item["control_success"] for item in windows)
    updated["attempts_used"] = attempt
    return updated


def control_attempt_snapshot(
    report: dict[str, Any],
    attempt: int,
    *,
    previous_snapshot: dict[str, Any] | None = None,
    revised_keys: set[tuple[str, str]] | None = None,
    agent_call_usage: list[dict[str, Any]] | None = None,
    proposal_phase: str | None = None,
    triggered_by_attempt: int | None = None,
) -> dict[str, Any]:
    phase = proposal_phase or ("initial" if attempt == 1 else "frozen")
    revised = revised_keys or set()
    previous_windows = {
        window_key(window): window
        for window in (previous_snapshot or {}).get("windows", [])
        if isinstance(window, dict)
    }
    usage, usage_calls = build_zone_attempt_agent_usage(
        agent_call_usage or [],
        attempt=attempt,
        proposal_phase=phase,
        triggered_by_attempt=triggered_by_attempt,
    )
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if not isinstance(window, dict):
            continue
        key = window_key(window)
        previous = previous_windows.get(key, {})
        proposal_status = (
            "initial"
            if attempt == 1
            else "revised"
            if key in revised
            else "frozen"
        )
        previous_price = optional_number(
            window.get("mean_energy_price")
            if attempt == 1
            else previous.get("proposed_energy_price")
        )
        current_price = optional_number(window.get("proposed_energy_price"))
        previous_shift = (
            0.0
            if attempt == 1
            else optional_number(previous.get("suggested_price_shift_pct"))
        )
        current_shift = optional_number(window.get("suggested_price_shift_pct"))
        price_change_amount = numeric_difference(current_price, previous_price)
        price_change_pct = (
            round((price_change_amount / previous_price) * 100.0, 4)
            if price_change_amount is not None
            and previous_price is not None
            and previous_price > 0
            else None
        )
        shift_change = numeric_difference(current_shift, previous_shift)
        trigger_failure_reason = (
            previous.get("control_failure_reason")
            if proposal_status == "revised"
            else None
        )
        reforecast_load = window.get("price_conditioned_baseline_load_kwh")
        reforecast_position = window.get(
            "price_conditioned_load_range_position_pct"
        )
        reforecast_stress = window.get("price_conditioned_load_stress_level")
        windows.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "proposal_status": proposal_status,
                "mean_energy_price": window.get("mean_energy_price"),
                "previous_proposed_energy_price": previous_price,
                "proposed_energy_price": current_price,
                "price_change_amount": price_change_amount,
                "price_change_pct_vs_previous": price_change_pct,
                "previous_suggested_price_shift_pct": previous_shift,
                "suggested_price_shift_pct": current_shift,
                "shift_change_percentage_points": shift_change,
                "price_changed": numeric_value_changed(
                    previous_price,
                    current_price,
                ),
                "trigger_failure_reason": trigger_failure_reason,
                "historical_price_change_percentile": window.get(
                    "historical_price_change_percentile"
                ),
                "historical_price_change_p95_pct": window.get(
                    "historical_price_change_p95_pct"
                ),
                "exceeds_historical_p95": window.get(
                    "exceeds_historical_p95"
                ),
                "price_conditioned_baseline_load_kwh": reforecast_load,
                "price_conditioned_load_range_position_pct": reforecast_position,
                "price_conditioned_load_stress_level": reforecast_stress,
                "price_conditioned_baseline_source": window.get(
                    "price_conditioned_baseline_source"
                ),
                "reforecast_load_kwh": reforecast_load,
                "reforecast_load_range_position_pct": reforecast_position,
                "reforecast_load_stress_level": reforecast_stress,
                "control_success": window.get("control_success"),
                "control_failure_reason": window.get("control_failure_reason"),
                "price_rationale": window.get("price_rationale"),
            }
        )
    return {
        "attempt": attempt,
        "proposal_phase": phase,
        "triggered_by_attempt": triggered_by_attempt,
        "control_status": report.get("control_status"),
        "failed_window_count": report.get("failed_window_count"),
        "agent_usage": usage,
        "agent_call_usage": usage_calls,
        "agent_discussion_round_count": report.get(
            "agent_discussion_round_count",
            0,
        ),
        "agent_discussion_round_limit": report.get(
            "agent_discussion_round_limit",
            0,
        ),
        "agent_discussion_converged": bool(
            report.get("agent_discussion_converged", False)
        ),
        "agent_discussion_stop_reason": report.get(
            "agent_discussion_stop_reason"
        ),
        "agent_discussion_rounds": copy.deepcopy(
            report.get("agent_discussion_rounds") or []
        ),
        "grid_reasoning_summary": report.get("grid_reasoning_summary"),
        "behavior_reasoning_summary": report.get("behavior_reasoning_summary"),
        "economist_reasoning_summary": report.get("economist_reasoning_summary"),
        "price_rationale": report.get("price_rationale"),
        "price_conditioned_forecast_metadata": report.get(
            "price_conditioned_forecast_metadata"
        ),
        "windows": windows,
    }


def build_zone_attempt_agent_usage(
    records: list[dict[str, Any]],
    *,
    attempt: int,
    proposal_phase: str,
    triggered_by_attempt: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = copy.deepcopy(record)
        if item.get("token_usage_complete") is None:
            item["token_usage_complete"] = all(
                item.get(field) is not None
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        item["attempt"] = attempt
        item["proposal_phase"] = proposal_phase
        item["triggered_by_attempt"] = triggered_by_attempt
        calls.append(item)
    summarized = summarize_agent_call_usage(calls)
    usage = {
        "agent_invoked": summarized["agent_invoked"],
        "agent_call_count": summarized["agent_call_count"],
        "prompt_tokens": summarized["prompt_tokens"],
        "completion_tokens": summarized["completion_tokens"],
        "total_tokens": summarized["total_tokens"],
        "token_usage_complete": summarized["token_usage_complete"],
        "proposal_phase": proposal_phase,
        "triggered_by_attempt": triggered_by_attempt,
    }
    return usage, calls


def build_global_agent_round_usage(
    *,
    attempt: int,
    zone_usage: list[dict[str, Any]],
    proposal_phase: str,
    triggered_by_attempt: int | None,
) -> dict[str, Any]:
    invoked = [item for item in zone_usage if item.get("agent_invoked")]
    return {
        "attempt": attempt,
        "proposal_phase": proposal_phase,
        "triggered_by_attempt": triggered_by_attempt,
        "agent_invoked": bool(invoked),
        "agent_invoked_zone_count": len(invoked),
        "agent_call_count": sum(
            usage_integer(item.get("agent_call_count")) for item in zone_usage
        ),
        "prompt_tokens": sum(
            usage_integer(item.get("prompt_tokens")) for item in zone_usage
        ),
        "completion_tokens": sum(
            usage_integer(item.get("completion_tokens")) for item in zone_usage
        ),
        "total_tokens": sum(
            usage_integer(item.get("total_tokens")) for item in zone_usage
        ),
        "token_usage_complete": all(
            bool(item.get("token_usage_complete")) for item in zone_usage
        ),
        "invoked_zone_ids": [item.get("zone_id") for item in invoked],
    }


def cumulative_zone_agent_usage(
    attempts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usage_rows = [
        attempt.get("agent_usage")
        for attempt in attempts
        if isinstance(attempt, dict) and isinstance(attempt.get("agent_usage"), dict)
    ]
    calls = [
        copy.deepcopy(call)
        for attempt in attempts
        if isinstance(attempt, dict)
        for call in (attempt.get("agent_call_usage") or [])
        if isinstance(call, dict)
    ]
    summary = summarize_agent_call_usage(calls)
    return (
        {
            "agent_attempt_count": len(attempts),
            "agent_invoked_attempt_count": sum(
                bool(item.get("agent_invoked")) for item in usage_rows
            ),
            "agent_invoked": summary["agent_invoked"],
            "agent_call_count": summary["agent_call_count"],
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"],
            "total_tokens": summary["total_tokens"],
            "token_usage_complete": all(
                bool(item.get("token_usage_complete")) for item in usage_rows
            ),
        },
        calls,
    )


def cumulative_global_agent_usage(
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "agent_round_count": len(rounds),
        "agent_invoked_round_count": sum(
            bool(item.get("agent_invoked")) for item in rounds
        ),
        "agent_invoked": any(bool(item.get("agent_invoked")) for item in rounds),
        "agent_invoked_zone_count": sum(
            usage_integer(item.get("agent_invoked_zone_count")) for item in rounds
        ),
        "agent_call_count": sum(
            usage_integer(item.get("agent_call_count")) for item in rounds
        ),
        "prompt_tokens": sum(
            usage_integer(item.get("prompt_tokens")) for item in rounds
        ),
        "completion_tokens": sum(
            usage_integer(item.get("completion_tokens")) for item in rounds
        ),
        "total_tokens": sum(
            usage_integer(item.get("total_tokens")) for item in rounds
        ),
        "token_usage_complete": all(
            bool(item.get("token_usage_complete")) for item in rounds
        ),
    }


def token_total_summary(usage: dict[str, Any]) -> dict[str, Any]:
    """Expose a compact final token-only total without timing fields."""
    return {
        "agent_call_count": usage_integer(usage.get("agent_call_count")),
        "prompt_tokens": usage_integer(usage.get("prompt_tokens")),
        "completion_tokens": usage_integer(usage.get("completion_tokens")),
        "total_tokens": usage_integer(usage.get("total_tokens")),
        "token_usage_complete": bool(usage.get("token_usage_complete")),
    }


def retry_proposal_phase(agent_mode: str) -> str:
    return {
        "multi_agent_economist_retry": "economist_retry",
        "multi_agent_full_retry": "full_multi_agent_retry",
        "multi_agent_discussion_3rounds": "three_round_agent_discussion_retry",
        "single_agent_price_retry": "single_agent_price_retry",
        "single_agent_full_retry": "single_agent_full_retry",
    }.get(agent_mode, "price_retry")


def numeric_difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return round(current - previous, 4)


def numeric_value_changed(previous: float | None, current: float | None) -> bool:
    if previous is None or current is None:
        return previous is not None or current is not None
    return not math.isclose(previous, current, rel_tol=0.0, abs_tol=1e-9)


def usage_integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_retry_context(
    original_context: dict[str, Any],
    report: dict[str, Any],
    *,
    attempt: int,
    replace_forecast: bool,
) -> dict[str, Any]:
    context = copy.deepcopy(original_context)
    report_windows = {
        window_key(window): window
        for window in report.get("price_change_windows_3h") or []
    }
    feedback: list[dict[str, Any]] = []
    updated_windows: list[dict[str, Any]] = []
    for original in context.get("pricing_windows_3h") or []:
        latest = report_windows.get(window_key(original), {})
        item = dict(original)
        if replace_forecast and latest.get("price_conditioned_baseline_load_kwh") is not None:
            item["sum_predicted_kwh"] = latest.get("price_conditioned_baseline_load_kwh")
            item["mean_predicted_kwh"] = latest.get("price_conditioned_mean_predicted_kwh")
            item["peak_predicted_kwh"] = latest.get("price_conditioned_peak_predicted_kwh")
            item["load_range_position_pct"] = latest.get(
                "price_conditioned_load_range_position_pct"
            )
            item["load_stress_level"] = latest.get("price_conditioned_load_stress_level")
            item["grid_stress_level"] = latest.get("price_conditioned_load_stress_level")
            item["stress_load_3h_kwh"] = latest.get("price_conditioned_baseline_load_kwh")
        updated_windows.append(item)
        if latest and not latest.get("control_success"):
            feedback.append(
                {
                    "attempt": attempt,
                    "window_start": latest.get("window_start"),
                    "window_end": latest.get("window_end"),
                    "previous_price_shift_pct": latest.get("suggested_price_shift_pct"),
                    "previous_energy_price": latest.get("proposed_energy_price"),
                    "reforecast_load_kwh": latest.get("price_conditioned_baseline_load_kwh"),
                    "load_range_position_pct": latest.get(
                        "price_conditioned_load_range_position_pct"
                    ),
                    "load_stress_level": latest.get("price_conditioned_load_stress_level"),
                    "failure_reason": latest.get("control_failure_reason"),
                }
            )
    context["pricing_windows_3h"] = updated_windows
    context["retry_feedback"] = feedback
    if replace_forecast:
        loads = [optional_number(item.get("sum_predicted_kwh")) for item in updated_windows]
        valid_loads = [value for value in loads if value is not None]
        context["forecast_total_kwh"] = round(sum(valid_loads), 4)
        context["forecast_peak_kwh"] = round(
            max(
                (
                    optional_number(item.get("peak_predicted_kwh")) or 0.0
                    for item in updated_windows
                ),
                default=0.0,
            ),
            4,
        )
        context["grid_stress_level"] = max_stress_level(
            [item.get("load_stress_level") for item in updated_windows]
        )
    return context


def merge_failed_window_revision(
    previous: dict[str, Any],
    revision: dict[str, Any],
    failed_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    revised_by_key = {
        window_key(window): window
        for window in revision.get("price_change_windows_3h") or []
    }
    windows: list[dict[str, Any]] = []
    for old in previous.get("price_change_windows_3h") or []:
        key = window_key(old)
        if key not in failed_keys or key not in revised_by_key:
            windows.append(dict(old))
            continue
        new = revised_by_key[key]
        merged = dict(old)
        for field in (
            "suggested_price_shift_pct",
            "proposed_energy_price",
            "predicted_energy_price",
            "price_valid",
            "price_validation_error",
            "action_label",
            "price_rationale",
        ):
            if field in new:
                merged[field] = new.get(field)
        windows.append(merged)
    updated = dict(previous)
    for field in (
        "grid_reasoning_summary",
        "behavior_reasoning_summary",
        "economist_reasoning_summary",
        "agent_reasoning",
        "action_label",
        "price_rationale",
        "grid_output",
        "behavior_output",
        "economist_output",
        "agent_discussion_round_count",
        "agent_discussion_round_limit",
        "agent_discussion_converged",
        "agent_discussion_stop_reason",
        "agent_discussion_rounds",
        "source",
    ):
        if field in revision:
            updated[field] = revision[field]
    updated["price_change_windows_3h"] = windows
    updated["suggested_price_shift_pct"] = round(
        sum(optional_number(item.get("suggested_price_shift_pct")) or 0.0 for item in windows)
        / len(windows),
        4,
    ) if windows else 0.0
    return updated


def window_key(window: dict[str, Any]) -> tuple[str, str]:
    return str(window.get("window_start")), str(window.get("window_end"))


def energy_price_with_predicted_windows(
    energy_price: pd.DataFrame,
    zone_id: str,
    windows: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = energy_price.copy()
    if "time" not in frame or zone_id not in frame:
        return frame
    frame["time"] = pd.to_datetime(frame["time"])
    for window in windows:
        if not isinstance(window, dict):
            continue
        predicted_price = predicted_energy_price(window)
        if predicted_price is None:
            continue
        start = parse_datetime_24h(window.get("window_start"))
        end = parse_datetime_24h(window.get("window_end"))
        mask = (frame["time"] >= start) & (frame["time"] <= end)
        frame.loc[mask, zone_id] = predicted_price
    return frame


def predicted_energy_price(window: dict[str, Any]) -> float | None:
    proposed = optional_number(
        window.get("proposed_energy_price")
        if window.get("proposed_energy_price") is not None
        else window.get("predicted_energy_price")
    )
    if proposed is not None:
        return round(proposed, 4) if proposed >= 0 else None
    base_price = optional_number(window.get("mean_energy_price"))
    shift_pct = optional_number(window.get("suggested_price_shift_pct"))
    if base_price is None or shift_pct is None:
        return None
    return round(base_price * (1 + shift_pct / 100), 4)


def attach_price_conditioned_baselines(
    report: dict[str, Any],
    conditioned_windows: list[dict[str, Any]],
    *,
    forecast_model: str,
    forecast_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conditioned_by_window = {
        (str(window.get("window_start")), str(window.get("window_end"))): window
        for window in conditioned_windows
    }
    updated_windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if not isinstance(window, dict):
            continue
        updated = dict(window)
        key = (str(window.get("window_start")), str(window.get("window_end")))
        conditioned = conditioned_by_window.get(key)
        if conditioned is None:
            updated["price_conditioned_baseline_source"] = "missing_price_conditioned_forecast_window"
        else:
            updated["price_conditioned_baseline_load_kwh"] = conditioned.get("sum_predicted_kwh")
            updated["price_conditioned_mean_predicted_kwh"] = conditioned.get("mean_predicted_kwh")
            updated["price_conditioned_peak_predicted_kwh"] = conditioned.get("peak_predicted_kwh")
            updated["price_conditioned_load_stress_level"] = conditioned.get("load_stress_level")
            updated["price_conditioned_load_range_position_pct"] = conditioned.get(
                "load_range_position_pct"
            )
            updated["price_conditioned_energy_price"] = conditioned.get("mean_energy_price")
            updated["price_conditioned_baseline_source"] = (
                f"{forecast_model}_native_backend_reforecast_with_proposed_energy_price"
            )
        updated_windows.append(updated)

    updated_report = dict(report)
    updated_report["price_change_windows_3h"] = updated_windows
    updated_report["price_conditioned_forecast_metadata"] = dict(
        forecast_metadata or {}
    )
    return updated_report


def mark_price_conditioned_baseline_unavailable(
    report: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    updated = dict(report)
    windows: list[dict[str, Any]] = []
    for window in report.get("price_change_windows_3h") or []:
        if isinstance(window, dict):
            item = dict(window)
            item["price_conditioned_baseline_source"] = reason
            windows.append(item)
    updated["price_change_windows_3h"] = windows
    return updated


def optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def build_contexts(
    *,
    pipeline_data,
    selected_zones: pd.DataFrame,
    forecast_start: str | None,
    horizon_days: int,
    history_days: int,
    validation_days: int,
    forecast_model: str,
    zone_load_thresholds: pd.DataFrame,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_step_horizon: int,
    timesfm_exog_cols: list[str] | None,
    timesfm_diurnal_blend_alpha: float,
    timesfm_roll_actuals: bool,
    ar_diurnal_blend_alpha: float,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_step_horizon: int,
    chronos_diurnal_blend_alpha: float,
    chronos_device: str,
    chronos_roll_actuals: bool,
    lstm_context_hours: int,
    lstm_step_horizon: int,
    lstm_exog_cols: list[str] | None,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    lstm_epochs: int,
    lstm_learning_rate: float,
    lstm_batch_size: int,
    lstm_diurnal_blend_alpha: float,
    lstm_device: str,
    lstm_roll_actuals: bool,
    lstm_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, ForecastResult]]:
    start = pd.Timestamp(forecast_start) if forecast_start else None
    contexts = []
    forecast_results = {}
    profiles_by_zone = pipeline_data.profiles.set_index("zone_id", drop=False)
    stress_thresholds_by_zone = zone_load_thresholds.set_index("zone_id", drop=False)
    for row in selected_zones.to_dict(orient="records"):
        zone_id = str(row["zone_id"])
        profile = profiles_by_zone.loc[zone_id].to_dict()
        zone_stress_thresholds = load_stress_thresholds(
            stress_thresholds_by_zone,
            zone_id,
            profile=profile,
        )
        try:
            raw_result = forecast_zone(
                zone_id=zone_id,
                category=row["category"],
                load=pipeline_data.load,
                service_price=pipeline_data.service_price,
                energy_price=pipeline_data.energy_price,
                occupancy=pipeline_data.occupancy,
                weather=pipeline_data.weather,
                profile=profile,
                forecast_start=start,
                horizon_days=horizon_days,
                history_days=history_days,
                validation_days=validation_days,
                forecast_model=forecast_model,
                timesfm_repo=timesfm_repo,
                timesfm_context_hours=timesfm_context_hours,
                timesfm_step_horizon=timesfm_step_horizon,
                timesfm_exog_cols=timesfm_exog_cols,
                timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
                timesfm_roll_actuals=timesfm_roll_actuals,
                ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
                chronos_repo=chronos_repo,
                chronos_context_hours=chronos_context_hours,
                chronos_step_horizon=chronos_step_horizon,
                chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
                chronos_device=chronos_device,
                chronos_roll_actuals=chronos_roll_actuals,
                lstm_context_hours=lstm_context_hours,
                lstm_step_horizon=lstm_step_horizon,
                lstm_exog_cols=lstm_exog_cols,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_epochs=lstm_epochs,
                lstm_learning_rate=lstm_learning_rate,
                lstm_batch_size=lstm_batch_size,
                lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
                lstm_device=lstm_device,
                lstm_roll_actuals=lstm_roll_actuals,
                lstm_seed=lstm_seed,
            )
        except Exception as exc:
            raise PipelineStageError(
                stage="forecast",
                zone_id=zone_id,
                original=exc,
            ) from exc
        pricing_windows_3h = build_pricing_windows_3h(
            raw_result.hourly,
            stress_thresholds=zone_stress_thresholds,
            include_evaluation=False,
        )
        summary = apply_load_policy_stress(raw_result.summary, zone_stress_thresholds, pricing_windows_3h)
        result = ForecastResult(
            hourly=raw_result.hourly,
            summary=summary,
            reusable_forecaster=raw_result.reusable_forecaster,
        )
        forecast_results[zone_id] = result
        hourly_forecast = build_agent_hourly_data(result.hourly)
        hourly_averages = build_hourly_averages(result.hourly)
        horizon_text = format_horizon_days(summary.get("forecast_horizon_days", horizon_days))
        context = {
            **decision_summary(summary),
            "selection_reason": row["selection_reason"],
            "hourly_averages": hourly_averages,
            "hourly_forecast": hourly_forecast,
            "pricing_windows_3h": pricing_windows_3h,
            "instructions": {
                "forecast_task": f"Predict next {horizon_text} of EV charging load.",
                "behavior_task": "Explain demand using POI, weather, and temporal markers.",
                "pricing_task": "Suggest energy-price shifts for each 3-hour pricing window.",
            },
        }
        contexts.append(context)
    return contexts, forecast_results


def load_stress_thresholds(
    thresholds_by_zone: pd.DataFrame,
    zone_id: str,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or {}
    zone_mean_energy_price = safe_float(profile.get("mean_energy_price"))
    price_fields = {"zone_mean_energy_price": zone_mean_energy_price}
    if zone_id not in thresholds_by_zone.index:
        return {
            "available": False,
            "source_file": "volume.csv",
            "window_hours": 3,
            "historical_windows": 0,
            "historical_min_load_kwh": 0.0,
            "historical_max_load_kwh": 0.0,
            "historical_load_range_kwh": 0.0,
            "low_medium_threshold_pct": LOW_MAX_LOAD_PCT,
            "medium_high_threshold_pct": MEDIUM_MAX_LOAD_PCT,
            "high_extremely_high_threshold_pct": HIGH_MAX_LOAD_PCT,
            "low_medium_threshold_kwh": 0.0,
            "medium_high_threshold_kwh": 0.0,
            "high_extremely_high_threshold_kwh": 0.0,
            **price_fields,
        }
    row = thresholds_by_zone.loc[zone_id]
    historical_min = safe_float(row.get("historical_min_load_3h_kwh"))
    historical_max = safe_float(row.get("historical_max_load_3h_kwh"))
    return {
        "available": True,
        "source_file": str(row.get("stress_source_file", "volume.csv")),
        "window_hours": int(row.get("stress_window_hours", 3) or 3),
        "historical_windows": int(row.get("historical_3h_windows", 0) or 0),
        "historical_min_load_kwh": historical_min,
        "historical_max_load_kwh": historical_max,
        "historical_load_range_kwh": safe_float(
            row.get("historical_load_range_3h_kwh", historical_max - historical_min)
        ),
        "low_medium_threshold_pct": safe_float(
            row.get("low_medium_threshold_pct", LOW_MAX_LOAD_PCT)
        ),
        "medium_high_threshold_pct": safe_float(
            row.get("medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT)
        ),
        "high_extremely_high_threshold_pct": safe_float(
            row.get("high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT)
        ),
        "low_medium_threshold_kwh": safe_float(
            row.get("load_3h_low_medium_threshold_kwh")
        ),
        "medium_high_threshold_kwh": safe_float(
            row.get("load_3h_medium_high_threshold_kwh")
        ),
        "high_extremely_high_threshold_kwh": safe_float(
            row.get("load_3h_high_extremely_high_threshold_kwh")
        ),
        **price_fields,
    }


def classify_load_stress(load_value: Any, thresholds: dict[str, Any]) -> str:
    if not thresholds.get("available", True):
        return LOW_STRESS
    position_pct = load_range_position_pct(
        load_value,
        thresholds.get("historical_min_load_kwh"),
        thresholds.get("historical_max_load_kwh"),
    )
    return classify_load_percentage(position_pct)


def apply_load_policy_stress(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    pricing_windows_3h: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(summary)
    stress_loads = [safe_float(window.get("sum_predicted_kwh")) for window in pricing_windows_3h]
    stress_load = max(stress_loads) if stress_loads else 0.0
    range_positions = [
        safe_float(window.get("load_range_position_pct")) for window in pricing_windows_3h
    ]
    window_levels = [window.get("load_stress_level") for window in pricing_windows_3h]
    updated["grid_stress_level"] = max_stress_level(window_levels) if window_levels else classify_load_stress(stress_load, thresholds)
    updated["grid_stress_basis"] = (
        "forecast_3h_sum_predicted_kwh_position_in_zone_historical_3h_min_max_range"
    )
    updated["grid_stress_load_kwh"] = stress_load
    updated["grid_stress_load_range_position_pct"] = (
        max(range_positions) if range_positions else 0.0
    )
    updated["grid_stress_source_file"] = thresholds.get("source_file", "volume.csv")
    updated["grid_stress_window_hours"] = thresholds.get("window_hours", 3)
    updated["grid_stress_historical_windows"] = thresholds.get("historical_windows", 0)
    updated["historical_min_load_3h_kwh"] = thresholds.get("historical_min_load_kwh", 0.0)
    updated["historical_max_load_3h_kwh"] = thresholds.get("historical_max_load_kwh", 0.0)
    updated["historical_load_range_3h_kwh"] = thresholds.get(
        "historical_load_range_kwh", 0.0
    )
    updated["low_medium_threshold_pct"] = thresholds.get(
        "low_medium_threshold_pct", LOW_MAX_LOAD_PCT
    )
    updated["medium_high_threshold_pct"] = thresholds.get(
        "medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT
    )
    updated["high_extremely_high_threshold_pct"] = thresholds.get(
        "high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT
    )
    updated["load_3h_low_medium_threshold_kwh"] = thresholds.get(
        "low_medium_threshold_kwh", 0.0
    )
    updated["load_3h_medium_high_threshold_kwh"] = thresholds.get(
        "medium_high_threshold_kwh", 0.0
    )
    updated["load_3h_high_extremely_high_threshold_kwh"] = thresholds.get(
        "high_extremely_high_threshold_kwh", 0.0
    )
    updated["zone_mean_energy_price"] = thresholds.get("zone_mean_energy_price", 0.0)
    return updated


def build_agent_hourly_data(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "time",
        "predicted_kwh",
        "q10_kwh",
        "q50_kwh",
        "q90_kwh",
        "s_price",
        "e_price",
        "occupancy",
        "T",
        "U",
        "nRAIN",
        "hour",
        "is_weekend",
    ]
    frame = hourly[[col for col in columns if col in hourly.columns]].copy()
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return [{key: clean_context_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def build_hourly_averages(hourly: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "predicted_kwh": "mean_predicted_kwh",
        "s_price": "mean_service_price",
        "e_price": "mean_energy_price",
        "occupancy": "mean_occupancy",
        "T": "mean_temp_c",
        "U": "mean_humidity",
        "nRAIN": "mean_rain",
    }
    averages: dict[str, Any] = {}
    for source_col, output_col in columns.items():
        if source_col in hourly:
            value = pd.to_numeric(hourly[source_col], errors="coerce").mean(skipna=True)
            averages[output_col] = clean_context_value(value)
    if "predicted_kwh" in hourly:
        predicted = pd.to_numeric(hourly["predicted_kwh"], errors="coerce")
        averages["peak_predicted_kwh"] = clean_context_value(predicted.max(skipna=True))
        averages["total_predicted_kwh"] = clean_context_value(predicted.sum(skipna=True))
    return averages


def build_pricing_windows_3h(
    hourly: pd.DataFrame,
    *,
    stress_thresholds: dict[str, Any] | None = None,
    include_evaluation: bool = False,
) -> list[dict[str, Any]]:
    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    windows: list[dict[str, Any]] = []
    for idx in range(0, len(frame), 3):
        chunk = frame.iloc[idx : idx + 3]
        if chunk.empty:
            continue
        window: dict[str, Any] = {
            "window_start": chunk["time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": chunk["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "hours": int(len(chunk)),
        }
        aggregations = {
            "predicted_kwh": ("mean_predicted_kwh", "sum_predicted_kwh", "peak_predicted_kwh"),
        }
        if include_evaluation:
            aggregations["actual_kwh"] = (
                "mean_actual_kwh",
                "sum_actual_kwh",
                "peak_actual_kwh",
            )
        for source_col, (mean_col, sum_col, max_col) in aggregations.items():
            if source_col in chunk:
                values = pd.to_numeric(chunk[source_col], errors="coerce")
                window[mean_col] = clean_context_value(values.mean(skipna=True))
                window[sum_col] = clean_context_value(values.sum(skipna=True) if values.notna().any() else None)
                window[max_col] = clean_context_value(values.max(skipna=True))
        mean_columns = {
            "s_price": "mean_service_price",
            "e_price": "mean_energy_price",
            "occupancy": "mean_occupancy",
            "T": "mean_temp_c",
            "U": "mean_humidity",
        }
        for source_col, output_col in mean_columns.items():
            if source_col in chunk:
                window[output_col] = clean_context_value(pd.to_numeric(chunk[source_col], errors="coerce").mean(skipna=True))
        if "nRAIN" in chunk:
            window["total_rain"] = clean_context_value(pd.to_numeric(chunk["nRAIN"], errors="coerce").sum(skipna=True))
        if stress_thresholds is not None:
            stress_load = window.get("sum_predicted_kwh")
            stress_level = classify_load_stress(stress_load, stress_thresholds)
            actual_stress_load = window.get("sum_actual_kwh") if include_evaluation else None
            actual_stress_level = (
                classify_load_stress(actual_stress_load, stress_thresholds)
                if actual_stress_load is not None
                else None
            )
            window["load_stress_level"] = stress_level
            window["grid_stress_level"] = stress_level
            window["stress_load_3h_kwh"] = clean_context_value(stress_load)
            window["actual_load_stress_level"] = actual_stress_level
            window["actual_grid_stress_level"] = actual_stress_level
            window["actual_stress_load_3h_kwh"] = clean_context_value(actual_stress_load)
            predicted_rank = stress_rank(stress_level)
            actual_rank = stress_rank(actual_stress_level)
            window["stress_correct"] = predicted_rank == actual_rank if actual_rank is not None else None
            window["stress_missed"] = actual_rank > predicted_rank if actual_rank is not None and predicted_rank is not None else None
            window["stress_source_file"] = stress_thresholds.get("source_file", "volume.csv")
            window["stress_window_hours"] = stress_thresholds.get("window_hours", 3)
            historical_min_load = safe_float(
                stress_thresholds.get("historical_min_load_kwh")
            )
            historical_max_load = safe_float(
                stress_thresholds.get("historical_max_load_kwh")
            )
            historical_load_range = max(0.0, historical_max_load - historical_min_load)
            window["historical_min_load_3h_kwh"] = historical_min_load
            window["historical_max_load_3h_kwh"] = historical_max_load
            window["historical_load_range_3h_kwh"] = historical_load_range
            window["load_range_position_pct"] = round(
                load_range_position_pct(stress_load, historical_min_load, historical_max_load),
                4,
            )
            window["actual_load_range_position_pct"] = (
                round(
                    load_range_position_pct(
                        actual_stress_load,
                        historical_min_load,
                        historical_max_load,
                    ),
                    4,
                )
                if actual_stress_load is not None
                else None
            )
            window["low_medium_threshold_pct"] = stress_thresholds.get(
                "low_medium_threshold_pct", LOW_MAX_LOAD_PCT
            )
            window["medium_high_threshold_pct"] = stress_thresholds.get(
                "medium_high_threshold_pct", MEDIUM_MAX_LOAD_PCT
            )
            window["high_extremely_high_threshold_pct"] = stress_thresholds.get(
                "high_extremely_high_threshold_pct", HIGH_MAX_LOAD_PCT
            )
            window["load_3h_low_medium_threshold_kwh"] = stress_thresholds.get(
                "low_medium_threshold_kwh",
                historical_min_load + historical_load_range * LOW_MAX_LOAD_PCT / 100,
            )
            window["load_3h_medium_high_threshold_kwh"] = stress_thresholds.get(
                "medium_high_threshold_kwh",
                historical_min_load + historical_load_range * MEDIUM_MAX_LOAD_PCT / 100,
            )
            window["load_3h_high_extremely_high_threshold_kwh"] = stress_thresholds.get(
                "high_extremely_high_threshold_kwh",
                historical_min_load + historical_load_range * HIGH_MAX_LOAD_PCT / 100,
            )
            window["zone_mean_energy_price"] = stress_thresholds.get(
                "zone_mean_energy_price",
                0.0,
            )
        windows.append(window)
    return windows


def decision_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the allowlisted forecast information that may reach an agent."""

    return {
        key: value
        for key, value in summary.items()
        if key in DECISION_SUMMARY_KEYS
    }


def clean_context_value(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, ndigits)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return round(number, 4)


def max_stress_level(levels: list[Any]) -> str:
    normalized = [str(level) for level in levels if stress_rank(level) is not None]
    if not normalized:
        return "Low"
    return max(normalized, key=lambda level: STRESS_LEVEL_ORDER[level])


def stress_rank(level: Any) -> int | None:
    return STRESS_LEVEL_ORDER.get(str(level))


def format_horizon_days(value: Any) -> str:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if days == 1 else f"{days} days"


def normalize_zone_ids(zone_ids: str | Iterable[str] | None) -> list[str]:
    if zone_ids is None:
        return []

    raw_values = [zone_ids] if isinstance(zone_ids, str) else list(zone_ids)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            zone_id = part.strip()
            if zone_id and zone_id not in seen:
                normalized.append(zone_id)
                seen.add(zone_id)
    return normalized


def select_requested_zones(profiles: pd.DataFrame, zone_ids: Iterable[str]) -> pd.DataFrame:
    requested = normalize_zone_ids(zone_ids)
    if not requested:
        raise ValueError("At least one zone id is required.")

    frame = profiles.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    available = set(frame["zone_id"])
    missing = [zone_id for zone_id in requested if zone_id not in available]
    if missing:
        examples = ", ".join(sorted(available)[:10])
        raise ValueError(
            f"Unknown zone id(s): {', '.join(missing)}. "
            f"Available zone id examples: {examples}"
        )

    selected = frame.set_index("zone_id", drop=False).loc[requested].reset_index(drop=True)
    selected.insert(0, "category", "User-selected")
    selected.insert(2, "selection_score", None)
    selected.insert(3, "selection_reason", "User-specified zone for direct validation.")

    preferred_columns = [
        "category",
        "zone_id",
        "selection_score",
        "selection_reason",
        "longitude",
        "latitude",
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "noon_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
        "mean_energy_price",
    ]
    return selected[[col for col in preferred_columns if col in selected.columns]]
