from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.chains import run_all_zone_chains
from src.agents.client import AgentChatClient, AgentStageError
from src.config.agent import AgentConfig, agent_config_profile, normalize_agent_mode
from src.config.run import VALIDATION_BIAS_DAYS, normalize_pipeline_stage, normalize_zone_selection
from src.config.training import TrainingConfig
from src.config.values import normalize_forecast_model_name
from src.control.loop import run_control_loop
from src.control.windows import build_zone_3h_load_policy_thresholds
from src.data.adapters.registry import load_canonical_dataset, validate_forecaster_dataset_source
from src.data.canonical import DatasetSpec
from src.data.eligibility import random_training_candidates
from src.data.profiles import load_pipeline_data, profiles_at_origin
from src.data.selection import (
    filter_training_profiles,
    normalize_zone_ids,
    select_experiment_zone_ids,
    select_requested_zones,
    select_zone_categories,
)
from src.forecast.artifact import NativeForecasterArtifact
from src.forecast.parameters import forecast_runtime_parameters, make_forecast_parameters
from src.paths import forecast_output_dir
from src.pipeline.context import build_contexts
from src.pipeline.errors import PipelineStageError
from src.results.control import write_agent_outputs
from src.results.price_range import annotate_final_price_ranges
from src.results.forecast import write_forecaster_outputs
from src.results.manifest import (
    ForecasterBundle,
    build_agent_manifest,
    build_forecaster_manifest,
    load_forecaster_bundle,
    resolve_forecaster_output_dir,
)
from src.results.schema import MODEL_PROVENANCE_FIELDS
from src.results.window_cache import (
    apply_precomputed_window_data,
    load_precomputed_window_data,
    partition_precomputed_contexts,
)
from src.training.prepare import eligible_training_zone_ids, prepare_offline_models


def run_training_stage(*, data_dir: Path, backends: Iterable[str] = ("lstm", "xgboost"),
                       dataset_spec: DatasetSpec | None = None, cache_dir: Path | None = None,
                       training_config: TrainingConfig | None = None,
                       training_cli_overrides: dict[str, Any] | None = None,
                       zone_ids: str | Iterable[str] | None = None,
                       experiment_zone_count: int = 12,
                       experiment_zone_selection: str = "representative",
                       experiment_zone_seed: int = 42,
                       history_days: int = 35, horizon_days: int = 1,
                       force_train: bool = False, force_cache: bool = False,
                       backend_parameters: dict[str, Any] | None = None) -> dict[str, Path]:
    training_config = training_config or TrainingConfig()
    selected = training_config.enabled_backends(backends, cli_overrides=training_cli_overrides)
    if not selected:
        raise ValueError("The train stage requires lstm, xgboost, timesfm_lora or chronos_lora")
    spec = dataset_spec or DatasetSpec(path=data_dir, cache_dir=cache_dir)
    selected_zone_ids = normalize_zone_ids(zone_ids)
    if not selected_zone_ids and normalize_zone_selection(experiment_zone_selection) == "random":
        spec = replace(spec, allow_zero_energy_price=True)
    dataset = load_canonical_dataset(spec, force_cache=force_cache)
    eligible = eligible_training_zone_ids(
        dataset, training_config, zone_ids=selected_zone_ids,
        history_days=history_days, horizon_days=horizon_days,
    )
    if not selected_zone_ids:
        selection = normalize_zone_selection(experiment_zone_selection)
        if selection == "random":
            eligible = random_training_candidates(dataset, eligible)
            if len(eligible) < experiment_zone_count:
                raise ValueError(f"Only {len(eligible)} eligible zones remain for {experiment_zone_count} requested random zones")
            profiles = pd.DataFrame({"zone_id": eligible})
        else:
            selection_start = pd.Timestamp(dataset.timeseries["timestamp"].max()) + pd.Timedelta(hours=1)
            _, _, profiles = profiles_at_origin(dataset, Path(dataset.cache_dir), selection_start, force_cache=force_cache)
            profiles = filter_training_profiles(profiles, eligible)
        selected_zone_ids = select_experiment_zone_ids(
            profiles, count=experiment_zone_count, selection=selection, seed=experiment_zone_seed,
        )
    prepare_offline_models(dataset, selected, training_config, training_cli_overrides,
                           zone_ids=selected_zone_ids, history_days=history_days, horizon_days=horizon_days,
                           force_train=force_train, data_dir=data_dir, backend_parameters=backend_parameters)
    outputs = {}
    for backend in selected:
        directory = Path(dataset.cache_dir) / "weight" / backend
        outputs[f"{backend}_weight_dir"] = directory
        outputs[f"{backend}_training_index"] = directory / "index.json"
    return outputs


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
    forecast_origin: str | None = None,
    horizon_days: int = 1,
    history_days: int = 35,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    zone_ids: str | Iterable[str] | None = None,
    experiment_zone_count: int = 12,
    agent_mode: str = "multi_agent_economist_retry",
    backend: str = "timesfm",
    diurnal_blend_alpha: float = 0.0,
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 840,
    timesfm_exog_cols: list[str] | None = None,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 840,
    chronos_device: str = "auto",
    temperature: float = 0.2,
    precomputed_window_data: Path | str | None = None,
    pipeline_stage: str = "full",
    forecaster_output_dir: Path | str | None = None,
    agent_output_dir: Path | str | None = None,
    dataset_spec: DatasetSpec | None = None,
    training_config: TrainingConfig | None = None,
    training_cli_overrides: dict[str, Any] | None = None,
    force_train: bool = False,
    zone_model_references: dict[str, dict[str, dict[str, Any]]] | None = None,
    raw_prediction_cache: dict | None = None,
) -> dict[str, Path]:
    backend = normalize_forecast_model_name(backend)
    pipeline_stage = normalize_pipeline_stage(pipeline_stage)
    training_config = training_config or TrainingConfig()
    offline_backend = training_config.backend_enabled(backend, cli_overrides=training_cli_overrides)
    backend_parameters = {
        "timesfm_repo": timesfm_repo, "timesfm_context_hours": timesfm_context_hours,
        "timesfm_exog_cols": timesfm_exog_cols, "chronos_repo": chronos_repo,
        "chronos_context_hours": chronos_context_hours,
    }
    if pipeline_stage == "train":
        return run_training_stage(data_dir=data_dir, dataset_spec=dataset_spec,
                                  backends=[backend], training_config=training_config,
                                  training_cli_overrides=training_cli_overrides,
                                  backend_parameters=backend_parameters,
                                  zone_ids=zone_ids, experiment_zone_count=experiment_zone_count,
                                  history_days=history_days, horizon_days=horizon_days,
                                  force_train=force_train, force_cache=force_cache, cache_dir=cache_dir)
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
    run_output_dir = forecast_output_dir(output_dir, backend, diurnal_blend_alpha,
                                         validation_bias=validation_bias, validation_days=validation_days)
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
        forecast_origin=forecast_origin,
        horizon_days=horizon_days,
        history_days=history_days,
        validation_bias=validation_bias,
        validation_days=validation_days,
        backend=backend,
        diurnal_blend_alpha=diurnal_blend_alpha,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_exog_cols=timesfm_exog_cols,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_device=chronos_device,
    )

    if pipeline_stage == "agent":
        try:
            bundle = load_forecaster_bundle(forecaster_dir)
            requested = normalize_zone_ids(zone_ids)
            if dataset_spec.allow_zero_energy_price and requested and set(requested) != set(bundle.manifest["zone_ids"]):
                raise ValueError("Saved forecaster zones differ from the current random selection; rerun the forecaster stage.")
            if bundle.forecast_parameters["validation_bias"] != validation_bias:
                raise ValueError(
                    f"Saved forecaster validation_bias={bundle.forecast_parameters['validation_bias']} "
                    f"differs from the requested validation_bias={validation_bias} for {data_dir}."
                )
            if bundle.forecast_parameters.get("validation_days") != forecast_parameters.get("validation_days"):
                raise ValueError(
                    f"Saved forecaster validation_days={bundle.forecast_parameters.get('validation_days')} "
                    f"differs from the requested validation_days={validation_days} for {data_dir}."
                )
        except Exception as exc:
            raise PipelineStageError(stage="load_forecaster_output", original=exc) from exc
        selected_zones = bundle.selected_zones
        contexts = bundle.contexts
        forecast_parameters = bundle.forecast_parameters
        backend = str(forecast_parameters["backend"])
        source = bundle.manifest.get("data_source") or {}
        validate_forecaster_dataset_source(dataset_spec, source)
        recorded_data_dir = source.get("data_dir")
        if recorded_data_dir:
            data_dir = Path(str(recorded_data_dir))
        weather_file = str(source.get("weather_file") or weather_file)
        dataset_spec = replace(
            dataset_spec,
            path=data_dir,
            weather_file=weather_file,
            allow_zero_energy_price=(bundle.manifest.get("feature_manifest") or {}).get("allow_zero_energy_price", False),
        )
        if cache_dir is None and dataset_spec.cache_dir is None:
            shared_cache_dir = dataset_spec.resolved_cache_dir
        forecaster_outputs = bundle.outputs
        scenario_artifacts = bundle.scenario_artifacts

    canonical_dataset = load_canonical_dataset(dataset_spec, force_cache=force_cache)
    raw_prediction_cache = {} if raw_prediction_cache is None else raw_prediction_cache
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
        pd.Timestamp(forecast_parameters.get("forecast_origin"))
        if forecast_parameters.get("forecast_origin")
        else pd.Timestamp(canonical_dataset.timeseries["timestamp"].max())
        - pd.Timedelta(hours=int(forecast_parameters["horizon_days"]) * 24 - 1)
    )
    forecast_parameters["forecast_origin"] = effective_forecast_start.isoformat()
    split_key, split_cache_dir, profiles = profiles_at_origin(
        canonical_dataset, shared_cache_dir, effective_forecast_start, force_cache=force_cache
    )
    zone_load_thresholds = build_zone_3h_load_policy_thresholds(
        split_cache_dir,
        force_cache=force_cache,
        source_file="canonical_dataset",
        forecast_origin=effective_forecast_start,
        load_frame=canonical_dataset.wide_frame(
            "load_kwh",
            profiles["zone_id"].astype(str).tolist(),
        ),
    )
    if pipeline_stage != "agent":
        requested_zone_ids = normalize_zone_ids(zone_ids)
        selection_profiles = profiles
        if offline_backend and (not requested_zone_ids or zone_model_references is None):
            eligible = eligible_training_zone_ids(
                canonical_dataset, training_config, zone_ids=requested_zone_ids,
                history_days=history_days, horizon_days=horizon_days,
            )
            selection_profiles = filter_training_profiles(
                profiles, eligible, empty_error=f"No eligible offline {backend} training zones have forecast history",
            )
        if not requested_zone_ids and (len(selection_profiles) < 5 or "station_type" in selection_profiles):
            requested_zone_ids = selection_profiles["zone_id"].astype(str).tolist()
        try:
            selected_zones = (
                select_requested_zones(profiles, requested_zone_ids)
                if requested_zone_ids
                else select_zone_categories(selection_profiles)
            )
        except Exception as exc:
            raise PipelineStageError(stage="zone_selection", original=exc) from exc
    selected_zone_ids = selected_zones["zone_id"].astype(str).tolist()
    if pipeline_stage != "agent" and zone_model_references is None:
        zone_model_references = prepare_offline_models(
            canonical_dataset, [backend], training_config, training_cli_overrides,
            zone_ids=selected_zone_ids, history_days=history_days, horizon_days=horizon_days,
            force_train=force_train, data_dir=data_dir,
            backend_parameters=backend_parameters,
        )
    current_model_references = ((bundle.manifest.get("model_provenance") or {}) if bundle is not None
                                else (zone_model_references or {}).get(backend, {}))
    selected_model_references = {zone: current_model_references[zone] for zone in selected_zone_ids
                                if zone in current_model_references}
    if pipeline_stage != "agent" and offline_backend and set(selected_model_references) != set(selected_zone_ids):
        raise ValueError(f"Missing prepared offline {backend} weights for selected zones")
    selected_model_provenance = dict(selected_model_references) if bundle is not None else {}
    if selected_model_references:
        forecast_parameters["model_ids"] = {zone: reference["model_id"]
                                            for zone, reference in selected_model_references.items()}
    try:
        pipeline_data = load_pipeline_data(
            profiles,
            selected_zone_ids,
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
                zone_model_references=(zone_model_references or {}).get(backend, {}),
                raw_prediction_cache=raw_prediction_cache,
                **forecast_runtime_parameters(forecast_parameters),
            )
            scenario_artifacts = {
                str(zone_id): NativeForecasterArtifact.from_forecast_result(result)
                for zone_id, result in forecast_results.items()
            }
            selected_model_provenance = {
                str(zone): {key: result.summary.get(key) for key in (*MODEL_PROVENANCE_FIELDS, "model_parameters")}
                for zone, result in forecast_results.items() if result.summary.get("model_id")
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
                    zone_model_references=(zone_model_references or {}).get(backend, {}),
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
            backend=backend,
            agent_mode=agent_mode,
        )
    except Exception as exc:
        raise PipelineStageError(stage="precomputed_window_data", original=exc) from exc

    metrics_outputs: dict[str, Path] = {}

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
                        if not config.client_api_key:
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
                        await client.start_metrics(resolved_agent_output_dir, context={
                            "agent_mode": agent_mode, "forecast_backend": backend,
                            "forecast_origin": str(forecast_parameters.get("forecast_origin")),
                            "dataset": data_dir.name, "dataset_fingerprint": canonical_dataset.dataset_fingerprint,
                            "zone_ids": [str(zone) for zone in selected_zone_ids],
                        })
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
                metrics_outputs.update(client.metrics_outputs)

    reports = asyncio.run(execute_agent_control_stage())
    try:
        annotate_final_price_ranges(reports, pipeline_data.energy_price, effective_forecast_start)
        agent_manifest = build_agent_manifest(
            agent_mode=agent_mode,
            backend=backend,
            validation_bias=forecast_parameters["validation_bias"],
            validation_days=validation_days,
            forecaster_dir=forecaster_dir,
            selected_zone_ids=selected_zone_ids,
            forecast_origin=forecast_parameters.get("forecast_origin"),
            model_ids=forecast_parameters.get("model_ids"),
            model_provenance=selected_model_provenance,
            dataset_fingerprint=canonical_dataset.dataset_fingerprint,
            cache_keys=pipeline_data.cache_keys or {},
            feature_manifest=pipeline_data.feature_manifest or {},
        )
        if metrics_outputs:
            agent_manifest["sglang_metrics"] = {
                "scope": "server",
                "artifacts": {name: path.relative_to(resolved_agent_output_dir).as_posix()
                              for name, path in metrics_outputs.items()},
            }
        agent_outputs = write_agent_outputs(
            output_dir=resolved_agent_output_dir,
            reports=reports,
            backend=backend,
            agent_mode=agent_mode,
            manifest=agent_manifest,
        )
        return {**forecaster_outputs, **agent_outputs, **metrics_outputs}
    except Exception as exc:
        raise PipelineStageError(stage="write_outputs", original=exc) from exc


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
