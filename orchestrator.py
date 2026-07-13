from __future__ import annotations

import asyncio
import traceback
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from agents import (
    AgentChatClient,
    AgentStageError,
    recompute_report_nash,
    run_all_zone_chains,
    summarize_nash_equilibrium,
)
from config import AgentConfig, normalize_agent_mode, normalize_forecast_model_name
from data_loader import build_zone_3h_load_quantiles, build_zone_profiles, load_pipeline_data
from forecasting import DEFAULT_TIMESFM_EXOG_COLS, ForecastResult, forecast_zone
from load_policy import (
    EXTREMELY_HIGH_STRESS,
    HIGH_MAX_LOAD_PCT,
    HIGH_STRESS,
    LOW_MAX_LOAD_PCT,
    LOW_STRESS,
    MEDIUM_MAX_LOAD_PCT,
    MEDIUM_STRESS,
    classify_load_percentage,
    load_percentage,
)
from reporting import safe_filename, write_outputs
from zone_selection import select_zone_categories


STRESS_LEVEL_ORDER = {
    LOW_STRESS: 0,
    MEDIUM_STRESS: 1,
    HIGH_STRESS: 2,
    EXTREMELY_HIGH_STRESS: 3,
}
ERROR_CODE_BY_STAGE = {
    "zone_selection": "MAPF-E010",
    "load_data": "MAPF-E020",
    "forecast": "MAPF-E100",
    "build_contexts": "MAPF-E110",
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
    agent_mode: str = "agents",
    forecast_model: str = "timesfm",
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 168,
    timesfm_step_horizon: int = 24,
    timesfm_exog_cols: list[str] | None = None,
    timesfm_diurnal_blend_alpha: float = 1.0,
    timesfm_roll_actuals: bool = True,
    ar_diurnal_blend_alpha: float = 0.0,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 512,
    chronos_step_horizon: int = 24,
    chronos_diurnal_blend_alpha: float = 0.0,
    chronos_device: str = "auto",
    chronos_roll_actuals: bool = True,
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
    lstm_roll_actuals: bool = True,
    lstm_seed: int = 42,
    temperature: float = 0.2,
    precomputed_window_data: Path | str | None = None,
) -> dict[str, Path]:
    forecast_model = normalize_forecast_model_name(forecast_model)
    agent_mode = normalize_agent_mode(agent_mode)
    apply_nash = agent_mode != "agents_no_nash"
    chain_mode = "single_model" if agent_mode == "single_model" else "agents"
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir = cache_dir or output_dir / "cache"
    run_output_dir = forecast_output_dir(output_dir, forecast_model)
    profiles = build_zone_profiles(
        data_dir,
        shared_cache_dir,
        force_cache=force_cache,
        max_poi_rows=max_poi_rows,
    )
    zone_load_quantiles = build_zone_3h_load_quantiles(
        data_dir,
        shared_cache_dir,
        force_cache=force_cache,
    )
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
        )
    except Exception as exc:
        raise PipelineStageError(stage="load_data", original=exc) from exc
    try:
        contexts, forecast_results = build_contexts(
            pipeline_data=pipeline_data,
            selected_zones=selected_zones,
            forecast_start=forecast_start,
            horizon_days=horizon_days,
            history_days=history_days,
            validation_days=validation_days,
            forecast_model=forecast_model,
            zone_load_quantiles=zone_load_quantiles,
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
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(stage="build_contexts", original=exc) from exc

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

    reports_by_zone: dict[str, dict[str, Any]] = {}
    try:
        if cached_contexts:
            cached_reports = asyncio.run(
                run_all_zone_chains(
                    cached_contexts,
                    client=None,
                    temperature=temperature,
                    heuristic_source="precomputed_window_data",
                    chain_mode=chain_mode,
                    apply_nash=False,
                )
            )
            cached_reports = apply_precomputed_window_data(
                cached_reports,
                cached_windows_by_zone,
                source_path=precomputed_window_data,
            )
            reports_by_zone.update({str(report.get("zone_id")): report for report in cached_reports})

        if live_contexts:
            if agent_mode == "rules":
                client = None
                heuristic_source = "rules"
            elif dry_run:
                client = None
                heuristic_source = f"dry-run_{agent_mode}" if agent_mode != "agents" else "dry-run"
            else:
                config = AgentConfig.from_file(config_path, model=model, required=True)
                if not config.api_key:
                    raise RuntimeError("agent.api_key is required in config.yaml, or pass --dry-run")
                config = select_agent_config_for_mode(config, agent_mode=agent_mode, cli_model=model)
                client = AgentChatClient(config)
                heuristic_source = "dry-run"
            live_reports = asyncio.run(
                run_all_zone_chains(
                    live_contexts,
                    client=client,
                    temperature=temperature,
                    heuristic_source=heuristic_source,
                    chain_mode=chain_mode,
                    apply_nash=apply_nash,
                )
            )
            reports_by_zone.update({str(report.get("zone_id")): report for report in live_reports})
        reports = [reports_by_zone[str(context.get("zone_id"))] for context in contexts]
    except AgentStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(stage="agent_chain", original=exc) from exc
    try:
        live_zone_ids = {str(context.get("zone_id")) for context in live_contexts}
        live_reports = [report for report in reports if str(report.get("zone_id")) in live_zone_ids]
        live_reports = apply_price_conditioned_baseline_forecasts(
            reports=live_reports,
            contexts=live_contexts,
            pipeline_data=pipeline_data,
            zone_load_quantiles=zone_load_quantiles,
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
            ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
            chronos_repo=chronos_repo,
            chronos_context_hours=chronos_context_hours,
            chronos_step_horizon=chronos_step_horizon,
            chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
            chronos_device=chronos_device,
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
            lstm_seed=lstm_seed,
            apply_nash=apply_nash,
        )
        reports_by_zone.update({str(report.get("zone_id")): report for report in live_reports})
        reports = [reports_by_zone[str(context.get("zone_id"))] for context in contexts]
    except Exception as exc:
        raise PipelineStageError(stage="price_conditioned_forecast", original=exc) from exc
    try:
        return write_outputs(
            output_dir=run_output_dir,
            selected_zones=selected_zones,
            contexts=contexts,
            reports=reports,
            forecast_results=forecast_results,
            forecast_model=forecast_model,
            agent_mode=agent_mode,
        )
    except Exception as exc:
        raise PipelineStageError(stage="write_outputs", original=exc) from exc


def select_agent_config_for_mode(
    config: AgentConfig,
    *,
    agent_mode: str,
    cli_model: str | None,
) -> AgentConfig:
    if agent_mode == "single_model" and cli_model is None and config.single_model_model:
        return replace(config, model=config.single_model_model)
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
    modes = normalize_agent_modes(agent_modes, pipeline_kwargs.get("agent_mode", "agents"))
    blend_alphas = normalize_blend_alphas(diurnal_blend_alphas)
    selected_zone_ids = normalize_zone_ids(zone_ids)
    shared_cache_dir = output_dir / "cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    base_force_cache = bool(pipeline_kwargs.pop("force_cache", False))
    if not selected_zone_ids:
        profiles = build_zone_profiles(
            data_dir,
            shared_cache_dir,
            force_cache=base_force_cache,
            max_poi_rows=pipeline_kwargs.get("max_poi_rows"),
        )
        selected_zone_ids = select_representative_zone_ids(
            profiles,
            count=experiment_zone_count,
        )
    experiment_dir = output_dir / "experiments" / (
        experiment_name
        or experiment_slug(
            zone_count=len(selected_zone_ids),
            time_count=len(starts),
            mode_count=len(modes),
            blend_count=len(blend_alphas),
        )
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    runs_path = experiment_dir / "experiment_runs.csv"
    metrics_path = experiment_dir / "experiment_forecast_metrics.csv"
    price_path = experiment_dir / "experiment_price_comparison_summary.csv"
    rationale_path = experiment_dir / "experiment_rationale_trace.csv"
    explainability_path = experiment_dir / "experiment_explainability_review_packet.csv"
    forecast_summary_path = experiment_dir / "experiment_forecast_summary.csv"
    price_summary_path = experiment_dir / "experiment_price_summary.csv"
    rationale_summary_path = experiment_dir / "experiment_rationale_summary.csv"
    decision_summary_path = experiment_dir / "experiment_decision_quality_summary.csv"

    run_records: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    rationale_frames: list[pd.DataFrame] = []
    explainability_frames: list[pd.DataFrame] = []
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
                        except Exception as exc:
                            record["status"] = "failed"
                            record.update(
                                experiment_error_record(
                                    exc,
                                    experiment_dir=experiment_dir,
                                    record=record,
                                    run_index=len(run_records) + 1,
                                )
                            )
                            record["completed_at"] = pd.Timestamp.now().isoformat()
                            raise
                        finally:
                            cache_attempted = True
                            run_records.append(record)
                            pd.DataFrame(run_records).to_csv(runs_path, index=False)

    metrics = write_experiment_summary(metrics_frames, metrics_path)
    prices = write_experiment_summary(price_frames, price_path)
    rationales = write_experiment_summary(rationale_frames, rationale_path)
    write_experiment_summary(explainability_frames, explainability_path)
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
            "avg_predicted_minus_actual_service_price",
            "avg_predicted_vs_actual_pct",
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
    traceback_path = write_error_traceback(
        exc,
        experiment_dir=experiment_dir,
        record=record,
        run_index=run_index,
        error_code=details["error_code"],
    )
    error = (
        f"{details['error_code']} {details['error_stage']}: "
        f"{details['error_type']}: {details['error_message']}"
    )
    return {
        "error": error,
        "error_code": details["error_code"],
        "error_stage": details["error_stage"],
        "error_zone_id": details["error_zone_id"],
        "error_agent": details["error_agent"],
        "error_type": details["error_type"],
        "error_message": details["error_message"],
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
        "error_code": ERROR_CODE_BY_STAGE.get(stage, ERROR_CODE_BY_STAGE["unexpected"]),
        "error_stage": stage,
        "error_zone_id": zone_id,
        "error_agent": agent,
        "error_type": type(source).__name__,
        "error_message": str(source),
    }


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
    metadata = "\n".join(f"{key}: {value}" for key, value in record.items())
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
                    "avg_predicted_minus_actual_service_price",
                    "avg_predicted_vs_actual_pct",
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
        "predicted_service_price",
        "price_conditioned_load_kwh",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Precomputed window data is missing required column(s): " + ", ".join(missing)
        )
    frame = frame.copy()
    frame["zone_id"] = frame["zone_id"].astype("string").str.strip()
    for column in ("window_start", "window_end"):
        parsed = pd.to_datetime(frame[column], errors="coerce")
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
                or optional_number(row.get("predicted_service_price")) is None
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
        "load_pct_of_q95",
        "actual_load_pct_of_q95",
        "price_conditioned_load_stress_level",
        "load_3h_q95_kwh",
        "load_low_max_pct",
        "load_medium_max_pct",
        "load_high_max_pct",
        "historical_min_service_price",
        "historical_max_service_price",
        "nash_equilibrium_reached",
        "nash_status",
        "nash_iterations",
        "target_peak_reduction_pct",
        "baseline_load_pct_of_q95",
        "target_load_kwh",
        "target_load_pct_of_q95",
        "medium_load_min_kwh",
        "medium_load_max_kwh",
        "elasticity_factor",
        "capacity_limit_kwh",
        "capacity_limit_source",
        "expected_load_kwh",
        "expected_load_pct_of_q95",
        "load_in_medium_band",
        "price_within_historical_bounds",
        "grid_safe",
        "user_tolerant",
        "price_stable",
        "discomfort_score",
        "max_discomfort_score",
        "price_stability_epsilon_pct",
        "action_label",
        "price_rationale",
    )
    boolean_fields = {
        "nash_equilibrium_reached",
        "grid_safe",
        "user_tolerant",
        "price_stable",
        "load_in_medium_band",
        "price_within_historical_bounds",
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
            predicted_price = optional_number(row.get("predicted_service_price"))
            conditioned_load = optional_number(row.get("price_conditioned_load_kwh"))
            actual_price = optional_number(updated.get("mean_service_price"))
            final_shift = optional_number(row.get("final_price_shift_pct"))
            if final_shift is None and actual_price not in (None, 0) and predicted_price is not None:
                final_shift = ((predicted_price / actual_price) - 1) * 100
            pre_nash_shift = optional_number(row.get("pre_nash_price_shift_pct"))
            if pre_nash_shift is None:
                pre_nash_shift = final_shift

            forecast_load = optional_number(row.get("forecast_load_kwh"))
            if forecast_load is not None:
                updated["sum_predicted_kwh"] = forecast_load
            updated["predicted_service_price"] = predicted_price
            updated["suggested_price_shift_pct"] = round(final_shift, 4) if final_shift is not None else None
            updated["pre_nash_suggested_price_shift_pct"] = (
                round(pre_nash_shift, 4) if pre_nash_shift is not None else None
            )
            updated["price_conditioned_baseline_load_kwh"] = conditioned_load
            updated["price_conditioned_mean_predicted_kwh"] = optional_number(
                row.get("price_conditioned_mean_load_kwh")
            )
            updated["price_conditioned_peak_predicted_kwh"] = optional_number(
                row.get("price_conditioned_peak_load_kwh")
            )
            forecaster_price = optional_number(row.get("forecaster_input_predicted_service_price"))
            updated["price_conditioned_service_price"] = (
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
            if row.get("nash_status") is None:
                updated["nash_equilibrium_reached"] = None
                updated["nash_status"] = "precomputed"
                updated["nash_iterations"] = 0
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
        if updated_windows and all(window.get("nash_status") == "precomputed" for window in updated_windows):
            nash_summary = {
                "nash_equilibrium_reached": None,
                "nash_equilibrium_windows": len(updated_windows),
                "nash_equilibrium_reached_windows": 0,
                "nash_equilibrium_rounds": 0,
                "nash_equilibrium_summary": f"Reused precomputed results for {len(updated_windows)} pricing windows",
            }
        else:
            nash_summary = summarize_nash_equilibrium(updated_windows)
        updated_report.update(nash_summary)
        updated_report["agent_reasoning"] = f"Reused window load and service-price predictions from {source_name}."
        updated_report["action_label"] = "Reuse precomputed window predictions"
        updated_report["price_rationale"] = f"Loaded complete 3-hour window predictions from {source_name}."
        updated_report["agent_time_cost_seconds"] = 0.0
        updated_report["agent_prompt_tokens"] = 0
        updated_report["agent_completion_tokens"] = 0
        updated_report["agent_total_tokens"] = 0
        updated_report["agent_call_usage"] = []
        updated_report["source"] = "precomputed_window_data"
        updated_report["precomputed_window_data_source"] = str(source_path)
        updated_reports.append(updated_report)
    return updated_reports


def precomputed_window_key(zone_id: Any, window_start: Any, window_end: Any) -> tuple[str, str, str]:
    return (
        str(zone_id).strip(),
        pd.Timestamp(window_start).isoformat(),
        pd.Timestamp(window_end).isoformat(),
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
    zone_load_quantiles: pd.DataFrame,
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
    apply_nash: bool = True,
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
    thresholds_by_zone = zone_load_quantiles.copy()
    thresholds_by_zone["zone_id"] = thresholds_by_zone["zone_id"].astype(str)
    thresholds_by_zone = thresholds_by_zone.set_index("zone_id", drop=False)

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

        price_conditioned_service_price = service_price_with_predicted_windows(
            pipeline_data.service_price,
            zone_id,
            report.get("price_change_windows_3h") or [],
        )
        conditioned_result = forecast_zone(
            zone_id=zone_id,
            category=str(report.get("category") or context.get("category") or "User-selected"),
            load=pipeline_data.load,
            service_price=price_conditioned_service_price,
            energy_price=pipeline_data.energy_price,
            occupancy=pipeline_data.occupancy,
            weather=pipeline_data.weather,
            profile=profiles_by_zone.loc[zone_id].to_dict(),
            forecast_start=context.get("forecast_start") or forecast_start,
            horizon_days=int(context.get("forecast_horizon_days") or horizon_days),
            history_days=history_days,
            validation_days=validation_days,
            forecast_model=normalized_model,
            timesfm_repo=timesfm_repo,
            timesfm_context_hours=timesfm_context_hours,
            timesfm_step_horizon=timesfm_step_horizon,
            timesfm_exog_cols=ensure_service_price_exog_cols(timesfm_exog_cols),
            timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
            timesfm_roll_actuals=False,
            ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
            chronos_repo=chronos_repo,
            chronos_context_hours=chronos_context_hours,
            chronos_step_horizon=chronos_step_horizon,
            chronos_exog_cols=ensure_service_price_exog_cols(DEFAULT_TIMESFM_EXOG_COLS),
            chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
            chronos_device=chronos_device,
            chronos_roll_actuals=False,
            lstm_context_hours=lstm_context_hours,
            lstm_step_horizon=lstm_step_horizon,
            lstm_exog_cols=ensure_service_price_exog_cols(lstm_exog_cols),
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_epochs=lstm_epochs,
            lstm_learning_rate=lstm_learning_rate,
            lstm_batch_size=lstm_batch_size,
            lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
            lstm_device=lstm_device,
            lstm_roll_actuals=False,
            lstm_seed=lstm_seed,
        )
        thresholds = load_stress_thresholds(
            thresholds_by_zone,
            zone_id,
            profile=profiles_by_zone.loc[zone_id].to_dict(),
        )
        conditioned_windows = build_pricing_windows_3h(
            conditioned_result.hourly,
            stress_thresholds=thresholds,
        )
        updated_report = attach_price_conditioned_baselines(
            report,
            conditioned_windows,
            forecast_model=normalized_model,
        )
        updated_reports.append(recompute_report_nash(context, updated_report, apply_nash=apply_nash))
    return updated_reports


def ensure_service_price_exog_cols(values: list[str] | None) -> list[str]:
    cols = list(values) if values else list(DEFAULT_TIMESFM_EXOG_COLS)
    if "s_price" in cols:
        return cols
    if "e_price" in cols:
        cols.insert(cols.index("e_price") + 1, "s_price")
    else:
        cols.append("s_price")
    return cols


def service_price_with_predicted_windows(
    service_price: pd.DataFrame,
    zone_id: str,
    windows: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = service_price.copy()
    if "time" not in frame or zone_id not in frame:
        return frame
    frame["time"] = pd.to_datetime(frame["time"])
    for window in windows:
        if not isinstance(window, dict):
            continue
        predicted_price = predicted_service_price(window)
        if predicted_price is None:
            continue
        start = pd.Timestamp(window.get("window_start"))
        end = pd.Timestamp(window.get("window_end"))
        mask = (frame["time"] >= start) & (frame["time"] <= end)
        frame.loc[mask, zone_id] = predicted_price
    return frame


def predicted_service_price(window: dict[str, Any]) -> float | None:
    base_price = optional_number(window.get("mean_service_price"))
    shift_pct = optional_number(window.get("suggested_price_shift_pct"))
    if shift_pct is None:
        shift_pct = optional_number(window.get("pre_nash_suggested_price_shift_pct"))
    if base_price is None or shift_pct is None:
        return None
    return round(base_price * (1 + shift_pct / 100), 4)


def attach_price_conditioned_baselines(
    report: dict[str, Any],
    conditioned_windows: list[dict[str, Any]],
    *,
    forecast_model: str,
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
            updated["price_conditioned_service_price"] = conditioned.get("mean_service_price")
            updated["price_conditioned_baseline_source"] = (
                f"{forecast_model}_forecast_with_predicted_service_price_and_observed_conditions"
            )
        updated_windows.append(updated)

    updated_report = dict(report)
    updated_report["price_change_windows_3h"] = updated_windows
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
    zone_load_quantiles: pd.DataFrame,
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
    stress_thresholds_by_zone = zone_load_quantiles.set_index("zone_id", drop=False)
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
        pricing_windows_3h = build_pricing_windows_3h(raw_result.hourly, stress_thresholds=zone_stress_thresholds)
        summary = apply_load_quantile_stress(raw_result.summary, zone_stress_thresholds, pricing_windows_3h)
        result = ForecastResult(hourly=raw_result.hourly, summary=summary)
        forecast_results[zone_id] = result
        hourly_forecast = build_agent_hourly_data(result.hourly)
        hourly_averages = build_hourly_averages(result.hourly)
        horizon_text = format_horizon_days(summary.get("forecast_horizon_days", horizon_days))
        context = {
            **summary,
            "selection_reason": row["selection_reason"],
            "hourly_averages": hourly_averages,
            "hourly_forecast": hourly_forecast,
            "pricing_windows_3h": pricing_windows_3h,
            "instructions": {
                "forecast_task": f"Predict next {horizon_text} of EV charging load.",
                "behavior_task": "Explain demand using POI, weather, and temporal markers.",
                "pricing_task": "Suggest service-price shifts for each 3-hour pricing window.",
            },
        }
        contexts.append(context)
    return contexts, forecast_results


def load_stress_thresholds(
    quantiles_by_zone: pd.DataFrame,
    zone_id: str,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or {}
    price_fields = {
        "historical_min_service_price": safe_float(profile.get("historical_min_service_price")),
        "historical_max_service_price": safe_float(profile.get("historical_max_service_price")),
    }
    if zone_id not in quantiles_by_zone.index:
        return {
            "available": False,
            "source_file": "volume.csv",
            "window_hours": 3,
            "historical_windows": 0,
            "q50": 0.0,
            "q80": 0.0,
            "q95": 0.0,
            "reference_load_kwh": 0.0,
            "low_max_pct": LOW_MAX_LOAD_PCT,
            "medium_max_pct": MEDIUM_MAX_LOAD_PCT,
            "high_max_pct": HIGH_MAX_LOAD_PCT,
            **price_fields,
        }
    row = quantiles_by_zone.loc[zone_id]
    q95 = safe_float(row.get("load_3h_q95_kwh"))
    return {
        "available": True,
        "source_file": str(row.get("stress_source_file", "volume.csv")),
        "window_hours": int(row.get("stress_window_hours", 3) or 3),
        "historical_windows": int(row.get("historical_3h_windows", 0) or 0),
        "q50": safe_float(row.get("load_3h_q50_kwh")),
        "q80": safe_float(row.get("load_3h_q80_kwh")),
        "q95": q95,
        "reference_load_kwh": q95,
        "low_max_pct": LOW_MAX_LOAD_PCT,
        "medium_max_pct": MEDIUM_MAX_LOAD_PCT,
        "high_max_pct": HIGH_MAX_LOAD_PCT,
        **price_fields,
    }


def classify_load_stress(load_value: Any, thresholds: dict[str, Any]) -> str:
    if not thresholds.get("available", True):
        return LOW_STRESS
    reference_load = thresholds.get("reference_load_kwh") or thresholds.get("q95", 0.0)
    return classify_load_percentage(load_percentage(load_value, reference_load))


def apply_load_quantile_stress(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    pricing_windows_3h: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(summary)
    stress_loads = [safe_float(window.get("sum_predicted_kwh")) for window in pricing_windows_3h]
    stress_load = max(stress_loads) if stress_loads else 0.0
    stress_percentages = [safe_float(window.get("load_pct_of_q95")) for window in pricing_windows_3h]
    window_levels = [window.get("load_stress_level") for window in pricing_windows_3h]
    updated["grid_stress_level"] = max_stress_level(window_levels) if window_levels else classify_load_stress(stress_load, thresholds)
    updated["grid_stress_basis"] = "forecast_3h_sum_predicted_kwh_as_pct_of_zone_historical_3h_q95"
    updated["grid_stress_load_kwh"] = stress_load
    updated["grid_stress_load_pct_of_q95"] = max(stress_percentages) if stress_percentages else 0.0
    updated["grid_stress_source_file"] = thresholds.get("source_file", "volume.csv")
    updated["grid_stress_window_hours"] = thresholds.get("window_hours", 3)
    updated["grid_stress_historical_windows"] = thresholds.get("historical_windows", 0)
    updated["grid_stress_q50_kwh"] = thresholds.get("q50", 0.0)
    updated["grid_stress_q80_kwh"] = thresholds.get("q80", 0.0)
    updated["grid_stress_q95_kwh"] = thresholds.get("q95", 0.0)
    updated["load_low_max_pct"] = thresholds.get("low_max_pct", LOW_MAX_LOAD_PCT)
    updated["load_medium_max_pct"] = thresholds.get("medium_max_pct", MEDIUM_MAX_LOAD_PCT)
    updated["load_high_max_pct"] = thresholds.get("high_max_pct", HIGH_MAX_LOAD_PCT)
    updated["historical_min_service_price"] = thresholds.get("historical_min_service_price", 0.0)
    updated["historical_max_service_price"] = thresholds.get("historical_max_service_price", 0.0)
    updated.update(stress_evaluation_metrics(pricing_windows_3h))
    return updated


def stress_evaluation_metrics(pricing_windows_3h: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_windows = []
    for window in pricing_windows_3h:
        predicted_rank = stress_rank(window.get("load_stress_level") or window.get("grid_stress_level"))
        actual_rank = stress_rank(window.get("actual_load_stress_level") or window.get("actual_grid_stress_level"))
        if predicted_rank is None or actual_rank is None:
            continue
        evaluated_windows.append((window, predicted_rank, actual_rank))

    if not evaluated_windows:
        return {
            "actual_grid_stress_level": None,
            "actual_grid_stress_load_kwh": None,
            "stress_accuracy": None,
            "miss_stress_rate": None,
            "stress_eval_windows": 0,
            "stress_miss_count": None,
        }

    correct_count = sum(1 for _, predicted_rank, actual_rank in evaluated_windows if predicted_rank == actual_rank)
    miss_count = sum(1 for _, predicted_rank, actual_rank in evaluated_windows if actual_rank > predicted_rank)
    actual_levels = [
        window.get("actual_load_stress_level") or window.get("actual_grid_stress_level")
        for window, _, _ in evaluated_windows
    ]
    actual_loads = [
        safe_float(window.get("actual_stress_load_3h_kwh") or window.get("sum_actual_kwh"))
        for window, _, _ in evaluated_windows
    ]
    total = len(evaluated_windows)
    return {
        "actual_grid_stress_level": max_stress_level(actual_levels),
        "actual_grid_stress_load_kwh": max(actual_loads) if actual_loads else None,
        "stress_accuracy": round(correct_count / total, 4),
        "miss_stress_rate": round(miss_count / total, 4),
        "stress_eval_windows": total,
        "stress_miss_count": miss_count,
    }


def build_agent_hourly_data(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "time",
        "predicted_kwh",
        "q10_kwh",
        "q50_kwh",
        "q90_kwh",
        "actual_kwh",
        "error_kwh",
        "abs_pct_error",
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
        "actual_kwh": "mean_actual_kwh",
        "s_price": "mean_service_price",
        "e_price": "mean_energy_price",
        "occupancy": "mean_occupancy",
        "T": "mean_temp_c",
        "U": "mean_humidity",
        "nRAIN": "mean_rain",
        "abs_pct_error": "mean_abs_pct_error",
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
            "actual_kwh": ("mean_actual_kwh", "sum_actual_kwh", "peak_actual_kwh"),
        }
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
            "abs_pct_error": "mean_abs_pct_error",
        }
        for source_col, output_col in mean_columns.items():
            if source_col in chunk:
                window[output_col] = clean_context_value(pd.to_numeric(chunk[source_col], errors="coerce").mean(skipna=True))
        if "nRAIN" in chunk:
            window["total_rain"] = clean_context_value(pd.to_numeric(chunk["nRAIN"], errors="coerce").sum(skipna=True))
        if stress_thresholds is not None:
            stress_load = window.get("sum_predicted_kwh")
            stress_level = classify_load_stress(stress_load, stress_thresholds)
            actual_stress_load = window.get("sum_actual_kwh")
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
            window["load_3h_q50_kwh"] = stress_thresholds.get("q50", 0.0)
            window["load_3h_q80_kwh"] = stress_thresholds.get("q80", 0.0)
            window["load_3h_q95_kwh"] = stress_thresholds.get("q95", 0.0)
            reference_load = stress_thresholds.get("reference_load_kwh") or stress_thresholds.get("q95", 0.0)
            window["load_pct_of_q95"] = round(load_percentage(stress_load, reference_load), 4)
            window["actual_load_pct_of_q95"] = (
                round(load_percentage(actual_stress_load, reference_load), 4)
                if actual_stress_load is not None
                else None
            )
            window["load_low_max_pct"] = stress_thresholds.get("low_max_pct", LOW_MAX_LOAD_PCT)
            window["load_medium_max_pct"] = stress_thresholds.get("medium_max_pct", MEDIUM_MAX_LOAD_PCT)
            window["load_high_max_pct"] = stress_thresholds.get("high_max_pct", HIGH_MAX_LOAD_PCT)
            window["historical_min_service_price"] = stress_thresholds.get(
                "historical_min_service_price",
                0.0,
            )
            window["historical_max_service_price"] = stress_thresholds.get(
                "historical_max_service_price",
                0.0,
            )
        windows.append(window)
    return windows


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
    ]
    return selected[[col for col in preferred_columns if col in selected.columns]]
