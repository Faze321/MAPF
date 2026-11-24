from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.agent import normalize_agent_mode
from src.config.run import (
    BACKEND_BLEND_PARAMETER_NAMES,
    BACKEND_PARAMETER_NAMES,
    RunConfig,
    VALIDATION_BIAS_DAYS,
    normalize_pipeline_stage,
    normalize_zone_selection,
    resolve_backend_blend_alphas,
    validation_bias_days,
)
from src.config.training import TrainingConfig
from src.config.values import normalize_float_list, normalize_forecast_model_name, positive_integer
from src.data.adapters.registry import load_canonical_dataset
from src.data.canonical import DatasetSpec
from src.data.eligibility import load_zone_eligibility
from src.data.profiles import profiles_at_origin
from src.data.selection import filter_training_profiles, normalize_zone_ids, select_experiment_zone_ids
from src.paths import forecast_output_dir, forecast_start_slug, normalize_experiment_name, safe_filename
from src.pipeline.errors import experiment_error_record
from src.pipeline.stages import run_pipeline, run_training_stage
from src.results.price_range import PRICE_RANGE_TABLES
from src.results.experiment import (
    append_experiment_frame,
    write_decision_quality_summary,
    write_experiment_summary,
    write_numeric_summary,
)
from src.training.prepare import eligible_training_zone_ids, prepare_offline_models


def run_experiment_matrix(
    *,
    data_dir: Path,
    output_dir: Path,
    forecast_starts: Iterable[str],
    forecast_models: Iterable[str],
    zone_ids: str | Iterable[str] | None = None,
    experiment_zone_count: int = 12,
    experiment_zone_selection: str = "representative",
    experiment_zone_seed: int = 42,
    experiment_seeds: Iterable[int] | None = None,
    agent_modes: Iterable[str] | None = None,
    diurnal_blend_alphas: Iterable[float] | None = None,
    experiment_name: str | None = None,
    max_dataset_errors: int = RunConfig.max_dataset_errors,
    **pipeline_kwargs: Any,
) -> dict[str, Path]:
    max_dataset_errors = positive_integer(max_dataset_errors, "run.max_dataset_errors")
    models = normalize_forecast_models(forecast_models)
    matrix_stage = normalize_pipeline_stage(pipeline_kwargs.get("pipeline_stage", "full"))
    if matrix_stage == "train":
        return run_training_stage(
            data_dir=data_dir, backends=models, dataset_spec=pipeline_kwargs.get("dataset_spec"),
            cache_dir=pipeline_kwargs.get("cache_dir"), training_config=pipeline_kwargs.get("training_config"),
            training_cli_overrides=pipeline_kwargs.get("training_cli_overrides"),
            backend_parameters={name: pipeline_kwargs.get(name, getattr(RunConfig(), name))
                                for name in BACKEND_PARAMETER_NAMES},
            zone_ids=zone_ids, experiment_zone_count=experiment_zone_count,
            experiment_zone_selection=experiment_zone_selection, experiment_zone_seed=experiment_zone_seed,
            history_days=pipeline_kwargs.get("history_days", RunConfig.history_days),
            horizon_days=pipeline_kwargs.get("horizon_days", RunConfig.horizon_days),
            force_train=bool(pipeline_kwargs.get("force_train", False)),
            force_cache=bool(pipeline_kwargs.get("force_cache", False)),
        )
    validation_bias = pipeline_kwargs.get("validation_bias", False)
    validation_days = pipeline_kwargs.get("validation_days", VALIDATION_BIAS_DAYS)
    effective_validation_days = validation_bias_days(validation_bias, validation_days)
    starts = normalize_forecast_starts(forecast_starts)
    seeds = normalize_experiment_seeds(experiment_seeds)
    requested_modes = normalize_agent_modes(
        agent_modes,
        pipeline_kwargs.get("agent_mode", "multi_agent_economist_retry"),
    )
    modes = requested_modes
    if matrix_stage == "forecaster":
        modes = modes[:1]
    global_blend_alphas = normalize_blend_alphas(diurnal_blend_alphas)
    blend_overrides = {name: normalize_float_list(pipeline_kwargs.pop(name, None))
                       for name in BACKEND_BLEND_PARAMETER_NAMES}
    model_blend_alphas = {
        backend: resolve_backend_blend_alphas(backend, global_blend_alphas, blend_overrides)
        for backend in models
    }
    selected_zone_ids = normalize_zone_ids(zone_ids)
    supplied_spec = pipeline_kwargs.get("dataset_spec")
    shared_cache_dir = Path(pipeline_kwargs.pop("cache_dir", None)
                            or (supplied_spec or DatasetSpec(path=data_dir)).resolved_cache_dir)
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    base_force_cache = bool(pipeline_kwargs.pop("force_cache", False))
    selection_artifacts: dict[str, Path] = {}
    offline_backends = (pipeline_kwargs.get("training_config") or TrainingConfig()).enabled_backends(
        models, cli_overrides=pipeline_kwargs.get("training_cli_overrides"))
    canonical_dataset = None
    if not selected_zone_ids or (offline_backends and matrix_stage != "agent"):
        matrix_dataset_spec = supplied_spec or DatasetSpec(
            path=data_dir,
            cache_dir=shared_cache_dir,
            weather_file=str(pipeline_kwargs.get("weather_file") or "weather_airport.csv"),
        )
        if not selected_zone_ids and normalize_zone_selection(experiment_zone_selection) == "random":
            matrix_dataset_spec = replace(matrix_dataset_spec, allow_zero_energy_price=True)
            pipeline_kwargs["dataset_spec"] = matrix_dataset_spec
        canonical_dataset = load_canonical_dataset(
            matrix_dataset_spec,
            force_cache=base_force_cache,
        )
    training_eligible_zones = None
    if offline_backends and matrix_stage != "agent":
        training_eligible_zones = set(eligible_training_zone_ids(
            canonical_dataset, pipeline_kwargs.get("training_config"), zone_ids=selected_zone_ids,
            history_days=pipeline_kwargs.get("history_days", RunConfig.history_days),
            horizon_days=pipeline_kwargs.get("horizon_days", RunConfig.horizon_days),
        ))
    if not selected_zone_ids:
        if normalize_zone_selection(experiment_zone_selection) == "random":
            eligibility = load_zone_eligibility(
                canonical_dataset,
                forecast_starts=starts,
                history_days=pipeline_kwargs.get("history_days", RunConfig.history_days),
                validation_bias=validation_bias,
                validation_days=validation_days,
                force_cache=base_force_cache,
            )
            selection_artifacts = {
                "zone_eligibility_json": Path(eligibility["cache_path"]),
                "zone_eligibility_csv": Path(eligibility["csv_path"]),
            }
            eligible_ids = eligibility["eligible_zone_ids"]
            if training_eligible_zones is not None:
                eligible_ids = [zone for zone in eligible_ids if str(zone) in training_eligible_zones]
            requested_count = max(1, int(experiment_zone_count))
            if len(eligible_ids) < requested_count:
                raise ValueError(
                    f"Only {len(eligible_ids)} eligible zones remain for {requested_count} requested random zones. "
                    "Both load and electricity price must vary in the eligible pre-forecast history, "
                    "zero loads and energy prices must be within their respective eligibility limits, "
                    "all historical windows must be complete and enough offline training samples must be available. "
                    "Choose another dataset/date or reduce its configured zone count "
                    "(run.experiment_zone_counts or run.experiment_zone_count). "
                    f"See eligibility reasons: {eligibility['cache_path']}"
                )
            profiles = pd.DataFrame({"zone_id": eligible_ids})
        else:
            selection_start = pd.Timestamp(min(starts))
            _, _, profiles = profiles_at_origin(
                canonical_dataset, shared_cache_dir, selection_start, force_cache=base_force_cache,
            )
        if training_eligible_zones is not None:
            profiles = filter_training_profiles(
                profiles, training_eligible_zones, empty_error="No candidate zones have enough complete sliding windows for offline training",
            )
        selected_zone_ids = select_experiment_zone_ids(
            profiles,
            count=experiment_zone_count,
            selection=experiment_zone_selection,
            seed=experiment_zone_seed,
        )
    if matrix_stage != "agent":
        if pipeline_kwargs.get("zone_model_references") is None:
            pipeline_kwargs["zone_model_references"] = prepare_offline_models(
                canonical_dataset, models, pipeline_kwargs.get("training_config"),
                pipeline_kwargs.get("training_cli_overrides"), zone_ids=selected_zone_ids,
                history_days=pipeline_kwargs.get("history_days", RunConfig.history_days),
                horizon_days=pipeline_kwargs.get("horizon_days", RunConfig.horizon_days),
                force_train=bool(pipeline_kwargs.get("force_train", False)), data_dir=data_dir,
                backend_parameters={name: pipeline_kwargs.get(name, getattr(RunConfig(), name))
                                    for name in BACKEND_PARAMETER_NAMES},
            )
        pipeline_kwargs.setdefault("raw_prediction_cache", {})
        # A force request applies to preparation, never to each experiment cell.
        pipeline_kwargs["force_train"] = False
    del canonical_dataset
    resolved_experiment_name = normalize_experiment_name(experiment_name, default="experiment")
    # main resolves <output_folder>/<experiment_name>/<dataset> once for every stage.
    experiment_dir = output_dir
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
    dataset_error_count = 0
    metrics_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    rationale_frames: list[pd.DataFrame] = []
    explainability_frames: list[pd.DataFrame] = []
    agent_attempt_usage_frames: list[pd.DataFrame] = []
    agent_step_token_usage_frames: list[pd.DataFrame] = []
    control_attempt_trace_frames: list[pd.DataFrame] = []
    experiment_tables = {
        "forecast_metrics_csv": (metrics_frames, metrics_path),
        "price_comparison_summary_csv": (price_frames, price_path),
        "rationale_trace_csv": (rationale_frames, rationale_path),
        "explainability_review_packet_csv": (explainability_frames, explainability_path),
        "agent_attempt_usage_csv": (agent_attempt_usage_frames, agent_attempt_usage_path),
        "agent_step_token_usage_csv": (agent_step_token_usage_frames, agent_step_token_usage_path),
        "control_attempt_trace_csv": (control_attempt_trace_frames, control_attempt_trace_path),
    }
    price_range_outputs = {
        f"{name}_csv": experiment_dir / f"experiment_{name}.csv"
        for name in PRICE_RANGE_TABLES if matrix_stage in {"agent", "full"}
    }
    for key, path in price_range_outputs.items():
        experiment_tables[key] = ([], path)
    cache_attempted = False
    add_seed_folder = len(seeds) > 1
    add_mode_folder = len(modes) > 1

    for forecast_origin in starts:
        time_output_dir = experiment_dir / forecast_start_slug(forecast_origin)
        for backend in models:
            for seed in seeds:
                for agent_mode in modes:
                    for blend_alpha in model_blend_alphas[backend]:
                        run_base_dir = time_output_dir
                        if add_seed_folder and seed is not None:
                            run_base_dir = run_base_dir / f"seed_{seed}"
                        if add_mode_folder:
                            run_base_dir = run_base_dir / f"agent_{safe_filename(agent_mode)}"
                        run_output_dir = forecast_output_dir(run_base_dir, backend, blend_alpha,
                                                             validation_bias=validation_bias, validation_days=validation_days)
                        run_kwargs = dict(pipeline_kwargs)
                        run_kwargs["agent_mode"] = agent_mode
                        run_kwargs["diurnal_blend_alpha"] = blend_alpha
                        if matrix_stage == "agent":
                            shared_run_base_dir = time_output_dir
                            if add_seed_folder and seed is not None:
                                shared_run_base_dir = shared_run_base_dir / f"seed_{seed}"
                            if not run_kwargs.get("forecaster_output_dir"):
                                mode_forecaster = run_output_dir / "forecaster"
                                shared_forecaster = forecast_output_dir(
                                    shared_run_base_dir, backend, blend_alpha,
                                    validation_bias=validation_bias, validation_days=validation_days
                                ) / "forecaster"
                                run_kwargs["forecaster_output_dir"] = (
                                    mode_forecaster
                                    if (mode_forecaster / "forecaster_manifest.json").is_file()
                                    else shared_forecaster
                                )
                            if not run_kwargs.get("agent_output_dir"):
                                run_kwargs["agent_output_dir"] = run_output_dir / "agent"
                        record: dict[str, Any] = {
                            "experiment_name": resolved_experiment_name,
                            "forecast_origin": forecast_origin,
                            "backend": backend,
                            "experiment_seed": seed,
                            "agent_mode": agent_mode,
                            "validation_bias": validation_bias,
                            "validation_days": effective_validation_days,
                            "diurnal_blend_alpha": blend_alpha,
                            "zone_count": len(selected_zone_ids),
                            "zone_ids": ",".join(selected_zone_ids),
                            "zone_eligibility_cache": str(selection_artifacts.get("zone_eligibility_json", "")),
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
                            "dataset_error_count": dataset_error_count,
                        }
                        try:
                            outputs = run_pipeline(
                                data_dir=data_dir,
                                output_dir=run_base_dir,
                                cache_dir=shared_cache_dir,
                                force_cache=base_force_cache and not cache_attempted,
                                forecast_origin=forecast_origin,
                                backend=backend,
                                zone_ids=selected_zone_ids,
                                **run_kwargs,
                            )
                            record["status"] = "success"
                            record["completed_at"] = pd.Timestamp.now().isoformat()
                            metadata = {
                                "forecast_origin": forecast_origin,
                                "backend": backend,
                                "experiment_seed": seed,
                                "agent_mode": agent_mode,
                                "validation_bias": validation_bias,
                                "validation_days": effective_validation_days,
                                "diurnal_blend_alpha": blend_alpha,
                                "run_output_dir": str(run_output_dir),
                            }
                            for output_key, (frames, _) in experiment_tables.items():
                                append_experiment_frame(frames, outputs.get(output_key), metadata)
                        except Exception as exc:
                            dataset_error_count += 1
                            record["status"] = "failed"
                            record["completed_at"] = pd.Timestamp.now().isoformat()
                            record["dataset_error_count"] = dataset_error_count
                            record.update(
                                experiment_error_record(
                                    exc,
                                    experiment_dir=experiment_dir,
                                    record=record,
                                    run_index=len(run_records) + 1,
                                )
                            )
                            limit_reached = dataset_error_count >= max_dataset_errors
                            action = (
                                "Stopping this dataset: error limit reached."
                                if limit_reached
                                else "Skipping this combination and continuing this dataset."
                            )
                            print(
                                f"[{data_dir}] Experiment failed "
                                f"({dataset_error_count}/{max_dataset_errors}): {record['error']}\n"
                                f"{action} Traceback: {record['traceback_file']}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if limit_reached:
                                raise
                        finally:
                            cache_attempted = True
                            run_records.append(record)
                            pd.DataFrame(run_records).to_csv(runs_path, index=False)

    summaries = {
        key: write_experiment_summary(frames, path)
        for key, (frames, path) in experiment_tables.items()
    }
    metrics = summaries["forecast_metrics_csv"]
    prices = summaries["price_comparison_summary_csv"]
    rationales = summaries["rationale_trace_csv"]
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
        **selection_artifacts,
        "experiment_dir": experiment_dir,
        "experiment_runs_csv": runs_path,
        "experiment_forecast_metrics_csv": metrics_path,
        "experiment_price_comparison_summary_csv": price_path,
        "experiment_rationale_trace_csv": rationale_path,
        "experiment_explainability_review_packet_csv": explainability_path,
        "experiment_agent_attempt_usage_csv": agent_attempt_usage_path,
        "experiment_agent_step_token_usage_csv": agent_step_token_usage_path,
        "experiment_control_attempt_trace_csv": control_attempt_trace_path,
        **{f"experiment_{key}": path for key, path in price_range_outputs.items()},
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


def normalize_blend_alphas(diurnal_blend_alphas: Iterable[float] | None) -> list[float]:
    values = normalize_float_list(list(diurnal_blend_alphas) if diurnal_blend_alphas is not None else None)
    return values if values is not None else [0.0]
