from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    AppConfig,
    normalize_agent_mode,
    normalize_agent_mode_list,
    normalize_float_list,
    normalize_forecast_model_list,
    normalize_int_list,
    normalize_pipeline_stage,
    normalize_string_list,
)
from dataset_adapter import DatasetSpec
from orchestrator import format_failure_message, run_experiment_matrix, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Run Multi-Agent Prescriptive Forecasting on UrbanEV data.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file for run and model settings.")
    parser.add_argument("--data-dir", default=None, help="Directory containing UrbanEV CSV files.")
    parser.add_argument(
        "--dataset-adapter",
        default=None,
        help="Dataset adapter: urbanev or long_format.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Dataset feature cache directory. Defaults to <data-dir>/cache.",
    )
    parser.add_argument(
        "--timeseries-file",
        default=None,
        help="Long-format time-series CSV relative to data-dir.",
    )
    parser.add_argument(
        "--output-folder",
        "--output-dir",
        dest="output_folder",
        default=None,
        help="Root folder for generated outputs. Defaults to run.output_folder or output.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Name of this experiment under <output-folder>. If omitted, "
            "the existing automatic matrix name is used."
        ),
    )
    parser.add_argument("--weather-file", default=None, help="Weather CSV file under data-dir.")
    parser.add_argument(
        "--forecast-model",
        default=None,
        help="Forecast model: timesfm, chronos, lstm, or AR.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the model for the active agent profile only, for example "
            "openai/gpt-4o-mini."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip LLM calls and use deterministic heuristics.",
    )
    parser.add_argument(
        "--force-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rebuild cached zone profiles and POI assignment.",
    )
    parser.add_argument(
        "--pipeline-stage",
        "--stage",
        dest="pipeline_stage",
        default=None,
        help="Execution stage: full, forecaster, or agent.",
    )
    parser.add_argument(
        "--forecaster-output-dir",
        default=None,
        help=(
            "Forecaster output directory (or forecaster_manifest.json) consumed by an "
            "agent-only run. Defaults to <output-folder>/<forecast-model>/forecaster."
        ),
    )
    parser.add_argument(
        "--agent-output-dir",
        default=None,
        help="Optional agent-stage output directory. Defaults to a sibling of forecaster/.",
    )
    parser.add_argument(
        "--precomputed-window-data",
        default=None,
        help=(
            "CSV produced as window_load_price_cache.csv. Complete zone-window matches reuse "
            "cached energy-price and price-conditioned load predictions."
        ),
    )
    parser.add_argument("--max-poi-rows", type=int, default=None, help="Limit POI rows for quick experiments.")
    parser.add_argument("--forecast-start", default=None, help="ISO timestamp for the forecast window start.")
    parser.add_argument(
        "--forecast-starts",
        nargs="+",
        default=None,
        help="ISO timestamp(s) for an experiment matrix. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument("--horizon-days", type=int, default=None, help="Forecast horizon.")
    parser.add_argument("--history-days", type=int, default=None, help="History window used for the zone snippets.")
    parser.add_argument("--validation-days", type=int, default=None, help="Validation window used for bias calibration.")
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help=(
            "UrbanEV zone id(s) to validate directly. If omitted, the pipeline keeps the automatic five-zone category selection."
        ),
    )
    parser.add_argument(
        "--forecast-models",
        nargs="+",
        default=None,
        help="Forecast model(s) for an experiment matrix: timesfm, chronos, lstm, or AR.",
    )
    parser.add_argument(
        "--experiment-zone-count",
        type=int,
        default=None,
        help="Representative zone count for experiment matrices when --zones is omitted.",
    )
    parser.add_argument(
        "--experiment-seeds",
        nargs="+",
        default=None,
        help="Seed(s) for repeated experiment runs, mainly affecting LSTM training.",
    )
    parser.add_argument(
        "--agent-mode",
        default=None,
        help=(
            "Agent mode: multi_agent_economist_retry, multi_agent_full_retry, "
            "multi_agent_discussion_3rounds, single_agent_price_retry, or "
            "single_agent_full_retry."
        ),
    )
    parser.add_argument(
        "--agent-modes",
        nargs="+",
        default=None,
        help="One or more supported control-loop agent modes.",
    )
    parser.add_argument(
        "--diurnal-blend-alpha",
        type=float,
        default=None,
        help="Uniform diurnal blend weight for all forecasters.",
    )
    parser.add_argument(
        "--diurnal-blend-alphas",
        nargs="+",
        default=None,
        help="Uniform diurnal blend weights to sweep in an experiment matrix.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="LLM sampling temperature.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    app_config = AppConfig.from_file(config_path, required=False, load_agent=False)
    run_config = app_config.run
    data_config = app_config.data
    pipeline_stage = normalize_pipeline_stage(
        args.pipeline_stage if args.pipeline_stage is not None else run_config.pipeline_stage
    )
    forecaster_output_dir = (
        args.forecaster_output_dir
        if args.forecaster_output_dir is not None
        else run_config.forecaster_output_dir
    )
    agent_output_dir = (
        args.agent_output_dir
        if args.agent_output_dir is not None
        else run_config.agent_output_dir
    )
    experiment_name = (
        args.experiment_name
        if args.experiment_name is not None
        else run_config.experiment_name
    )

    cli_starts = normalize_string_list(args.forecast_starts)
    if cli_starts:
        forecast_starts = cli_starts
    elif args.forecast_start:
        forecast_starts = [args.forecast_start]
    elif run_config.forecast_starts:
        forecast_starts = run_config.forecast_starts
    else:
        forecast_starts = [run_config.forecast_start] if run_config.forecast_start else []

    cli_models = normalize_forecast_model_list(args.forecast_models)
    if cli_models:
        forecast_models = cli_models
    elif args.forecast_model:
        forecast_models = normalize_forecast_model_list([args.forecast_model]) or [args.forecast_model]
    elif run_config.forecast_models:
        forecast_models = run_config.forecast_models
    else:
        forecast_models = [run_config.forecast_model]

    cli_seeds = normalize_int_list(args.experiment_seeds)
    experiment_seeds = cli_seeds if cli_seeds else run_config.experiment_seeds

    cli_modes = normalize_agent_mode_list(args.agent_modes)
    if cli_modes:
        agent_modes = cli_modes
    elif args.agent_mode:
        agent_modes = [normalize_agent_mode(args.agent_mode)]
    else:
        agent_modes = run_config.agent_modes

    cli_alphas = normalize_float_list(args.diurnal_blend_alphas)
    if cli_alphas:
        diurnal_blend_alphas = cli_alphas
    elif args.diurnal_blend_alpha is not None:
        diurnal_blend_alphas = [float(args.diurnal_blend_alpha)]
    else:
        diurnal_blend_alphas = run_config.diurnal_blend_alphas
    run_matrix = pipeline_stage != "agent" or not forecaster_output_dir
    run_matrix = run_matrix and (
        len(forecast_starts) > 1
        or len(forecast_models) > 1
        or bool(experiment_seeds and len(experiment_seeds) > 1)
        or bool(agent_modes and len(agent_modes) > 1)
        or bool(diurnal_blend_alphas and len(diurnal_blend_alphas) > 1)
        or bool(experiment_name)
    )

    if args.agent_mode:
        agent_mode = normalize_agent_mode(args.agent_mode)
    elif agent_modes and len(agent_modes) == 1:
        agent_mode = agent_modes[0]
    else:
        agent_mode = run_config.agent_mode
    diurnal_blend_alpha = (
        args.diurnal_blend_alpha
        if args.diurnal_blend_alpha is not None
        else diurnal_blend_alphas[0]
        if diurnal_blend_alphas and len(diurnal_blend_alphas) == 1
        else run_config.diurnal_blend_alpha
    )

    resolved_data_dir = Path(args.data_dir or run_config.data_dir)
    dataset_spec = DatasetSpec(
        path=resolved_data_dir,
        adapter=args.dataset_adapter or data_config.adapter,
        weather_file=args.weather_file or run_config.weather_file,
        cache_dir=Path(args.cache_dir or data_config.cache_dir)
        if (args.cache_dir or data_config.cache_dir)
        else None,
        timeseries_file=args.timeseries_file or data_config.timeseries_file,
        column_mapping=data_config.column_mapping,
        static_file=data_config.static_file,
        static_mapping=data_config.static_mapping,
        unit_conversions=data_config.unit_conversions,
    )
    common_kwargs = {
        "data_dir": resolved_data_dir,
        "dataset_spec": dataset_spec,
        "cache_dir": dataset_spec.resolved_cache_dir,
        "output_dir": Path(args.output_folder or run_config.output_folder),
        "config_path": config_path,
        "model": args.model,
        "weather_file": args.weather_file or run_config.weather_file,
        "dry_run": args.dry_run if args.dry_run is not None else run_config.dry_run,
        "force_cache": args.force_cache if args.force_cache is not None else run_config.force_cache,
        "pipeline_stage": pipeline_stage,
        "forecaster_output_dir": forecaster_output_dir,
        "agent_output_dir": agent_output_dir,
        "precomputed_window_data": (
            args.precomputed_window_data
            if args.precomputed_window_data is not None
            else run_config.precomputed_window_data
        ),
        "max_poi_rows": args.max_poi_rows if args.max_poi_rows is not None else run_config.max_poi_rows,
        "horizon_days": args.horizon_days if args.horizon_days is not None else run_config.horizon_days,
        "history_days": args.history_days if args.history_days is not None else run_config.history_days,
        "validation_days": args.validation_days if args.validation_days is not None else run_config.validation_days,
        "zone_ids": args.zones if args.zones is not None else run_config.zone_ids,
        "experiment_zone_count": (
            args.experiment_zone_count
            if args.experiment_zone_count is not None
            else run_config.experiment_zone_count
        ),
        "agent_mode": agent_mode,
        "timesfm_repo": run_config.timesfm_repo,
        "timesfm_context_hours": run_config.timesfm_context_hours,
        "timesfm_step_horizon": run_config.timesfm_step_horizon,
        "timesfm_exog_cols": run_config.timesfm_exog_cols,
        "timesfm_diurnal_blend_alpha": run_config.timesfm_diurnal_blend_alpha,
        "timesfm_roll_actuals": run_config.timesfm_roll_actuals,
        "ar_diurnal_blend_alpha": run_config.ar_diurnal_blend_alpha,
        "chronos_repo": run_config.chronos_repo,
        "chronos_context_hours": run_config.chronos_context_hours,
        "chronos_step_horizon": run_config.chronos_step_horizon,
        "chronos_diurnal_blend_alpha": run_config.chronos_diurnal_blend_alpha,
        "chronos_device": run_config.chronos_device,
        "chronos_roll_actuals": run_config.chronos_roll_actuals,
        "lstm_context_hours": run_config.lstm_context_hours,
        "lstm_step_horizon": run_config.lstm_step_horizon,
        "lstm_exog_cols": run_config.lstm_exog_cols,
        "lstm_hidden_size": run_config.lstm_hidden_size,
        "lstm_num_layers": run_config.lstm_num_layers,
        "lstm_epochs": run_config.lstm_epochs,
        "lstm_learning_rate": run_config.lstm_learning_rate,
        "lstm_batch_size": run_config.lstm_batch_size,
        "lstm_diurnal_blend_alpha": run_config.lstm_diurnal_blend_alpha,
        "lstm_device": run_config.lstm_device,
        "lstm_roll_actuals": run_config.lstm_roll_actuals,
        "lstm_seed": run_config.lstm_seed,
        "temperature": args.temperature if args.temperature is not None else run_config.temperature,
    }
    if diurnal_blend_alpha is not None:
        common_kwargs["timesfm_diurnal_blend_alpha"] = float(diurnal_blend_alpha)
        common_kwargs["ar_diurnal_blend_alpha"] = float(diurnal_blend_alpha)
        common_kwargs["chronos_diurnal_blend_alpha"] = float(diurnal_blend_alpha)
        common_kwargs["lstm_diurnal_blend_alpha"] = float(diurnal_blend_alpha)

    if run_matrix:
        if not forecast_starts:
            raise ValueError("Experiment matrix requires at least one forecast start.")
        outputs = run_experiment_matrix(
            experiment_name=experiment_name,
            forecast_starts=forecast_starts,
            forecast_models=forecast_models,
            experiment_seeds=experiment_seeds,
            agent_modes=agent_modes,
            diurnal_blend_alphas=diurnal_blend_alphas,
            **common_kwargs,
        )
    else:
        outputs = run_pipeline(
            forecast_start=forecast_starts[0] if forecast_starts else None,
            forecast_model=forecast_models[0],
            **common_kwargs,
        )
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


def cli(argv: list[str] | None = None) -> int:
    try:
        main(argv)
    except Exception as exc:
        print(format_failure_message(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
