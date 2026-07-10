from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PRICE_ERROR_THRESHOLD_RATIO = 0.08
PRICE_ERROR_THRESHOLD_PCT = PRICE_ERROR_THRESHOLD_RATIO * 100
ECONOMIST_AGENT_OUTPUT_KEY = "_economist_agent_output"
EXPLAINABILITY_RUBRIC_TEXT = """# Explainability Evaluation Rubric

Score each criterion from 1 to 5.

1. Forecast grounding: the rationale cites forecast load, stress, price, weather, occupancy, or temporal evidence present in the packet.
2. Temporal specificity: the rationale names the relevant window or time-of-day pattern instead of only giving a generic statement.
3. Decision consistency: the recommended price action is consistent with predicted stress, demand level, and service/energy price context.
4. Actionability: an operator could turn the explanation into a concrete pricing or grid-management decision.
5. No leakage or hallucination: the rationale avoids actual future load, forecast error, evaluation labels, and unsupported external facts.

Recommended protocol: use at least two independent raters, then add an operator sanity-check note for rationales that are technically inconsistent, unsafe, or operationally implausible.
"""


TRACE_COLUMNS = [
    "category",
    "zone_id",
    "predicted_load_kwh",
    "predicted_change_pct",
    "actual_load_kwh",
    "mae_kwh",
    "rmse_kwh",
    "mape_pct",
    "rae",
    "wape_pct",
    "grid_stress_level",
    "actual_grid_stress_level",
    "stress_accuracy",
    "miss_stress_rate",
    "stress_eval_windows",
    "stress_miss_count",
    "agent_reasoning",
    "suggested_price_shift_pct",
    "action_label",
    "price_rationale",
    "nash_equilibrium_reached",
    "nash_equilibrium_windows",
    "nash_equilibrium_reached_windows",
    "nash_equilibrium_rounds",
    "nash_equilibrium_summary",
    "agent_time_cost_seconds",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_total_tokens",
    "source",
]
TRACE_OUTPUT_RENAMES = {
    "predicted_load_kwh": "forecast_total_kwh",
    "actual_load_kwh": "actual_total_kwh",
    "suggested_price_shift_pct": "final_price_shift_pct",
}

WINDOW_LOAD_PRICE_CACHE_COLUMNS = [
    "zone_id",
    "category",
    "forecast_model",
    "agent_mode",
    "source",
    "window_start",
    "window_end",
    "forecast_load_kwh",
    "price_conditioned_load_kwh",
    "price_conditioned_mean_load_kwh",
    "price_conditioned_peak_load_kwh",
    "predicted_service_price",
    "forecaster_input_predicted_service_price",
    "actual_service_price",
    "actual_energy_price",
    "pre_nash_price_shift_pct",
    "final_price_shift_pct",
    "load_stress_level",
    "price_conditioned_load_stress_level",
    "load_3h_q95_kwh",
    "nash_equilibrium_reached",
    "nash_status",
    "nash_iterations",
    "baseline_load_kwh",
    "baseline_load_source",
    "target_peak_reduction_pct",
    "elasticity_factor",
    "capacity_limit_kwh",
    "capacity_limit_source",
    "expected_load_kwh",
    "grid_safe",
    "user_tolerant",
    "price_stable",
    "discomfort_score",
    "max_discomfort_score",
    "price_stability_epsilon_pct",
    "action_label",
    "price_rationale",
]


def write_outputs(
    *,
    output_dir: Path,
    selected_zones: pd.DataFrame,
    contexts: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    forecast_results: dict[str, Any],
    forecast_model: str | None = None,
    agent_mode: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_zones.csv"
    contexts_path = output_dir / "context_snippets.json"
    trace_csv = output_dir / "rationale_trace.csv"
    trace_md = output_dir / "rationale_trace.md"
    trace_json = output_dir / "rationale_trace.json"
    agent_debug_outputs_json = output_dir / "agent_debug_outputs.json"
    metrics_csv = output_dir / "forecast_metrics.csv"
    metrics_md = output_dir / "forecast_metrics.md"
    price_schedule_csv = output_dir / "price_schedule_3h.csv"
    price_schedule_md = output_dir / "price_schedule_3h.md"
    price_comparison_csv = output_dir / "price_comparison_summary.csv"
    price_comparison_md = output_dir / "price_comparison_summary.md"
    window_load_price_cache_csv = output_dir / "window_load_price_cache.csv"
    explainability_rubric_md = output_dir / "explainability_rubric.md"
    explainability_review_packet_csv = output_dir / "explainability_review_packet.csv"
    details_dir = output_dir / "forecast_details"

    selected_zones.to_csv(selected_path, index=False)
    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_reports, agent_debug_outputs = split_agent_debug_outputs(reports)
    agent_debug_outputs_json.write_text(json.dumps(agent_debug_outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    trace = pd.DataFrame(trace_reports)
    trace = trace[[col for col in TRACE_COLUMNS if col in trace.columns]]
    trace = trace.rename(columns=TRACE_OUTPUT_RENAMES)
    trace.to_csv(trace_csv, index=False)
    trace_json.write_text(json.dumps(trace_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_md.write_text(markdown_table(trace), encoding="utf-8")
    write_price_schedule_outputs(price_schedule_csv, price_schedule_md, price_comparison_csv, price_comparison_md, trace_reports)
    write_window_load_price_cache(
        window_load_price_cache_csv,
        trace_reports,
        forecast_model=forecast_model,
        agent_mode=agent_mode,
    )
    write_explainability_review_outputs(
        explainability_rubric_md,
        explainability_review_packet_csv,
        trace_reports,
    )
    metrics = write_forecast_outputs(details_dir, metrics_csv, metrics_md, forecast_results)

    outputs = {
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "rationale_trace_csv": trace_csv,
        "rationale_trace_md": trace_md,
        "rationale_trace_json": trace_json,
        "agent_debug_outputs_json": agent_debug_outputs_json,
        "forecast_metrics_csv": metrics_csv,
        "forecast_metrics_md": metrics_md,
        "price_schedule_3h_csv": price_schedule_csv,
        "price_schedule_3h_md": price_schedule_md,
        "price_comparison_summary_csv": price_comparison_csv,
        "price_comparison_summary_md": price_comparison_md,
        "window_load_price_cache_csv": window_load_price_cache_csv,
        "explainability_rubric_md": explainability_rubric_md,
        "explainability_review_packet_csv": explainability_review_packet_csv,
        "forecast_details_dir": details_dir,
    }
    outputs.update(metrics)
    return outputs


def split_agent_debug_outputs(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_reports: list[dict[str, Any]] = []
    agent_debug_outputs: list[dict[str, Any]] = []
    for report in reports:
        clean_report = {key: value for key, value in report.items() if key != ECONOMIST_AGENT_OUTPUT_KEY}
        trace_reports.append(clean_report)
        debug = report.get(ECONOMIST_AGENT_OUTPUT_KEY)
        if isinstance(debug, dict):
            agent_debug_outputs.append(debug)
    return trace_reports, agent_debug_outputs


def split_economist_agent_outputs(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return split_agent_debug_outputs(reports)


def write_forecast_outputs(
    details_dir: Path,
    metrics_csv: Path,
    metrics_md: Path,
    forecast_results: dict[str, Any],
) -> dict[str, Path]:
    details_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    plot_paths = {}

    for zone_id, result in forecast_results.items():
        safe_zone = safe_filename(zone_id)
        hourly = result.hourly.copy()
        hourly.insert(0, "zone_id", zone_id)
        hourly.insert(1, "category", result.summary.get("category"))
        hourly_path = details_dir / f"zone_{safe_zone}_forecast_vs_actual.csv"
        plot_path = details_dir / f"zone_{safe_zone}_forecast_plot.png"
        old_svg_path = details_dir / f"zone_{safe_zone}_forecast_plot.svg"
        hourly.to_csv(hourly_path, index=False)
        if old_svg_path.exists():
            old_svg_path.unlink()
        write_zone_plot(plot_path, result.summary, hourly)
        plot_paths[f"zone_{safe_zone}_forecast_csv"] = hourly_path
        plot_paths[f"zone_{safe_zone}_forecast_plot"] = plot_path

        metrics = result.summary.get("metrics", {}) or {}
        metric_rows.append(
            {
                "zone_id": zone_id,
                "category": result.summary.get("category"),
                "forecast_model": result.summary.get("forecast_model"),
                "diurnal_blend_alpha": result.summary.get("diurnal_blend_alpha"),
                "calibration_enabled": (result.summary.get("calibration") or {}).get("enabled"),
                "bias_mean": (result.summary.get("calibration") or {}).get("bias_mean"),
                "bias_max_abs": (result.summary.get("calibration") or {}).get("bias_max_abs"),
                "forecast_start": result.summary.get("forecast_start"),
                "forecast_end": result.summary.get("forecast_end"),
                "n": metrics.get("n"),
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "MAPE_pct": metrics.get("MAPE_pct"),
                "RAE": metrics.get("RAE"),
                "WAPE_pct": metrics.get("WAPE_pct"),
                "forecast_total_kwh": result.summary.get("forecast_total_kwh"),
                "actual_total_kwh": result.summary.get("actual_total_kwh"),
                "forecast_peak_kwh": result.summary.get("forecast_peak_kwh"),
                "actual_peak_kwh": result.summary.get("actual_peak_kwh"),
                "grid_stress_level": result.summary.get("grid_stress_level"),
                "actual_grid_stress_level": result.summary.get("actual_grid_stress_level"),
                "stress_accuracy": result.summary.get("stress_accuracy"),
                "miss_stress_rate": result.summary.get("miss_stress_rate"),
                "stress_eval_windows": result.summary.get("stress_eval_windows"),
                "stress_miss_count": result.summary.get("stress_miss_count"),
                "grid_stress_basis": result.summary.get("grid_stress_basis"),
                "grid_stress_load_kwh": result.summary.get("grid_stress_load_kwh"),
                "actual_grid_stress_load_kwh": result.summary.get("actual_grid_stress_load_kwh"),
                "grid_stress_source_file": result.summary.get("grid_stress_source_file"),
                "grid_stress_window_hours": result.summary.get("grid_stress_window_hours"),
                "grid_stress_historical_windows": result.summary.get("grid_stress_historical_windows"),
                "grid_stress_q50_kwh": result.summary.get("grid_stress_q50_kwh"),
                "grid_stress_q80_kwh": result.summary.get("grid_stress_q80_kwh"),
                "grid_stress_q95_kwh": result.summary.get("grid_stress_q95_kwh"),
                "lstm_seed": result.summary.get("lstm_seed"),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(metrics_csv, index=False)
    metrics_md.write_text(markdown_table(metrics_frame), encoding="utf-8")
    return plot_paths


def write_price_schedule_outputs(
    price_schedule_csv: Path,
    price_schedule_md: Path,
    price_comparison_csv: Path,
    price_comparison_md: Path,
    reports: list[dict[str, Any]],
) -> None:
    rows = []
    for report in reports:
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            row = {
                "zone_id": report.get("zone_id"),
                "category": report.get("category"),
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                "sum_actual_kwh": window.get("sum_actual_kwh"),
                "actual_service_price": window.get("mean_service_price"),
                "actual_energy_price": window.get("mean_energy_price"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "actual_load_stress_level": window.get("actual_load_stress_level") or window.get("actual_grid_stress_level"),
                "stress_correct": window.get("stress_correct"),
                "stress_missed": window.get("stress_missed"),
                "load_3h_q50_kwh": window.get("load_3h_q50_kwh"),
                "load_3h_q80_kwh": window.get("load_3h_q80_kwh"),
                "load_3h_q95_kwh": window.get("load_3h_q95_kwh"),
                "model_price_shift_pct": window.get("pre_nash_suggested_price_shift_pct"),
                "forecaster_input_predicted_service_price": window.get("price_conditioned_service_price"),
                "price_conditioned_baseline_load_kwh": window.get("price_conditioned_baseline_load_kwh"),
                "price_conditioned_mean_predicted_kwh": window.get("price_conditioned_mean_predicted_kwh"),
                "price_conditioned_peak_predicted_kwh": window.get("price_conditioned_peak_predicted_kwh"),
                "price_conditioned_load_stress_level": window.get("price_conditioned_load_stress_level"),
                "price_conditioned_baseline_source": window.get("price_conditioned_baseline_source"),
                "final_price_shift_pct": window.get("suggested_price_shift_pct"),
                "nash_equilibrium_reached": window.get("nash_equilibrium_reached"),
                "nash_status": window.get("nash_status"),
                "nash_iterations": window.get("nash_iterations"),
                "baseline_load_kwh": window.get("baseline_load_kwh"),
                "baseline_load_source": window.get("baseline_load_source"),
                "target_peak_reduction_pct": window.get("target_peak_reduction_pct"),
                "elasticity_factor": window.get("elasticity_factor"),
                "capacity_limit_kwh": window.get("capacity_limit_kwh"),
                "capacity_limit_source": window.get("capacity_limit_source"),
                "expected_load_kwh": window.get("expected_load_kwh"),
                "grid_safe": window.get("grid_safe"),
                "user_tolerant": window.get("user_tolerant"),
                "price_stable": window.get("price_stable"),
                "discomfort_score": window.get("discomfort_score"),
                "max_discomfort_score": window.get("max_discomfort_score"),
                "price_stability_epsilon_pct": window.get("price_stability_epsilon_pct"),
                "action_label": window.get("action_label"),
                "price_rationale": window.get("price_rationale"),
                "source": report.get("source"),
            }
            row.update(
                price_comparison_fields(
                    row["actual_service_price"],
                    row["final_price_shift_pct"],
                    window.get("predicted_service_price"),
                )
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(price_schedule_csv, index=False)
    price_schedule_md.write_text(markdown_table(frame), encoding="utf-8")
    summary = build_price_comparison_summary(frame)
    summary.to_csv(price_comparison_csv, index=False)
    price_comparison_md.write_text(markdown_table(summary), encoding="utf-8")


def write_window_load_price_cache(
    cache_csv: Path,
    reports: list[dict[str, Any]],
    *,
    forecast_model: str | None = None,
    agent_mode: str | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            actual_service_price = window.get("mean_service_price")
            comparison = price_comparison_fields(
                actual_service_price,
                window.get("suggested_price_shift_pct"),
                window.get("predicted_service_price"),
            )
            rows.append(
                {
                    "zone_id": report.get("zone_id"),
                    "category": report.get("category"),
                    "forecast_model": forecast_model,
                    "agent_mode": agent_mode,
                    "source": report.get("source"),
                    "window_start": window.get("window_start"),
                    "window_end": window.get("window_end"),
                    "forecast_load_kwh": window.get("sum_predicted_kwh"),
                    "price_conditioned_load_kwh": window.get("price_conditioned_baseline_load_kwh"),
                    "price_conditioned_mean_load_kwh": window.get("price_conditioned_mean_predicted_kwh"),
                    "price_conditioned_peak_load_kwh": window.get("price_conditioned_peak_predicted_kwh"),
                    "predicted_service_price": comparison["predicted_service_price"],
                    "forecaster_input_predicted_service_price": window.get("price_conditioned_service_price"),
                    "actual_service_price": actual_service_price,
                    "actual_energy_price": window.get("mean_energy_price"),
                    "pre_nash_price_shift_pct": window.get("pre_nash_suggested_price_shift_pct"),
                    "final_price_shift_pct": window.get("suggested_price_shift_pct"),
                    "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                    "price_conditioned_load_stress_level": window.get("price_conditioned_load_stress_level"),
                    "load_3h_q95_kwh": window.get("load_3h_q95_kwh"),
                    "nash_equilibrium_reached": window.get("nash_equilibrium_reached"),
                    "nash_status": window.get("nash_status"),
                    "nash_iterations": window.get("nash_iterations"),
                    "baseline_load_kwh": window.get("baseline_load_kwh"),
                    "baseline_load_source": window.get("baseline_load_source"),
                    "target_peak_reduction_pct": window.get("target_peak_reduction_pct"),
                    "elasticity_factor": window.get("elasticity_factor"),
                    "capacity_limit_kwh": window.get("capacity_limit_kwh"),
                    "capacity_limit_source": window.get("capacity_limit_source"),
                    "expected_load_kwh": window.get("expected_load_kwh"),
                    "grid_safe": window.get("grid_safe"),
                    "user_tolerant": window.get("user_tolerant"),
                    "price_stable": window.get("price_stable"),
                    "discomfort_score": window.get("discomfort_score"),
                    "max_discomfort_score": window.get("max_discomfort_score"),
                    "price_stability_epsilon_pct": window.get("price_stability_epsilon_pct"),
                    "action_label": window.get("action_label"),
                    "price_rationale": window.get("price_rationale"),
                }
            )
    pd.DataFrame(rows, columns=WINDOW_LOAD_PRICE_CACHE_COLUMNS).to_csv(cache_csv, index=False)


def write_explainability_review_outputs(
    rubric_md: Path,
    review_packet_csv: Path,
    reports: list[dict[str, Any]],
) -> None:
    rubric_md.write_text(EXPLAINABILITY_RUBRIC_TEXT, encoding="utf-8")
    rows = []
    for report in reports:
        rows.append(explainability_review_row(report, None))
        for window in report.get("price_change_windows_3h") or []:
            if isinstance(window, dict):
                rows.append(explainability_review_row(report, window))
    pd.DataFrame(rows).to_csv(review_packet_csv, index=False)


def explainability_review_row(report: dict[str, Any], window: dict[str, Any] | None) -> dict[str, Any]:
    is_window = window is not None
    item = window or {}
    return {
        "record_type": "price_window_3h" if is_window else "zone_summary",
        "zone_id": report.get("zone_id"),
        "category": report.get("category"),
        "source": report.get("source"),
        "window_start": item.get("window_start"),
        "window_end": item.get("window_end"),
        "forecast_load_kwh": item.get("sum_predicted_kwh") if is_window else report.get("predicted_load_kwh"),
        "observed_load_kwh": item.get("sum_actual_kwh") if is_window else report.get("actual_load_kwh"),
        "predicted_stress_level": item.get("load_stress_level") or report.get("grid_stress_level"),
        "actual_stress_level": item.get("actual_load_stress_level") or report.get("actual_grid_stress_level"),
        "final_price_shift_pct": item.get("suggested_price_shift_pct")
        if is_window
        else report.get("suggested_price_shift_pct"),
        "action_label": item.get("action_label") if is_window else report.get("action_label"),
        "rationale": item.get("price_rationale") if is_window else report.get("price_rationale"),
        "behavior_reasoning": report.get("agent_reasoning"),
        "rater_1_forecast_grounding_1_5": "",
        "rater_1_temporal_specificity_1_5": "",
        "rater_1_decision_consistency_1_5": "",
        "rater_1_actionability_1_5": "",
        "rater_1_no_leakage_1_5": "",
        "rater_1_total": "",
        "rater_2_forecast_grounding_1_5": "",
        "rater_2_temporal_specificity_1_5": "",
        "rater_2_decision_consistency_1_5": "",
        "rater_2_actionability_1_5": "",
        "rater_2_no_leakage_1_5": "",
        "rater_2_total": "",
        "operator_sanity_check": "",
        "operator_notes": "",
    }


def price_comparison_fields(
    actual_price: Any,
    shift_pct: Any,
    predicted_price: Any = None,
) -> dict[str, Any]:
    actual = optional_float(actual_price)
    shift = optional_float(shift_pct)
    predicted = optional_float(predicted_price)
    if predicted is None and actual is not None and shift is not None:
        predicted = actual * (1 + shift / 100)
    if actual is None or predicted is None:
        return {
            "predicted_service_price": None,
            "predicted_minus_actual_service_price": None,
            "predicted_vs_actual_pct": None,
        }
    diff = predicted - actual
    return {
        "predicted_service_price": round(predicted, 4),
        "predicted_minus_actual_service_price": round(diff, 4),
        "predicted_vs_actual_pct": round((diff / actual) * 100, 4) if actual != 0 else None,
    }


def build_price_comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zone_id",
        "category",
        "price_windows",
        "price_error_threshold_pct",
        "price_pass_windows",
        "price_accuracy",
        "avg_actual_service_price",
        "avg_predicted_service_price",
        "avg_predicted_minus_actual_service_price",
        "avg_predicted_vs_actual_pct",
    ]
    if frame.empty or "predicted_service_price" not in frame:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        "actual_service_price",
        "predicted_service_price",
        "predicted_minus_actual_service_price",
        "predicted_vs_actual_pct",
    ]
    working = frame.copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["actual_service_price", "predicted_service_price"])
    if working.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (zone_id, category), group in working.groupby(["zone_id", "category"], dropna=False):
        rows.append(price_summary_row(zone_id, category, group))
    return pd.DataFrame(rows, columns=columns)


def price_summary_row(zone_id: Any, category: Any, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "category": category,
        "price_windows": int(len(group)),
        "price_error_threshold_pct": PRICE_ERROR_THRESHOLD_PCT,
        "price_pass_windows": int(price_pass_mask(group).sum()),
        "price_accuracy": round(float(price_pass_mask(group).mean()), 4),
        "avg_actual_service_price": round(float(group["actual_service_price"].mean()), 4),
        "avg_predicted_service_price": round(float(group["predicted_service_price"].mean()), 4),
        "avg_predicted_minus_actual_service_price": round(
            float(group["predicted_minus_actual_service_price"].mean()),
            4,
        ),
        "avg_predicted_vs_actual_pct": round(float(group["predicted_vs_actual_pct"].mean()), 4),
    }


def price_pass_mask(group: pd.DataFrame) -> pd.Series:
    threshold = group["actual_service_price"].abs() * PRICE_ERROR_THRESHOLD_RATIO
    return group["predicted_minus_actual_service_price"].abs() <= threshold


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def write_zone_plot(path: Path, summary: dict[str, Any], hourly: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for forecast plots. Install it with: pip install matplotlib") from exc

    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame["hour_index"] = range(len(frame))

    fig = plt.figure(figsize=(16, 9), dpi=140)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[3.2, 1.0], hspace=0.46, wspace=0.22)
    ax_main = fig.add_subplot(gs[0, :])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    ax_main.plot(
        frame["time"],
        frame["actual_kwh"],
        color="#185FA5",
        lw=2.0,
        marker="o",
        ms=3.2,
        label="Actual",
    )
    if {"q10_kwh", "q90_kwh"}.issubset(frame.columns) and frame[["q10_kwh", "q90_kwh"]].notna().any().any():
        ax_main.fill_between(
            frame["time"],
            frame["q10_kwh"],
            frame["q90_kwh"],
            color="#D85A30",
            alpha=0.14,
            label="P10-P90",
        )
    ax_main.plot(
        frame["time"],
        frame["predicted_kwh"],
        color="#D85A30",
        lw=2.0,
        ls="--",
        marker="s",
        ms=3.0,
        label="Predicted",
    )
    ax_main.set_title(f"Zone {summary.get('zone_id')} Forecast vs Actual", fontsize=15, fontweight="bold")
    ax_main.set_xlabel("Time")
    ax_main.set_ylabel("Load (kWh)")
    ax_main.grid(axis="y", alpha=0.25)
    ax_main.legend(loc="upper left", frameon=False)

    errors = frame["error_kwh"].astype(float)
    colors = ["#0F6E56" if value >= 0 else "#D85A30" for value in errors]
    ax_resid.bar(frame["time"], errors, color=colors, width=0.03, alpha=0.82)
    ax_resid.axhline(0, color="#6B7280", lw=0.9)
    ax_resid.set_title("Residuals (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax_resid.set_xlabel("Time")
    ax_resid.set_ylabel("Error (kWh)")
    ax_resid.grid(axis="y", alpha=0.22)

    metrics = summary.get("metrics", {}) or {}
    metric_lines = [
        ("MAE", format_metric(metrics.get("MAE"))),
        ("RMSE", format_metric(metrics.get("RMSE"))),
        ("MAPE", f"{format_metric(metrics.get('MAPE_pct'))}%"),
        ("RAE", format_metric(metrics.get("RAE"))),
        ("WAPE", f"{format_metric(metrics.get('WAPE_pct'))}%"),
    ]
    ax_metrics.axis("off")
    ax_metrics.set_title("Evaluation", fontsize=11, fontweight="bold", loc="left")
    for idx, (label, value) in enumerate(metric_lines):
        y = 0.88 - idx * 0.16
        ax_metrics.text(0.02, y, label, fontsize=11, color="#374151", transform=ax_metrics.transAxes)
        ax_metrics.text(0.98, y, value, fontsize=11, fontweight="bold", ha="right", transform=ax_metrics.transAxes)

    for axis in (ax_main, ax_resid):
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.DateFormatter("%m-%d %H:%M")
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
        axis.tick_params(axis="x", labelbottom=True)
        axis.tick_params(axis="x", rotation=30)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for col in display.columns:
        formatted_values = []
        for value in display[col]:
            if pd.isna(value):
                formatted_values.append("")
                continue
            text = str(value).replace("\n", " ")
            formatted_values.append(text[:220] + "..." if len(text) > 223 else text)
        display[col] = formatted_values
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in display[col]))
        for col in display.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in display.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"
