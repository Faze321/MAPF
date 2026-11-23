from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast.artifact import NATIVE_ARTIFACT_SCHEMA_VERSION, NativeForecasterArtifact
from src.paths import safe_filename
from src.results.formatting import markdown_table
from src.results.schema import MODEL_PROVENANCE_FIELDS


def write_forecaster_outputs(
    *,
    output_dir: Path,
    selected_zones: pd.DataFrame,
    contexts: list[dict[str, Any]],
    forecast_results: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_zones.csv"
    contexts_path = output_dir / "context_snippets.json"
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = evaluation_dir / "forecast_metrics.csv"
    metrics_md = evaluation_dir / "forecast_metrics.md"
    details_dir = evaluation_dir / "forecast_details"
    manifest_path = output_dir / "forecaster_manifest.json"
    artifact_path = output_dir / "forecaster_artifact.json"

    selected_zones.to_csv(selected_path, index=False)
    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = write_forecast_outputs(details_dir, metrics_csv, metrics_md, forecast_results)
    artifact_payload = {
        "schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native_backend_reforecast",
        "artifacts": {
            str(zone_id): NativeForecasterArtifact.from_forecast_result(result).to_dict()
            for zone_id, result in forecast_results.items()
        },
    }
    artifact_path.write_text(
        json.dumps(artifact_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    outputs = {
        "forecaster_output_dir": output_dir,
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "forecast_metrics_csv": metrics_csv,
        "forecast_metrics_md": metrics_md,
        "forecast_details_dir": details_dir,
        "evaluation_dir": evaluation_dir,
        "forecaster_artifact_json": artifact_path,
    }
    outputs.update(metrics)
    if manifest is not None:
        provenance = {
            str(zone): {key: result.summary.get(key) for key in (*MODEL_PROVENANCE_FIELDS, "model_parameters")}
            for zone, result in forecast_results.items() if result.summary.get("model_id")
        }
        model_ids = {zone: values["model_id"] for zone, values in provenance.items()}
        manifest_payload = {
            **manifest,
            "model_ids": model_ids,
            "model_provenance": provenance,
            "forecast_parameters": {**(manifest.get("forecast_parameters") or {}), "model_ids": model_ids},
            "artifacts": {
                "selected_zones": selected_path.name,
                "context_snippets": contexts_path.name,
                "forecast_metrics_csv": str(metrics_csv.relative_to(output_dir)),
                "forecast_metrics_md": str(metrics_md.relative_to(output_dir)),
                "forecast_details_dir": str(details_dir.relative_to(output_dir)),
                "forecaster_artifact_json": artifact_path.name,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs["forecaster_manifest_json"] = manifest_path
    return outputs


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
        calibration = result.summary.get("calibration") or {}
        validation_bias = result.summary.get("validation_bias", calibration.get("enabled", False))
        validation_days = calibration["validation_days"] if validation_bias else 0
        hourly.insert(0, "zone_id", zone_id)
        hourly.insert(1, "category", result.summary.get("category"))
        hourly["validation_bias"] = validation_bias
        hourly["validation_days"] = validation_days
        for key in MODEL_PROVENANCE_FIELDS:
            if key in result.summary:
                hourly[key] = result.summary[key]
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
                **{key: result.summary.get(key) for key in MODEL_PROVENANCE_FIELDS},
                "category": result.summary.get("category"),
                "backend": result.summary.get("backend"),
                "diurnal_blend_alpha": result.summary.get("diurnal_blend_alpha"),
                "validation_bias": validation_bias,
                "validation_days": validation_days,
                **({"calibration_method": calibration["method"], "bias_mean": calibration["bias_mean"],
                    "bias_max_abs": calibration["bias_max_abs"]} if calibration.get("enabled") else {}),
                "forecast_origin": result.summary.get("forecast_origin"),
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
                "historical_min_load_3h_kwh": result.summary.get(
                    "historical_min_load_3h_kwh"
                ),
                "historical_max_load_3h_kwh": result.summary.get(
                    "historical_max_load_3h_kwh"
                ),
                "historical_load_range_3h_kwh": result.summary.get(
                    "historical_load_range_3h_kwh"
                ),
                "extremely_low_low_threshold_pct": result.summary.get("extremely_low_low_threshold_pct"),
                "low_medium_threshold_pct": result.summary.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": result.summary.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": result.summary.get(
                    "high_extremely_high_threshold_pct"
                ),
                "load_3h_extremely_low_low_threshold_kwh": result.summary.get(
                    "load_3h_extremely_low_low_threshold_kwh"
                ),
                "load_3h_low_medium_threshold_kwh": result.summary.get(
                    "load_3h_low_medium_threshold_kwh"
                ),
                "load_3h_medium_high_threshold_kwh": result.summary.get(
                    "load_3h_medium_high_threshold_kwh"
                ),
                "load_3h_high_extremely_high_threshold_kwh": result.summary.get(
                    "load_3h_high_extremely_high_threshold_kwh"
                ),
                "lstm_seed": result.summary.get("lstm_seed"),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(metrics_csv, index=False)
    metrics_md.write_text(markdown_table(metrics_frame), encoding="utf-8")
    return plot_paths


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
