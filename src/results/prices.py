from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.usage import CACHE_USAGE_FIELDS
from src.results.formatting import markdown_table
from src.results.schema import MODEL_PROVENANCE_FIELDS


PRICE_ERROR_THRESHOLD_RATIO = 0.08
PRICE_ERROR_THRESHOLD_PCT = PRICE_ERROR_THRESHOLD_RATIO * 100

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
    *MODEL_PROVENANCE_FIELDS,
    "predicted_load_kwh",
    "predicted_change_pct",
    "grid_stress_level",
    "agent_reasoning",
    "suggested_price_shift_pct",
    "action_label",
    "price_rationale",
    "control_status",
    "attempts_used",
    "failed_window_count",
    "agent_invoked",
    "agent_call_count",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_total_tokens",
    "agent_token_usage_complete",
    *[f"agent_{field}" for field in CACHE_USAGE_FIELDS],
    "source",
]

TRACE_OUTPUT_RENAMES = {
    "predicted_load_kwh": "forecast_total_kwh",
    "suggested_price_shift_pct": "final_price_shift_pct",
}


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
                "known_service_price": window.get("mean_service_price"),
                "baseline_energy_price": window.get("mean_energy_price"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "load_range_position_pct": window.get("load_range_position_pct"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "historical_min_load_3h_kwh": window.get("historical_min_load_3h_kwh"),
                "historical_max_load_3h_kwh": window.get("historical_max_load_3h_kwh"),
                "historical_load_range_3h_kwh": window.get(
                    "historical_load_range_3h_kwh"
                ),
                "extremely_low_low_threshold_pct": window.get("extremely_low_low_threshold_pct"),
                "low_medium_threshold_pct": window.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": window.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": window.get(
                    "high_extremely_high_threshold_pct"
                ),
                "load_3h_extremely_low_low_threshold_kwh": window.get(
                    "load_3h_extremely_low_low_threshold_kwh"
                ),
                "load_3h_low_medium_threshold_kwh": window.get(
                    "load_3h_low_medium_threshold_kwh"
                ),
                "load_3h_medium_high_threshold_kwh": window.get(
                    "load_3h_medium_high_threshold_kwh"
                ),
                "load_3h_high_extremely_high_threshold_kwh": window.get(
                    "load_3h_high_extremely_high_threshold_kwh"
                ),
                "zone_mean_energy_price": window.get("zone_mean_energy_price"),
                "model_price_shift_pct": window.get("suggested_price_shift_pct"),
                "forecaster_input_predicted_energy_price": window.get("price_conditioned_energy_price"),
                "price_conditioned_baseline_load_kwh": window.get("price_conditioned_baseline_load_kwh"),
                "price_conditioned_mean_predicted_kwh": window.get("price_conditioned_mean_predicted_kwh"),
                "price_conditioned_peak_predicted_kwh": window.get("price_conditioned_peak_predicted_kwh"),
                "price_conditioned_load_stress_level": window.get("price_conditioned_load_stress_level"),
                "price_conditioned_baseline_source": window.get("price_conditioned_baseline_source"),
                "final_price_shift_pct": window.get("suggested_price_shift_pct"),
                "baseline_load_kwh": window.get("baseline_load_kwh"),
                "baseline_load_range_position_pct": window.get(
                    "baseline_load_range_position_pct"
                ),
                "baseline_load_source": window.get("baseline_load_source"),
                "target_load_kwh": window.get("target_load_kwh"),
                "target_load_range_position_pct": window.get(
                    "target_load_range_position_pct"
                ),
                "medium_load_min_kwh": window.get("medium_load_min_kwh"),
                "medium_load_max_kwh": window.get("medium_load_max_kwh"),
                "target_peak_reduction_pct": window.get("target_peak_reduction_pct"),
                "elasticity_factor": window.get("elasticity_factor"),
                "load_range_source": window.get("load_range_source"),
                "expected_load_kwh": window.get("expected_load_kwh"),
                "expected_load_range_position_pct": window.get(
                    "expected_load_range_position_pct"
                ),
                "load_in_medium_band": window.get("load_in_medium_band"),
                "price_valid": window.get("price_valid"),
                "historical_price_change_percentile": window.get(
                    "historical_price_change_percentile"
                ),
                "historical_price_change_p95_pct": window.get(
                    "historical_price_change_p95_pct"
                ),
                "exceeds_historical_p95": window.get("exceeds_historical_p95"),
                "control_success": window.get("control_success"),
                "control_failure_reason": window.get("control_failure_reason"),
                "action_label": window.get("action_label"),
                "price_rationale": window.get("price_rationale"),
                "source": report.get("source"),
            }
            row.update(
                price_comparison_fields(
                    row["baseline_energy_price"],
                    row["final_price_shift_pct"],
                    window.get("predicted_energy_price"),
                )
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(price_schedule_csv, index=False)
    price_schedule_md.write_text(markdown_table(frame), encoding="utf-8")
    summary = build_price_comparison_summary(frame)
    summary.to_csv(price_comparison_csv, index=False)
    price_comparison_md.write_text(markdown_table(summary), encoding="utf-8")


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
        "predicted_stress_level": item.get("load_stress_level") or report.get("grid_stress_level"),
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
    baseline_price: Any,
    shift_pct: Any,
    predicted_price: Any = None,
) -> dict[str, Any]:
    baseline = optional_float(baseline_price)
    shift = optional_float(shift_pct)
    predicted = optional_float(predicted_price)
    if predicted is None and baseline is not None and shift is not None:
        predicted = baseline * (1 + shift / 100)
    if baseline is None or predicted is None:
        return {
            "predicted_energy_price": None,
            "predicted_minus_baseline_energy_price": None,
            "predicted_vs_baseline_pct": None,
        }
    diff = predicted - baseline
    return {
        "predicted_energy_price": round(predicted, 4),
        "predicted_minus_baseline_energy_price": round(diff, 4),
        "predicted_vs_baseline_pct": round((diff / baseline) * 100, 4)
        if baseline != 0
        else None,
    }


def build_price_comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zone_id",
        "category",
        "price_windows",
        "price_error_threshold_pct",
        "price_pass_windows",
        "price_accuracy",
        "avg_baseline_energy_price",
        "avg_predicted_energy_price",
        "avg_predicted_minus_baseline_energy_price",
        "avg_predicted_vs_baseline_pct",
    ]
    if frame.empty or "predicted_energy_price" not in frame:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        "baseline_energy_price",
        "predicted_energy_price",
        "predicted_minus_baseline_energy_price",
        "predicted_vs_baseline_pct",
    ]
    working = frame.copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["baseline_energy_price", "predicted_energy_price"])
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
        "avg_baseline_energy_price": round(float(group["baseline_energy_price"].mean()), 4),
        "avg_predicted_energy_price": round(float(group["predicted_energy_price"].mean()), 4),
        "avg_predicted_minus_baseline_energy_price": round(
            float(group["predicted_minus_baseline_energy_price"].mean()),
            4,
        ),
        "avg_predicted_vs_baseline_pct": round(float(group["predicted_vs_baseline_pct"].mean()), 4),
    }


def price_pass_mask(group: pd.DataFrame) -> pd.Series:
    threshold = group["baseline_energy_price"].abs() * PRICE_ERROR_THRESHOLD_RATIO
    return group["predicted_minus_baseline_energy_price"].abs() <= threshold


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
