from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.reports import ECONOMIST_AGENT_OUTPUT_KEY
from src.agents.usage import (
    CONTROL_AGENT_USAGE_KEY,
    cumulative_global_agent_usage as derive_global_cumulative_usage,
    token_total_summary as token_total_payload,
    usage_integer as reporting_usage_integer,
)
from src.control.policy import LOAD_POLICY_ID
from src.results.formatting import markdown_table
from src.results.price_range import PRICE_RANGE_FIELDS, PRICE_RANGE_TABLES, write_price_range_tables
from src.results.prices import (
    TRACE_COLUMNS,
    TRACE_OUTPUT_RENAMES,
    write_explainability_review_outputs,
    write_price_schedule_outputs,
)
from src.results.schema import CONTROL_RESULT_SCHEMA_VERSION, MODEL_PROVENANCE_FIELDS
from src.results.usage import (
    AGENT_ATTEMPT_USAGE_COLUMNS,
    AGENT_STEP_TOKEN_USAGE_COLUMNS,
    agent_attempt_usage_row,
    agent_step_token_usage_row,
    derive_global_agent_round_usage,
    extract_control_agent_usage,
    normalized_attempt_usage,
    normalized_zone_cumulative_usage,
    token_only_call_usage,
    token_only_cumulative_usage,
    token_only_round_usage,
)
from src.results.window_cache import write_window_load_price_cache


def write_agent_outputs(
    *,
    output_dir: Path,
    reports: list[dict[str, Any]],
    backend: str | None = None,
    agent_mode: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = output_dir / "rationale_trace.csv"
    trace_md = output_dir / "rationale_trace.md"
    trace_json = output_dir / "rationale_trace.json"
    agent_debug_outputs_json = output_dir / "agent_debug_outputs.json"
    price_schedule_csv = output_dir / "price_schedule_3h.csv"
    price_schedule_md = output_dir / "price_schedule_3h.md"
    price_comparison_csv = output_dir / "price_comparison_summary.csv"
    price_comparison_md = output_dir / "price_comparison_summary.md"
    window_load_price_cache_csv = output_dir / "window_load_price_cache.csv"
    explainability_rubric_md = output_dir / "explainability_rubric.md"
    explainability_review_packet_csv = output_dir / "explainability_review_packet.csv"
    manifest_path = output_dir / "agent_manifest.json"
    control_results_json = output_dir / "control_results.json"
    control_final_windows_csv = output_dir / "control_final_windows.csv"
    control_attempt_trace_csv = output_dir / "control_attempt_trace.csv"
    agent_attempt_usage_csv = output_dir / "agent_attempt_usage.csv"
    agent_step_token_usage_csv = output_dir / "agent_step_token_usage.csv"
    experiment_aggregate_csv = output_dir / "control_experiment_summary.csv"

    control_agent_usage = extract_control_agent_usage(reports)
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
        backend=backend,
        agent_mode=agent_mode,
    )
    write_explainability_review_outputs(
        explainability_rubric_md,
        explainability_review_packet_csv,
        trace_reports,
    )
    write_control_outputs(
        control_results_json,
        control_final_windows_csv,
        control_attempt_trace_csv,
        agent_attempt_usage_csv,
        agent_step_token_usage_csv,
        experiment_aggregate_csv,
        trace_reports,
        backend=backend,
        agent_mode=agent_mode,
        manifest=manifest,
        control_agent_usage=control_agent_usage,
    )

    outputs = {
        "agent_output_dir": output_dir,
        "rationale_trace_csv": trace_csv,
        "rationale_trace_md": trace_md,
        "rationale_trace_json": trace_json,
        "agent_debug_outputs_json": agent_debug_outputs_json,
        "price_schedule_3h_csv": price_schedule_csv,
        "price_schedule_3h_md": price_schedule_md,
        "price_comparison_summary_csv": price_comparison_csv,
        "price_comparison_summary_md": price_comparison_md,
        "window_load_price_cache_csv": window_load_price_cache_csv,
        "explainability_rubric_md": explainability_rubric_md,
        "explainability_review_packet_csv": explainability_review_packet_csv,
        "control_results_json": control_results_json,
        "control_final_windows_csv": control_final_windows_csv,
        **{f"{name}_csv": output_dir / f"{name}.csv" for name in PRICE_RANGE_TABLES},
        "control_attempt_trace_csv": control_attempt_trace_csv,
        "agent_attempt_usage_csv": agent_attempt_usage_csv,
        "agent_step_token_usage_csv": agent_step_token_usage_csv,
        "control_experiment_summary_csv": experiment_aggregate_csv,
    }
    if manifest is not None:
        manifest_payload = {
            **manifest,
            "artifacts": {name: path.name for name, path in outputs.items() if name != "agent_output_dir"},
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs["agent_manifest_json"] = manifest_path
    return outputs


def write_control_outputs(
    results_json: Path,
    final_windows_csv: Path,
    attempt_trace_csv: Path,
    agent_attempt_usage_csv: Path,
    agent_step_token_usage_csv: Path,
    experiment_summary_csv: Path,
    reports: list[dict[str, Any]],
    *,
    backend: str | None,
    agent_mode: str | None,
    manifest: dict[str, Any] | None,
    control_agent_usage: dict[str, Any] | None = None,
) -> None:
    """Write the no-leakage, versioned authority for the control stage."""

    metadata = dict(manifest or {})
    final_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    zone_usage_rows: list[dict[str, Any]] = []
    step_token_rows: list[dict[str, Any]] = []
    zone_usage_by_attempt: dict[int, list[dict[str, Any]]] = {}
    zones: list[dict[str, Any]] = []
    usage_metadata = dict(control_agent_usage or {})
    global_round_usage = [
        token_only_round_usage(item)
        for item in usage_metadata.get("agent_round_usage") or []
        if isinstance(item, dict)
    ]
    global_cumulative_usage = usage_metadata.get("agent_cumulative_usage")
    if not isinstance(global_cumulative_usage, dict):
        global_cumulative_usage = {}
    else:
        global_cumulative_usage = token_only_cumulative_usage(
            global_cumulative_usage
        )
    safe_window_keys = (
        "window_start",
        "window_end",
        "sum_predicted_kwh",
        "load_range_position_pct",
        "load_stress_level",
        "mean_energy_price",
        "suggested_price_shift_pct",
        "proposed_energy_price",
        "price_valid",
        "price_conditioned_baseline_load_kwh",
        "price_conditioned_load_range_position_pct",
        "price_conditioned_load_stress_level",
        "price_conditioned_baseline_source",
        "historical_price_change_percentile",
        "historical_price_change_p95_pct",
        "exceeds_historical_p95",
        *PRICE_RANGE_FIELDS,
        "control_success",
        "control_failure_reason",
        "direction_deviation", "hours",
        "extremely_low_low_threshold_pct", "low_medium_threshold_pct",
        "medium_high_threshold_pct", "high_extremely_high_threshold_pct",
        "load_3h_extremely_low_low_threshold_kwh", "load_3h_low_medium_threshold_kwh",
        "load_3h_medium_high_threshold_kwh", "load_3h_high_extremely_high_threshold_kwh",
        "action_label",
        "price_rationale",
    )

    for report in reports:
        zone_id = report.get("zone_id")
        saved_provenance = (metadata.get("model_provenance") or {}).get(str(zone_id), {})
        provenance = {key: report.get(key, saved_provenance.get(key)) for key in MODEL_PROVENANCE_FIELDS}
        if provenance["model_id"] is None:
            provenance["model_id"] = (metadata.get("model_ids") or {}).get(str(zone_id))
        final_windows = [
            {key: window.get(key) for key in safe_window_keys if key in window}
            for window in report.get("price_change_windows_3h") or []
            if isinstance(window, dict)
        ]
        attempts = report.get("control_attempt_trace") or []
        zone_cumulative_usage = normalized_zone_cumulative_usage(report, attempts)
        zone_call_usage = [
            token_only_call_usage(call)
            for call in report.get("agent_call_usage") or []
            if isinstance(call, dict)
        ]
        zones.append(
            {
                "zone_id": zone_id,
                **provenance,
                "category": report.get("category"),
                "status": report.get("control_status", "fail"),
                "attempts_used": report.get("attempts_used", len(attempts)),
                "selected_attempt": report.get("selected_attempt"),
                "failed_window_count": report.get("failed_window_count"),
                "initial_forecast_total_kwh": report.get("predicted_load_kwh"),
                "initial_grid_stress": report.get("grid_stress_level"),
                "selected_forecast_total_kwh": report.get("selected_forecast_total_kwh"),
                "selected_grid_stress": report.get("selected_grid_stress"),
                "grid_reasoning_summary": report.get("grid_reasoning_summary"),
                "behavior_reasoning_summary": report.get("behavior_reasoning_summary"),
                "economist_reasoning_summary": report.get("economist_reasoning_summary"),
                "final_rationale": report.get("price_rationale"),
                "price_change_reference": report.get("historical_price_change_reference"),
                "price_conditioned_forecast_metadata": report.get(
                    "price_conditioned_forecast_metadata"
                ),
                "agent_cumulative_usage": zone_cumulative_usage,
                "agent_token_totals": token_total_payload(zone_cumulative_usage),
                "agent_call_usage": zone_call_usage,
                "final_agent_discussion_round_count": report.get(
                    "agent_discussion_round_count",
                    0,
                ),
                "final_agent_discussion_round_limit": report.get(
                    "agent_discussion_round_limit",
                    0,
                ),
                "final_agent_discussion_converged": bool(
                    report.get("agent_discussion_converged", False)
                ),
                "final_agent_discussion_stop_reason": report.get(
                    "agent_discussion_stop_reason"
                ),
                "final_agent_discussion_rounds": report.get(
                    "agent_discussion_rounds"
                )
                or [],
                "attempt_trace": attempts,
                "final_windows": final_windows,
            }
        )
        for window in final_windows:
            final_rows.append(
                {
                    "backend": backend,
                    "agent_mode": agent_mode,
                    "zone_id": zone_id,
                    **provenance,
                    "category": report.get("category"),
                    "zone_status": report.get("control_status", "fail"),
                    "attempts_used": report.get("attempts_used", len(attempts)),
                    "selected_attempt": report.get("selected_attempt"),
                    **window,
                }
            )
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_number = reporting_usage_integer(attempt.get("attempt"))
            attempt_usage = normalized_attempt_usage(attempt)
            attempt_calls = [
                token_only_call_usage(call)
                for call in attempt.get("agent_call_usage") or []
                if isinstance(call, dict)
            ]
            for step_index, call in enumerate(attempt_calls, start=1):
                step_token_rows.append(
                    agent_step_token_usage_row(
                        backend=backend,
                        agent_mode=agent_mode,
                        zone_id=zone_id,
                        category=report.get("category"),
                        attempt=attempt_number,
                        proposal_phase=attempt.get("proposal_phase"),
                        triggered_by_attempt=attempt.get(
                            "triggered_by_attempt"
                        ),
                        step_index=step_index,
                        call=call,
                    )
                )
            zone_usage_by_attempt.setdefault(attempt_number, []).append(
                {"zone_id": zone_id, **attempt_usage}
            )
            zone_usage_rows.append(
                agent_attempt_usage_row(
                    record_scope="zone",
                    backend=backend,
                    agent_mode=agent_mode,
                    zone_id=zone_id,
                    category=report.get("category"),
                    attempt=attempt_number,
                    usage=attempt_usage,
                    calls=attempt_calls,
                )
            )
            for window in attempt.get("windows") or []:
                if not isinstance(window, dict):
                    continue
                attempt_rows.append(
                    {
                        "backend": backend,
                        "agent_mode": agent_mode,
                        "zone_id": zone_id,
                        **provenance,
                        "category": report.get("category"),
                        "attempt": attempt_number,
                        **{k: attempt.get(k) for k in ("parent_attempt", "accepted", "selected_attempt", "rejection_reason")},
                        "proposal_phase": attempt.get("proposal_phase"),
                        "triggered_by_attempt": attempt.get(
                            "triggered_by_attempt"
                        ),
                        "attempt_status": attempt.get("control_status"),
                        "attempt_failed_window_count": attempt.get("failed_window_count"),
                        "grid_reasoning_summary": attempt.get("grid_reasoning_summary"),
                        "behavior_reasoning_summary": attempt.get("behavior_reasoning_summary"),
                        "economist_reasoning_summary": attempt.get("economist_reasoning_summary"),
                        **window,
                    }
                )

    if not global_round_usage:
        global_round_usage = derive_global_agent_round_usage(zone_usage_by_attempt)
    if not global_cumulative_usage:
        global_cumulative_usage = derive_global_cumulative_usage(global_round_usage)
    global_usage_rows = [
        agent_attempt_usage_row(
            record_scope="global",
            backend=backend,
            agent_mode=agent_mode,
            zone_id=None,
            category=None,
            attempt=reporting_usage_integer(round_usage.get("attempt")),
            usage=round_usage,
            calls=[],
        )
        for round_usage in global_round_usage
    ]
    global_status = "success" if zones and all(z["status"] == "success" for z in zones) else "fail"
    payload = {
        "schema_version": CONTROL_RESULT_SCHEMA_VERSION,
        "load_policy_id": LOAD_POLICY_ID,
        "result_type": "forecaster_agent_control",
        "status": global_status,
        "forecast_origin": metadata.get("forecast_origin"),
        **{key: metadata[key] for key in ("validation_bias", "validation_days") if key in metadata},
        "backend": backend,
        "agent_mode": agent_mode,
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "cache_keys": metadata.get("cache_keys") or {},
        "feature_manifest": metadata.get("feature_manifest") or {},
        "model_ids": {str(zone["zone_id"]): zone["model_id"] for zone in zones if zone.get("model_id")},
        "model_provenance": {str(zone["zone_id"]): {key: zone.get(key) for key in MODEL_PROVENANCE_FIELDS}
                             for zone in zones if zone.get("model_id")},
        "attempt_limit": 3,
        "attempts_used": max((int(zone["attempts_used"] or 0) for zone in zones), default=0),
        "success_policy": "all zones and all 3-hour windows are Medium",
        "agent_round_usage": global_round_usage,
        "agent_cumulative_usage": global_cumulative_usage,
        "agent_token_totals": token_total_payload(global_cumulative_usage),
        "agent_step_token_usage": step_token_rows,
        "zones": zones,
    }
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(final_rows).to_csv(final_windows_csv, index=False)
    write_price_range_tables(final_windows_csv.parent, final_rows, metadata.get("forecast_origin"))
    pd.DataFrame(attempt_rows).to_csv(attempt_trace_csv, index=False)
    pd.DataFrame(
        [*global_usage_rows, *zone_usage_rows],
        columns=AGENT_ATTEMPT_USAGE_COLUMNS,
    ).to_csv(agent_attempt_usage_csv, index=False)
    pd.DataFrame(
        step_token_rows,
        columns=AGENT_STEP_TOKEN_USAGE_COLUMNS,
    ).to_csv(agent_step_token_usage_csv, index=False)
    pd.DataFrame(
        [
            {
                "backend": backend,
                "agent_mode": agent_mode,
                "status": global_status,
                "zone_count": len(zones),
                "successful_zone_count": sum(z["status"] == "success" for z in zones),
                "failed_zone_count": sum(z["status"] != "success" for z in zones),
                "max_attempts_used": max((int(z["attempts_used"] or 0) for z in zones), default=0),
            }
        ]
    ).to_csv(experiment_summary_csv, index=False)


def split_agent_debug_outputs(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_reports: list[dict[str, Any]] = []
    agent_debug_outputs: list[dict[str, Any]] = []
    for report in reports:
        clean_report = {
            key: value
            for key, value in report.items()
            if key not in {ECONOMIST_AGENT_OUTPUT_KEY, CONTROL_AGENT_USAGE_KEY}
        }
        trace_reports.append(clean_report)
        debug = report.get(ECONOMIST_AGENT_OUTPUT_KEY)
        if isinstance(debug, dict):
            agent_debug_outputs.append(debug)
    return trace_reports, agent_debug_outputs
