from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from global_forecaster import NATIVE_ARTIFACT_SCHEMA_VERSION, NativeForecasterArtifact


PRICE_ERROR_THRESHOLD_RATIO = 0.08
PRICE_ERROR_THRESHOLD_PCT = PRICE_ERROR_THRESHOLD_RATIO * 100
ECONOMIST_AGENT_OUTPUT_KEY = "_economist_agent_output"
CONTROL_AGENT_USAGE_KEY = "_control_agent_usage"
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
    "grid_stress_level",
    "agent_reasoning",
    "suggested_price_shift_pct",
    "action_label",
    "price_rationale",
    "control_status",
    "attempts_used",
    "failed_window_count",
    "agent_time_cost_seconds",
    "agent_invoked",
    "agent_call_count",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_total_tokens",
    "agent_token_usage_complete",
    "source",
]
TRACE_OUTPUT_RENAMES = {
    "predicted_load_kwh": "forecast_total_kwh",
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
    "predicted_energy_price",
    "forecaster_input_predicted_energy_price",
    "baseline_energy_price",
    "known_service_price",
    "final_price_shift_pct",
    "load_stress_level",
    "load_range_position_pct",
    "price_conditioned_load_stress_level",
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
    "baseline_load_kwh",
    "baseline_load_range_position_pct",
    "baseline_load_source",
    "target_load_kwh",
    "target_load_range_position_pct",
    "medium_load_min_kwh",
    "medium_load_max_kwh",
    "target_peak_reduction_pct",
    "elasticity_factor",
    "load_range_source",
    "expected_load_kwh",
    "expected_load_range_position_pct",
    "load_in_medium_band",
    "price_valid",
    "historical_price_change_percentile",
    "historical_price_change_p95_pct",
    "exceeds_historical_p95",
    "control_success",
    "control_failure_reason",
    "action_label",
    "price_rationale",
]

AGENT_ATTEMPT_USAGE_COLUMNS = [
    "record_scope",
    "forecast_model",
    "agent_mode",
    "zone_id",
    "category",
    "attempt",
    "proposal_phase",
    "triggered_by_attempt",
    "agent_invoked",
    "agent_invoked_zone_count",
    "agent_call_count",
    "agent_batch_wall_time_seconds",
    "agent_elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_complete",
    "agent_stages",
    "agent_names",
    "agent_call_usage_json",
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
    outputs = write_forecaster_outputs(
        output_dir=output_dir,
        selected_zones=selected_zones,
        contexts=contexts,
        forecast_results=forecast_results,
    )
    outputs.update(
        write_agent_outputs(
            output_dir=output_dir,
            reports=reports,
            forecast_model=forecast_model,
            agent_mode=agent_mode,
        )
    )
    return outputs


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
        manifest_payload = {
            **manifest,
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


def write_agent_outputs(
    *,
    output_dir: Path,
    reports: list[dict[str, Any]],
    forecast_model: str | None = None,
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
        forecast_model=forecast_model,
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
        experiment_aggregate_csv,
        trace_reports,
        forecast_model=forecast_model,
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
        "control_attempt_trace_csv": control_attempt_trace_csv,
        "agent_attempt_usage_csv": agent_attempt_usage_csv,
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
    experiment_summary_csv: Path,
    reports: list[dict[str, Any]],
    *,
    forecast_model: str | None,
    agent_mode: str | None,
    manifest: dict[str, Any] | None,
    control_agent_usage: dict[str, Any] | None = None,
) -> None:
    """Write the no-leakage, versioned authority for the control stage."""

    metadata = dict(manifest or {})
    final_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    zone_usage_rows: list[dict[str, Any]] = []
    zone_usage_by_attempt: dict[int, list[dict[str, Any]]] = {}
    zones: list[dict[str, Any]] = []
    usage_metadata = dict(control_agent_usage or {})
    global_round_usage = [
        dict(item)
        for item in usage_metadata.get("agent_round_usage") or []
        if isinstance(item, dict)
    ]
    global_cumulative_usage = usage_metadata.get("agent_cumulative_usage")
    if not isinstance(global_cumulative_usage, dict):
        global_cumulative_usage = {}
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
        "control_success",
        "control_failure_reason",
        "action_label",
        "price_rationale",
    )

    for report in reports:
        zone_id = report.get("zone_id")
        final_windows = [
            {key: window.get(key) for key in safe_window_keys if key in window}
            for window in report.get("price_change_windows_3h") or []
            if isinstance(window, dict)
        ]
        attempts = report.get("control_attempt_trace") or []
        zone_cumulative_usage = normalized_zone_cumulative_usage(report, attempts)
        zone_call_usage = [
            dict(call)
            for call in report.get("agent_call_usage") or []
            if isinstance(call, dict)
        ]
        zones.append(
            {
                "zone_id": zone_id,
                "category": report.get("category"),
                "status": report.get("control_status", "fail"),
                "attempts_used": report.get("attempts_used", len(attempts)),
                "failed_window_count": report.get("failed_window_count"),
                "initial_forecast_total_kwh": report.get("predicted_load_kwh"),
                "initial_grid_stress": report.get("grid_stress_level"),
                "grid_reasoning_summary": report.get("grid_reasoning_summary"),
                "behavior_reasoning_summary": report.get("behavior_reasoning_summary"),
                "economist_reasoning_summary": report.get("economist_reasoning_summary"),
                "final_rationale": report.get("price_rationale"),
                "price_change_reference": report.get("historical_price_change_reference"),
                "price_conditioned_forecast_metadata": report.get(
                    "price_conditioned_forecast_metadata"
                ),
                "agent_cumulative_usage": zone_cumulative_usage,
                "agent_call_usage": zone_call_usage,
                "final_agent_discussion_round_count": report.get(
                    "agent_discussion_round_count",
                    0,
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
                    "forecast_model": forecast_model,
                    "agent_mode": agent_mode,
                    "zone_id": zone_id,
                    "category": report.get("category"),
                    "zone_status": report.get("control_status", "fail"),
                    "attempts_used": report.get("attempts_used", len(attempts)),
                    **window,
                }
            )
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_number = reporting_usage_integer(attempt.get("attempt"))
            attempt_usage = normalized_attempt_usage(attempt)
            attempt_calls = [
                dict(call)
                for call in attempt.get("agent_call_usage") or []
                if isinstance(call, dict)
            ]
            zone_usage_by_attempt.setdefault(attempt_number, []).append(
                {"zone_id": zone_id, **attempt_usage}
            )
            zone_usage_rows.append(
                agent_attempt_usage_row(
                    record_scope="zone",
                    forecast_model=forecast_model,
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
                        "forecast_model": forecast_model,
                        "agent_mode": agent_mode,
                        "zone_id": zone_id,
                        "category": report.get("category"),
                        "attempt": attempt_number,
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
            forecast_model=forecast_model,
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
        "schema_version": 2,
        "result_type": "forecaster_agent_control",
        "status": global_status,
        "forecast_origin": metadata.get("forecast_origin"),
        "forecast_model": forecast_model,
        "agent_mode": agent_mode,
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "cache_keys": metadata.get("cache_keys") or {},
        "feature_manifest": metadata.get("feature_manifest") or {},
        "attempt_limit": 3,
        "attempts_used": max((int(zone["attempts_used"] or 0) for zone in zones), default=0),
        "success_policy": "all zones and all 3-hour windows are Medium",
        "agent_round_usage": global_round_usage,
        "agent_cumulative_usage": global_cumulative_usage,
        "zones": zones,
    }
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(final_rows).to_csv(final_windows_csv, index=False)
    pd.DataFrame(attempt_rows).to_csv(attempt_trace_csv, index=False)
    pd.DataFrame(
        [*global_usage_rows, *zone_usage_rows],
        columns=AGENT_ATTEMPT_USAGE_COLUMNS,
    ).to_csv(agent_attempt_usage_csv, index=False)
    pd.DataFrame(
        [
            {
                "forecast_model": forecast_model,
                "agent_mode": agent_mode,
                "status": global_status,
                "zone_count": len(zones),
                "successful_zone_count": sum(z["status"] == "success" for z in zones),
                "failed_zone_count": sum(z["status"] != "success" for z in zones),
                "max_attempts_used": max((int(z["attempts_used"] or 0) for z in zones), default=0),
            }
        ]
    ).to_csv(experiment_summary_csv, index=False)


def extract_control_agent_usage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    for report in reports:
        value = report.get(CONTROL_AGENT_USAGE_KEY)
        if isinstance(value, dict):
            return value
    return {}


def normalized_attempt_usage(attempt: dict[str, Any]) -> dict[str, Any]:
    raw = attempt.get("agent_usage")
    raw = dict(raw) if isinstance(raw, dict) else {}
    calls = [
        call
        for call in attempt.get("agent_call_usage") or []
        if isinstance(call, dict)
    ]
    agent_call_count = reporting_usage_integer(
        raw.get("agent_call_count", len(calls))
    )
    token_usage_complete = raw.get("token_usage_complete")
    if token_usage_complete is None:
        token_usage_complete = all(
            bool(call.get("token_usage_complete"))
            if call.get("token_usage_complete") is not None
            else all(
                call.get(field) is not None
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            for call in calls
        )
    return {
        "proposal_phase": raw.get("proposal_phase")
        or attempt.get("proposal_phase")
        or ("initial" if reporting_usage_integer(attempt.get("attempt")) == 1 else "frozen"),
        "triggered_by_attempt": raw.get("triggered_by_attempt")
        if "triggered_by_attempt" in raw
        else attempt.get("triggered_by_attempt"),
        "agent_invoked": bool(raw.get("agent_invoked", agent_call_count > 0)),
        "agent_invoked_zone_count": reporting_usage_integer(
            raw.get("agent_invoked_zone_count", 1 if agent_call_count > 0 else 0)
        ),
        "agent_call_count": agent_call_count,
        "agent_batch_wall_time_seconds": reporting_optional_float(
            raw.get("agent_batch_wall_time_seconds")
        ),
        "agent_elapsed_seconds": round(
            reporting_optional_float(
                raw.get(
                    "agent_elapsed_seconds",
                    sum(reporting_optional_float(call.get("elapsed_seconds")) or 0.0 for call in calls),
                )
            )
            or 0.0,
            4,
        ),
        "prompt_tokens": reporting_usage_integer(
            raw.get(
                "prompt_tokens",
                sum(reporting_usage_integer(call.get("prompt_tokens")) for call in calls),
            )
        ),
        "completion_tokens": reporting_usage_integer(
            raw.get(
                "completion_tokens",
                sum(reporting_usage_integer(call.get("completion_tokens")) for call in calls),
            )
        ),
        "total_tokens": reporting_usage_integer(
            raw.get(
                "total_tokens",
                sum(reporting_usage_integer(call.get("total_tokens")) for call in calls),
            )
        ),
        "token_usage_complete": bool(token_usage_complete),
    }


def normalized_zone_cumulative_usage(
    report: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    value = report.get("agent_cumulative_usage")
    if isinstance(value, dict):
        return dict(value)
    usage_rows = [
        normalized_attempt_usage(attempt)
        for attempt in attempts
        if isinstance(attempt, dict)
    ]
    if usage_rows:
        return {
            "agent_attempt_count": len(usage_rows),
            "agent_invoked_attempt_count": sum(
                bool(item.get("agent_invoked")) for item in usage_rows
            ),
            "agent_invoked": any(
                bool(item.get("agent_invoked")) for item in usage_rows
            ),
            "agent_call_count": sum(
                reporting_usage_integer(item.get("agent_call_count"))
                for item in usage_rows
            ),
            "agent_elapsed_seconds": round(
                sum(
                    reporting_optional_float(item.get("agent_elapsed_seconds")) or 0.0
                    for item in usage_rows
                ),
                4,
            ),
            "prompt_tokens": sum(
                reporting_usage_integer(item.get("prompt_tokens"))
                for item in usage_rows
            ),
            "completion_tokens": sum(
                reporting_usage_integer(item.get("completion_tokens"))
                for item in usage_rows
            ),
            "total_tokens": sum(
                reporting_usage_integer(item.get("total_tokens"))
                for item in usage_rows
            ),
            "token_usage_complete": all(
                bool(item.get("token_usage_complete")) for item in usage_rows
            ),
        }
    calls = [
        call
        for call in report.get("agent_call_usage") or []
        if isinstance(call, dict)
    ]
    fallback_token_usage_complete = all(
        bool(call.get("token_usage_complete"))
        if call.get("token_usage_complete") is not None
        else all(
            call.get(field) is not None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        for call in calls
    )
    return {
        "agent_attempt_count": reporting_usage_integer(report.get("attempts_used")),
        "agent_invoked_attempt_count": 1 if calls else 0,
        "agent_invoked": bool(calls),
        "agent_call_count": len(calls),
        "agent_elapsed_seconds": round(
            reporting_optional_float(report.get("agent_time_cost_seconds")) or 0.0,
            4,
        ),
        "prompt_tokens": reporting_usage_integer(report.get("agent_prompt_tokens")),
        "completion_tokens": reporting_usage_integer(
            report.get("agent_completion_tokens")
        ),
        "total_tokens": reporting_usage_integer(report.get("agent_total_tokens")),
        "token_usage_complete": bool(
            report.get(
                "agent_token_usage_complete",
                fallback_token_usage_complete,
            )
        ),
    }


def derive_global_agent_round_usage(
    zone_usage_by_attempt: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for attempt, usage_rows in sorted(zone_usage_by_attempt.items()):
        invoked = [item for item in usage_rows if item.get("agent_invoked")]
        active_phases = {
            str(item.get("proposal_phase"))
            for item in usage_rows
            if item.get("proposal_phase") not in (None, "", "frozen")
        }
        proposal_phase = (
            next(iter(active_phases))
            if len(active_phases) == 1
            else "mixed"
            if active_phases
            else "frozen"
        )
        triggered_values = [
            item.get("triggered_by_attempt")
            for item in usage_rows
            if item.get("triggered_by_attempt") is not None
        ]
        rounds.append(
            {
                "attempt": attempt,
                "proposal_phase": proposal_phase,
                "triggered_by_attempt": triggered_values[0]
                if triggered_values
                else None,
                "agent_invoked": bool(invoked),
                "agent_invoked_zone_count": len(invoked),
                "agent_call_count": sum(
                    reporting_usage_integer(item.get("agent_call_count"))
                    for item in usage_rows
                ),
                "agent_batch_wall_time_seconds": 0.0,
                "agent_elapsed_seconds": round(
                    sum(
                        reporting_optional_float(item.get("agent_elapsed_seconds"))
                        or 0.0
                        for item in usage_rows
                    ),
                    4,
                ),
                "prompt_tokens": sum(
                    reporting_usage_integer(item.get("prompt_tokens"))
                    for item in usage_rows
                ),
                "completion_tokens": sum(
                    reporting_usage_integer(item.get("completion_tokens"))
                    for item in usage_rows
                ),
                "total_tokens": sum(
                    reporting_usage_integer(item.get("total_tokens"))
                    for item in usage_rows
                ),
                "token_usage_complete": all(
                    bool(item.get("token_usage_complete")) for item in usage_rows
                ),
                "invoked_zone_ids": [item.get("zone_id") for item in invoked],
            }
        )
    return rounds


def derive_global_cumulative_usage(
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "agent_round_count": len(rounds),
        "agent_invoked_round_count": sum(
            bool(item.get("agent_invoked")) for item in rounds
        ),
        "agent_invoked": any(bool(item.get("agent_invoked")) for item in rounds),
        "agent_invoked_zone_count": sum(
            reporting_usage_integer(item.get("agent_invoked_zone_count"))
            for item in rounds
        ),
        "agent_call_count": sum(
            reporting_usage_integer(item.get("agent_call_count")) for item in rounds
        ),
        "agent_batch_wall_time_seconds": round(
            sum(
                reporting_optional_float(item.get("agent_batch_wall_time_seconds"))
                or 0.0
                for item in rounds
            ),
            4,
        ),
        "agent_elapsed_seconds": round(
            sum(
                reporting_optional_float(item.get("agent_elapsed_seconds")) or 0.0
                for item in rounds
            ),
            4,
        ),
        "prompt_tokens": sum(
            reporting_usage_integer(item.get("prompt_tokens")) for item in rounds
        ),
        "completion_tokens": sum(
            reporting_usage_integer(item.get("completion_tokens")) for item in rounds
        ),
        "total_tokens": sum(
            reporting_usage_integer(item.get("total_tokens")) for item in rounds
        ),
        "token_usage_complete": all(
            bool(item.get("token_usage_complete")) for item in rounds
        ),
    }


def agent_attempt_usage_row(
    *,
    record_scope: str,
    forecast_model: str | None,
    agent_mode: str | None,
    zone_id: Any,
    category: Any,
    attempt: int,
    usage: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = list(dict.fromkeys(str(call.get("stage")) for call in calls if call.get("stage")))
    names = list(dict.fromkeys(str(call.get("agent")) for call in calls if call.get("agent")))
    return {
        "record_scope": record_scope,
        "forecast_model": forecast_model,
        "agent_mode": agent_mode,
        "zone_id": zone_id,
        "category": category,
        "attempt": attempt,
        "proposal_phase": usage.get("proposal_phase"),
        "triggered_by_attempt": usage.get("triggered_by_attempt"),
        "agent_invoked": bool(usage.get("agent_invoked")),
        "agent_invoked_zone_count": reporting_usage_integer(
            usage.get(
                "agent_invoked_zone_count",
                1 if usage.get("agent_invoked") else 0,
            )
        ),
        "agent_call_count": reporting_usage_integer(usage.get("agent_call_count")),
        "agent_batch_wall_time_seconds": usage.get(
            "agent_batch_wall_time_seconds"
        ),
        "agent_elapsed_seconds": usage.get("agent_elapsed_seconds"),
        "prompt_tokens": reporting_usage_integer(usage.get("prompt_tokens")),
        "completion_tokens": reporting_usage_integer(
            usage.get("completion_tokens")
        ),
        "total_tokens": reporting_usage_integer(usage.get("total_tokens")),
        "token_usage_complete": bool(usage.get("token_usage_complete")),
        "agent_stages": "|".join(stages),
        "agent_names": "|".join(names),
        "agent_call_usage_json": json.dumps(calls, ensure_ascii=False),
    }


def reporting_usage_integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def reporting_optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


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
                "historical_min_load_3h_kwh": result.summary.get(
                    "historical_min_load_3h_kwh"
                ),
                "historical_max_load_3h_kwh": result.summary.get(
                    "historical_max_load_3h_kwh"
                ),
                "historical_load_range_3h_kwh": result.summary.get(
                    "historical_load_range_3h_kwh"
                ),
                "low_medium_threshold_pct": result.summary.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": result.summary.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": result.summary.get(
                    "high_extremely_high_threshold_pct"
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
                "low_medium_threshold_pct": window.get("low_medium_threshold_pct"),
                "medium_high_threshold_pct": window.get("medium_high_threshold_pct"),
                "high_extremely_high_threshold_pct": window.get(
                    "high_extremely_high_threshold_pct"
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
            baseline_energy_price = window.get("mean_energy_price")
            comparison = price_comparison_fields(
                baseline_energy_price,
                window.get("suggested_price_shift_pct"),
                window.get("predicted_energy_price"),
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
                    "predicted_energy_price": comparison["predicted_energy_price"],
                    "forecaster_input_predicted_energy_price": window.get("price_conditioned_energy_price"),
                    "baseline_energy_price": baseline_energy_price,
                    "known_service_price": window.get("mean_service_price"),
                    "final_price_shift_pct": window.get("suggested_price_shift_pct"),
                    "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                    "load_range_position_pct": window.get("load_range_position_pct"),
                    "price_conditioned_load_stress_level": window.get("price_conditioned_load_stress_level"),
                    "historical_min_load_3h_kwh": window.get(
                        "historical_min_load_3h_kwh"
                    ),
                    "historical_max_load_3h_kwh": window.get(
                        "historical_max_load_3h_kwh"
                    ),
                    "historical_load_range_3h_kwh": window.get(
                        "historical_load_range_3h_kwh"
                    ),
                    "low_medium_threshold_pct": window.get("low_medium_threshold_pct"),
                    "medium_high_threshold_pct": window.get("medium_high_threshold_pct"),
                    "high_extremely_high_threshold_pct": window.get(
                        "high_extremely_high_threshold_pct"
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
