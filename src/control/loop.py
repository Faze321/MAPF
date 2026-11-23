from __future__ import annotations

import asyncio
import copy
from typing import Any

import pandas as pd

from src.agents.chains import retry_zone_prices, run_zone_chain
from src.agents.usage import (
    CONTROL_AGENT_USAGE_KEY,
    cache_usage_fields,
    cumulative_global_agent_usage,
    token_total_summary,
)
from src.control.feedback import build_retry_context, merge_failed_window_revision
from src.control.policy import STRESS_LEVELS
from src.control.pricing import (
    attach_price_change_diagnostics,
    final_price_action_label,
    window_key,
)
from src.control.scenario import (
    apply_price_conditioned_baseline_forecasts,
    attach_price_conditioned_baselines,
)
from src.control.scoring import assess_control_report, schedule_score
from src.control.trace import (
    build_global_agent_round_usage,
    control_attempt_snapshot,
    cumulative_zone_agent_usage,
    retry_proposal_phase,
)
from src.control.windows import max_stress_level, optional_number
from src.forecast.artifact import NativeForecasterArtifact
from src.forecast.parameters import forecast_runtime_parameters


MAX_CONTROL_ATTEMPTS = 3


def original_schedule(report, context):
    reason = "R0: keep the original hourly energy prices."
    windows = []
    for source in context["pricing_windows_3h"]:
        w = dict(source)
        price = optional_number(w.get("mean_energy_price"))
        w.update(suggested_price_shift_pct=0.0, action_label=final_price_action_label(0),
                 predicted_energy_price=price, proposed_energy_price=price,
                 price_valid=price is not None and price >= 0, price_rationale=reason)
        windows.append(w)
    base = {k: report.get(k) for k in ("zone_id", "category", "predicted_load_kwh",
            "predicted_peak_kwh", "grid_stress_level", "predicted_change_pct")}
    base.update(price_change_windows_3h=windows, suggested_price_shift_pct=0.0,
                source="original_price_schedule", agent_reasoning=reason, price_rationale=reason,
                action_label=final_price_action_label(0), economist_reasoning_summary=reason,
                economist_output={"reasoning_summary": reason}, selected_attempt=0)
    base = attach_price_conditioned_baselines(base, windows, backend="baseline")
    for w in base["price_change_windows_3h"]:
        w["price_conditioned_baseline_source"] = "original_hourly_price_forecast"
    return assess_control_report(base, attempt=0)


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

    best = {}
    for report in current:
        zone = str(report["zone_id"])
        baseline = original_schedule(report, contexts_by_zone[zone])
        best[zone] = attach_price_change_diagnostics(baseline, price_change_reference)
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
            **forecast_runtime_parameters(forecast_parameters),
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
            parent = best[zone_id]
            parent_attempt = parent["selected_attempt"]
            parent_windows = parent["price_change_windows_3h"]
            for w, old in zip(report["price_change_windows_3h"], parent_windows, strict=True):
                delta = (optional_number(w.get("suggested_price_shift_pct")) or 0) - old["suggested_price_shift_pct"]
                stress = old["price_conditioned_load_stress_level"]
                w["direction_deviation"] = (
                    delta >= 0 if stress in STRESS_LEVELS[:2] else
                    delta <= 0 if stress in STRESS_LEVELS[3:] else delta != 0
                )
            previous_snapshot = (traces[zone_id][parent_attempt - 1] if parent_attempt
                                 else {"windows": parent_windows})
            snapshot = control_attempt_snapshot(
                report,
                attempt,
                previous_snapshot=previous_snapshot,
                revised_keys=pending_revised_keys_by_zone.get(zone_id, set()),
                agent_call_usage=pending_usage_by_zone.get(zone_id, []),
                proposal_phase=pending_phase_by_zone.get(zone_id, "frozen"),
                triggered_by_attempt=attempt - 1 if attempt > 1 else None,
            )
            score = schedule_score(report["price_change_windows_3h"])
            prior_score = schedule_score(parent_windows)
            accept = score is not None and (prior_score is None or score < prior_score)
            snapshot.update(parent_attempt=parent_attempt, accepted=accept,
                            selected_attempt=attempt if accept else parent_attempt,
                            rejection_reason=None if accept else "incomplete_or_invalid" if score is None
                            else "no_improvement_over_accepted")
            if accept:
                best[zone_id] = dict(report, selected_attempt=attempt)
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

        current = [best[str(r["zone_id"])] for r in current]
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
                last_trial=traces[zone_id][-1],
                previous_trial=traces[zone_id][-2] if attempt > 1 else None,
                replace_forecast=True,
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
        updated.pop("_conditioned_hourly", None)
        updated["control_attempt_trace"] = traces[zone_id]
        updated["attempts_used"] = len(traces[zone_id])
        final_windows = updated.get("price_change_windows_3h") or []
        selected_loads = [optional_number(w.get("price_conditioned_baseline_load_kwh")) for w in final_windows]
        updated["selected_forecast_total_kwh"] = (
            sum(selected_loads) if selected_loads and all(v is not None for v in selected_loads) else None
        )
        updated["selected_grid_stress"] = max_stress_level([
            w.get("price_conditioned_load_stress_level") for w in final_windows
        ]) if updated["selected_forecast_total_kwh"] is not None else None
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
        updated.update({f"agent_{key}": value for key, value in cache_usage_fields(cumulative_usage).items()})
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
