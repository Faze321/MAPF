from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from src.agents.reports import ECONOMIST_AGENT_OUTPUT_KEY
from src.control.pricing import final_price_action_label, window_key
from src.control.windows import max_stress_level, optional_number
from src.forecast.engine import daily_totals, pct_change
from src.pipeline.context import build_hourly_averages


def build_retry_context(
    original_context: dict[str, Any],
    report: dict[str, Any],
    *,
    attempt: int,
    replace_forecast: bool,
    last_trial: dict | None = None,
) -> dict[str, Any]:
    context = copy.deepcopy(original_context)
    report_windows = {
        window_key(window): window
        for window in report.get("price_change_windows_3h") or []
    }
    feedback: list[dict[str, Any]] = []
    updated_windows: list[dict[str, Any]] = []
    for original in context.get("pricing_windows_3h") or []:
        latest = report_windows.get(window_key(original), {})
        item = dict(original)
        item["accepted_shift_pct"] = latest.get("suggested_price_shift_pct")
        item["accepted_mean_energy_price"] = latest.get("proposed_energy_price")
        if replace_forecast and latest.get("price_conditioned_baseline_load_kwh") is not None:
            item["sum_predicted_kwh"] = latest.get("price_conditioned_baseline_load_kwh")
            item["mean_predicted_kwh"] = latest.get("price_conditioned_mean_predicted_kwh")
            item["peak_predicted_kwh"] = latest.get("price_conditioned_peak_predicted_kwh")
            item["load_range_position_pct"] = latest.get(
                "price_conditioned_load_range_position_pct"
            )
            item["load_stress_level"] = latest.get("price_conditioned_load_stress_level")
            item["grid_stress_level"] = latest.get("price_conditioned_load_stress_level")
            item["stress_load_3h_kwh"] = latest.get("price_conditioned_baseline_load_kwh")
        updated_windows.append(item)
        if latest and not latest.get("control_success"):
            feedback.append(
                {
                    "attempt": attempt,
                    "window_start": latest.get("window_start"),
                    "window_end": latest.get("window_end"),
                    "previous_price_shift_pct": latest.get("suggested_price_shift_pct"),
                    "previous_energy_price": latest.get("proposed_energy_price"),
                    "reforecast_load_kwh": latest.get("price_conditioned_baseline_load_kwh"),
                    "load_range_position_pct": latest.get(
                        "price_conditioned_load_range_position_pct"
                    ),
                    "load_stress_level": latest.get("price_conditioned_load_stress_level"),
                    "failure_reason": latest.get("control_failure_reason"),
                }
            )
    if last_trial:
        rejected = {window_key(w): w for w in last_trial["windows"]} if not last_trial["accepted"] else {}
        lost = sum(bool(w["control_success"]) and not rejected.get(k, {}).get("control_success", True)
                   for k, w in report_windows.items())
        gained = sum(not w["control_success"] and bool(rejected.get(k, {}).get("control_success"))
                     for k, w in report_windows.items())
        for item in updated_windows:
            trial = rejected.get(window_key(item))
            if trial:
                item.update(rejected_attempt=last_trial["attempt"], rejected_shift_pct=trial["suggested_price_shift_pct"],
                            rejected_mean_energy_price=trial.get("proposed_energy_price"),
                            rejected_load_kwh=trial["reforecast_load_kwh"], rejected_stress=trial["reforecast_load_stress_level"])
        for item in feedback:
            item["accepted_attempt"] = report["selected_attempt"]
            trial = rejected.get(window_key(item))
            if trial:
                item.update(rejected_shift_pct=trial["suggested_price_shift_pct"],
                            rejected_mean_energy_price=trial.get("proposed_energy_price"),
                            rejected_attempt=last_trial["attempt"],
                            rejected_load_kwh=trial["reforecast_load_kwh"],
                            rejected_stress=trial["reforecast_load_stress_level"],
                            rejection_reason=last_trial["rejection_reason"],
                            rejected_lost_medium_count=lost, rejected_gained_medium_count=gained)
    context["pricing_windows_3h"] = updated_windows
    context["retry_feedback"] = feedback
    if replace_forecast:
        loads = [optional_number(item.get("sum_predicted_kwh")) for item in updated_windows]
        valid_loads = [value for value in loads if value is not None]
        context["forecast_total_kwh"] = round(sum(valid_loads), 4)
        context["forecast_peak_kwh"] = round(
            max(
                (
                    optional_number(item.get("peak_predicted_kwh")) or 0.0
                    for item in updated_windows
                ),
                default=0.0,
            ),
            4,
        )
        context["grid_stress_level"] = max_stress_level(
            [item.get("load_stress_level") for item in updated_windows]
        )
        if report.get("_conditioned_hourly"):
            rows = report["_conditioned_hourly"]
            frame = pd.DataFrame(rows)
            frame["time"] = pd.to_datetime(frame["time"])
            context["hourly_forecast"] = rows
            context["hourly_averages"] = build_hourly_averages(frame)
            context["daily_forecast_kwh"] = daily_totals(frame, "predicted_kwh")
            capacity = context.get("capacity_kw_proxy")
            context["peak_capacity_ratio"] = context["forecast_peak_kwh"] / capacity if capacity else 0.0
            prior_total = context.get("previous_comparable_total_kwh")
            if prior_total is not None:
                context["predicted_change_pct"] = pct_change(context["forecast_total_kwh"], prior_total)
            context["grid_stress_load_kwh"] = max(valid_loads, default=0)
            context["grid_stress_load_range_position_pct"] = max(
                (w.get("load_range_position_pct") or 0 for w in updated_windows), default=0)
    return context


def merge_failed_window_revision(
    previous: dict[str, Any],
    revision: dict[str, Any],
    failed_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    revised_by_key = {
        window_key(window): window
        for window in revision.get("price_change_windows_3h") or []
    }
    windows: list[dict[str, Any]] = []
    for old in previous.get("price_change_windows_3h") or []:
        key = window_key(old)
        if key not in failed_keys or key not in revised_by_key:
            windows.append(dict(old))
            continue
        new = revised_by_key[key]
        merged = dict(old)
        for field in (
            "suggested_price_shift_pct",
            "proposed_energy_price",
            "predicted_energy_price",
            "price_valid",
            "price_validation_error",
            "action_label",
            "price_rationale",
        ):
            if field in new:
                merged[field] = new.get(field)
        windows.append(merged)
    updated = dict(previous)
    for field in (
        "grid_reasoning_summary",
        "behavior_reasoning_summary",
        "economist_reasoning_summary",
        "agent_reasoning",
        "action_label",
        "price_rationale",
        "grid_output",
        "behavior_output",
        "economist_output",
        ECONOMIST_AGENT_OUTPUT_KEY,
        "agent_discussion_round_count",
        "agent_discussion_round_limit",
        "agent_discussion_converged",
        "agent_discussion_stop_reason",
        "agent_discussion_rounds",
        "source",
    ):
        if field in revision:
            updated[field] = revision[field]
    updated["price_change_windows_3h"] = windows
    updated["suggested_price_shift_pct"] = round(
        sum(optional_number(item.get("suggested_price_shift_pct")) or 0.0 for item in windows)
        / len(windows),
        4,
    ) if windows else 0.0
    updated["action_label"] = final_price_action_label(updated["suggested_price_shift_pct"])
    return updated
