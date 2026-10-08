"""Offline result analysis. Run: python evaluate.py --input output/my_experiment.

No forecasting models or provider clients are loaded. An optional callable can
perform future LLM analysis of each completed control result.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from usage import cache_usage_fields, summarize_cache_usage
from evaluation_artifacts import (
    clean_json, dataset_metadata, find_forecast_context, load_forecast_context,
    pairing_metadata, read_json, timestamp,
)
from evaluation_windows import extract_window_rounds, summarize_windows, build_window_tables
from evaluation_usage import extract_usage_details, summarize_usage, build_usage_tables
from evaluation_comparison import build_comparisons


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
DEFINITIONS = {
    "forecast_scope": "Pool finite hourly actual/predicted pairs across Zones; never average Zone metrics.",
    "MAE": "mean(abs(actual - predicted)), kWh",
    "RMSE": "sqrt(mean((actual - predicted)**2)), kWh",
    "RAE": "sum(abs(error)) / sum(abs(actual - global_mean_actual)), ratio",
    "MAPE_pct": "100 * mean(abs(error / actual)); exclude abs(actual) <= 1e-9",
    "WAPE_pct": "100 * sum(abs(error)) / sum(abs(actual))",
    "undefined": "Undefined metrics and missing observations are null, never zero.",
    "first_success": "Success after proposal 1, not the uncontrolled baseline or ever-success.",
    "final_success": "Authoritative status/control_success in control_results.json; predicted control outcome, not observed demand.",
    "transitions": "Match run, Zone and window identity; report retained, gained, lost, never and unknown.",
    "calls": "Recorded agent_call_count, including recorded repair/retry calls; proposals are counted separately.",
    "tokens": "Known provider tokens only; full totals/averages are null if any usage is incomplete.",
    "cache": "Cached input tokens / all input tokens, pooled across calls; null when any cache count is unavailable. Independent of completion-token accounting.",
    "averages": "Calls/tokens per run and per Zone; tokens per call = pooled tokens / pooled calls. All runs, including failures, are included.",
    "forecast_deduplication": "Within each model/blend, identical samples with the same dataset, forecast parameters, Zone, time and predictions are counted once across copied Agent-mode outputs. Per-run metrics retain all rows.",
    "medium_new": "Originally non-Medium and currently Medium; separate from adjacent entries and first-ever entries.",
    "revenue": "Predicted electricity revenue only: window mean energy price * window total predicted kWh. No additional factor of 3, no service fee or costs. Each round is an alternative scenario, not additive income.",
    "revenue_baseline": "Original mean_energy_price * sum_predicted_kwh, a window-mean approximation. Proposed price * same-round reforecast load for control outcomes.",
    "revenue_baseline_hourly": "baseline_hourly_revenue = sum(original hourly e_price * original hourly predicted_kwh) within each window, including both endpoints. Requires complete unique hourly records; never substitute actual load or the window-mean approximation.",
    "revenue_changes": "revenue_change and revenue_change_pct use baseline_revenue (window mean). revenue_change_vs_hourly and revenue_change_vs_hourly_pct use baseline_hourly_revenue. Each percentage is 100 * change / its own baseline. The baseline phase retains window-mean revenue; its hourly-relative change shows the difference between the two baselines.",
    "revenue_pooling": "Revenue is pooled only within dataset and source currency; missing observations make full totals unknown. Zero baseline has undefined relative change.",
    "cache_tokens": "Cached tokens are included in prompt_tokens, not additional or subtracted tokens. Missing cache reports are unknown; known_cached_tokens is only the observed subtotal.",
    "fair_comparison": "Compare complete matched runs with identical forecast parameters, baseline prices/loads/thresholds and hourly observations/predictions, including hourly energy prices when saved. Batch labels are user-supplied, not verified LLM model IDs.",
}


class AgentAnalyzer(Protocol):
    """Reserved LLM hook: accept a JSON-ready payload, return a JSON-ready dict.

    Payload contains control_result and quantitative_analysis. Evaluation output
    must not be fed back into the control loop or its Agent context.
    """

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def evaluate_forecaster(hourly: pd.DataFrame) -> dict[str, Any]:
    """Compute micro/global metrics from raw observations, with explicit coverage."""
    values = hourly[["actual_kwh", "predicted_kwh"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    actual, predicted = values.loc[valid].to_numpy(dtype=float).T
    result: dict[str, Any] = {
        "n_total": len(values), "n": len(actual), "n_excluded": int((~valid).sum()),
        "n_mape": 0, "MAE": None, "RMSE": None, "RAE": None,
        "MAPE_pct": None, "WAPE_pct": None,
    }
    if not len(actual):
        return result
    error = np.abs(actual - predicted)
    nonzero = np.abs(actual) > 1e-9
    rae_denominator = float(np.abs(actual - actual.mean()).sum())
    wape_denominator = float(np.abs(actual).sum())
    result.update(
        MAE=float(error.mean()), RMSE=float(np.sqrt(np.mean(error ** 2))),
        RAE=float(error.sum() / rae_denominator) if rae_denominator > 1e-9 else None,
        MAPE_pct=float(np.mean(error[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else None,
        WAPE_pct=float(error.sum() / wape_denominator * 100) if wape_denominator > 1e-9 else None,
        n_mape=int(nonzero.sum()),
    )
    return result


def _status(value: Any) -> bool | None:
    return {"success": True, "fail": False}.get(value)


def _all_success(states: list[bool | None]) -> bool | None:
    if not states:
        return None
    if False in states:
        return False
    return None if None in states else True


def _transition_counts(pairs: list[tuple[bool | None, bool | None]]) -> dict[str, Any]:
    known_first = [a for a, _ in pairs if a is not None]
    known_final = [b for _, b in pairs if b is not None]
    result = {
        "count": len(pairs), "first_known_count": len(known_first),
        "final_known_count": len(known_final),
        "first_success_count": sum(known_first), "final_success_count": sum(known_final),
        "first_success_rate_pct": 100 * sum(known_first) / len(known_first) if known_first else None,
        "final_success_rate_pct": 100 * sum(known_final) / len(known_final) if known_final else None,
    }
    for name, pair in {"retained": (True, True), "gained": (False, True),
                       "lost": (True, False), "never": (False, False)}.items():
        result[f"{name}_count"] = sum(item == pair for item in pairs)
    result["unknown_transition_count"] = sum(a is None or b is None for a, b in pairs)
    return result


def _number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None


def _usage(result: dict[str, Any]) -> dict[str, Any]:
    # Use one authoritative total, never sum global + Zone + attempt records.
    raw = result.get("agent_cumulative_usage")
    if not isinstance(raw, dict) or not raw:
        return {"agent_call_count": None, "token_usage_complete": False,
                **{field: None for field in TOKEN_FIELDS},
                **cache_usage_fields({"agent_call_count": None})}
    values = {field: _number(raw.get(field)) for field in TOKEN_FIELDS}
    return {
        "agent_call_count": _number(raw.get("agent_call_count")), **values,
        "token_usage_complete": raw.get("token_usage_complete") is True
        and all(value is not None for value in values.values()),
        **cache_usage_fields({**raw, "agent_call_count": _number(raw.get("agent_call_count"))}),
    }


def _window_states(windows: list[dict[str, Any]]) -> dict[tuple[str, str], bool | None]:
    states = {}
    for window in windows:
        key = (str(timestamp(window["window_start"])), str(timestamp(window["window_end"])))
        if key in states:
            raise ValueError(f"Duplicate window identity: {key}")
        value = window.get("control_success")
        states[key] = value if isinstance(value, bool) else None
    return states


def evaluate_agent(
    control_result: dict[str, Any], *, analyzer: AgentAnalyzer | None = None,
    baseline_hourly: pd.DataFrame | Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a schema 2/3 result with optional original hourly baseline records.

    ``baseline_hourly`` uses zone_id/time/e_price/predicted_kwh; absent hourly
    records leave the additional hourly revenue baseline unknown.
    """
    if control_result.get("schema_version") not in (2, 3):
        raise ValueError("Expected control_results.json schema version 2 or 3")
    zones = []
    seen_zones: set[str] = set()
    for zone in control_result["zones"]:
        zone_id = str(zone["zone_id"])
        if zone_id in seen_zones:
            raise ValueError(f"Duplicate Zone: {zone_id}")
        seen_zones.add(zone_id)
        attempts = zone.get("attempt_trace") or []
        first = next((a for a in attempts if a.get("attempt") == 1), {})
        initial = _window_states(first.get("windows") or [])
        final = _window_states(zone.get("final_windows") or [])
        windows = [
            {"window_start": key[0], "window_end": key[1],
             "first_success": initial.get(key), "final_success": final.get(key)}
            for key in sorted(initial.keys() | final.keys())
        ]
        zones.append({
            "zone_id": zone_id, "first_success": _status(first.get("control_status")),
            "final_success": _status(zone.get("status")),
            "attempts_used": _number(zone.get("attempts_used")),
            "usage": _usage(zone), "windows": windows,
            "window_summary": _transition_counts([
                (w["first_success"], w["final_success"]) for w in windows
            ]),
        })
    analysis = {
        "forecast_model": control_result.get("forecast_model"),
        "agent_mode": control_result.get("agent_mode"),
        "forecast_origin": control_result.get("forecast_origin"),
        "first_success": _all_success([z["first_success"] for z in zones]),
        "final_success": _status(control_result.get("status")),
        "attempts_used": _number(control_result.get("attempts_used")),
        "usage": _usage(control_result), "zones": zones,
    }
    analysis["summary"] = summarize_agents([analysis])
    analysis["window_rounds"] = extract_window_rounds(control_result, baseline_hourly=baseline_hourly)
    analysis["usage_details"] = extract_usage_details(control_result)
    analysis["outcome_summary"] = summarize_windows(
        [row for row in analysis["window_rounds"] if row["phase"] == "final"], include_revenue=True)
    if analyzer is None:
        analysis["llm_analysis"] = {"status": "not_configured"}
    else:
        response = analyzer(copy.deepcopy({
            "control_result": control_result, "quantitative_analysis": analysis,
        }))
        if not isinstance(response, dict):
            raise TypeError("Agent analyzer must return a JSON-serializable dict")
        json.dumps(response, allow_nan=False)
        analysis["llm_analysis"] = {"status": "completed", "result": response}
    return analysis


def summarize_agents(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool outcomes and resource totals before computing means."""
    zones = [zone for run in runs for zone in run["zones"]]
    windows = [window for zone in zones for window in zone["windows"]]
    result: dict[str, Any] = {}
    for scope, records in (("run", runs), ("zone", zones), ("window", windows)):
        counts = _transition_counts([(r["first_success"], r["final_success"]) for r in records])
        result.update({f"{scope}_{key}": value for key, value in counts.items()})
    for scope, records in (("run", runs), ("zone", zones)):
        attempts = [r["attempts_used"] for r in records]
        result[f"mean_attempts_per_{scope}"] = (
            sum(attempts) / len(attempts) if attempts and None not in attempts else None
        )
    usages = [run["usage"] for run in runs]
    calls = [u["agent_call_count"] for u in usages]
    call_total = sum(calls) if calls and None not in calls else None
    result["agent_call_count"] = call_total
    for scope, count in (("run", len(runs)), ("zone", len(zones))):
        result[f"mean_calls_per_{scope}"] = call_total / count if count and call_total is not None else None
    complete = bool(usages) and all(u["token_usage_complete"] for u in usages)
    result["token_usage_complete"] = complete
    result["incomplete_usage_run_count"] = sum(not u["token_usage_complete"] for u in usages)
    result.update(summarize_cache_usage(usages, aggregated=True))
    result["incomplete_cache_usage_run_count"] = sum(not u["cache_usage_complete"] for u in usages)
    for field in TOKEN_FIELDS:
        observed = [u[field] for u in usages if u[field] is not None]
        known = sum(observed) if observed else None
        result[f"known_{field}"] = known
        result[field] = known if complete else None
        for scope, count in (("run", len(runs)), ("zone", len(zones)), ("call", call_total)):
            result[f"mean_{field}_per_{scope}"] = known / count if complete and count else None
    return result


def _group_rows(rows: list[dict], keys: tuple[str, ...]):
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    for key in sorted(groups, key=lambda value: tuple(str(x) for x in value)):
        yield dict(zip(keys, key)), groups[key]


def _forecast_summary(frame: pd.DataFrame) -> dict:
    result = evaluate_forecaster(frame)
    values = frame[["actual_kwh", "predicted_kwh"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    totals = values.loc[valid].sum()
    actual = float(totals.actual_kwh) if valid.any() else None
    predicted = float(totals.predicted_kwh) if valid.any() else None
    result.update(actual_total_kwh=actual, predicted_total_kwh=predicted,
                  bias_pct=100 * (predicted - actual) / actual if actual else None)
    if all(key in frame for key in ("q10_kwh", "q90_kwh")):
        interval = frame[["actual_kwh", "q10_kwh", "q90_kwh"]].apply(pd.to_numeric, errors="coerce")
        covered = np.isfinite(interval.to_numpy(dtype=float)).all(axis=1)
        subset = interval.loc[covered]
        result["interval_coverage_pct"] = float(((subset.actual_kwh >= subset.q10_kwh) &
                                               (subset.actual_kwh <= subset.q90_kwh)).mean() * 100) if len(subset) else None
    return result


def _ppt_tables(runs: list[dict], windows: list[dict], rounds: list[dict], common_ids: list[str]) -> dict:
    selected = set(common_ids)
    medium, entries, tokens = [], [], []
    for meta, group in _group_rows([r for r in runs if r["run_id"] in selected], ("batch", "agent_mode")):
        ids = {r["run_id"] for r in group}
        rows = [w for w in windows if w["run_id"] in ids]
        baseline = summarize_windows([w for w in rows if w["phase"] == "baseline"])
        final = summarize_windows([w for w in rows if w["phase"] == "final"])
        usage = summarize_agents(group)
        record = {**meta, "run_count": len(group), "window_count": baseline["window_count"],
                  "original_medium": baseline["medium_count"], "final_medium": final["medium_count"],
                  "original_unknown": baseline["medium_unknown_count"], "final_unknown": final["medium_unknown_count"],
                  "net_change_vs_original": final["net_medium_count"]}
        first = {**meta, "run_count": len(group), "ever_newly_medium": final["ever_newly_medium_count"],
                 "ever_medium_final_exit": final["ever_medium_final_exit_count"],
                 "ever_newly_medium_final_exit": final["ever_newly_medium_final_exit_count"],
                 "first_entry_unknown_count": final["first_entry_unknown_count"],
                 "total_tokens": usage["total_tokens"], "cached_tokens": usage["cached_tokens"],
                 "known_cached_tokens": usage["known_cached_tokens"], "cache_usage_complete": usage["cache_usage_complete"]}
        token_row = {**meta, "run_count": len(group), "total_tokens": usage["total_tokens"],
                     "all_rounds_per_run": usage["mean_total_tokens_per_run"],
                     "cached_tokens": usage["cached_tokens"], "known_cached_tokens": usage["known_cached_tokens"],
                     "cache_hit_ratio": usage["cache_hit_ratio"], "cache_usage_complete": usage["cache_usage_complete"],
                     "mean_cached_tokens_per_run": usage["cached_tokens"] / len(group) if usage["cached_tokens"] is not None else None}
        round_ids = sorted({w["round"] for w in rows if w["phase"] == "round"})
        for number in round_ids:
            current = summarize_windows([w for w in rows if w["phase"] == "round" and w["round"] == number])
            record.update({f"round_{number}_total": current["medium_count"],
                           f"round_{number}_new": current["new_medium_count"],
                           f"round_{number}_window_count": current["window_count"],
                           f"round_{number}_participating_run_count": current["run_count"],
                           f"round_{number}_unknown": current["medium_unknown_count"] + baseline["window_count"] - current["window_count"]})
            first[f"first_in_round_{number}"] = final["first_entry_round_distribution"].get(str(number), 0)
            usage_rows = [u for u in rounds if u["run_id"] in ids and u["round"] == number]
            round_usage = summarize_usage(usage_rows, aggregated=True)
            # Runs with a shorter configured horizon do not silently count as zero calls.
            full_cohort = len({u["run_id"] for u in usage_rows}) == len(ids)
            for field in TOKEN_FIELDS + ("cached_tokens",):
                value = round_usage[field]
                token_row[f"round_{number}_mean_{field}"] = value / len(group) if value is not None and full_cohort else None
            token_row[f"round_{number}_cache_hit_ratio"] = round_usage["cache_hit_ratio"] if full_cohort else None
        round_slots = len([u for u in rounds if u["run_id"] in ids])
        token_row["mean_per_round"] = usage["total_tokens"] / round_slots if usage["total_tokens"] is not None and round_slots else None
        medium.append(record)
        entries.append(first)
        tokens.append(token_row)
    return {"ppt_medium": medium, "ppt_first_entries": entries, "ppt_tokens": tokens}


def _evaluate_inputs(inputs: Mapping[str, Path | str], *, analyzer: AgentAnalyzer | None,
                     section: str, multiple: bool) -> dict[str, Any]:
    if section not in ("all", "forecaster", "agent"):
        raise ValueError(f"Unknown section: {section}")
    if not inputs:
        raise ValueError("At least one input directory is required")
    roots = {label: Path(path).resolve() for label, path in inputs.items()}
    if any(not isinstance(label, str) or not label.strip() for label in roots):
        raise ValueError("Batch labels must be non-empty strings")
    if len(set(roots.values())) != len(roots):
        raise ValueError("The same directory cannot be evaluated as multiple batches")
    if any(a != b and a in b.parents for a in roots.values() for b in roots.values()):
        raise ValueError("Batch directories must not overlap (parent/child inputs would compare a result with itself)")
    for root in roots.values():
        if not root.is_dir():
            raise ValueError(f"Input directory does not exist: {root}")
    forecast_runs, frames, agent_runs, windows, calls, rounds, audits = [], [], [], [], [], [], []
    contexts: dict[Path, dict] = {}
    for label, root in roots.items():
        if section != "agent":
            detail_dirs = sorted({p.parent for p in root.rglob("zone_*_forecast_vs_actual.csv")})
            for detail_dir in detail_dirs:
                directory = detail_dir.parent.parent.resolve()
                if directory not in contexts:
                    contexts[directory] = load_forecast_context(directory)
                context = contexts[directory]
                frame = context["frame"].copy()
                meta = {"batch": label, **context["metadata"]}
                run_id = str(detail_dir.relative_to(root))
                if multiple:
                    run_id = f"{label}/{run_id}"
                forecast_runs.append({"run_id": run_id, **meta, **_forecast_summary(frame)})
                for key, value in meta.items():
                    frame[key] = value
                frame["_forecast_identity"] = context["identity"]
                frames.append(frame)
        if section != "forecaster":
            for path in sorted(root.rglob("control_results.json")):
                raw = read_json(path)
                context = find_forecast_context(path, contexts)
                try:
                    analysis = evaluate_agent(raw, analyzer=analyzer,
                                              baseline_hourly=(context or {}).get("frame"))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Cannot evaluate {path}: {exc}") from exc
                meta = {"batch": label, **dataset_metadata((context or {}).get("manifest", {}), raw),
                        "forecast_model": raw.get("forecast_model"), "agent_mode": raw.get("agent_mode"),
                        "forecast_origin": timestamp(raw.get("forecast_origin")),
                        "run_id": f"{label}/{path.relative_to(root)}" if multiple else str(path.relative_to(root))}
                analysis.update(meta)
                analysis.update(pairing_metadata(raw, context))
                analysis["pairing_ready"] &= meta["dataset_provenance_known"]
                analysis["label_source"] = "user_supplied_batch_label" if multiple else "input_directory_name"
                analysis["source_file"] = str(path)
                run_windows = [{**w, **meta} for w in analysis.pop("window_rounds")]
                detail = analysis.pop("usage_details")
                run_calls = [{**c, **meta} for c in detail["calls"]]
                run_rounds = [{**u, **meta} for u in detail["rounds"]]
                audits.extend({**a, **meta} for a in detail["audit"])
                analysis["outcome_summary"] = summarize_windows([w for w in run_windows if w["phase"] == "final"], include_revenue=True)
                analysis["pairing_ready"] &= analysis["outcome_summary"]["run_complete_count"] == 1
                windows.extend(run_windows)
                calls.extend(run_calls)
                rounds.extend(run_rounds)
                agent_runs.append(analysis)
    if not forecast_runs and not agent_runs:
        raise ValueError(f"No {section} result artifacts found under the input directories")
    tables = build_window_tables(windows)
    tables.update(build_usage_tables(calls, rounds))
    for name in ("forecast_by_dataset", "forecast_by_origin", "forecast_by_zone_origin", "ppt_forecast", "forecast_conflicts"):
        tables[name] = []
    tables["window_round_details"] = windows
    tables["usage_audit"] = audits
    forecast_summary = []
    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        keys = ["batch", "_forecast_identity", "zone_id", "time", "actual_kwh", "predicted_kwh"]
        unique = pooled.drop_duplicates(keys) if all(k in pooled for k in keys) else pooled
        dimensions = {
            "forecast_by_dataset": ["batch", "dataset", "dataset_identity", "forecast_model", "diurnal_blend_alpha"],
            "forecast_by_origin": ["batch", "dataset", "dataset_identity", "forecast_model", "diurnal_blend_alpha", "forecast_origin"],
            "forecast_by_zone_origin": ["batch", "dataset", "dataset_identity", "forecast_model", "diurnal_blend_alpha", "forecast_origin", "zone_id"],
        }
        for name, dimensions_ in dimensions.items():
            rows = []
            for key, group in unique.groupby(dimensions_, dropna=False, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                rows.append({**dict(zip(dimensions_, key)), **_forecast_summary(group)})
            tables[name] = rows
        tables["ppt_forecast"] = tables["forecast_by_dataset"]
        dimensions_ = ["batch", "forecast_model", "diurnal_blend_alpha"]
        for key, group in pooled.groupby(dimensions_, dropna=False, sort=True):
            dedup = group.drop_duplicates(keys) if all(k in group for k in keys) else group
            meta = dict(zip(dimensions_, key))
            run_count = sum(all(run.get(k) == v for k, v in meta.items()) for run in forecast_runs)
            forecast_summary.append({**meta, "run_count": run_count,
                                     "duplicate_sample_count": len(group) - len(dedup), **_forecast_summary(dedup)})
        identity_keys = [k for k in keys if k not in ("actual_kwh", "predicted_kwh")]
        if all(k in pooled for k in keys):
            conflict = pooled.groupby(identity_keys, dropna=False)[["actual_kwh", "predicted_kwh"]].nunique(dropna=False)
            bad = conflict[(conflict > 1).any(axis=1)].reset_index()
            tables["forecast_conflicts"] = [{**row, "reason": "conflicting predictions/observations for the same forecast sample"}
                                            for row in bad.to_dict("records")]
    comparison = build_comparisons(agent_runs)
    tables.update(_ppt_tables(agent_runs, windows, rounds, comparison["common_run_ids"]))
    tables["comparison_audit"] = comparison["audit"]
    tables["paired_runs"] = comparison["pairs"]
    tables["paired_summary"] = comparison["paired_summary"]
    tables["control_inventory"] = [{key: run.get(key) for key in
        ("batch", "dataset", "dataset_identity", "forecast_model", "agent_mode", "forecast_origin", "run_id",
         "source_file", "zone_ids", "pairing_ready", "label_source", "first_success", "final_success", "attempts_used")}
        for run in agent_runs]
    group_keys = ("batch", "forecast_model", "agent_mode") if multiple else ("forecast_model", "agent_mode")
    report = {
        "schema_version": 1, "input_dir": str(next(iter(roots.values()))) if not multiple else None,
        "inputs": {label: str(path) for label, path in roots.items()}, "definitions": DEFINITIONS,
        "forecaster": {"runs": forecast_runs, "global_by_model": forecast_summary},
        "agent": {"runs": agent_runs, "overall": summarize_agents(agent_runs),
                  "by_model_and_mode": [{**meta, **summarize_agents(group)} for meta, group in _group_rows(agent_runs, group_keys)]},
        "comparison": comparison, "tables": tables,
    }
    return clean_json(report)


def evaluate_directory(input_dir: Path | str, *, analyzer: AgentAnalyzer | None = None,
                       section: str = "all") -> dict[str, Any]:
    """Analyze one directory, retaining the original public result fields."""
    root = Path(input_dir).resolve()
    return _evaluate_inputs({root.name: root}, analyzer=analyzer, section=section, multiple=False)


def evaluate_experiments(inputs: Mapping[str, Path | str], *, analyzer: AgentAnalyzer | None = None,
                         section: str = "all") -> dict[str, Any]:
    """Compare labelled batches on verified common runs; labels do not verify LLM IDs."""
    return _evaluate_inputs(inputs, analyzer=analyzer, section=section, multiple=True)


def _markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "暂无满足条件的数据。"
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    def cell(value):
        if value is None:
            return "未知"
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows)])


def render_evaluation_report(report: dict[str, Any]) -> str:
    """Deterministic Chinese report using exactly the exported table values."""
    tables = report.get("tables", {})
    comparison = report.get("comparison", {})
    medium_view, entry_view, token_view = [], [], []
    for row in tables.get("ppt_medium", []):
        view = {"批次": row["batch"], "模式": row["agent_mode"], "运行数": row["run_count"],
                "窗口数": row["window_count"], "原始 Medium": row["original_medium"]}
        for number in sorted(int(key.split("_")[1]) for key in row if key.startswith("round_") and key.endswith("_total")):
            value = f"{row[f'round_{number}_total']} / {row[f'round_{number}_new']}"
            if row[f"round_{number}_unknown"]:
                value += f"（未知 {row[f'round_{number}_unknown']}）"
            view[f"第 {number} 轮 Total / new"] = value
        view.update({"最终 Medium": row["final_medium"], "最终净增": row["net_change_vs_original"]})
        medium_view.append(view)
    for row in tables.get("ppt_first_entries", []):
        view = {"批次": row["batch"], "模式": row["agent_mode"]}
        for key in row:
            if key.startswith("first_in_round_"):
                view[f"首次进入第 {key.rsplit('_', 1)[-1]} 轮"] = row[key]
        view.update({"曾新增达标": row["ever_newly_medium"], "新增后最终退出": row["ever_newly_medium_final_exit"],
                     "首次轮次未知": row["first_entry_unknown_count"], "总 tokens": row["total_tokens"], "缓存命中 tokens": row["cached_tokens"]})
        entry_view.append(view)
    for row in tables.get("ppt_tokens", []):
        view = {"批次": row["batch"], "模式": row["agent_mode"]}
        for key in row:
            if key.startswith("round_") and key.endswith("_mean_total_tokens"):
                view[f"第 {key.split('_')[1]} 轮"] = row[key]
        view.update({"全部轮次/运行": row["all_rounds_per_run"], "平均每轮": row["mean_per_round"],
                     "缓存命中总 tokens": row["cached_tokens"], "平均缓存命中/运行": row["mean_cached_tokens_per_run"],
                     "输入缓存命中比例": row["cache_hit_ratio"]})
        token_view.append(view)
    views = {"ppt_medium": medium_view, "ppt_first_entries": entry_view, "ppt_tokens": token_view}
    sections = ["# 实验评估报告", "本报告离线读取保存结果；控制效果和收益均为模型预测，未代表真实调价后的观测。",
                "批次标签用于区分输入目录；旧结果未记录实际 LLM 型号时，标签不构成型号验证。",
                f"读取 {len(report['agent']['runs'])} 个控制结果；同批模式共同样本包含 "
                f"{len(comparison.get('common_run_ids', []))} 个运行；跨批成功配对 "
                f"{len(comparison.get('pairs', []))} 对。业务控制失败的完整运行也参与分析。"]
    for title, name, columns in (
        ("预测指标", "ppt_forecast", ["batch", "dataset", "forecast_model", "diurnal_blend_alpha", "n", "MAE", "RMSE", "RAE", "MAPE_pct", "WAPE_pct", "bias_pct"]),
        ("Medium 逐轮数量（共同样本）", "ppt_medium", None),
        ("首次进入 Medium 与总 tokens（共同样本）", "ppt_first_entries", None),
        ("每轮平均 tokens 与缓存命中（共同样本）", "ppt_tokens", None),
    ):
        sections.extend([f"## {title}", _markdown_table(views.get(name, tables.get(name, [])), columns)])
    sections.extend(["new 指原始非 Medium、当前轮为 Medium；相邻轮次进入/退出和首次进入另表统计。控制轮次与 Agent 内部讨论轮次分别计数。",
                     "每轮均值以共同样本的运行数为分母；明确成功早停后的轮次为零调用，缺失记录为未知。缓存命中 tokens 已包含在输入 tokens 中。"])
    usage = tables.get("usage_by_mode", [])
    sections.extend(["## 批次调用总览（全量，模式数量可能不同）", _markdown_table(tables.get("usage_by_batch", []), [
        "batch", "agent_call_count", "prompt_tokens", "completion_tokens", "total_tokens",
        "mean_prompt_tokens_per_call", "mean_completion_tokens_per_call", "mean_total_tokens_per_call",
        "cached_tokens", "known_cached_tokens", "mean_cached_tokens_per_call", "cache_hit_ratio", "cache_reported_call_count"])])
    sections.extend(["## 每次调用与缓存统计（全量）", _markdown_table(usage, [
        "batch", "agent_mode", "agent_call_count", "mean_prompt_tokens_per_call", "mean_completion_tokens_per_call",
        "mean_total_tokens_per_call", "median_total_tokens_per_call", "p95_total_tokens_per_call",
        "cached_tokens", "known_cached_tokens", "mean_cached_tokens_per_call", "cache_hit_ratio",
        "cache_reported_call_count", "cache_usage_complete"]),
        "缓存命中比例为累计命中输入 tokens ÷ 累计输入 tokens（0–1）。服务端未返回时完整统计为未知；known_cached_tokens 仅代表已知部分，不能把部分数据视为完整总量。"])
    revenue = tables.get("revenue_by_dataset_model_mode", [])
    sections.extend(["## 电价收益（全量，按数据集分开）", "同时使用两个基准：窗口基准 baseline_revenue = 窗口均价 × 原始预测总 kWh；逐小时基准 baseline_hourly_revenue = Σ（原始小时电价 e_price × 同小时预测负荷 predicted_kwh），按窗口起止小时（含两端）求和。每轮及最终收益 = 实际采用价格 × 同轮重预测总 kWh。无需再乘 3，各轮是独立方案，不累加为收入。仅含电费、不含服务费、未扣成本；未改变 Medium 达标规则。",
                     _markdown_table(revenue, ["batch", "dataset", "forecast_model", "agent_mode", "revenue_unit", "phase", "round",
                                               "baseline_revenue", "baseline_hourly_revenue", "revenue", "revenue_change", "revenue_change_pct",
                                               "revenue_change_vs_hourly", "revenue_change_vs_hourly_pct"]),
                     "revenue_change / revenue_change_pct 相对窗口基准；revenue_change_vs_hourly / revenue_change_vs_hourly_pct 相对逐小时基准，各自除以对应基准计算百分比。baseline 行的 revenue 保留窗口基准，因此该行相对逐小时的差额表示两种基准的差异。不同数据集/币种不直接合计收益。基准为零时其增减比例未知。缺少完整小时价格/预测负荷时逐小时基准未知，不用窗口均价或实际负荷代替。"])
    incomplete_revenue = [row for row in revenue if not all(row.get(f"{field}_complete") for field in
                                                          ("baseline_revenue", "baseline_hourly_revenue", "revenue"))]
    if incomplete_revenue:
        sections.extend(["### 收益数据覆盖（仅列不完整项）", _markdown_table(incomplete_revenue, [
            "batch", "dataset", "forecast_model", "agent_mode", "revenue_unit", "phase", "round", "window_count",
            "known_baseline_revenue", "baseline_revenue_coverage_pct", "known_baseline_hourly_revenue",
            "baseline_hourly_revenue_coverage_pct", "known_revenue", "revenue_coverage_pct"]),
            "known_* 仅为完整有效窗口的已知部分和，不能作为完整收益或基准；覆盖率以窗口数计算，窗口内缺小时不计为完整窗口。"])
    paired = []
    for row in tables.get("paired_summary", []):
        flattened = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flattened.update({f"{key}_{name}": item for name, item in value.items() if name in
                                  ("medium_count", "total_tokens", "cached_tokens", "revenue", "baseline_revenue",
                                   "baseline_hourly_revenue", "revenue_change", "revenue_change_vs_hourly")})
            else:
                flattened[key] = value
        paired.append(flattened)
    sections.extend(["## 跨批配对比较", _markdown_table(paired),
                     "仅比较同模式、同预测参数、同区域时间范围，且基线价格、负荷、阈值与小时预测（包括已保存的小时电价）一致的完整运行。差值方向见表中标记；不混用两种和五种模式的总体均值。"])
    conclusions = []
    for meta, rows in _group_rows(tables.get("ppt_medium", []), ("batch",)):
        if rows and all(row.get("final_medium") is not None and row.get("final_unknown") == 0 for row in rows):
            best = max(row["final_medium"] for row in rows)
            names = "、".join(row["agent_mode"] for row in rows if row["final_medium"] == best)
            conclusions.append(f"- {meta['batch']}：共同样本中 {names} 的最终 Medium 数量最多，为 {best}。")
    if any(not row.get("cache_usage_complete") for row in usage):
        conclusions.append("- 部分批次缺少完整缓存统计，不能据此比较完整缓存命中数量或加速效果。")
    sections.extend(["## 数据覆盖与结论", "\n".join(conclusions) or "当前样本不足以生成模式排名。",
                     f"配对诊断记录 {len(tables.get('comparison_audit', []))} 条；用量校验记录 {len(tables.get('usage_audit', []))} 条；"
                     f"预测冲突记录 {len(tables.get('forecast_conflicts', []))} 条。",
                     "全部逐窗口、逐调用、分数据集/模型/起点/区域统计和未配对原因见同目录 CSV 与 evaluation.json。"])
    return "\n\n".join(sections) + "\n"


def write_evaluation(report: dict[str, Any], output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "evaluation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tables = {
        "forecaster_global_metrics.csv": report["forecaster"]["global_by_model"],
        "forecaster_run_metrics.csv": report["forecaster"]["runs"],
        "agent_summary.csv": report["agent"]["by_model_and_mode"],
        "agent_run_metrics.csv": [
            {"run_id": r["run_id"], "forecast_model": r["forecast_model"],
             "agent_mode": r["agent_mode"], **r["summary"]}
            for r in report["agent"]["runs"]
        ],
    }
    tables.update({f"{name}.csv": rows for name, rows in report.get("tables", {}).items()})
    for filename, rows in tables.items():
        if rows:
            csv_rows = [{key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows]
            pd.DataFrame(csv_rows).to_csv(output / filename, index=False, encoding="utf-8-sig")
        else:
            # Remove only this evaluator's obsolete table, never source artifacts.
            (output / filename).unlink(missing_ok=True)
    (output / "evaluation_report_zh.md").write_text(render_evaluation_report(report), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Run or experiment directory (default: output)")
    source.add_argument("--batch", action="append", metavar="LABEL=PATH", help="Repeat to compare labelled experiment directories")
    parser.add_argument("--output", type=Path, help="Default: <input>/analysis")
    parser.add_argument("--section", choices=("all", "forecaster", "agent"), default="all")
    parser.add_argument("--agent-analyzer", help="Optional callable module:function for LLM analysis")
    args = parser.parse_args()
    analyzer = None
    if args.agent_analyzer:
        module_name, separator, attribute = args.agent_analyzer.partition(":")
        if not separator or not module_name or not attribute:
            parser.error("--agent-analyzer must use module:function")
        if args.section == "forecaster":
            parser.error("--agent-analyzer requires the agent section")
        analyzer = getattr(importlib.import_module(module_name), attribute)
        if not callable(analyzer):
            parser.error("--agent-analyzer must reference a callable")
    if args.batch:
        if args.output is None:
            parser.error("--batch requires --output")
        inputs = {}
        for value in args.batch:
            label, separator, path = value.partition("=")
            if not separator or not label.strip() or not path.strip() or label in inputs:
                parser.error("Each --batch must be a unique LABEL=PATH")
            inputs[label] = Path(path)
        report = evaluate_experiments(inputs, analyzer=analyzer, section=args.section)
        destination = args.output
    else:
        input_dir = args.input or Path("output")
        report = evaluate_directory(input_dir, analyzer=analyzer, section=args.section)
        destination = args.output or input_dir / "analysis"
    target = write_evaluation(report, destination)
    print(f"Evaluated {len(report['forecaster']['runs'])} forecaster runs and "
          f"{len(report['agent']['runs'])} agent runs. Report: {target.resolve()}")


if __name__ == "__main__":
    main()
