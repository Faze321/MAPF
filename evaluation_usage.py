"""Offline, coverage-aware accounting of logical Agent calls and control rounds.

Provider retries remain one recorded logical call: the saved result cannot
reconstruct the distribution or tokens of unreported HTTP requests.
"""
from __future__ import annotations

from collections import defaultdict
from math import ceil, floor
from numbers import Integral
import re
from statistics import median
from typing import Any, Iterable

from usage import CACHE_USAGE_FIELDS, cache_usage_fields, summarize_cache_usage


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_CONTEXT = ("batch", "dataset", "forecast_model", "agent_mode")


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def _records(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _normalize_usage(raw: dict[str, Any], *, aggregated: bool) -> dict[str, Any]:
    count = _integer(raw.get("agent_call_count")) if aggregated else 1
    if aggregated and count is None and raw.get("agent_invoked") is False:
        count = 0
    values = {field: _integer(raw.get(field)) for field in TOKEN_FIELDS}
    if count == 0:
        # An explicit zero-call row (e.g. a frozen proposal) is observed zero.
        values = {field: 0 if value is None else value for field, value in values.items()}
    retry_count = _integer(raw.get("provider_attempt_count"))
    permitted = (raw.get("token_usage_complete") is None or raw.get("token_usage_complete") is True) and not (retry_count and retry_count > 1)
    complete = permitted and count is not None and all(value is not None for value in values.values())
    known_count = count if count is not None else _integer(raw.get("known_agent_call_count")) or 0
    result = {"agent_call_count": count, "known_agent_call_count": known_count,
              **values, "token_usage_complete": complete,
              "call_population_complete": raw.get("call_population_complete")}
    cache = {**raw, **result}
    if count is None and known_count:
        cache.update(agent_call_count=known_count, cache_usage_complete=False)
    if retry_count and retry_count > 1:
        # Preserve observed counts before marking missing retry usage incomplete.
        cache.update(cache_usage_fields(cache))
        cache["cache_usage_complete"] = False
    result.update(cache_usage_fields(cache))
    return result


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = floor(position), ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_usage(rows: Iterable[dict[str, Any]], *, aggregated: bool = False) -> dict[str, Any]:
    """Pool counts first; unavailable observations never become measured zeros.

    Aggregate rows allow pooled means per logical call, but cannot supply
    medians/P95 of individual calls. Known distributions are explicitly partial.
    """
    records = [_normalize_usage(row, aggregated=aggregated) for row in rows]
    counts = [row["agent_call_count"] for row in records]
    known_counts = [row["known_agent_call_count"] for row in records]
    population_complete = all(row.get("call_population_complete") is not False for row in records)
    count_complete = population_complete and all(value is not None for value in counts)
    call_count = sum(known_counts) if count_complete else None
    complete = count_complete and all(row["token_usage_complete"] for row in records)
    result: dict[str, Any] = {
        "source_row_count": len(records), "agent_call_count": call_count,
        "known_agent_call_count": sum(known_counts), "call_count_complete": count_complete,
        "token_usage_complete": complete,
        "incomplete_usage_row_count": sum(not row["token_usage_complete"] for row in records),
        "call_population_complete": population_complete,
        "call_distribution_available": not aggregated and complete,
        "distribution_unit": "logical_call" if not aggregated else None,
    }
    for field in TOKEN_FIELDS:
        observed = [row[field] for row in records if row[field] is not None]
        known_total = sum(observed) if observed or not records else None
        result[field] = known_total if complete else None
        result[f"known_{field}"] = known_total
        result[f"known_{field}_row_count"] = len(observed)
        result[f"mean_{field}_per_call"] = known_total / call_count if complete and call_count else None
        result[f"mean_{field}_per_row"] = known_total / len(records) if complete and records else None
        result[f"median_{field}_per_call"] = median(observed) if not aggregated and complete and observed else None
        result[f"p95_{field}_per_call"] = _percentile(observed, .95) if not aggregated and complete else None
        result[f"known_mean_{field}_per_call"] = sum(observed) / len(observed) if not aggregated and observed else None
        result[f"known_median_{field}_per_call"] = median(observed) if not aggregated and observed else None
        result[f"known_p95_{field}_per_call"] = _percentile(observed, .95) if not aggregated else None
        result[f"known_{field}_call_count"] = len(observed) if not aggregated else None
    cache_records = [
        {**row, "agent_call_count": row["known_agent_call_count"], "cache_usage_complete": False}
        if row["agent_call_count"] is None and row["known_agent_call_count"] else row
        for row in records
    ]
    result.update(summarize_cache_usage(cache_records, aggregated=True))
    if not population_complete:
        result.update(cached_tokens=None, cache_usage_complete=False, cache_hit_ratio=None)
    result["mean_cached_tokens_per_call"] = (
        result["cached_tokens"] / call_count if result["cache_usage_complete"] and call_count else None
    )
    result["cache_reported_call_fraction"] = result["cache_reported_call_count"] / call_count if call_count else None
    result["incomplete_cache_usage_row_count"] = sum(not row["cache_usage_complete"] for row in records)
    return result


def _totals(rows: list[dict[str, Any]], *, aggregated: bool) -> dict[str, Any]:
    """Keep known numeric subtotals with an explicit incomplete flag internally."""
    summary = summarize_usage(rows, aggregated=aggregated)
    return {
        "agent_call_count": summary["agent_call_count"],
        "known_agent_call_count": summary["known_agent_call_count"],
        **{field: summary[f"known_{field}"] for field in TOKEN_FIELDS},
        "token_usage_complete": summary["token_usage_complete"],
        **{field: summary[field] for field in CACHE_USAGE_FIELDS},
    }


def _call(raw: dict[str, Any], *, source: str, zone_id: Any = None,
          attempt: Any = None, index: int | None = None) -> dict[str, Any]:
    stage = raw.get("stage")
    match = re.search(r"(?:^|\.)round_(\d+)(?:\.|$)", str(stage or ""))
    attempt = _integer(raw.get("attempt", attempt))
    discussion = _integer(raw.get("discussion_round"))
    return {
        **raw, **_normalize_usage(raw, aggregated=False),
        "attempt": attempt, "round": attempt,
        "zone_id": str(raw.get("zone_id", zone_id)) if raw.get("zone_id", zone_id) is not None else None,
        "stage": stage, "agent": raw.get("agent"),
        "discussion_round": discussion if discussion is not None else int(match.group(1)) if match else None,
        "is_repair": "repair" in str(stage or "").lower() or "repair" in str(raw.get("agent") or "").lower(),
        "provider_attempt_count": _integer(raw.get("provider_attempt_count", 1)),
        "step_index": raw.get("step_index", index),
        "usage_source": source, "record_scope": "call",
    }


def _compare(left: dict[str, Any], right: dict[str, Any], *, check: str,
             attempt: int | None = None, zone_id: str | None = None) -> dict[str, Any]:
    mismatches = []
    unknown = []
    for field in ("agent_call_count", *TOKEN_FIELDS, "cached_tokens"):
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            unknown.append(field)
        elif a != b:
            mismatches.append({"field": field, "detail_value": a, "authoritative_value": b})
    return {"check": check, "attempt": attempt, "zone_id": zone_id,
            "status": "mismatch" if mismatches else "partial" if unknown else "match",
            "mismatches": mismatches, "unavailable_fields": unknown}


def extract_usage_details(control_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read one source at each level; never add calls to their aggregate totals."""
    zones = _records(control_result.get("zones"))
    audit: list[dict[str, Any]] = []
    has_step_source = isinstance(control_result.get("agent_step_token_usage"), list)
    calls = []
    if has_step_source:
        calls = [_call(raw, source="agent_step_token_usage", index=i)
                 for i, raw in enumerate(_records(control_result["agent_step_token_usage"]), 1)]
    else:
        for zone in zones:
            traces = _records(zone.get("attempt_trace"))
            has_trace_calls = any(isinstance(trace.get("agent_call_usage"), list) for trace in traces)
            if has_trace_calls:
                for trace in traces:
                    calls.extend(_call(raw, source="attempt_trace.agent_call_usage", zone_id=zone.get("zone_id"),
                                       attempt=trace.get("attempt"), index=i)
                                 for i, raw in enumerate(_records(trace.get("agent_call_usage")), 1))
            else:
                calls.extend(_call(raw, source="zone.agent_call_usage", zone_id=zone.get("zone_id"), index=i)
                             for i, raw in enumerate(_records(zone.get("agent_call_usage")), 1))
    by_attempt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if call["attempt"] is not None:
            by_attempt[call["attempt"]].append(call)
        else:
            audit.append({"check": "call_round_identity", "status": "partial", "zone_id": call["zone_id"],
                          "message": "Call has no recorded control attempt; it is not assigned to a guessed round."})
        if all(call[field] is not None for field in TOKEN_FIELDS) and call["total_tokens"] != call["prompt_tokens"] + call["completion_tokens"]:
            audit.append({"check": "call_token_sum", "status": "mismatch", "attempt": call["attempt"],
                          "zone_id": call["zone_id"], "message": "Recorded total differs from prompt + completion; retained as reported."})

    source_rounds: dict[int, dict[str, Any]] = {}
    for raw in _records(control_result.get("agent_round_usage")):
        number = _integer(raw.get("attempt"))
        if number is None:
            continue
        if number in source_rounds:
            audit.append({"check": "round_identity", "status": "mismatch", "attempt": number,
                          "message": "Duplicate global round row; first record retained."})
            continue
        source_rounds[number] = {**raw, **_normalize_usage(raw, aggregated=True), "usage_source": "agent_round_usage"}
    traces_by_zone = [(_integer(zone.get("attempts_used")), zone,
                      {_integer(t.get("attempt")): t for t in _records(zone.get("attempt_trace"))})
                     for zone in zones]
    observed = set(source_rounds) | set(by_attempt)
    observed.update(n for _, _, traces in traces_by_zone for n in traces if n is not None)
    used = _integer(control_result.get("attempts_used"))
    limit = _integer(control_result.get("attempt_limit"))
    last = max([n for n in observed if n is not None] + [used or 0, limit or 0])
    stopped = control_result.get("status") == "success" and used is not None
    rounds = []
    for number in range(1, last + 1):
        if number in source_rounds:
            row = dict(source_rounds[number])
        elif stopped and number > used:
            row = {**_normalize_usage({"agent_call_count": 0}, aggregated=True),
                   "agent_invoked": False, "usage_source": "success_early_stop_zero"}
        else:
            zone_rows = []
            for zone_used, zone, traces in traces_by_zone:
                trace = traces.get(number)
                raw = trace.get("agent_usage") if trace is not None else None
                if isinstance(raw, dict) and raw:
                    zone_rows.append(_normalize_usage(raw, aggregated=True))
                elif trace is not None and isinstance(trace.get("agent_call_usage"), list):
                    zone_calls = [_call(c, source="attempt_trace.agent_call_usage") for c in _records(trace["agent_call_usage"])]
                    zone_rows.append(_totals(zone_calls, aggregated=False))
                elif zone.get("status") == "success" and zone_used is not None and number > zone_used:
                    zone_rows.append(_normalize_usage({"agent_call_count": 0}, aggregated=True))
                else:
                    zone_rows.append(_normalize_usage({}, aggregated=True))
            if zone_rows and any(row["agent_call_count"] is not None for row in zone_rows):
                row = {**_totals(zone_rows, aggregated=True), "usage_source": "zone_attempt_usage"}
            elif has_step_source and number in by_attempt:
                row = {**_totals(by_attempt[number], aggregated=False), "usage_source": "agent_step_token_usage"}
            else:
                row = {**_normalize_usage({}, aggregated=True), "usage_source": "missing_round_usage"}
        round_calls = by_attempt.get(number, [])
        if any((_integer(call.get("provider_attempt_count")) or 0) > 1 for call in round_calls):
            row["token_usage_complete"] = False
            row["cache_usage_complete"] = False
            row.update(cache_usage_fields(row))
        row.update(attempt=number, round=number, record_scope="global", zone_id=None,
                   carried_forward=row["usage_source"] == "success_early_stop_zero")
        rounds.append(row)
        if row["usage_source"] != "success_early_stop_zero":
            details = summarize_usage(round_calls)
            audit.append(_compare(details, summarize_usage([row], aggregated=True),
                                  check="calls_vs_round", attempt=number))
    cumulative = control_result.get("agent_cumulative_usage")
    if isinstance(cumulative, dict) and cumulative:
        authority = summarize_usage([cumulative], aggregated=True)
        audit.append(_compare(summarize_usage(rounds, aggregated=True), authority, check="rounds_vs_cumulative"))
        audit.append(_compare(summarize_usage(calls), authority, check="calls_vs_cumulative"))
    else:
        audit.append({"check": "cumulative_usage", "status": "partial", "message": "No authoritative cumulative usage recorded."})
    for zone in zones:
        cumulative = zone.get("agent_cumulative_usage")
        if isinstance(cumulative, dict) and cumulative:
            zone_id = str(zone.get("zone_id"))
            audit.append(_compare(summarize_usage([c for c in calls if c["zone_id"] == zone_id]),
                                  summarize_usage([cumulative], aggregated=True),
                                  check="calls_vs_zone_cumulative", zone_id=zone_id))
    cumulative = control_result.get("agent_cumulative_usage")
    expected_count = _integer(cumulative.get("agent_call_count")) if isinstance(cumulative, dict) else None
    if expected_count is None:
        round_counts = [_integer(row.get("agent_call_count")) for row in rounds]
        if round_counts and all(count is not None for count in round_counts):
            expected_count = sum(round_counts)
    population_complete = len(calls) == expected_count if expected_count is not None else False
    for call in calls:
        call.update(call_population_complete=population_complete,
                    run_expected_call_count=expected_count, run_observed_call_count=len(calls))
    if not population_complete:
        # Totals saved independently of calls stay authoritative. Derived totals
        # cannot acquire full coverage merely because the surviving rows agree.
        for row in rounds:
            if row["usage_source"] not in ("agent_round_usage", "success_early_stop_zero"):
                row["call_population_complete"] = False
        audit.append({"check": "call_detail_coverage", "status": "partial",
                      "expected_call_count": expected_count, "recorded_call_count": len(calls),
                      "message": "Call details do not cover the full recorded run; role/zone distributions describe known calls only."})
    return {"calls": calls, "rounds": rounds, "audit": audit}


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def _group_summary(rows: list[dict[str, Any]], *, aggregated: bool,
                   calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result = summarize_usage(rows, aggregated=aggregated)
    runs = {row.get("run_id") for row in rows if row.get("run_id") is not None}
    result["run_count"] = len(runs)
    result["run_count_complete"] = all(row.get("run_id") is not None for row in rows)
    result["mean_denominator"] = "participating_runs_in_group; includes explicit early-stop zeros"
    detail_rows = calls if aggregated else rows
    result["recorded_call_count"] = len(detail_rows) if detail_rows is not None else None
    result["call_details_complete"] = (
        detail_rows is not None and result["agent_call_count"] is not None
        and len(detail_rows) == result["agent_call_count"]
        and all(row.get("call_population_complete") is not False for row in detail_rows)
    )
    for field in (*TOKEN_FIELDS, "cached_tokens"):
        value = result[field]
        result[f"mean_{field}_per_run"] = value / len(runs) if value is not None and runs and result["run_count_complete"] else None
    if aggregated and calls is not None:
        call_summary = summarize_usage(calls)
        distribution_complete = (call_summary["agent_call_count"] == result["agent_call_count"]
                                 and call_summary["token_usage_complete"] and result["token_usage_complete"]
                                 and all(call_summary[field] == result[field] for field in TOKEN_FIELDS))
        result["call_distribution_available"] = distribution_complete
        result["distribution_unit"] = "logical_call" if distribution_complete else None
        for field in TOKEN_FIELDS:
            for stat in ("median", "p95"):
                result[f"{stat}_{field}_per_call"] = call_summary[f"{stat}_{field}_per_call"] if distribution_complete else None
            for stat in ("mean", "median", "p95"):
                result[f"known_{stat}_{field}_per_call"] = call_summary[f"known_{stat}_{field}_per_call"]
            result[f"known_{field}_call_count"] = call_summary[f"known_{field}_call_count"]
    return result


def build_usage_tables(calls: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build portable tables; global totals use rounds, call roles use details."""
    tables: dict[str, list[dict[str, Any]]] = {"call_details": calls, "usage_per_run_round": []}
    by_run_round = _group(calls, ("run_id", "attempt"))
    for row in rounds:
        tables["usage_per_run_round"].append({
            **row, **_group_summary([row], aggregated=True,
                                   calls=by_run_round.get((row.get("run_id"), row.get("attempt")), [])),
        })
    aggregate_groups = {
        "usage_by_batch": ("batch",),
        "usage_by_dataset": ("batch", "dataset"),
        "usage_by_mode": ("batch", "agent_mode"),
        "usage_by_dataset_model_mode": _CONTEXT,
        "usage_by_origin": (*_CONTEXT, "forecast_origin"),
        "tokens_by_mode_round": (*_CONTEXT, "attempt"),
    }
    for name, keys in aggregate_groups.items():
        call_groups = _group(calls, keys)
        tables[name] = [{**dict(zip(keys, key)), **_group_summary(group, aggregated=True, calls=call_groups.get(key, []))}
                        for key, group in sorted(_group(rounds, keys).items(), key=lambda item: str(item[0]))]
    detail_groups = {
        "usage_by_zone": (*_CONTEXT, "zone_id"),
        "usage_by_stage": (*_CONTEXT, "stage", "is_repair"),
        "usage_by_agent": (*_CONTEXT, "agent"),
        "usage_by_discussion_round": (*_CONTEXT, "discussion_round"),
        "usage_by_repair": (*_CONTEXT, "is_repair"),
    }
    for name, keys in detail_groups.items():
        tables[name] = [{**dict(zip(keys, key)), **_group_summary(group, aggregated=False)}
                        for key, group in sorted(_group(calls, keys).items(), key=lambda item: str(item[0]))]
    return tables
