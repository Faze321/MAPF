"""Fair, whole-run cohorts for offline experiment comparisons.

Candidate identity deliberately excludes the Agent mode and batch.  Matching
identities alone are insufficient: forecast inputs and uncontrolled baselines
must also be known and equal before outcomes or whole-run tokens are compared.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
from numbers import Integral, Real
from typing import Any

from usage import cache_usage_fields, summarize_cache_usage


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_COUNT_FIELDS = (
    "window_count", "medium_known_count", "medium_unknown_count", "medium_count",
    "zone_count", "zone_complete_count", "zone_unknown_count", "zone_all_medium_count",
    "run_count", "run_complete_count", "run_unknown_count", "run_all_medium_count",
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    return value


def _count(value: Any) -> int | None:
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def _sum_complete(values: list[Any], *, count: bool = False) -> int | float | None:
    parsed = [(_count(value) if count else _number(value)) for value in values]
    return sum(parsed) if parsed and all(value is not None for value in parsed) else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _identity(run: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]] | None:
    fields = [run.get(name) for name in ("dataset_identity", "forecast_origin", "forecast_model")]
    zones = run.get("zone_ids")
    if not all(isinstance(value, str) and value for value in fields):
        return None
    if not isinstance(zones, list) or not zones or not all(isinstance(zone, str) and zone for zone in zones):
        return None
    if len(set(zones)) != len(zones):
        return None
    return (*fields, tuple(sorted(zones)))


def _metadata(key: tuple[str, str, str, tuple[str, ...]]) -> dict[str, Any]:
    return dict(zip(("dataset_identity", "forecast_origin", "forecast_model", "zone_ids"), (*key[:3], list(key[3]))))


def _unknown_fields(run: dict[str, Any]) -> list[str]:
    missing = []
    if run.get("pairing_ready") is not True:
        missing.append("pairing_ready")
    if not isinstance(run.get("forecast_parameters"), dict) or not run["forecast_parameters"]:
        missing.append("forecast_parameters")
    else:
        try:
            _json(run["forecast_parameters"])
        except (TypeError, ValueError, RecursionError):
            missing.append("forecast_parameters")
    for field in ("baseline_signature", "hourly_signature"):
        if not isinstance(run.get(field), str) or not run[field]:
            missing.append(field)
    return missing


def _match_status(runs: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    evidence = [(run["run_id"], _unknown_fields(run)) for run in runs]
    missing = {run_id: fields for run_id, fields in evidence if fields}
    if missing:
        return "unknown", {"reason": "missing_pairing_evidence", "missing_fields": missing}
    mismatches = []
    for field in ("forecast_parameters", "baseline_signature", "hourly_signature"):
        if len({_json(run[field]) for run in runs}) > 1:
            mismatches.append(field)
    # The same numeric price in different known currencies is not the same baseline.
    units = {run.get("revenue_unit") for run in runs if run.get("revenue_unit") is not None}
    if len(units) > 1:
        mismatches.append("revenue_unit")
    return ("inconsistent", {"reason": "baseline_mismatch", "mismatched_fields": mismatches}) if mismatches else ("matched", {})


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [run.get("outcome_summary") or {} for run in runs]
    usages = [run.get("usage") or {} for run in runs]
    result: dict[str, Any] = {"matched_run_count": len(runs)}
    result.update({field: _sum_complete([row.get(field) for row in outcomes], count=True) for field in _COUNT_FIELDS})
    known_medium = result["medium_known_count"]
    result["medium_rate_pct"] = (
        100 * result["medium_count"] / known_medium
        if known_medium and result["medium_count"] is not None else None
    )
    result["agent_call_count"] = _sum_complete([row.get("agent_call_count") for row in usages], count=True)
    complete_flags = [
        row.get("token_usage_complete") is True and all(_count(row.get(field)) is not None for field in TOKEN_FIELDS)
        for row in usages
    ]
    complete = bool(usages) and all(complete_flags)
    result["token_usage_complete"] = complete
    result["incomplete_usage_run_count"] = sum(not flag for flag in complete_flags)
    for field in TOKEN_FIELDS:
        known = [_count(row.get(field)) for row in usages]
        result[f"known_{field}"] = sum(value for value in known if value is not None) if any(value is not None for value in known) else None
        result[field] = result[f"known_{field}"] if complete else None
        result[f"mean_{field}_per_call"] = result[field] / result["agent_call_count"] if complete and result["agent_call_count"] else None
    result.update(summarize_cache_usage(usages, aggregated=True))
    result["incomplete_cache_usage_run_count"] = sum(not cache_usage_fields({**row, "agent_call_count": row.get("agent_call_count")})["cache_usage_complete"] for row in usages)
    result["mean_cached_tokens_per_call"] = (
        result["cached_tokens"] / result["agent_call_count"]
        if result["cache_usage_complete"] and result["agent_call_count"] else None
    )
    for prefix in ("revenue", "baseline_revenue", "baseline_hourly_revenue"):
        valid = [row.get(f"{prefix}_complete") is True and _number(row.get(prefix)) is not None for row in outcomes]
        result[f"{prefix}_complete"] = bool(valid) and all(valid)
        result[prefix] = sum(row[prefix] for row in outcomes) if result[f"{prefix}_complete"] else None
        known = [_number(row.get(f"known_{prefix}")) for row in outcomes]
        result[f"known_{prefix}"] = sum(value for value in known if value is not None) if any(value is not None for value in known) else None
        for suffix in ("known_count", "unknown_count"):
            result[f"{prefix}_{suffix}"] = _sum_complete([row.get(f"{prefix}_{suffix}") for row in outcomes], count=True)
        count = result[f"{prefix}_known_count"]
        result[f"{prefix}_coverage_pct"] = (
            100 * count / result["window_count"]
            if count is not None and result["window_count"] else None
        )
    revenue, baseline = result["revenue"], result["baseline_revenue"]
    result["revenue_change"] = revenue - baseline if revenue is not None and baseline is not None else None
    result["revenue_change_pct"] = 100 * result["revenue_change"] / baseline if baseline and result["revenue_change"] is not None else None
    hourly_baseline = result["baseline_hourly_revenue"]
    result["revenue_hourly_comparable_count"] = _sum_complete(
        [row.get("revenue_hourly_comparable_count") for row in outcomes], count=True,
    )
    known_changes = [_number(row.get("known_revenue_change_vs_hourly")) for row in outcomes]
    result["known_revenue_change_vs_hourly"] = (
        sum(value for value in known_changes if value is not None)
        if any(value is not None for value in known_changes) else None
    )
    result["revenue_change_vs_hourly"] = (
        revenue - hourly_baseline if revenue is not None and hourly_baseline is not None else None
    )
    result["revenue_change_vs_hourly_pct"] = (
        100 * result["revenue_change_vs_hourly"] / hourly_baseline
        if hourly_baseline and result["revenue_change_vs_hourly"] is not None else None
    )
    return result


def _paired_summaries(pairs: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        left, right = by_id[pair["left_run_id"]], by_id[pair["right_run_id"]]
        key = (
            pair["left_batch"], pair["right_batch"], pair["dataset_identity"],
            pair["forecast_model"], pair["agent_mode"], left.get("revenue_unit"), right.get("revenue_unit"),
        )
        grouped[key].append(pair)
    summaries = []
    for key, records in sorted(grouped.items(), key=lambda item: _json(item[0])):
        left = _summarize([by_id[record["left_run_id"]] for record in records])
        right = _summarize([by_id[record["right_run_id"]] for record in records])
        delta = {}
        monetary_fields = ("revenue", "revenue_change", "baseline_hourly_revenue",
                           "revenue_change_vs_hourly", "revenue_change_vs_hourly_pct")
        for field in ("medium_count", "medium_rate_pct", "zone_all_medium_count", "run_all_medium_count", "agent_call_count", *TOKEN_FIELDS, "cached_tokens", "cache_hit_ratio", *monetary_fields):
            a, b = _number(left.get(field)), _number(right.get(field))
            delta[field] = b - a if a is not None and b is not None else None
        if left.get("medium_unknown_count") != 0 or right.get("medium_unknown_count") != 0:
            delta["medium_count"] = delta["medium_rate_pct"] = None
        for scope in ("zone", "run"):
            if left.get(f"{scope}_unknown_count") != 0 or right.get(f"{scope}_unknown_count") != 0:
                delta[f"{scope}_all_medium_count"] = None
        units_agree = key[5] == key[6]
        if not units_agree:
            delta.update({field: None for field in monetary_fields})
        first_left = by_id[records[0]["left_run_id"]]
        summaries.append({
            "left_batch": key[0], "right_batch": key[1], "dataset_identity": key[2],
            "dataset": first_left.get("dataset"), "forecast_model": key[3], "agent_mode": key[4],
            "revenue_unit": key[5] if units_agree else None,
            "left_revenue_unit": key[5], "right_revenue_unit": key[6],
            "pair_count": len(records), "left": left, "right": right, "delta": delta,
            "delta_direction": "right_minus_left",
        })
    return summaries


def build_comparisons(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build within-batch mode cohorts and pairwise cross-batch comparisons.

    ``run_id`` must be globally unique, including across batches.  ``pairs``
    contains accepted matches; ``audit`` contains every rejected candidate
    combination with its reason.  No pairing ever slices a run's tokens across
    windows or resolves duplicate candidates arbitrarily.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Every comparison run requires a nonempty run_id")
        if run_id in by_id:
            raise ValueError(f"Duplicate comparison run_id: {run_id}")
        by_id[run_id] = run
    modes: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    audit: list[dict[str, Any]] = []
    batches: set[str] = set()
    for run in runs:
        batch, mode = run.get("batch"), run.get("agent_mode")
        key = _identity(run)
        if isinstance(batch, str) and batch:
            batches.add(batch)
            if isinstance(mode, str) and mode:
                modes[batch].add(mode)
        if not isinstance(batch, str) or not batch or not isinstance(mode, str) or not mode or key is None:
            audit.append({"scope": "identity", "status": "unknown", "reason": "missing_or_invalid_run_identity", "batch": batch, "agent_mode": mode, "run_ids": [run["run_id"]]})
            continue
        candidates[batch][key][mode].append(run)
    common = []
    for batch in sorted(batches):
        for key, per_mode in sorted(candidates[batch].items()):
            base = {"scope": "within_batch", "batch": batch, **_metadata(key), "run_ids": sorted(run["run_id"] for group in per_mode.values() for run in group)}
            missing_modes = sorted(modes[batch] - per_mode.keys())
            ambiguous = {mode: sorted(run["run_id"] for run in group) for mode, group in per_mode.items() if len(group) != 1}
            if missing_modes or ambiguous:
                audit.append({**base, "status": "ambiguous" if ambiguous else "missing", "reason": "duplicate_candidates" if ambiguous else "missing_modes", "missing_modes": missing_modes, "ambiguous_modes": ambiguous})
                continue
            cohort = [per_mode[mode][0] for mode in sorted(modes[batch])]
            status, details = _match_status(cohort)
            if status == "matched":
                common.extend(run["run_id"] for run in cohort)
            else:
                audit.append({**base, "status": status, **details})
    pairs = []
    for left_batch, right_batch in combinations(sorted(batches), 2):
        for key in sorted(candidates[left_batch].keys() | candidates[right_batch].keys()):
            left_modes, right_modes = candidates[left_batch].get(key, {}), candidates[right_batch].get(key, {})
            for mode in sorted(left_modes.keys() | right_modes.keys()):
                left, right = left_modes.get(mode, []), right_modes.get(mode, [])
                base = {"scope": "cross_batch", "left_batch": left_batch, "right_batch": right_batch, "agent_mode": mode, **_metadata(key)}
                identifiers = {"left_run_ids": sorted(run["run_id"] for run in left), "right_run_ids": sorted(run["run_id"] for run in right)}
                if len(left) > 1 or len(right) > 1:
                    audit.append({**base, **identifiers, "status": "ambiguous", "reason": "duplicate_candidates"})
                elif not left or not right:
                    audit.append({**base, **identifiers, "status": "missing", "reason": "missing_matching_run"})
                else:
                    status, details = _match_status([left[0], right[0]])
                    if status == "matched":
                        pairs.append({**base, "status": status, "left_run_id": left[0]["run_id"], "right_run_id": right[0]["run_id"]})
                    else:
                        audit.append({**base, **identifiers, "status": status, **details})
    return {
        "common_run_ids": sorted(set(common)), "pairs": pairs, "audit": audit,
        "paired_summary": _paired_summaries(pairs, by_id),
    }
