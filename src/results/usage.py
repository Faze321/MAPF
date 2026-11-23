from __future__ import annotations

import json
import re
from typing import Any

from src.agents.usage import (
    CACHE_USAGE_FIELDS,
    CONTROL_AGENT_USAGE_KEY,
    aggregate_usage,
    cache_usage_fields,
    summarize_cache_usage,
    usage_integer as reporting_usage_integer,
)


AGENT_ATTEMPT_USAGE_COLUMNS = [
    "record_scope",
    "backend",
    "agent_mode",
    "zone_id",
    "category",
    "attempt",
    "proposal_phase",
    "triggered_by_attempt",
    "agent_invoked",
    "agent_invoked_zone_count",
    "agent_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_complete",
    *CACHE_USAGE_FIELDS,
    "agent_stages",
    "agent_names",
    "agent_call_usage_json",
]

AGENT_STEP_TOKEN_USAGE_COLUMNS = [
    "backend",
    "agent_mode",
    "zone_id",
    "category",
    "attempt",
    "proposal_phase",
    "triggered_by_attempt",
    "step_index",
    "discussion_round",
    "stage",
    "agent",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage_complete",
    *CACHE_USAGE_FIELDS,
    "provider_attempt_count",
]


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
    result = {
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
    # Prefer the authoritative aggregate; older records may expose only calls.
    if any(field in raw for field in CACHE_USAGE_FIELDS):
        result.update(cache_usage_fields({**result, **raw}))
    else:
        result.update(cache_usage_fields({**result, **summarize_cache_usage(calls)}))
    return result


def normalized_zone_cumulative_usage(
    report: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    value = report.get("agent_cumulative_usage")
    if isinstance(value, dict):
        return token_only_cumulative_usage(value)
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
            **aggregate_usage(usage_rows),
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
        **summarize_cache_usage(calls),
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
                **aggregate_usage(usage_rows),
                "invoked_zone_ids": [item.get("zone_id") for item in invoked],
            }
        )
    return rounds


def agent_attempt_usage_row(
    *,
    record_scope: str,
    backend: str | None,
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
        "backend": backend,
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
        "prompt_tokens": reporting_usage_integer(usage.get("prompt_tokens")),
        "completion_tokens": reporting_usage_integer(
            usage.get("completion_tokens")
        ),
        "total_tokens": reporting_usage_integer(usage.get("total_tokens")),
        "token_usage_complete": bool(usage.get("token_usage_complete")),
        **cache_usage_fields(usage),
        "agent_stages": "|".join(stages),
        "agent_names": "|".join(names),
        "agent_call_usage_json": json.dumps(calls, ensure_ascii=False),
    }


def agent_step_token_usage_row(
    *,
    backend: str | None,
    agent_mode: str | None,
    zone_id: Any,
    category: Any,
    attempt: int,
    proposal_phase: Any,
    triggered_by_attempt: Any,
    step_index: int,
    call: dict[str, Any],
) -> dict[str, Any]:
    stage = str(call.get("stage") or "")
    match = re.search(r"(?:^|\.)round_(\d+)(?:\.|$)", stage)
    return {
        "backend": backend,
        "agent_mode": agent_mode,
        "zone_id": zone_id,
        "category": category,
        "attempt": attempt,
        "proposal_phase": proposal_phase,
        "triggered_by_attempt": triggered_by_attempt,
        "step_index": step_index,
        "discussion_round": int(match.group(1)) if match else None,
        "stage": stage,
        "agent": call.get("agent"),
        "prompt_tokens": reporting_usage_integer(call.get("prompt_tokens")),
        "completion_tokens": reporting_usage_integer(
            call.get("completion_tokens")
        ),
        "total_tokens": reporting_usage_integer(call.get("total_tokens")),
        "token_usage_complete": bool(call.get("token_usage_complete")),
        **cache_usage_fields(call),
        "provider_attempt_count": reporting_usage_integer(
            call.get("provider_attempt_count", 1)
        ),
    }


def token_only_call_usage(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "stage",
        "agent",
        "zone_id",
        "attempt",
        "proposal_phase",
        "triggered_by_attempt",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_usage_complete",
        "provider_attempt_count",
    )
    return {**{key: value.get(key) for key in allowed if key in value},
            **cache_usage_fields(value)}


def token_only_round_usage(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "attempt",
        "proposal_phase",
        "triggered_by_attempt",
        "agent_invoked",
        "agent_invoked_zone_count",
        "agent_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_usage_complete",
        "invoked_zone_ids",
    )
    return {**{key: value.get(key) for key in allowed if key in value},
            **cache_usage_fields(value)}


def token_only_cumulative_usage(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    allowed = (
        "agent_round_count",
        "agent_attempt_count",
        "agent_invoked_round_count",
        "agent_invoked_attempt_count",
        "agent_invoked",
        "agent_invoked_zone_count",
        "agent_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_usage_complete",
    )
    return {**{key: value.get(key) for key in allowed if key in value},
            **cache_usage_fields(value)}
