"""Token accounting shared by Agent execution and output reporting."""
from __future__ import annotations

from typing import Any


def optional_int_usage(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def call_token_usage_complete(record: dict[str, Any]) -> bool:
    explicit = record.get("token_usage_complete")
    if explicit is not None:
        return bool(explicit)
    return all(
        optional_int_usage(record.get(field)) is not None
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def sum_token_usage(records: list[dict[str, Any]], field: str) -> int:
    return sum(usage_integer(record.get(field)) for record in records)


def summarize_agent_call_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "agent_invoked": bool(records),
        "agent_call_count": len(records),
        "prompt_tokens": sum_token_usage(records, "prompt_tokens"),
        "completion_tokens": sum_token_usage(records, "completion_tokens"),
        "total_tokens": sum_token_usage(records, "total_tokens"),
        "token_usage_complete": all(
            call_token_usage_complete(record)
            for record in records
        ),
    }


def usage_integer(value: Any) -> int:
    return optional_int_usage(value) or 0


def cumulative_global_agent_usage(
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "agent_round_count": len(rounds),
        "agent_invoked_round_count": sum(
            bool(item.get("agent_invoked")) for item in rounds
        ),
        "agent_invoked": any(bool(item.get("agent_invoked")) for item in rounds),
        "agent_invoked_zone_count": sum(
            usage_integer(item.get("agent_invoked_zone_count")) for item in rounds
        ),
        **aggregate_usage(rounds),
    }


def token_total_summary(usage: dict[str, Any]) -> dict[str, Any]:
    """Expose a compact final token-only total without timing fields."""
    return {
        "agent_call_count": usage_integer(usage.get("agent_call_count")),
        "prompt_tokens": usage_integer(usage.get("prompt_tokens")),
        "completion_tokens": usage_integer(usage.get("completion_tokens")),
        "total_tokens": usage_integer(usage.get("total_tokens")),
        "token_usage_complete": bool(usage.get("token_usage_complete")),
    }


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum recorded counts without claiming missing provider usage is complete."""
    return {
        **{
            field: sum(usage_integer(row.get(field)) for row in rows)
            for field in ("agent_call_count", "prompt_tokens", "completion_tokens", "total_tokens")
        },
        "token_usage_complete": all(bool(row.get("token_usage_complete")) for row in rows),
    }
