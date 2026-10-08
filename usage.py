"""Token accounting shared by Agent execution and output reporting."""
from __future__ import annotations

from numbers import Integral
from typing import Any


CACHE_USAGE_FIELDS = (
    "cached_tokens", "known_cached_tokens", "cache_reported_call_count",
    "cache_usage_complete", "cache_hit_ratio",
)


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


def cache_usage_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize reported cache usage without treating unavailable counts as zero.

    A call defaults to one invocation; aggregate rows supply agent_call_count.
    A ratio is a pooled fraction of input tokens, not a request hit rate.
    """
    count = _nonnegative_integer(value.get("agent_call_count", 1))
    prompt = _nonnegative_integer(value.get("prompt_tokens"))
    cached = _nonnegative_integer(value.get("cached_tokens"))
    known = _nonnegative_integer(value.get("known_cached_tokens"))
    reported = _nonnegative_integer(value.get("cache_reported_call_count"))
    if prompt is None or (cached is not None and cached > prompt):
        cached = None
    if prompt is None or (known is not None and known > prompt):
        known = None
    if "cache_reported_call_count" in value and (
        reported is None or count is None or reported > count
    ):
        reported = 0
    if count == 0:
        return {"cached_tokens": 0, "known_cached_tokens": 0,
                "cache_reported_call_count": 0, "cache_usage_complete": True,
                "cache_hit_ratio": None}
    permitted = value.get("cache_usage_complete") is None or value.get("cache_usage_complete") is True
    if reported is None:
        reported = count if cached is not None and permitted and count is not None else 0
    complete = cached is not None and count is not None and permitted and reported == count
    known = cached if complete or known is None else known
    if known is None:
        reported = 0
    return {
        "cached_tokens": cached if complete else None,
        "known_cached_tokens": known if known is not None else 0,
        "cache_reported_call_count": reported,
        "cache_usage_complete": complete,
        "cache_hit_ratio": cached / prompt if complete and prompt else None,
    }


def summarize_cache_usage(
    records: list[dict[str, Any]], *, aggregated: bool = False,
) -> dict[str, Any]:
    """Pool cache counts across calls, or across already aggregated usage rows."""
    normalized_records = [
        {
            **record,
            "agent_call_count": record.get(
                "agent_call_count", 0 if record.get("agent_invoked") is False else None,
            ) if aggregated else 1,
        }
        for record in records
    ]
    rows = [cache_usage_fields(record) for record in normalized_records]
    complete = all(row["cache_usage_complete"] for row in rows)
    known = sum(row["known_cached_tokens"] for row in rows)
    prompts = [
        0 if _nonnegative_integer(record["agent_call_count"]) == 0
        else _nonnegative_integer(record.get("prompt_tokens"))
        for record in normalized_records
    ]
    prompt_total = sum(prompts) if all(value is not None for value in prompts) else None
    return {
        "cached_tokens": known if complete else None,
        "known_cached_tokens": known,
        "cache_reported_call_count": sum(row["cache_reported_call_count"] for row in rows),
        "cache_usage_complete": complete,
        "cache_hit_ratio": known / prompt_total if complete and prompt_total else None,
    }


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
        **summarize_cache_usage(records),
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
        **cache_usage_fields(usage),
    }


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum recorded counts without claiming missing provider usage is complete."""
    return {
        **{
            field: sum(usage_integer(row.get(field)) for row in rows)
            for field in ("agent_call_count", "prompt_tokens", "completion_tokens", "total_tokens")
        },
        "token_usage_complete": all(bool(row.get("token_usage_complete")) for row in rows),
        **summarize_cache_usage(rows, aggregated=True),
    }
