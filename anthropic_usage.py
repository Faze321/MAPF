"""Normalize native Anthropic Messages usage into the shared Agent metrics.

Anthropic's ``input_tokens`` counts uncached input only. Cache creation and cache
reads must be included exactly once in prompt tokens; only reads count as hits.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

from usage import cache_usage_fields


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


def anthropic_response_token_usage(response: Any) -> dict[str, Any]:
    """Read a Messages response without guessing missing cache counts as zero.

    A complete input count requires uncached input, cache-write input, and
    cache-read input. Partial records retain their native components as
    diagnostics but cannot establish full prompt/total tokens or a cache ratio.
    ``known_cached_tokens`` follows the shared normalizer's validated-subtotal
    semantics; when the denominator is unavailable its native evidence remains
    in ``cache_read_input_tokens`` instead.
    """
    usage = _field(response, "usage")
    inputs = {
        name: _count(_field(usage, name))
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    }
    prompt = sum(inputs.values()) if all(value is not None for value in inputs.values()) else None
    completion = _count(_field(usage, "output_tokens"))
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        **inputs,
    }
    result.update(cache_usage_fields({
        **result,
        "cached_tokens": inputs["cache_read_input_tokens"],
    }))
    return result
