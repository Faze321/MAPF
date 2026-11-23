from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd


_END_OF_DAY_PATTERN = re.compile(
    r"^\s*(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})[ T]"
    r"24:00(?::00(?:\.0+)?)?(?P<timezone>Z|[+-]\d{2}:?\d{2})?\s*$"
)


def parse_datetime_24h(
    value: Any,
    *,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Timestamp:
    """Parse timestamps while treating an exact 24:00 as next-day midnight."""

    if errors not in {"raise", "coerce"}:
        raise ValueError(f"Unsupported errors policy: {errors}")
    if value is None or value is pd.NaT:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        match = _END_OF_DAY_PATTERN.match(value)
        if match:
            try:
                next_day = pd.Timestamp(match.group("date")) + pd.Timedelta(days=1)
                timezone = match.group("timezone") or ""
                return pd.Timestamp(
                    f"{next_day.strftime('%Y-%m-%d')} 00:00:00{timezone}"
                )
            except (TypeError, ValueError) as exc:
                if errors == "coerce":
                    return pd.NaT
                raise exc

    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        if errors == "coerce":
            return pd.NaT
        raise


def normalize_datetime_series_24h(
    values: pd.Series,
    *,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.Series:
    """Vector-friendly wrapper that preserves a Series index and name."""

    return values.map(lambda value: parse_datetime_24h(value, errors=errors))
