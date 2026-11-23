from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.config.run import VALIDATION_BIAS_DAYS, validation_bias_days
from src.data.adapters.registry import load_canonical_dataset
from src.data.canonical import CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN, DatasetSpec


def default_forecast_start(spec: DatasetSpec, horizon_days: int) -> str:
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    dataset = load_canonical_dataset(spec)
    end = dataset.timeseries[CANONICAL_TIME_COLUMN].max()
    return (end - pd.Timedelta(hours=horizon_days * 24 - 1)).isoformat()


def evenly_spaced_forecast_starts(
    timeseries: pd.DataFrame, *, count: int, history_days: int,
    horizon_days: int, zone_ids: Iterable[str] | None = None, validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
) -> list[str]:
    """Choose interior daily origins with complete hourly windows for every site."""
    if count < 1 or history_days < 1 or horizon_days < 1:
        raise ValueError("Origin count, history and horizon must be positive")
    frame = timeseries[[CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN]].copy()
    frame[CANONICAL_ZONE_COLUMN] = frame[CANONICAL_ZONE_COLUMN].astype(str)
    if zone_ids is not None:
        requested = {str(zone) for zone in zone_ids}
        missing = requested - set(frame[CANONICAL_ZONE_COLUMN])
        if missing:
            raise ValueError(f"Cannot select forecast origins for missing zones: {sorted(missing)}")
        frame = frame[frame[CANONICAL_ZONE_COLUMN].isin(requested)]
    if frame.empty:
        raise ValueError("Cannot select forecast origins from an empty dataset/zone selection")
    bounds = frame.groupby(CANONICAL_ZONE_COLUMN)[CANONICAL_TIME_COLUMN].agg(["min", "max"])
    past_hours = (history_days + validation_bias_days(validation_bias, validation_days)) * 24
    future_hours = horizon_days * 24
    earliest = (bounds["min"].max() + pd.Timedelta(hours=past_hours)).ceil("D")
    latest = (bounds["max"].min() - pd.Timedelta(hours=future_hours - 1)).floor("D")
    candidates = pd.date_range(earliest, latest, freq="D")
    complete = np.ones(len(candidates), dtype=bool)
    window_starts = candidates - pd.Timedelta(hours=past_hours)
    window_ends = candidates + pd.Timedelta(hours=future_hours)
    for zone, group in frame.groupby(CANONICAL_ZONE_COLUMN):
        timestamps = pd.DatetimeIndex(group[CANONICAL_TIME_COLUMN]).sort_values()
        if timestamps.has_duplicates or not timestamps.equals(timestamps.floor("h")):
            raise ValueError(f"Evenly spaced origins require unique whole-hour timestamps: {zone}")
        observed = timestamps.searchsorted(window_ends) - timestamps.searchsorted(window_starts)
        complete &= observed == past_hours + future_hours
    candidates = candidates[complete]
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} complete daily forecast origins are available for {count} requested; "
            "check the selected sites' common coverage and history/validation/horizon lengths"
        )
    # Interior quantiles avoid concentrating experiments at either dataset boundary.
    indices = [index * len(candidates) // (count + 1) for index in range(1, count + 1)]
    return [candidates[index].isoformat() for index in indices]
