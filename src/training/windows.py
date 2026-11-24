from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.canonical import valid_energy_prices

DEFAULT_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
KNOWN_EXOG_COLS = frozenset([*DEFAULT_EXOG_COLS, "hour", "occupancy"])
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def complete_target_indices(frame: pd.DataFrame, context_hours: int,
                            horizon_hours: int = 1, stride_hours: int = 1) -> np.ndarray:
    """First target positions with continuous [t-H,t+P), strided per valid segment."""
    context_hours = _positive_integer(context_hours, "context_hours")
    horizon_hours = _positive_integer(horizon_hours, "horizon_hours")
    stride_hours = _positive_integer(stride_hours, "stride_hours")
    if len(frame) < context_hours + horizon_hours:
        return np.empty(0, dtype=np.int64)
    times = pd.to_datetime(frame["time"], errors="raise")
    new_segment = (~times.diff().eq(pd.Timedelta(hours=1))).to_numpy()
    positions = np.arange(len(frame), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(new_segment, positions, 0))
    targets = np.arange(context_hours, len(frame) - horizon_hours + 1, dtype=np.int64)
    # Using the last target's segment start also excludes gaps inside the labels.
    starts = segment_starts[targets + horizon_hours - 1]
    complete = targets - starts >= context_hours
    aligned = (targets - starts - context_hours) % stride_hours == 0
    return targets[complete & aligned]


def checked_training_frame(frame: pd.DataFrame, *, actuals: bool) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["time"] = pd.to_datetime(result["time"], errors="raise").dt.as_unit("ns")
    if result["time"].isna().any() or not result["time"].is_monotonic_increasing or result["time"].duplicated().any():
        raise ValueError("Model frames must contain unique sorted valid timestamps")
    if actuals:
        result["actual_kwh"] = pd.to_numeric(result["actual_kwh"], errors="coerce")
        if not np.isfinite(result["actual_kwh"].to_numpy(dtype=float)).all():
            raise ValueError("Training/context load must be finite; remove invalid observations before model use")
    return result


def raw_covariates(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arrays = [pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
              for name in columns]
    result = np.column_stack(arrays) if arrays else np.empty((len(frame), 0), dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def canonical_zone_frame(dataset: Any, zone_id: str) -> pd.DataFrame:
    source = dataset.timeseries
    frame = source[source["zone_id"].astype(str) == str(zone_id)].copy()
    names = {"timestamp": "time", "load_kwh": "actual_kwh", "energy_price": "e_price",
             "service_price": "s_price", "temperature": "T", "humidity": "U", "rain": "nRAIN"}
    frame = frame.rename(columns={key: value for key, value in names.items() if value not in frame.columns})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["hour"] = frame["time"].dt.hour.astype(float)
    frame["is_weekend"] = frame["time"].dt.dayofweek.isin([5, 6]).astype(float)
    frame["temp_price_idx"] = (pd.to_numeric(frame["T"], errors="coerce")
                               * pd.to_numeric(frame["e_price"], errors="coerce")) if "T" in frame else np.nan
    columns = ["time", "actual_kwh", "e_price", "s_price", "occupancy", "T", "U", "nRAIN",
               "hour", "is_weekend", "temp_price_idx"]
    return frame.reindex(columns=columns).sort_values("time").reset_index(drop=True)


def inspect_training_window(
    frame: pd.DataFrame, *, history_days: int = 35, horizon_days: int = 1,
    train_ratio: float = 0.8, validation_ratio: float = 0.2,
    min_history_days: int = 45, min_samples: int = 512, sample_stride_hours: int = 1,
    allow_zero_energy_price: bool = False,
) -> dict[str, Any]:
    """Chronologically split supervised windows, purging overlapping target labels."""
    for key, value in (("history_days", history_days), ("horizon_days", horizon_days),
                       ("min_history_days", min_history_days), ("min_samples", min_samples),
                       ("sample_stride_hours", sample_stride_hours)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (not 0 < train_ratio < 1 or not 0 < validation_ratio < 1
            or not np.isclose(train_ratio + validation_ratio, 1, rtol=0, atol=1e-9)):
        raise ValueError("train_ratio and validation_ratio must be positive and sum to 1")
    input_hours, output_hours = history_days * 24, horizon_days * 24
    result: dict[str, Any] = {"eligible": False, "reason": None, "observed_hours": len(frame),
                              "input_hours": input_hours, "output_hours": output_hours,
                              "sample_stride_hours": sample_stride_hours,
                              "train_ratio": train_ratio, "validation_ratio": validation_ratio}
    if frame.empty:
        return {**result, "reason": "no_observations"}
    clean = frame.copy()
    clean["time"] = pd.to_datetime(clean["time"], errors="coerce")
    if clean["time"].dropna().duplicated().any():
        return {**result, "reason": "duplicate_timestamps"}
    for column in ("actual_kwh", "e_price"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    valid = (clean["time"].notna() & (clean["time"] == clean["time"].dt.floor("h"))
             & np.isfinite(clean["actual_kwh"]) & (clean["actual_kwh"] >= 0)
             & valid_energy_prices(clean["e_price"], allow_zero_energy_price))
    clean = clean[valid].sort_values("time").reset_index(drop=True)
    result.update(valid_hours=len(clean), invalid_hours=int((~valid).sum()))
    if len(clean) < min_history_days * 24:
        return {**result, "reason": "insufficient_valid_history"}
    for column, reason in (("actual_kwh", "constant_load"), ("e_price", "constant_energy_price")):
        if np.isclose(clean[column].min(), clean[column].max(), rtol=1e-9, atol=1e-12):
            return {**result, "reason": reason}
    candidates = complete_target_indices(clean, input_hours, output_hours, sample_stride_hours)
    count = len(candidates)
    result.update(sample_count=count, full_samples=count)
    if count < max(2, min_samples):
        return {**result, "reason": "insufficient_complete_samples"}
    split_index = max(1, min(count - 1, int(np.floor(count * train_ratio))))
    validation_targets = candidates[split_index:]
    validation_start = clean.iloc[validation_targets[0]]["time"]
    validation_end = clean.iloc[validation_targets[-1] + output_hours - 1]["time"] + pd.Timedelta(hours=1)
    train_candidates = candidates[:split_index]
    train_targets = train_candidates[train_candidates + output_hours <= validation_targets[0]]
    train_count, validation_count = len(train_targets), len(validation_targets)
    result.update(validation_start=validation_start, validation_end=validation_end,
                  train_samples=train_count, validation_samples=validation_count,
                  purged_samples=len(train_candidates) - train_count,
                  actual_train_ratio=train_count / (train_count + validation_count),
                  actual_validation_ratio=validation_count / (train_count + validation_count),
                  split_method="chronological_window_ratio_purged_targets",
                  fit_start=clean.iloc[0]["time"], fit_end_exclusive=validation_start)
    if not train_count:
        return {**result, "reason": "insufficient_training_samples_after_purge"}
    return {**result, "eligible": True, "frame": clean}


def assess_dataset_training(dataset: Any, training_config: Any, *, zone_ids=None,
                            history_days: int = 35, horizon_days: int = 1) -> dict[str, Any]:
    """Read-only eligibility for selection; no directories, models or cache writes."""
    available = set(dataset.timeseries["zone_id"].astype(str))
    selected = sorted(available) if zone_ids is None else list(dict.fromkeys(str(zone) for zone in zone_ids))
    excluded = {str(zone): reason for zone, reason in (dataset.feature_manifest.get("excluded_sites") or {}).items()}
    statuses = {}
    for zone in selected:
        if zone in excluded or zone not in available:
            statuses[zone] = {"eligible": False, "reason": "adapter_excluded" if zone in excluded else "unknown_zone"}
            continue
        assessment = inspect_training_window(canonical_zone_frame(dataset, zone),
            allow_zero_energy_price=dataset.feature_manifest.get("allow_zero_energy_price", False),
            history_days=history_days, horizon_days=horizon_days,
            train_ratio=training_config.train_ratio, validation_ratio=training_config.validation_ratio,
            min_history_days=training_config.min_history_days, min_samples=training_config.min_samples,
            sample_stride_hours=training_config.sample_stride_hours)
        statuses[zone] = {key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                          for key, value in assessment.items() if key != "frame"}
    return {"eligible_zone_ids": [zone for zone in selected if statuses[zone]["eligible"]], "zones": statuses}
