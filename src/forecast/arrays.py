from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def checked_raw_prediction(frame: pd.DataFrame, times) -> np.ndarray:
    if len(frame) != len(times) or not pd.DatetimeIndex(frame["time"]).as_unit("ns").equals(pd.DatetimeIndex(times).as_unit("ns")):
        raise ValueError("Backend prediction timestamps do not match the requested horizon")
    raw = frame["raw_predicted_kwh"].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Backend returned non-finite load predictions")
    return raw


def align_vector(values: np.ndarray, target: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == target:
        return arr
    if len(arr) > target:
        return arr[:target]
    if len(arr) == 0:
        return np.zeros(target, dtype=np.float64)
    return np.concatenate([arr, np.repeat(arr[-1], target - len(arr))])


def require_fitted_backend(state: dict[str, Any], expected_backend: str) -> None:
    actual = str(state.get("backend") or "").strip().lower().replace("_", "-")
    expected = str(expected_backend).strip().lower().replace("_", "-")
    if actual != expected:
        raise ValueError(
            f"Reusable forecaster state belongs to {actual or 'an unknown backend'}, "
            f"not {expected}."
        )


def state_matrix(
    state: dict[str, Any],
    key: str,
    column_count: int,
    *,
    row_count: int | None = None,
) -> np.ndarray:
    if key not in state:
        raise ValueError(f"Reusable forecaster state is missing {key!r}.")
    values = np.asarray(state[key], dtype=np.float64)
    if values.size == 0:
        return np.empty((int(row_count or 0), int(column_count)), dtype=np.float64)
    if values.ndim == 1 and int(column_count) == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != int(column_count):
        raise ValueError(
            f"Reusable forecaster state {key!r} has shape {values.shape}; "
            f"expected (*, {int(column_count)})."
        )
    if row_count is not None and values.shape[0] != int(row_count):
        raise ValueError(
            f"Reusable forecaster state {key!r} has {values.shape[0]} rows; "
            f"expected {int(row_count)}."
        )
    return values
