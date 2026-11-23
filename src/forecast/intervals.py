from __future__ import annotations

import numpy as np


def estimate_interval_half_width(residual: np.ndarray) -> float:
    values = np.abs(np.asarray(residual, dtype=np.float64))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.nanquantile(values, 0.9))


def deterministic_interval(point: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    point_arr = np.asarray(point, dtype=np.float64)
    if not np.isfinite(half_width):
        return np.full(len(point_arr), np.nan), np.full(len(point_arr), np.nan)
    if half_width <= 0:
        return point_arr.copy(), point_arr.copy()
    q10 = np.clip(point_arr - half_width, 0, None)
    q90 = point_arr + half_width
    return np.minimum(q10, point_arr), np.maximum(q90, point_arr)
