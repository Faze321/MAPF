from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if not len(actual):
        return dict.fromkeys(("MAE", "RMSE", "MAPE_pct", "RAE", "WAPE_pct"))
    error = np.abs(actual - predicted)
    magnitude = np.abs(actual)
    ratios = error[magnitude > 1e-9] / magnitude[magnitude > 1e-9]
    rae_denominator = float(np.abs(actual - actual.mean()).sum())
    wape_denominator = float(magnitude.sum())
    return {
        "MAE": float(error.mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_pct": float(np.nanmean(ratios) * 100) if ratios.size else None,
        "RAE": float(error.sum() / rae_denominator) if rae_denominator > 1e-9 else None,
        "WAPE_pct": float(error.sum() / wape_denominator * 100) if wape_denominator > 1e-9 else None,
    }


def evaluate_forecaster(hourly: pd.DataFrame) -> dict[str, Any]:
    """Pool only finite observation pairs, retaining coverage and precision."""
    values = hourly[["actual_kwh", "predicted_kwh"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    actual, predicted = values.loc[valid].to_numpy(dtype=float).T
    stats = _metrics(actual, predicted)
    return {
        "n_total": len(values), "n": len(actual), "n_excluded": int((~valid).sum()),
        "n_mape": int((np.abs(actual) > 1e-9).sum()),
        **{key: stats[key] for key in ("MAE", "RMSE", "RAE", "MAPE_pct", "WAPE_pct")},
    }


def compute_forecast_metrics(hourly: pd.DataFrame) -> dict[str, float | None]:
    values = hourly[["actual_kwh", "predicted_kwh"]].dropna()
    actual, predicted = values.to_numpy(dtype=float).T
    stats = _metrics(actual, predicted)
    if len(actual) and stats["MAPE_pct"] is None:
        stats["MAPE_pct"] = float("nan")
    return {**{key: round(value, 4) if value is not None else None for key, value in stats.items()}, "n": len(actual)}
