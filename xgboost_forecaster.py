"""Inference-only XGBoost entry point. Offline fitting lives in offline_models."""
from __future__ import annotations

import pandas as pd

DEFAULT_XGBOOST_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]


def xgboost_forecast(
    history: pd.DataFrame, validation: pd.DataFrame, full_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp, horizon_hours: int, *,
    model_reference: dict | None = None, diurnal_blend_alpha: float = 0.0,
    fitted_state: dict | None = None, raw_prediction_cache: dict | None = None,
) -> pd.DataFrame:
    from forecasting import forecast_load

    return forecast_load(
        history, validation, full_frame, forecast_origin, horizon_hours, backend="xgboost",
        model_reference=model_reference, diurnal_blend_alpha=diurnal_blend_alpha,
        fitted_state=fitted_state, raw_prediction_cache=raw_prediction_cache,
    )
