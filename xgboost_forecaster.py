"""Inference-only XGBoost entry point. Offline fitting lives in offline_models."""
from __future__ import annotations

import pandas as pd
from config import VALIDATION_BIAS_DAYS

DEFAULT_XGBOOST_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]


def xgboost_forecast(
    history: pd.DataFrame, full_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp, horizon_hours: int, *,
    model_reference: dict | None = None, diurnal_blend_alpha: float = 0.0,
    fitted_state: dict | None = None, raw_prediction_cache: dict | None = None,
    validation_bias: bool = False, validation_days: int = VALIDATION_BIAS_DAYS,
    validation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from forecasting import forecast_load

    return forecast_load(
        history, full_frame, forecast_origin, horizon_hours, backend="xgboost",
        validation_bias=validation_bias, validation_days=validation_days, validation=validation,
        model_reference=model_reference, diurnal_blend_alpha=diurnal_blend_alpha,
        fitted_state=fitted_state, raw_prediction_cache=raw_prediction_cache,
    )
