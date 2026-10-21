"""Fixed-origin, recursive XGBoost load forecasting with reusable model state."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting import (
    DEFAULT_TIMESFM_EXOG_COLS,
    build_diurnal_profile,
    compute_forecast_metrics,
    deterministic_interval,
    disabled_calibration,
    estimate_interval_half_width,
    require_fitted_backend,
    serialize_diurnal_profile,
    state_diurnal_profile,
    state_vector,
)

DEFAULT_XGBOOST_LAGS = [1, 2, 3, 24]
ALLOWED_EXOG_COLS = frozenset([*DEFAULT_TIMESFM_EXOG_COLS, "hour", "occupancy"])


def load_xgb_regressor():
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is required for forecast_model: xgboost. "
            "Install the project dependencies or run: pip install 'xgboost>=2.0'."
        ) from exc
    return XGBRegressor


def xgboost_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    """Keep missing covariates as NaN; XGBoost handles them without fitted scaling."""
    values = frame.reindex(columns=exog_cols).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    times = pd.DatetimeIndex(pd.to_datetime(frame["time"]))
    calendar = np.column_stack([
        np.sin(2 * np.pi * times.hour / 24),
        np.cos(2 * np.pi * times.hour / 24),
        np.sin(2 * np.pi * times.dayofweek / 7),
        np.cos(2 * np.pi * times.dayofweek / 7),
    ])
    return np.hstack([values, calendar])


def recursive_xgboost_prediction(
    model: Any,
    frame: pd.DataFrame,
    known_load: np.ndarray,
    *,
    lags: list[int],
    exog_cols: list[str],
    bias: float,
    diurnal: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    values = list(np.asarray(known_load, dtype=np.float64))
    exog = xgboost_exog_matrix(frame, exog_cols)
    rows = []
    for idx, timestamp in enumerate(pd.to_datetime(frame["time"])):
        features = np.concatenate([[values[-lag] for lag in lags], exog[idx]])
        raw = float(model.predict(features.reshape(1, -1))[0])
        if not np.isfinite(raw):
            raise ValueError("XGBoost returned a non-finite load prediction.")
        corrected = max(0.0, raw - bias)
        point = max(0.0, (1 - alpha) * corrected + alpha * float(diurnal.loc[timestamp.hour]))
        rows.append((timestamp, raw, corrected, point))
        # Only our prediction becomes a future lag, never frame.actual_kwh.
        values.append(point)
    return pd.DataFrame(rows, columns=[
        "time", "raw_predicted_kwh", "bias_corrected_kwh", "predicted_kwh",
    ])


def xgboost_forecast(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    lags: list[int] | None = None,
    exog_cols: list[str] | None = None,
    n_estimators: int = 300,
    max_depth: int = 9,
    learning_rate: float = 0.1,
    colsample_bytree: float = 0.9,
    tree_method: str = "approx",
    n_jobs: int = 1,
    diurnal_blend_alpha: float = 0.0,
    seed: int = 42,
    fitted_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    lags = list(DEFAULT_XGBOOST_LAGS if lags is None else lags)
    exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if exog_cols is None else exog_cols)
    if not lags or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("xgboost_lags must contain positive integer hour offsets.")
    if len(set(lags)) != len(lags):
        raise ValueError("xgboost_lags cannot contain duplicate offsets.")
    if set(exog_cols) - ALLOWED_EXOG_COLS or len(set(exog_cols)) != len(exog_cols):
        raise ValueError("xgboost_exog_cols must contain unique known covariates, never actual_kwh or load.")
    for name, value in (("n_estimators", n_estimators), ("max_depth", max_depth), ("n_jobs", n_jobs)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"xgboost_{name} must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("xgboost_learning_rate must be positive and finite.")
    if not 0 < colsample_bytree <= 1:
        raise ValueError("xgboost_colsample_bytree must be in (0, 1].")
    if tree_method not in {"approx", "hist", "exact"}:
        raise ValueError("xgboost_tree_method must be approx, hist, or exact.")
    alpha = float(diurnal_blend_alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("xgboost_diurnal_blend_alpha must be in [0, 1].")
    if horizon_hours < 1:
        raise ValueError("XGBoost forecast horizon must be positive.")
    origin = pd.Timestamp(forecast_start)
    regressor = load_xgb_regressor()

    if fitted_state is None:
        history = history.sort_values("time").reset_index(drop=True)
        validation = validation.sort_values("time").reset_index(drop=True)
        past = pd.concat([history, validation], ignore_index=True)
        times = pd.DatetimeIndex(pd.to_datetime(past["time"]))
        expected = pd.date_range(end=origin - pd.Timedelta(hours=1), periods=len(past), freq="h")
        if not times.equals(expected):
            raise ValueError("XGBoost history and validation must be consecutive hourly rows ending before forecast_start.")
        past_load = pd.to_numeric(past["actual_kwh"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(past_load).all():
            raise ValueError("XGBoost requires finite historical and validation load values.")
        load = past_load[:len(history)]
        max_lag = max(lags)
        if len(load) <= max_lag:
            raise ValueError(f"XGBoost requires at least {max_lag + 1} history hours for xgboost_lags={lags}.")
        exog = xgboost_exog_matrix(history, exog_cols)
        x_train = np.column_stack([
            *[load[max_lag - lag:len(load) - lag] for lag in lags],
            exog[max_lag:],
        ])
        model = regressor(
            objective="reg:squarederror", tree_method=tree_method,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, colsample_bytree=colsample_bytree,
            n_jobs=n_jobs, random_state=seed,
        )
        model.fit(x_train, load[max_lag:])
        diurnal = build_diurnal_profile(history)
        bias, half_width = 0.0, np.nan
        calibration = disabled_calibration(interval_half_width=None)
        if not validation.empty:
            val = recursive_xgboost_prediction(
                model, validation, load, lags=lags, exog_cols=exog_cols,
                bias=0.0, diurnal=diurnal, alpha=alpha,
            )
            actual = past_load[len(history):]
            residual = val["raw_predicted_kwh"].to_numpy() - actual
            bias = float(np.mean(residual))
            half_width = estimate_interval_half_width(val["predicted_kwh"].to_numpy() - actual)
            calibration = {
                "enabled": True, "bias_mean": bias,
                "bias_max_abs": float(np.max(np.abs(residual))),
                "interval_half_width": half_width,
                "metrics": compute_forecast_metrics(val.assign(actual_kwh=actual)),
            }
        state = {
            "backend": "xgboost",
            "model_json": model.get_booster().save_raw(raw_format="json").decode("utf-8"),
            "lags": lags, "exog_cols": exog_cols,
            "rolling_load": past_load[-max_lag:].tolist(),
            "bias": bias,
            "interval_half_width": float(half_width) if np.isfinite(half_width) else None,
            "diurnal_by_hour": serialize_diurnal_profile(diurnal),
            "calibration": calibration,
        }
    else:
        require_fitted_backend(fitted_state, "xgboost")
        state = fitted_state
        if state.get("lags") != lags or state.get("exog_cols") != exog_cols:
            raise ValueError("Reusable XGBoost feature schema does not match lags/exog_cols.")
        if not isinstance(state.get("model_json"), str):
            raise ValueError("Reusable XGBoost state is missing model_json.")
        model = regressor(n_jobs=n_jobs)
        model.load_model(bytearray(state["model_json"], "utf-8"))
        diurnal = state_diurnal_profile(state)
        bias = float(state["bias"])
        half_width = float(state["interval_half_width"]) if state["interval_half_width"] is not None else np.nan
        calibration = dict(state["calibration"])

    rolling_load = state_vector(state, "rolling_load")
    if len(rolling_load) < max(lags) or not np.isfinite(rolling_load).all():
        raise ValueError("Reusable XGBoost state has insufficient or non-finite historical lags.")
    times = pd.date_range(origin, periods=horizon_hours, freq="h")
    future = full_frame.set_index("time").reindex(times).rename_axis("time").reset_index()
    result = recursive_xgboost_prediction(
        model, future, rolling_load, lags=lags, exog_cols=exog_cols,
        bias=bias, diurnal=diurnal, alpha=alpha,
    )
    result["q10_kwh"], result["q90_kwh"] = deterministic_interval(result["predicted_kwh"], half_width)
    result["q50_kwh"] = result["predicted_kwh"]
    result.attrs["calibration"] = calibration
    result.attrs["reusable_forecaster_state"] = state
    return result
