"""Train-once XGBoost with raw recursive inference and shared hourly calibration."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from forecasting import DEFAULT_TIMESFM_EXOG_COLS

DEFAULT_XGBOOST_LAGS = [1, 2, 3, 24]
ALLOWED_EXOG_COLS = frozenset([*DEFAULT_TIMESFM_EXOG_COLS, "hour", "occupancy"])


def load_xgb_regressor():
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is required for backend: xgboost. "
            "Install the project dependencies or run: pip install 'xgboost>=2.0'."
        ) from exc
    return XGBRegressor


def xgboost_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    """XGBoost handles missing covariates without fitting a scaler."""
    values = frame.reindex(columns=exog_cols).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    times = pd.DatetimeIndex(pd.to_datetime(frame["time"]))
    calendar = np.column_stack([
        np.sin(2 * np.pi * times.hour / 24), np.cos(2 * np.pi * times.hour / 24),
        np.sin(2 * np.pi * times.dayofweek / 7), np.cos(2 * np.pi * times.dayofweek / 7),
    ])
    return np.hstack([values, calendar])


def recursive_xgboost_prediction(
    model: Any, frame: pd.DataFrame, known_load: np.ndarray,
    *, lags: list[int], exog_cols: list[str],
) -> pd.DataFrame:
    values = list(np.asarray(known_load, dtype=np.float64))
    if len(values) < max(lags):
        raise ValueError("XGBoost context has insufficient historical lags.")
    exog = xgboost_exog_matrix(frame, exog_cols)
    rows = []
    for idx, timestamp in enumerate(pd.to_datetime(frame["time"])):
        features = np.concatenate([[values[-lag] for lag in lags], exog[idx]])
        raw = float(model.predict(features.reshape(1, -1))[0])
        if not np.isfinite(raw):
            raise ValueError("XGBoost returned a non-finite load prediction.")
        raw = max(0.0, raw)
        rows.append((timestamp, raw))
        # Recursive inputs use model outputs; blending and bias happen once outside.
        values.append(raw)
    return pd.DataFrame(rows, columns=["time", "raw_predicted_kwh"])


def prepare_xgboost_backend(
    history: pd.DataFrame, parameters: dict[str, Any],
    *, fitted_state: dict[str, Any] | None = None,
) -> tuple[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], dict[str, Any]]:
    lags = parameters.get("xgboost_lags")
    lags = list(DEFAULT_XGBOOST_LAGS if lags is None else lags)
    exog_cols = parameters.get("xgboost_exog_cols")
    exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if exog_cols is None else exog_cols)
    if not lags or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("xgboost_lags must contain positive integer hour offsets.")
    if len(set(lags)) != len(lags):
        raise ValueError("xgboost_lags cannot contain duplicate offsets.")
    if set(exog_cols) - ALLOWED_EXOG_COLS or len(set(exog_cols)) != len(exog_cols):
        raise ValueError("xgboost_exog_cols must contain unique known covariates, never actual_kwh or load.")
    model_parameters = {name: parameters["xgboost_" + name] for name in (
        "n_estimators", "max_depth", "learning_rate", "colsample_bytree", "tree_method", "n_jobs",
    )}
    model_parameters["random_state"] = parameters["xgboost_seed"]
    for name in ("n_estimators", "max_depth", "n_jobs"):
        value = model_parameters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"xgboost_{name} must be a positive integer.")
    if not np.isfinite(model_parameters["learning_rate"]) or model_parameters["learning_rate"] <= 0:
        raise ValueError("xgboost_learning_rate must be positive and finite.")
    if not 0 < model_parameters["colsample_bytree"] <= 1:
        raise ValueError("xgboost_colsample_bytree must be in (0, 1].")
    if model_parameters["tree_method"] not in {"approx", "hist", "exact"}:
        raise ValueError("xgboost_tree_method must be approx, hist, or exact.")
    regressor = load_xgb_regressor()
    if fitted_state is None:
        load = history["actual_kwh"].to_numpy(dtype=np.float64)
        max_lag = max(lags)
        if len(load) <= max_lag:
            raise ValueError(f"XGBoost requires at least {max_lag + 1} history hours for xgboost_lags={lags}.")
        exog = xgboost_exog_matrix(history, exog_cols)
        x_train = np.column_stack([
            *[load[max_lag - lag:len(load) - lag] for lag in lags], exog[max_lag:],
        ])
        model = regressor(objective="reg:squarederror", **model_parameters)
        model.fit(x_train, load[max_lag:])
        state = {
            "model_json": model.get_booster().save_raw(raw_format="json").decode("utf-8"),
            "lags": lags, "exog_cols": exog_cols,
        }
    else:
        if fitted_state.get("lags") != lags or fitted_state.get("exog_cols") != exog_cols:
            raise ValueError("Reusable XGBoost feature schema does not match lags/exog_cols.")
        if not isinstance(fitted_state.get("model_json"), str):
            raise ValueError("Reusable XGBoost state is missing model_json.")
        model = regressor(n_jobs=model_parameters["n_jobs"])
        model.load_model(bytearray(fitted_state["model_json"], "utf-8"))
        state = {key: fitted_state[key] for key in ("model_json", "lags", "exog_cols")}

    def predict(context: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        return recursive_xgboost_prediction(
            model, future, context["actual_kwh"].to_numpy(dtype=np.float64), lags=lags, exog_cols=exog_cols,
        )

    return predict, state


def xgboost_forecast(
    history: pd.DataFrame, validation: pd.DataFrame, full_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp, horizon_hours: int, *,
    lags: list[int] | None = None, exog_cols: list[str] | None = None,
    n_estimators: int = 300, max_depth: int = 9, learning_rate: float = 0.1,
    colsample_bytree: float = 0.9, tree_method: str = "approx", n_jobs: int = 1,
    diurnal_blend_alpha: float = 0.0, seed: int = 42,
    fitted_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convenience entry point into the shared calibration protocol."""
    from forecasting import forecast_load

    return forecast_load(
        history, validation, full_frame, forecast_origin, horizon_hours,
        backend="xgboost", diurnal_blend_alpha=diurnal_blend_alpha,
        xgboost_lags=lags, xgboost_exog_cols=exog_cols,
        xgboost_n_estimators=n_estimators, xgboost_max_depth=max_depth,
        xgboost_learning_rate=learning_rate, xgboost_colsample_bytree=colsample_bytree,
        xgboost_tree_method=tree_method, xgboost_n_jobs=n_jobs, xgboost_seed=seed,
        fitted_state=fitted_state,
    )
