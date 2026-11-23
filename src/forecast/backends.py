from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pandas as pd

from src.config.training import TRAINABLE_BACKENDS, base_backend
from src.forecast.ar import estimate_energy_price_response, run_ar_prediction
from src.forecast.foundation import (
    DEFAULT_TIMESFM_EXOG_COLS,
    KNOWN_FUTURE_COVARIATES,
    build_exog_matrix,
    load_chronos_model,
    load_timesfm_model,
    run_chronos_prediction,
    run_timesfm_prediction,
)


def prepare_forecast_backend(
    backend: str, history: pd.DataFrame, parameters: dict[str, Any], max_horizon: int,
    *, fitted_state: dict[str, Any] | None = None,
    model_reference: dict[str, Any] | None = None,
) -> tuple[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], dict[str, Any]]:
    """Prepare one frozen model and return an uncalibrated inference callable."""
    p = parameters
    state = fitted_state or {}
    prefix = base_backend(backend).lower() + "_"
    cols = p.get(prefix + "exog_cols")
    exog_cols = list(DEFAULT_TIMESFM_EXOG_COLS if cols is None else cols)
    if set(exog_cols) - KNOWN_FUTURE_COVARIATES:
        raise ValueError("Forecaster covariates must be known future features, never load or evaluation fields")

    reference = model_reference or state.get("model_reference")
    if reference is not None:
        if backend not in TRAINABLE_BACKENDS:
            raise ValueError(f"Pretrained backend {backend} cannot use an offline model reference; select its LoRA forecaster")
        from src.training.weights import load_cached_predictor

        if not isinstance(reference, dict) or reference.get("backend") != backend:
            raise ValueError(f"The prepared model reference does not match backend {backend}")
        return load_cached_predictor(reference), {
            "model_reference": deepcopy(reference), "model_id": reference["model_id"],
        }

    if backend == "timesfm":
        context_hours = p["timesfm_context_hours"]
        model = load_timesfm_model(p["timesfm_repo"], context_hours, max(32, max_horizon))
        def predict(context, future):
            raw, q10, q90 = run_timesfm_prediction(
                model, context["actual_kwh"].to_numpy(dtype=float), build_exog_matrix(context, exog_cols),
                build_exog_matrix(future, exog_cols), len(future), context_hours, exog_cols,
            )
            return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw,
                                 "q10_kwh": q10, "q90_kwh": q90})
        return predict, {}
    if backend == "chronos":
        context_hours = p["chronos_context_hours"]
        model = load_chronos_model(p["chronos_repo"], p["chronos_device"])
        def predict(context, future):
            raw, q10, q90 = run_chronos_prediction(
                model, context["actual_kwh"].to_numpy(dtype=float), len(future), context_hours,
                exog_context=build_exog_matrix(context, exog_cols),
                exog_horizon=build_exog_matrix(future, exog_cols), exog_cols=exog_cols,
            )
            return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw,
                                 "q10_kwh": q10, "q90_kwh": q90})
        return predict, {}
    if backend in TRAINABLE_BACKENDS:
        raise ValueError(f"{backend} requires a prepared per-zone model_reference; run the train stage first")
    if backend == "AR":
        response = state["energy_price_response"] if fitted_state is not None else estimate_energy_price_response(history)
        def predict(context, future):
            return run_ar_prediction(context, future, training_history=history, fitted_response=response)
        return predict, {"energy_price_response": response}
    raise ValueError(f"Unsupported backend: {backend}")
