from __future__ import annotations

from typing import Any

from src.config.run import (
    BACKEND_PARAMETER_NAMES,
    CALIBRATION_METHOD,
    RunConfig,
    VALIDATION_BIAS_DAYS,
    validation_bias_days,
)
from src.config.training import FORECAST_BACKENDS, LORA_BACKENDS, TRAINABLE_BACKENDS, base_backend
from src.config.values import normalize_float_list, normalize_forecast_model_name


FORECAST_PARAMETER_NAMES = (
    'forecast_origin',
    'horizon_days',
    'history_days',
    'validation_bias',
    'validation_days',
    'backend',
    'diurnal_blend_alpha',
    *BACKEND_PARAMETER_NAMES,
)


def make_forecast_parameters(**values: Any) -> dict[str, Any]:
    """Persist only the resolved run and the active backend's hyperparameters."""
    from src.config.training import TRAINING_PARAMETER_NAMES
    moved = [name for name in values if name in TRAINING_PARAMETER_NAMES or name.startswith(("lstm_", "xgboost_"))]
    if moved:
        raise ValueError("Offline training parameters belong to training configuration and per-zone model references: "
                         + ", ".join(moved))
    defaults = RunConfig()
    common = ("forecast_origin", "backend", "history_days", "horizon_days", "diurnal_blend_alpha")
    missing = [name for name in common if name not in values]
    if missing:
        raise ValueError("Missing forecast parameter(s): " + ", ".join(missing))
    backend = normalize_forecast_model_name(str(values["backend"]))
    if backend not in FORECAST_BACKENDS:
        raise ValueError(f"Unsupported forecast backend: {backend}")
    alpha = values["diurnal_blend_alpha"]
    if isinstance(alpha, (list, tuple, dict)) or alpha is None:
        raise ValueError("A resolved forecast requires one scalar diurnal_blend_alpha")
    validation_bias = values.get("validation_bias", False)
    days = validation_bias_days(validation_bias, values.get("validation_days", VALIDATION_BIAS_DAYS))
    if "calibration_method" in values and (
        not validation_bias or values["calibration_method"] != CALIBRATION_METHOD
    ):
        raise ValueError("Unsupported forecast calibration method; rerun the forecaster stage")
    parameters = {name: values[name] for name in common}
    parameters["backend"] = backend
    parameters["validation_bias"] = validation_bias
    parameters["diurnal_blend_alpha"] = normalize_float_list(alpha)[0]
    parameters.update({name: values.get(name, getattr(defaults, name))
                       for name in BACKEND_PARAMETER_NAMES if name.startswith(base_backend(backend).lower() + "_")})
    if backend in LORA_BACKENDS:
        parameters[f"{base_backend(backend)}_context_hours"] = int(parameters["history_days"]) * 24
    if validation_bias:
        parameters["validation_days"] = days
        parameters["calibration_method"] = CALIBRATION_METHOD
    if values.get("model_ids"):
        if backend not in TRAINABLE_BACKENDS:
            raise ValueError("Pretrained forecasters cannot identify offline models; select the LoRA backend")
        model_ids = values["model_ids"]
        if not isinstance(model_ids, dict) or any(not isinstance(zone, str) or not zone
                                                  or not isinstance(model_id, str) or not model_id
                                                  for zone, model_id in model_ids.items()):
            raise ValueError("model_ids must map nonempty zone IDs to nonempty model IDs")
        parameters["model_ids"] = dict(model_ids)
    return parameters


def forecast_runtime_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Expand defaults only while dispatching; do not add them to provenance."""
    defaults = RunConfig()
    return {**{name: getattr(defaults, name) for name in BACKEND_PARAMETER_NAMES},
            **{name: parameters[name] for name in FORECAST_PARAMETER_NAMES if name in parameters}}
