from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.run import VALIDATION_BIAS_DAYS, validation_bias_days
from src.config.training import FORECAST_BACKENDS, TRAINABLE_BACKENDS


NATIVE_ARTIFACT_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class ForecastBatch:
    """Cross-zone predictions for one fixed forecast origin."""

    hourly: pd.DataFrame
    metadata: dict[str, Any]

    def for_zone(self, zone_id: Any) -> pd.DataFrame:
        return self.hourly[self.hourly["zone"].astype(str) == str(zone_id)].copy()


@dataclass(frozen=True)
class NativeForecasterArtifact:
    """Reusable state for running a new price scenario through the original backend."""

    zone: str
    backend: str
    forecast_origin: str
    horizon_hours: int
    history: tuple[dict[str, Any], ...]
    known_future: tuple[dict[str, Any], ...]
    forecast_parameters: dict[str, Any]
    fitted_state: dict[str, Any]
    validation: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        from src.forecast.calibration import check_validation_bias_state

        enabled = self.forecast_parameters.get("validation_bias", False)
        days = self.forecast_parameters.get("validation_days", VALIDATION_BIAS_DAYS)
        if len(self.validation) != validation_bias_days(enabled, days) * 24:
            raise ValueError("Native artifact validation history does not match its configured validation days")
        check_validation_bias_state(self.fitted_state, enabled, days)
        if self.backend not in FORECAST_BACKENDS:
            raise ValueError(f"Unsupported native forecaster backend: {self.backend}")
        if self.fitted_state.get("backend") not in {None, self.backend}:
            raise ValueError("Native artifact fitted backend does not match its forecaster identity")
        if self.backend not in TRAINABLE_BACKENDS and self.fitted_state.get("model_reference") is not None:
            raise ValueError("Pretrained forecaster artifacts cannot contain a LoRA/offline model reference")
        if self.backend in TRAINABLE_BACKENDS:
            from src.training.weights import validate_model_reference

            reference = self.fitted_state.get("model_reference")
            if not isinstance(reference, dict):
                raise ValueError("Native artifact requires an immutable per-zone model reference")
            if reference.get("backend") != self.backend or str(reference.get("zone_id")) != str(self.zone):
                raise ValueError("Native artifact model reference does not match its zone/backend")
            dimensions = reference.get("parameters", {})
            if dimensions.get("input_hours") != len(self.history) or dimensions.get("output_hours") != self.horizon_hours:
                raise ValueError("Native artifact history/horizon does not match its direct model dimensions")
            validate_model_reference(reference)
        forecast_origin = pd.Timestamp(self.forecast_origin)
        if self.horizon_hours <= 0:
            raise ValueError("Reusable forecaster horizon must be positive.")
        if any("actual_kwh" in row or "load" in row for row in self.known_future):
            raise ValueError(
                "Reusable known-future features cannot contain forecast-period actual load."
            )
        future = records_dataframe(self.known_future)
        if len(future) != self.horizon_hours:
            raise ValueError(
                "Reusable known-future feature count does not match forecast horizon."
            )
        if future.empty or (pd.to_datetime(future["time"]) < forecast_origin).any():
            raise ValueError("Reusable future features precede the fixed forecast origin.")
        for label, records in (("history", self.history), ("validation", self.validation)):
            frame = records_dataframe(records)
            if not frame.empty and (pd.to_datetime(frame["time"]) >= forecast_origin).any():
                raise ValueError(
                    f"Reusable {label} contains load at or after the forecast origin."
                )

    @classmethod
    def from_forecast_result(cls, result: Any) -> "NativeForecasterArtifact":
        reusable = getattr(result, "reusable_forecaster", None)
        if not isinstance(reusable, dict):
            raise ValueError(
                "Forecast result has no reusable native forecaster state. "
                "Run the forecaster stage again with the current artifact schema."
            )
        fitted_state = reusable.get("fitted_state")
        if not isinstance(fitted_state, dict):
            raise ValueError("Forecast result did not preserve fitted backend state.")
        return cls(
            zone=str(reusable["zone"]),
            backend=str(reusable["backend"]),
            forecast_origin=pd.Timestamp(reusable["forecast_origin"]).isoformat(),
            horizon_hours=int(reusable["horizon_hours"]),
            history=dataframe_records(reusable["history"]),
            validation=dataframe_records(reusable["validation"]) if "validation" in reusable else (),
            known_future=dataframe_records(reusable["known_future"]),
            forecast_parameters=json_safe_mapping(reusable["forecast_parameters"]),
            fitted_state=json_safe_mapping(fitted_state),
        )

    def predict(self, energy_price_schedule: pd.DataFrame) -> ForecastBatch:
        """Re-run backend inference with a complete, non-negative price schedule."""

        from src.forecast.engine import forecast_load

        history = records_dataframe(self.history)
        known_future = records_dataframe(self.known_future)
        if "time" not in known_future:
            raise ValueError("Reusable forecaster artifact has no future timestamps.")

        timestamps = pd.DatetimeIndex(pd.to_datetime(known_future["time"]))
        price_by_time = scenario_price_series(energy_price_schedule, self.zone, timestamps=timestamps)
        scenario_prices = price_by_time.reindex(timestamps)
        if scenario_prices.isna().any():
            missing = timestamps[scenario_prices.isna()]
            raise ValueError(
                f"Energy-price scenario is missing {len(missing)} forecast timestamp(s) "
                f"for zone {self.zone}; first missing timestamp: {missing[0].isoformat()}"
            )
        if (scenario_prices < 0).any():
            raise ValueError("Energy-price schedules cannot contain negative values.")

        known_future = known_future.copy()
        known_future["e_price"] = scenario_prices.to_numpy(dtype=np.float64)
        if "T" in known_future:
            temperature = pd.to_numeric(known_future["T"], errors="coerce")
            known_future["temp_price_idx"] = (
                temperature.to_numpy(dtype=np.float64)
                * known_future["e_price"].to_numpy(dtype=np.float64)
            )
        known_future = known_future.drop(columns=["actual_kwh"], errors="ignore")

        hourly = forecast_load(
            history,
            known_future,
            pd.Timestamp(self.forecast_origin),
            self.horizon_hours,
            backend=self.backend,
            validation=records_dataframe(self.validation) if self.validation else None,
            fitted_state=self.fitted_state,
            **self.forecast_parameters,
        )
        output = hourly[["time", "predicted_kwh"]].copy()
        output = output.rename(
            columns={"time": "timestamp", "predicted_kwh": "predicted_load"}
        )
        output["zone"] = self.zone
        output["energy_price"] = scenario_prices.to_numpy(dtype=np.float64)
        output["horizon_offset"] = np.arange(len(output), dtype=int)
        return ForecastBatch(
            output[
                [
                    "timestamp",
                    "zone",
                    "predicted_load",
                    "energy_price",
                    "horizon_offset",
                ]
            ],
            self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "native_backend_reforecast",
            "backend": self.backend,
            "model_id": (self.fitted_state.get("model_reference") or {}).get("model_id"),
            "zone": self.zone,
            "forecast_origin": self.forecast_origin,
            "inference_mode": "original_backend_reforecast",
            "price_schedule_scope": "complete_control_scenario",
            "artifact_reused": True,
            "model_retrained": False,
            "approximation_used": False,
            "validation_bias": self.forecast_parameters.get("validation_bias", False),
            **({"calibration": self.fitted_state["calibration"]} if self.validation else {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "horizon_hours": self.horizon_hours,
            "history": list(self.history),
            **({"validation": list(self.validation)} if self.validation else {}),
            "known_future": list(self.known_future),
            "forecast_parameters": {**self.forecast_parameters, "validation_bias": bool(self.validation)},
            "fitted_state": self.fitted_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NativeForecasterArtifact":
        if payload.get("schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported native forecaster artifact schema version. "
                "Run the forecaster stage again; legacy price-mapping artifacts "
                "cannot be used for closed-loop validation."
            )
        if payload.get("artifact_type") != "native_backend_reforecast":
            raise ValueError("Forecaster artifact is not a native backend artifact.")
        parameters = dict(payload["forecast_parameters"])
        if not isinstance(parameters.get("validation_bias"), bool) or payload.get("validation_bias") is not parameters["validation_bias"]:
            raise ValueError("Native artifact metadata and forecast parameters disagree on validation bias")
        if parameters["validation_bias"] and "validation_days" not in parameters:
            raise ValueError("Calibrated native artifact is missing validation_days")
        return cls(
            zone=str(payload["zone"]),
            backend=str(payload["backend"]),
            forecast_origin=pd.Timestamp(payload["forecast_origin"]).isoformat(),
            horizon_hours=int(payload["horizon_hours"]),
            history=tuple(dict(row) for row in payload["history"]),
            validation=tuple(dict(row) for row in payload.get("validation", ())),
            known_future=tuple(dict(row) for row in payload["known_future"]),
            forecast_parameters=parameters,
            fitted_state=dict(payload["fitted_state"]),
        )


def dataframe_records(frame: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Reusable forecaster frames must be pandas DataFrames.")
    return tuple(
        {
            str(column): json_safe_value(value)
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    )


def records_dataframe(records: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    frame = pd.DataFrame([dict(row) for row in records])
    for column in ("time", "timestamp"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def json_safe_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): json_safe_value(value)
        for key, value in payload.items()
    }


def json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_price_series(schedule: pd.DataFrame, zone: str, *, timestamps: pd.DatetimeIndex | None = None) -> pd.Series:
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("Energy-price schedule must be a pandas DataFrame.")
    frame = schedule.copy()
    if "timestamp" not in frame and "time" in frame:
        frame = frame.rename(columns={"time": "timestamp"})
    if "timestamp" not in frame:
        raise ValueError("Energy-price schedule has no timestamp column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps is not None:
        # Sites can begin operation on different dates. Only the forecast-period
        # schedule is needed for inference; unavailable pre-opening tariffs are not errors.
        frame = frame[frame["timestamp"].isin(timestamps)]
    if zone in frame:
        values = frame[["timestamp", zone]].rename(columns={zone: "energy_price"})
    else:
        zone_column = "zone" if "zone" in frame else "zone_id" if "zone_id" in frame else None
        price_column = (
            "energy_price"
            if "energy_price" in frame
            else "e_price"
            if "e_price" in frame
            else None
        )
        if zone_column is None or price_column is None:
            raise ValueError(f"Energy-price schedule has no values for zone {zone}.")
        values = frame[frame[zone_column].astype(str) == str(zone)][
            ["timestamp", price_column]
        ].rename(columns={price_column: "energy_price"})
    if values["timestamp"].duplicated().any():
        raise ValueError(f"Energy-price schedule has duplicate timestamps for zone {zone}.")
    numeric = pd.to_numeric(values["energy_price"], errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"Energy-price schedule contains non-numeric values for zone {zone}.")
    return pd.Series(
        numeric.to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(values["timestamp"]),
        dtype=np.float64,
    ).sort_index()
