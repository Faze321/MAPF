from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
NATIVE_ARTIFACT_SCHEMA_VERSION = 2
SUPPORTED_BACKENDS = {"timesfm", "chronos", "chronos_2", "lstm", "ar", "AR"}
CORE_COLUMNS = {"timestamp", "time", "zone", "zone_id", "load", "actual_kwh", "energy_price", "e_price"}


@dataclass(frozen=True)
class ForecastBatch:
    """Cross-zone predictions for one fixed forecast origin."""

    hourly: pd.DataFrame
    metadata: dict[str, Any]

    def for_zone(self, zone_id: Any) -> pd.DataFrame:
        return self.hourly[self.hourly["zone"].astype(str) == str(zone_id)].copy()


@dataclass(frozen=True)
class ForecasterArtifact:
    """Reusable fitted state. Calling predict never refits or reads actual future load."""

    backend: str
    coefficients: tuple[float, ...]
    zones: tuple[str, ...]
    numeric_covariates: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    history_values: dict[str, tuple[float, ...]]
    history_timestamps: dict[str, tuple[str, ...]]
    validation_bias: dict[str, tuple[float, ...]]
    forecast_origin: str
    feature_schema_version: int = SCHEMA_VERSION

    def predict(
        self,
        known_future: pd.DataFrame,
        energy_price_schedule: pd.DataFrame | pd.Series | None = None,
    ) -> ForecastBatch:
        future = canonicalize_long_frame(known_future, require_load=False)
        future = attach_energy_price_schedule(future, energy_price_schedule)
        if future.empty:
            return ForecastBatch(future.assign(predicted_load=pd.Series(dtype=float)), self.metadata())
        if "energy_price" not in future:
            raise ValueError("known future must include the baseline energy-price schedule")
        prices = pd.to_numeric(future["energy_price"], errors="coerce")
        if prices.isna().any() or (prices < 0).any():
            raise ValueError("energy-price schedules must be numeric and non-negative")

        predictions: list[dict[str, Any]] = []
        history = {zone: list(values) for zone, values in self.history_values.items()}
        zone_index = {zone: idx for idx, zone in enumerate(self.zones)}
        for timestamp, timestamp_rows in future.groupby("timestamp", sort=True):
            for _, row in timestamp_rows.sort_values("zone").iterrows():
                zone = str(row["zone"])
                if zone not in zone_index:
                    raise ValueError(f"unknown zone in known future: {zone}")
                prior = history.setdefault(zone, [])
                vector = artifact_feature_vector(
                    row,
                    zone=zone,
                    zones=self.zones,
                    numeric_covariates=self.numeric_covariates,
                    categorical_levels=self.categorical_levels,
                    lag_1=prior[-1] if prior else 0.0,
                    lag_24=prior[-24] if len(prior) >= 24 else (prior[-1] if prior else 0.0),
                )
                raw = float(np.dot(np.asarray(self.coefficients), vector))
                offset = len(predictions_for_zone(predictions, zone))
                bias = self.validation_bias.get(zone, ())
                calibrated = raw + (bias[offset % len(bias)] if bias else 0.0)
                predicted = max(0.0, calibrated)
                prior.append(predicted)
                predictions.append(
                    {
                        "timestamp": pd.Timestamp(timestamp),
                        "zone": zone,
                        "predicted_load": predicted,
                        "energy_price": float(row["energy_price"]),
                        "horizon_offset": offset,
                    }
                )
        return ForecastBatch(pd.DataFrame(predictions), self.metadata())

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.feature_schema_version,
            "backend": self.backend,
            "forecast_origin": self.forecast_origin,
            "zones": list(self.zones),
            "numeric_covariates": list(self.numeric_covariates),
            "categorical_levels": {
                key: list(values) for key, values in self.categorical_levels.items()
            },
            "validation_bias_offsets": {
                zone: len(values) for zone, values in self.validation_bias.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "coefficients": list(self.coefficients),
            "history_values": {key: list(value) for key, value in self.history_values.items()},
            "history_timestamps": {
                key: list(value) for key, value in self.history_timestamps.items()
            },
            "validation_bias": {key: list(value) for key, value in self.validation_bias.items()},
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "ForecasterArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported forecaster artifact schema version")
        return cls(
            backend=str(payload["backend"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            zones=tuple(str(value) for value in payload["zones"]),
            numeric_covariates=tuple(str(value) for value in payload["numeric_covariates"]),
            categorical_levels={
                str(key): tuple(str(value) for value in values)
                for key, values in (payload.get("categorical_levels") or {}).items()
            },
            history_values={
                str(key): tuple(float(value) for value in values)
                for key, values in payload["history_values"].items()
            },
            history_timestamps={
                str(key): tuple(str(value) for value in values)
                for key, values in payload["history_timestamps"].items()
            },
            validation_bias={
                str(key): tuple(float(value) for value in values)
                for key, values in payload["validation_bias"].items()
            },
            forecast_origin=str(payload["forecast_origin"]),
            feature_schema_version=int(payload["schema_version"]),
        )


class GlobalForecaster:
    """Shared cross-zone fixed-origin forecaster interface for every configured backend.

    The current lightweight artifact is a pooled ARX implementation. The backend name is
    retained so TimesFM, Chronos-2 and LSTM adapters can share this lifecycle without
    changing the control engine or artifact contract.
    """

    def __init__(self, backend: str = "AR", *, ridge: float = 1e-5) -> None:
        if backend not in SUPPORTED_BACKENDS and backend.lower() not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported global forecaster backend: {backend}")
        self.backend = backend
        self.ridge = float(ridge)

    def fit(self, history: pd.DataFrame, validation: pd.DataFrame) -> ForecasterArtifact:
        train = canonicalize_long_frame(history, require_load=True)
        valid = canonicalize_long_frame(validation, require_load=True)
        if train.empty:
            raise ValueError("history must contain at least one row")
        forecast_origin = valid["timestamp"].min() if not valid.empty else train["timestamp"].max() + pd.Timedelta(hours=1)
        train = train[train["timestamp"] < forecast_origin].copy()
        if train.empty:
            raise ValueError("fixed-origin truncation removed all training rows")
        valid = valid[valid["timestamp"] >= forecast_origin].copy()
        zones = tuple(sorted(train["zone"].astype(str).unique()))
        covariates = discover_numeric_covariates(train, valid)
        categorical_levels = discover_categorical_levels(train, valid)
        x, y = build_training_matrix(train, zones, covariates, categorical_levels)
        if len(y) == 0:
            raise ValueError("at least 25 hourly observations per zone are required")
        regularizer = np.eye(x.shape[1], dtype=float) * self.ridge
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.pinv(x.T @ x + regularizer) @ x.T @ y
        history_values, history_timestamps = artifact_history(train, zones)
        preliminary = ForecasterArtifact(
            backend=self.backend,
            coefficients=tuple(float(value) for value in coefficients),
            zones=zones,
            numeric_covariates=covariates,
            categorical_levels=categorical_levels,
            history_values=history_values,
            history_timestamps=history_timestamps,
            validation_bias={},
            forecast_origin=pd.Timestamp(forecast_origin).isoformat(),
        )
        bias = validation_bias(preliminary, valid)
        return ForecasterArtifact(**{**preliminary.__dict__, "validation_bias": bias})


@dataclass(frozen=True)
class NativeForecasterArtifact:
    """Reusable state for running a new price scenario through the original backend."""

    zone: str
    backend: str
    forecast_start: str
    horizon_hours: int
    history: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    known_future: tuple[dict[str, Any], ...]
    forecast_parameters: dict[str, Any]
    fitted_state: dict[str, Any]

    def __post_init__(self) -> None:
        forecast_origin = pd.Timestamp(self.forecast_start)
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
            forecast_start=pd.Timestamp(reusable["forecast_start"]).isoformat(),
            horizon_hours=int(reusable["horizon_hours"]),
            history=dataframe_records(reusable["history"]),
            validation=dataframe_records(reusable["validation"]),
            known_future=dataframe_records(reusable["known_future"]),
            forecast_parameters=json_safe_mapping(reusable["forecast_parameters"]),
            fitted_state=json_safe_mapping(fitted_state),
        )

    def predict(self, energy_price_schedule: pd.DataFrame) -> ForecastBatch:
        """Re-run backend inference with a complete, non-negative price schedule."""

        from forecasting import forecast_load

        history = records_dataframe(self.history)
        validation = records_dataframe(self.validation)
        known_future = records_dataframe(self.known_future)
        if "time" not in known_future:
            raise ValueError("Reusable forecaster artifact has no future timestamps.")

        price_by_time = scenario_price_series(energy_price_schedule, self.zone)
        timestamps = pd.DatetimeIndex(pd.to_datetime(known_future["time"]))
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
            temperature = pd.to_numeric(known_future["T"], errors="coerce").fillna(0.0)
            known_future["temp_price_idx"] = (
                temperature.to_numpy(dtype=np.float64)
                * known_future["e_price"].to_numpy(dtype=np.float64)
            )
        known_future = known_future.drop(columns=["actual_kwh"], errors="ignore")

        parameters = dict(self.forecast_parameters)
        hourly = forecast_load(
            history,
            validation,
            known_future,
            pd.Timestamp(self.forecast_start),
            self.horizon_hours,
            model_name=self.backend,
            timesfm_repo=str(parameters["timesfm_repo"]),
            timesfm_context_hours=int(parameters["timesfm_context_hours"]),
            timesfm_step_horizon=int(parameters["timesfm_step_horizon"]),
            timesfm_exog_cols=list(parameters["timesfm_exog_cols"]),
            timesfm_diurnal_blend_alpha=float(
                parameters["timesfm_diurnal_blend_alpha"]
            ),
            timesfm_roll_actuals=False,
            ar_diurnal_blend_alpha=float(parameters["ar_diurnal_blend_alpha"]),
            chronos_repo=str(parameters["chronos_repo"]),
            chronos_context_hours=int(parameters["chronos_context_hours"]),
            chronos_step_horizon=int(parameters["chronos_step_horizon"]),
            chronos_exog_cols=list(parameters["chronos_exog_cols"]),
            chronos_diurnal_blend_alpha=float(
                parameters["chronos_diurnal_blend_alpha"]
            ),
            chronos_device=str(parameters["chronos_device"]),
            chronos_roll_actuals=False,
            lstm_context_hours=int(parameters["lstm_context_hours"]),
            lstm_step_horizon=int(parameters["lstm_step_horizon"]),
            lstm_exog_cols=list(parameters["lstm_exog_cols"]),
            lstm_hidden_size=int(parameters["lstm_hidden_size"]),
            lstm_num_layers=int(parameters["lstm_num_layers"]),
            lstm_epochs=int(parameters["lstm_epochs"]),
            lstm_learning_rate=float(parameters["lstm_learning_rate"]),
            lstm_batch_size=int(parameters["lstm_batch_size"]),
            lstm_diurnal_blend_alpha=float(
                parameters["lstm_diurnal_blend_alpha"]
            ),
            lstm_device=str(parameters["lstm_device"]),
            lstm_roll_actuals=False,
            lstm_seed=int(parameters["lstm_seed"]),
            fitted_state=self.fitted_state,
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
            "zone": self.zone,
            "forecast_origin": self.forecast_start,
            "inference_mode": "original_backend_reforecast",
            "price_schedule_scope": "complete_control_scenario",
            "artifact_reused": True,
            "model_retrained": False,
            "approximation_used": False,
            "calibration": dict(self.fitted_state.get("calibration") or {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "horizon_hours": self.horizon_hours,
            "history": list(self.history),
            "validation": list(self.validation),
            "known_future": list(self.known_future),
            "forecast_parameters": self.forecast_parameters,
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
        return cls(
            zone=str(payload["zone"]),
            backend=str(payload["backend"]),
            forecast_start=pd.Timestamp(payload["forecast_origin"]).isoformat(),
            horizon_hours=int(payload["horizon_hours"]),
            history=tuple(dict(row) for row in payload["history"]),
            validation=tuple(dict(row) for row in payload["validation"]),
            known_future=tuple(dict(row) for row in payload["known_future"]),
            forecast_parameters=dict(payload["forecast_parameters"]),
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


def scenario_price_series(schedule: pd.DataFrame, zone: str) -> pd.Series:
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("Energy-price schedule must be a pandas DataFrame.")
    frame = schedule.copy()
    if "timestamp" not in frame and "time" in frame:
        frame = frame.rename(columns={"time": "timestamp"})
    if "timestamp" not in frame:
        raise ValueError("Energy-price schedule has no timestamp column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
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
    if numeric.isna().any():
        raise ValueError(f"Energy-price schedule contains non-numeric values for zone {zone}.")
    return pd.Series(
        numeric.to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(values["timestamp"]),
        dtype=np.float64,
    ).sort_index()


def canonicalize_long_frame(frame: pd.DataFrame, *, require_load: bool) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("forecaster inputs must be pandas DataFrames")
    renamed = frame.copy()
    aliases = {
        "time": "timestamp",
        "zone_id": "zone",
        "actual_kwh": "load",
        "e_price": "energy_price",
    }
    for source, target in aliases.items():
        if target not in renamed and source in renamed:
            renamed = renamed.rename(columns={source: target})
    required = {"timestamp", "zone"} | ({"load"} if require_load else set())
    missing = sorted(required.difference(renamed.columns))
    if missing:
        raise ValueError("missing forecaster column(s): " + ", ".join(missing))
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="raise")
    renamed["zone"] = renamed["zone"].astype(str)
    if "load" in renamed:
        renamed["load"] = pd.to_numeric(renamed["load"], errors="coerce")
    return renamed.sort_values(["timestamp", "zone"]).reset_index(drop=True)


def discover_numeric_covariates(*frames: pd.DataFrame) -> tuple[str, ...]:
    candidates: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column in CORE_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                candidates.add(str(column))
    candidates.add("energy_price")
    return tuple(sorted(candidates))


def discover_categorical_levels(*frames: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    levels: dict[str, tuple[str, ...]] = {}
    for column in combined.columns:
        if column in CORE_COLUMNS or pd.api.types.is_numeric_dtype(combined[column]):
            continue
        values = tuple(sorted(combined[column].dropna().astype(str).unique()))
        if values:
            levels[str(column)] = values
    return levels


def build_training_matrix(
    frame: pd.DataFrame,
    zones: tuple[str, ...],
    covariates: tuple[str, ...],
    categorical_levels: dict[str, tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    targets: list[float] = []
    for zone, group in frame.groupby("zone"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        loads = pd.to_numeric(group["load"], errors="coerce")
        for idx in range(24, len(group)):
            target = loads.iloc[idx]
            if pd.isna(target) or pd.isna(loads.iloc[idx - 1]) or pd.isna(loads.iloc[idx - 24]):
                continue
            vectors.append(
                artifact_feature_vector(
                    group.iloc[idx],
                    zone=str(zone),
                    zones=zones,
                    numeric_covariates=covariates,
                    categorical_levels=categorical_levels,
                    lag_1=float(loads.iloc[idx - 1]),
                    lag_24=float(loads.iloc[idx - 24]),
                )
            )
            targets.append(float(target))
    width = 7 + len(covariates) + len(zones) + sum(
        len(values) for values in categorical_levels.values()
    )
    return (
        np.vstack(vectors) if vectors else np.empty((0, width), dtype=float),
        np.asarray(targets, dtype=float),
    )


def artifact_feature_vector(
    row: pd.Series,
    *,
    zone: str,
    zones: tuple[str, ...],
    numeric_covariates: tuple[str, ...],
    categorical_levels: dict[str, tuple[str, ...]],
    lag_1: float,
    lag_24: float,
) -> np.ndarray:
    timestamp = pd.Timestamp(row["timestamp"])
    hour = timestamp.hour
    dow = timestamp.dayofweek
    numeric = []
    for name in numeric_covariates:
        value = pd.to_numeric(pd.Series([row.get(name, 0.0)]), errors="coerce").iloc[0]
        numeric.append(0.0 if pd.isna(value) else float(value))
    zone_one_hot = [1.0 if zone == candidate else 0.0 for candidate in zones]
    categorical = [
        1.0 if str(row.get(column)) == level else 0.0
        for column, levels in categorical_levels.items()
        for level in levels
    ]
    return np.asarray(
        [
            1.0,
            lag_1,
            lag_24,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            *numeric,
            *zone_one_hot,
            *categorical,
        ],
        dtype=float,
    )


def artifact_history(
    frame: pd.DataFrame,
    zones: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[str, ...]]]:
    values: dict[str, tuple[float, ...]] = {}
    timestamps: dict[str, tuple[str, ...]] = {}
    for zone in zones:
        group = frame[frame["zone"] == zone].sort_values("timestamp").tail(512)
        valid = group.dropna(subset=["load"])
        values[zone] = tuple(float(value) for value in valid["load"])
        timestamps[zone] = tuple(pd.Timestamp(value).isoformat() for value in valid["timestamp"])
    return values, timestamps


def attach_energy_price_schedule(
    future: pd.DataFrame,
    schedule: pd.DataFrame | pd.Series | None,
) -> pd.DataFrame:
    if schedule is None:
        return future
    if isinstance(schedule, pd.Series):
        updated = future.copy()
        if len(schedule) != len(updated):
            raise ValueError("energy price Series length must match known future")
        updated["energy_price"] = pd.to_numeric(schedule.to_numpy(), errors="coerce")
        return updated
    prices = canonicalize_long_frame(schedule, require_load=False)
    if "energy_price" not in prices:
        raise ValueError("energy price schedule is missing energy_price")
    updated = future.drop(columns=["energy_price"], errors="ignore").merge(
        prices[["timestamp", "zone", "energy_price"]],
        on=["timestamp", "zone"],
        how="left",
        validate="one_to_one",
    )
    return updated


def validation_bias(
    artifact: ForecasterArtifact,
    validation: pd.DataFrame,
) -> dict[str, tuple[float, ...]]:
    if validation.empty:
        return {}
    future = validation.drop(columns=["load"], errors="ignore")
    batch = artifact.predict(future)
    joined = validation[["timestamp", "zone", "load"]].merge(
        batch.hourly[["timestamp", "zone", "predicted_load", "horizon_offset"]],
        on=["timestamp", "zone"],
        how="inner",
    )
    bias: dict[str, tuple[float, ...]] = {}
    for zone, group in joined.groupby("zone"):
        ordered = group.sort_values("horizon_offset")
        bias[str(zone)] = tuple(
            float(actual - predicted)
            for actual, predicted in zip(ordered["load"], ordered["predicted_load"])
        )
    return bias


def predictions_for_zone(predictions: list[dict[str, Any]], zone: str) -> list[dict[str, Any]]:
    return [item for item in predictions if item["zone"] == zone]
