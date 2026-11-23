from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.run import CALIBRATION_METHOD, VALIDATION_BIAS_DAYS, validation_bias_days
from src.config.training import TRAINABLE_BACKENDS
from src.control.policy import LOAD_POLICY_ID
from src.forecast.artifact import NATIVE_ARTIFACT_SCHEMA_VERSION, NativeForecasterArtifact
from src.forecast.parameters import make_forecast_parameters
from src.results.schema import FORECASTER_MANIFEST_SCHEMA_VERSION, require_current_result


@dataclass(frozen=True)
class ForecasterBundle:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    selected_zones: pd.DataFrame
    contexts: list[dict[str, Any]]
    forecast_parameters: dict[str, Any]
    outputs: dict[str, Path]
    scenario_artifacts: dict[str, NativeForecasterArtifact]


def resolve_forecaster_output_dir(
    run_output_dir: Path,
    value: Path | str | None,
) -> Path:
    if value in (None, ""):
        return run_output_dir / "forecaster"
    path = Path(value)
    if path.name.lower() == "forecaster_manifest.json":
        return path.parent
    return path


def build_forecaster_manifest(
    *,
    data_dir: Path,
    weather_file: str,
    selected_zone_ids: list[str],
    forecast_parameters: dict[str, Any],
    dataset_fingerprint: str | None = None,
    cache_keys: dict[str, str] | None = None,
    feature_manifest: dict[str, Any] | None = None,
    dataset_adapter: str = "urbanev",
    zone_model_references: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    references = {zone: dict(zone_model_references[zone]) for zone in selected_zone_ids
                  if zone_model_references and zone in zone_model_references}
    model_ids = {zone: reference["model_id"] for zone, reference in references.items()}
    return {
        "schema_version": FORECASTER_MANIFEST_SCHEMA_VERSION,
        "stage": "forecaster",
        "load_policy_id": LOAD_POLICY_ID,
        "created_at": pd.Timestamp.now().isoformat(),
        "data_source": {
            "data_dir": str(data_dir.resolve()),
            "weather_file": weather_file,
            "adapter": dataset_adapter,
            "dataset_fingerprint": dataset_fingerprint,
            "cache_keys": cache_keys or {},
        },
        "feature_manifest": feature_manifest or {},
        "zone_ids": list(selected_zone_ids),
        "forecast_parameters": {**forecast_parameters, **({"model_ids": model_ids} if model_ids else {})},
        "model_ids": model_ids,
        "model_provenance": references,
    }


def build_agent_manifest(
    *,
    agent_mode: str,
    backend: str,
    validation_bias: bool = False,
    validation_days: int = VALIDATION_BIAS_DAYS,
    forecaster_dir: Path,
    selected_zone_ids: list[str],
    forecast_origin: Any = None,
    dataset_fingerprint: str | None = None,
    cache_keys: dict[str, str] | None = None,
    feature_manifest: dict[str, Any] | None = None,
    model_ids: dict[str, str] | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "stage": "agent",
        "load_policy_id": LOAD_POLICY_ID,
        "validation_bias": validation_bias,
        "validation_days": validation_bias_days(validation_bias, validation_days),
        "completed_at": pd.Timestamp.now().isoformat(),
        "agent_mode": agent_mode,
        "backend": backend,
        "forecast_origin": pd.Timestamp(forecast_origin).isoformat()
        if forecast_origin is not None
        else None,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_keys": cache_keys or {},
        "feature_manifest": feature_manifest or {},
        "forecaster_output_dir": str(forecaster_dir.resolve()),
        "zone_ids": list(selected_zone_ids),
        "model_ids": model_ids or {},
        "model_provenance": model_provenance or {},
    }


def load_forecaster_bundle(path: Path | str) -> ForecasterBundle:
    output_dir = resolve_forecaster_output_dir(Path("."), path)
    manifest_path = output_dir / "forecaster_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Forecaster manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_current_result(manifest, str(manifest_path), FORECASTER_MANIFEST_SCHEMA_VERSION)
    if manifest.get("stage") != "forecaster":
        raise ValueError(f"Invalid forecaster manifest stage: {manifest.get('stage')}")

    raw_parameters = manifest.get("forecast_parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("Forecaster manifest is missing forecast_parameters")
    if "validation_days" in raw_parameters and not raw_parameters.get("validation_bias"):
        raise ValueError("Unexpected validation_days in forecaster manifest")
    if not isinstance(raw_parameters.get("validation_bias"), bool):
        raise ValueError("Forecaster manifest must record its resolved validation_bias switch")
    elif raw_parameters["validation_bias"] and raw_parameters.get("calibration_method") != CALIBRATION_METHOD:
        raise ValueError("Forecaster manifest is missing the validation bias method")
    elif raw_parameters["validation_bias"] and "validation_days" not in raw_parameters:
        raise ValueError("Calibrated forecaster manifest is missing validation_days")
    forecast_parameters = make_forecast_parameters(**raw_parameters)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Forecaster manifest is missing artifacts")
    selected_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("selected_zones"),
        "selected_zones",
    )
    contexts_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("context_snippets"),
        "context_snippets",
    )
    model_artifact_path = forecaster_artifact_path(
        output_dir,
        artifacts.get("forecaster_artifact_json"),
        "forecaster_artifact_json",
    )
    selected_zones = pd.read_csv(selected_path, dtype={"zone_id": "string"})
    if "zone_id" not in selected_zones:
        raise ValueError(f"Forecaster selected_zones is missing zone_id: {selected_path}")
    selected_zones["zone_id"] = selected_zones["zone_id"].astype("string").str.strip()
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if not isinstance(contexts, list) or not all(isinstance(item, dict) for item in contexts):
        raise ValueError(f"Forecaster context_snippets must contain a JSON list: {contexts_path}")
    model_payload = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    if (
        not isinstance(model_payload, dict)
        or model_payload.get("schema_version") != NATIVE_ARTIFACT_SCHEMA_VERSION
        or model_payload.get("artifact_type") != "native_backend_reforecast"
    ):
        raise ValueError(
            "Forecaster artifact type or schema does not match its manifest. "
            "Run the forecaster stage again to regenerate its inference state."
        )
    raw_scenario_artifacts = model_payload.get("artifacts") if isinstance(model_payload, dict) else None
    if not isinstance(raw_scenario_artifacts, dict):
        raise ValueError("Forecaster artifact is missing per-zone artifacts")
    scenario_artifacts = {
        str(zone_id): NativeForecasterArtifact.from_dict(payload)
        for zone_id, payload in raw_scenario_artifacts.items()
        if isinstance(payload, dict)
    }

    selected_ids = selected_zones["zone_id"].astype(str).tolist()
    context_ids = [str(context.get("zone_id")) for context in contexts]
    manifest_ids = [str(value) for value in manifest.get("zone_ids") or []]
    if selected_ids != context_ids or selected_ids != manifest_ids:
        raise ValueError(
            "Forecaster output zone order mismatch between manifest, selected_zones, "
            "and context_snippets"
        )
    if set(scenario_artifacts) != set(selected_ids):
        raise ValueError(
            "Forecaster artifact Zone set does not match the no-leakage handoff."
        )
    model_ids = manifest.get("model_ids") or {}
    if forecast_parameters["backend"] in TRAINABLE_BACKENDS or model_ids:
        if not isinstance(model_ids, dict) or set(model_ids) != set(selected_ids):
            raise ValueError("Forecaster manifest must identify an offline model for every selected zone")
        if forecast_parameters.get("model_ids") != model_ids:
            raise ValueError("Forecaster model IDs differ between forecast parameters and manifest")
    provenance = manifest.get("model_provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("Forecaster model_provenance must contain a per-zone mapping")
    if forecast_parameters["backend"] not in TRAINABLE_BACKENDS and provenance:
        raise ValueError("Pretrained forecasters cannot contain offline model provenance")
    if set(provenance) != set(model_ids):
        raise ValueError("Forecaster model provenance must cover the same zones as its model IDs")
    for zone_id, entry in provenance.items():
        if not isinstance(entry, dict) or entry.get("model_id") != model_ids[zone_id]:
            raise ValueError(f"Forecaster model provenance model ID differs for zone {zone_id}")
        if entry.get("backend", forecast_parameters["backend"]) != forecast_parameters["backend"]:
            raise ValueError(f"Forecaster model provenance backend differs for zone {zone_id}")
    for zone_id, artifact in scenario_artifacts.items():
        metadata = artifact.metadata()
        if artifact.forecast_parameters.get("validation_bias", False) is not forecast_parameters["validation_bias"]:
            raise ValueError(f"Forecaster artifact validation_bias differs from the manifest for zone {zone_id}")
        if artifact.forecast_parameters.get("validation_days") != forecast_parameters.get("validation_days"):
            raise ValueError(f"Forecaster artifact validation_days differs from the manifest for zone {zone_id}")
        if metadata.get("backend") != forecast_parameters["backend"]:
            raise ValueError(f"Forecaster artifact backend differs from the manifest for zone {zone_id}")
        if pd.Timestamp(metadata.get("forecast_origin")) != pd.Timestamp(forecast_parameters["forecast_origin"]):
            raise ValueError(f"Forecaster artifact origin differs from the manifest for zone {zone_id}")
        if float(artifact.forecast_parameters.get("diurnal_blend_alpha", float("nan"))) != forecast_parameters["diurnal_blend_alpha"]:
            raise ValueError(f"Forecaster artifact blend alpha differs from the manifest for zone {zone_id}")
        if (model_ids or metadata.get("model_id")) and metadata.get("model_id") != model_ids.get(zone_id):
            raise ValueError(f"Forecaster artifact model ID differs from the manifest for zone {zone_id}")

    outputs = {
        "forecaster_output_dir": output_dir,
        "forecaster_manifest_json": manifest_path,
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "forecaster_artifact_json": model_artifact_path,
    }
    for name in (
        "forecast_metrics_csv",
        "forecast_metrics_md",
        "forecast_details_dir",
    ):
        value = artifacts.get(name)
        if value:
            outputs[name] = output_dir / str(value)
    return ForecasterBundle(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        selected_zones=selected_zones,
        contexts=contexts,
        forecast_parameters=forecast_parameters,
        outputs=outputs,
        scenario_artifacts=scenario_artifacts,
    )


def forecaster_artifact_path(output_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Forecaster manifest is missing artifact path: {name}")
    path = output_dir / value
    if not path.is_file():
        raise FileNotFoundError(f"Forecaster artifact not found: {path}")
    return path
