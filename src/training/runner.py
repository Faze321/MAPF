from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.config.training import TRAINABLE_BACKENDS, base_backend
from src.config.values import dataset_path_key, normalize_forecast_model_name
from src.files import atomic_write_json, process_file_lock
from src.training.weights import (
    DEFAULT_EXOG,
    FOUNDATION_BACKENDS,
    FOUNDATION_REPOS,
    MODEL_CACHE_SCHEMA_VERSION,
    TRAINING_IMPLEMENTATION_VERSION,
    _checked_model_files,
    _file_sha256,
    _json_bytes,
    _reference,
    _required_model_files,
    _zone_model_lock,
    training_fingerprint,
    training_intervals,
    validate_model_reference,
    zone_key,
)
from src.training.windows import assess_dataset_training, canonical_zone_frame, inspect_training_window


def _ensure_zone_model(
    dataset: Any, data_dir: Path, zone_id: str, backend: str, training_config: Any,
    cli_overrides: dict[str, Any] | None, force_train: bool, history_days: int, horizon_days: int,
    backend_parameters: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root = Path(dataset.cache_dir) / "weight" / backend
    folder = root / zone_key(zone_id)
    with _zone_model_lock(folder / ".training.lock"):
        assessment = inspect_training_window(
            canonical_zone_frame(dataset, zone_id), history_days=history_days, horizon_days=horizon_days,
            allow_zero_energy_price=dataset.feature_manifest.get("allow_zero_energy_price", False),
            train_ratio=training_config.train_ratio, validation_ratio=training_config.validation_ratio,
            min_history_days=training_config.min_history_days,
            min_samples=training_config.min_samples, sample_stride_hours=training_config.sample_stride_hours,
        )
        audit = {key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                 for key, value in assessment.items() if key != "frame"}
        if not assessment["eligible"]:
            return None, {"status": "unavailable", **audit}
        parameters = training_config.resolve(data_dir, zone_id, backend, cli_overrides=cli_overrides)
        parameters.update(input_hours=history_days * 24, output_hours=horizon_days * 24,
                          sample_stride_hours=training_config.sample_stride_hours,
                          train_ratio=training_config.train_ratio, validation_ratio=training_config.validation_ratio)
        if backend in FOUNDATION_BACKENDS:
            family = base_backend(backend)
            supplied = backend_parameters or {}
            parameters[f"{family}_repo"] = supplied.get(f"{family}_repo") or FOUNDATION_REPOS[family]
            # A fixed H-to-P adapter must retain the entire configured history.
            parameters[f"{family}_context_hours"] = history_days * 24
            if supplied.get(f"{family}_exog_cols") is not None:
                parameters[f"{family}_exog_cols"] = list(supplied[f"{family}_exog_cols"])
            else:
                parameters.setdefault(f"{family}_exog_cols", list(DEFAULT_EXOG))
            parameters["adapter_implementation"] = "native_lora_v2"
        frame = assessment["frame"]
        start, end = assessment["validation_start"], assessment["validation_end"]
        signature = training_fingerprint(frame, data_dir=data_dir, zone_id=zone_id, backend=backend,
                                         parameters=parameters, validation_start=start, validation_end=end)
        current_path = folder / "current.json"
        recovery_errors: list[str] = []
        # Recovery belongs only to the prepare stage. Artifact replay still calls
        # validate_model_reference directly and must fail on damaged/missing files.
        cache_errors = (FileNotFoundError, ValueError, KeyError, TypeError)
        if not force_train and current_path.is_file():
            try:
                current = json.loads(current_path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    raise ValueError("Offline current pointer must contain a JSON object")
                if current.get("training_fingerprint") == signature:
                    reference = current["model_reference"]
                    validate_model_reference(reference)
                    return reference, {**audit, "status": "cached", "model_id": reference["model_id"]}
            except cache_errors as exc:
                recovery_errors.append(f"current: {type(exc).__name__}: {exc}")
        # The stable initial version can be reused when returning to an earlier configuration.
        stable_dir = folder / signature[:24]
        if not force_train and stable_dir.exists():
            try:
                manifest = json.loads((stable_dir / "manifest.json").read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("Offline model manifest must contain a JSON object")
                reference = _reference(stable_dir, manifest)
                validate_model_reference(reference)
                if manifest.get("training_fingerprint") != signature:
                    raise ValueError("Offline model fingerprint collision")
            except cache_errors as exc:
                recovery_errors.append(f"stable: {type(exc).__name__}: {exc}")
            else:
                atomic_write_json(current_path, {"training_fingerprint": signature, "model_reference": reference})
                recovery = ({"reason": "recovered_invalid_cache", "cache_validation_errors": recovery_errors}
                            if recovery_errors else {})
                return reference, {**audit, **recovery, "status": "cached", "model_id": reference["model_id"]}
        recovery = ({"reason": "recovered_invalid_cache", "cache_validation_errors": recovery_errors}
                    if recovery_errors else {})
        model_id = signature[:24] + ("-" + uuid.uuid4().hex[:12] if force_train or stable_dir.exists() else "")
        destination = folder / model_id
        pending = folder / (".pending-" + uuid.uuid4().hex)
        pending.mkdir(parents=True)
        try:
            print(f"Training {backend} zone={zone_id}: {history_days * 24}->{horizon_days * 24} hours, "
                  f"{assessment['train_samples']} train/{assessment['validation_samples']} validation samples", flush=True)
            if backend in FOUNDATION_BACKENDS:
                from src.training.lora import FOUNDATION_PREPROCESS_SCHEMA_VERSION, train_foundation_model
                report = train_foundation_model(backend, frame, start, end, parameters, pending)
                preprocess_version = report.get("preprocess_schema_version", FOUNDATION_PREPROCESS_SCHEMA_VERSION)
                files = _checked_model_files(backend, report.get("artifact_files", _required_model_files(backend)))
            else:
                from src.training.direct import PREPROCESS_SCHEMA_VERSION, train_zone_model
                report = train_zone_model(backend, frame, start, end, parameters, pending)
                preprocess_version = PREPROCESS_SCHEMA_VERSION
                files = _checked_model_files(backend, _required_model_files(backend))
            manifest = {
                "schema_version": MODEL_CACHE_SCHEMA_VERSION, "implementation": TRAINING_IMPLEMENTATION_VERSION,
                "preprocess_schema_version": preprocess_version,
                "model_id": model_id, "training_fingerprint": signature,
                "backend": backend, "zone_id": str(zone_id), "dataset": dataset_path_key(data_dir),
                "dataset_fingerprint_at_training": dataset.dataset_fingerprint,
                "parameters": parameters, "training_data_scope": "fit_and_early_stopping_validation",
                "fit_start": frame["time"].min().isoformat(), "fit_end_exclusive": start.isoformat(),
                "training_start": frame["time"].min().isoformat(),
                "training_end_exclusive": (frame["time"].max() + pd.Timedelta(hours=1)).isoformat(),
                "training_intervals": training_intervals(frame), "eligibility": audit,
                "cache_recovery": recovery or None,
                "internal_validation_start": start.isoformat(), "internal_validation_end_exclusive": end.isoformat(),
                "training_report": report, "file_checksums": {name: _file_sha256(pending / name) for name in files},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _json_bytes(manifest)  # Reject invalid metadata before publishing anything.
            atomic_write_json(pending / "manifest.json", manifest)
            os.replace(pending, destination)
            reference = _reference(destination, manifest)
            atomic_write_json(current_path, {"training_fingerprint": signature, "model_reference": reference})
            return reference, {**audit, **recovery, "status": "trained", "model_id": model_id}
        finally:
            if pending.exists():
                if pending.resolve().parent != folder.resolve() or not pending.name.startswith(".pending-"):
                    raise ValueError("Unsafe temporary model cleanup path")
                shutil.rmtree(pending)


def ensure_dataset_models(
    dataset: Any, backends: list[str], training_config: Any, *, zone_ids: list[str],
    history_days: int = 35, horizon_days: int = 1, data_dir: Path | None = None,
    cli_overrides: dict[str, Any] | None = None, force_train: bool = False,
    backend_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Prepare only explicitly selected zones for the configured H-to-P windows."""
    data_dir = Path(data_dir) if data_dir is not None else Path(
        getattr(dataset, "source_path", None) or Path(dataset.cache_dir).parent
    )
    backends = list(dict.fromkeys(normalize_forecast_model_name(name) for name in backends))
    if any(name not in TRAINABLE_BACKENDS for name in backends):
        raise ValueError("Offline training supports lstm, xgboost, timesfm_lora and chronos_lora")
    backends = training_config.enabled_backends(backends, cli_overrides=cli_overrides)
    if not backends:
        return {}
    if not zone_ids or isinstance(zone_ids, (str, bytes)):
        raise ValueError("zone_ids must explicitly select at least one zone before training")
    zones = list(dict.fromkeys(str(zone) for zone in zone_ids))
    assessment = assess_dataset_training(dataset, training_config, zone_ids=zones,
                                         history_days=history_days, horizon_days=horizon_days)
    unavailable = {zone: status["reason"] for zone, status in assessment["zones"].items() if not status["eligible"]}
    if unavailable:
        raise ValueError(f"Selected zones cannot form valid training/validation samples: {unavailable}")
    results: dict[str, dict[str, dict[str, Any]]] = {}
    failures = []
    for backend in backends:
        root = Path(dataset.cache_dir) / "weight" / backend
        references: dict[str, dict[str, Any]] = {}
        audit: dict[str, Any] = {}
        workers = (1 if backend in FOUNDATION_BACKENDS else
                   training_config.lstm_workers if backend == "lstm" else training_config.xgboost_workers)
        # Concurrent LSTMs require separate RNG/CUDA contexts: manual_seed and
        # dropout use process-global state. Spawn is safe after CUDA initialization.
        isolated_lstm = backend == "lstm" and workers > 1
        pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                if isolated_lstm else ThreadPoolExecutor(max_workers=workers))
        with pool as executor:
            jobs = {executor.submit(_ensure_zone_model,
                                    _zone_training_payload(dataset, zone) if isolated_lstm else dataset,
                                    data_dir, zone, backend,
                                    training_config, cli_overrides, force_train, history_days, horizon_days,
                                    backend_parameters): zone
                    for zone in zones}
            for job in as_completed(jobs):
                zone = jobs[job]
                try:
                    reference, status = job.result()
                    audit[zone] = status
                    if reference is not None:
                        references[zone] = reference
                except Exception as exc:
                    audit[zone] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(f"{backend}/{zone}: {type(exc).__name__}: {exc}")
        with process_file_lock(root / ".index.lock", blocking=True):
            index_path = root / "index.json"
            previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
            previous_zones = previous.get("zones", {}) if previous.get("schema_version") == MODEL_CACHE_SCHEMA_VERSION else {}
            combined = {**previous_zones, **audit}
            atomic_write_json(root / "index.json", {
                "schema_version": MODEL_CACHE_SCHEMA_VERSION, "backend": backend,
                "dataset": dataset_path_key(data_dir), "dataset_fingerprint": dataset.dataset_fingerprint,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "zones": {zone: combined[zone] for zone in sorted(combined)},
                "last_selected_zone_ids": zones,
                "available_zone_ids": sorted(zone for zone, status in combined.items() if status.get("status") in {"trained", "cached"}),
            })
        results[backend] = {zone: references[zone] for zone in sorted(references)}
    if failures:
        raise RuntimeError("Offline training failed; successful models remain cached. " + "; ".join(failures))
    return results


def _zone_training_payload(dataset: Any, zone_id: str) -> SimpleNamespace:
    """Send only this site's records to a spawned training worker."""
    return SimpleNamespace(
        timeseries=dataset.timeseries[dataset.timeseries["zone_id"].astype(str) == str(zone_id)].copy(),
        cache_dir=Path(dataset.cache_dir), dataset_fingerprint=dataset.dataset_fingerprint,
        feature_manifest={"allow_zero_energy_price": getattr(dataset, "feature_manifest", {}).get("allow_zero_energy_price", False)},
    )
