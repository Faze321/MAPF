"""Offline per-zone training, immutable weight caches, and inference-only loading."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd

from config import dataset_path_key, normalize_forecast_model_name
from dataset_adapter import atomic_write_json, process_file_lock
from request_limits import _try_file_lock, _release_file_lock


MODEL_CACHE_SCHEMA_VERSION = 2
TRAINING_IMPLEMENTATION_VERSION = "selected_zone_direct_windows_v2"
TRAINABLE_BACKENDS = frozenset({"lstm", "xgboost"})
DEFAULT_EXOG = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
_PREDICTORS: OrderedDict[tuple[str, str], Any] = OrderedDict()
_PREDICTOR_LOCK = threading.Lock()


def zone_key(zone_id: str) -> str:
    original = str(zone_id)
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", original).strip("_.")[:48] or "zone"
    return f"zone_{label}_{hashlib.sha256(original.encode('utf-8')).hexdigest()[:12]}"


def canonical_zone_frame(dataset: Any, zone_id: str) -> pd.DataFrame:
    source = dataset.timeseries
    frame = source[source["zone_id"].astype(str) == str(zone_id)].copy()
    names = {"timestamp": "time", "load_kwh": "actual_kwh", "energy_price": "e_price",
             "service_price": "s_price", "temperature": "T", "humidity": "U", "rain": "nRAIN"}
    frame = frame.rename(columns={key: value for key, value in names.items() if value not in frame.columns})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["hour"] = frame["time"].dt.hour.astype(float)
    frame["is_weekend"] = frame["time"].dt.dayofweek.isin([5, 6]).astype(float)
    frame["temp_price_idx"] = (pd.to_numeric(frame["T"], errors="coerce")
                               * pd.to_numeric(frame["e_price"], errors="coerce")) if "T" in frame else np.nan
    columns = ["time", "actual_kwh", "e_price", "s_price", "occupancy", "T", "U", "nRAIN",
               "hour", "is_weekend", "temp_price_idx"]
    return frame.reindex(columns=columns).sort_values("time").reset_index(drop=True)


def complete_target_indices(frame: pd.DataFrame, context_hours: int,
                            horizon_hours: int = 1, stride_hours: int = 1) -> np.ndarray:
    """First target indices of complete continuous input/output windows."""
    from offline_models import complete_target_indices as model_target_indices
    return model_target_indices(frame, context_hours, horizon_hours, stride_hours)


def inspect_training_window(
    frame: pd.DataFrame, *, history_days: int = 30, horizon_days: int = 1,
    train_ratio: float = 0.8, validation_ratio: float = 0.2,
    min_history_days: int = 45, min_samples: int = 512, sample_stride_hours: int = 1,
) -> dict[str, Any]:
    """Chronologically split supervised windows, purging overlapping target labels."""
    for key, value in (("history_days", history_days), ("horizon_days", horizon_days),
                       ("min_history_days", min_history_days), ("min_samples", min_samples),
                       ("sample_stride_hours", sample_stride_hours)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (not 0 < train_ratio < 1 or not 0 < validation_ratio < 1
            or not np.isclose(train_ratio + validation_ratio, 1, rtol=0, atol=1e-9)):
        raise ValueError("train_ratio and validation_ratio must be positive and sum to 1")
    input_hours, output_hours = history_days * 24, horizon_days * 24
    result: dict[str, Any] = {"eligible": False, "reason": None, "observed_hours": len(frame),
                              "input_hours": input_hours, "output_hours": output_hours,
                              "sample_stride_hours": sample_stride_hours,
                              "train_ratio": train_ratio, "validation_ratio": validation_ratio}
    if frame.empty:
        return {**result, "reason": "no_observations"}
    clean = frame.copy()
    clean["time"] = pd.to_datetime(clean["time"], errors="coerce")
    if clean["time"].dropna().duplicated().any():
        return {**result, "reason": "duplicate_timestamps"}
    for column in ("actual_kwh", "e_price"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    valid = (clean["time"].notna() & (clean["time"] == clean["time"].dt.floor("h"))
             & np.isfinite(clean["actual_kwh"]) & (clean["actual_kwh"] >= 0)
             & np.isfinite(clean["e_price"]) & (clean["e_price"] > 0))
    clean = clean[valid].sort_values("time").reset_index(drop=True)
    result.update(valid_hours=len(clean), invalid_hours=int((~valid).sum()))
    if len(clean) < min_history_days * 24:
        return {**result, "reason": "insufficient_valid_history"}
    for column, reason in (("actual_kwh", "constant_load"), ("e_price", "constant_energy_price")):
        if np.isclose(clean[column].min(), clean[column].max(), rtol=1e-9, atol=1e-12):
            return {**result, "reason": reason}
    candidates = complete_target_indices(clean, input_hours, output_hours, sample_stride_hours)
    count = len(candidates)
    result.update(sample_count=count, full_samples=count)
    if count < max(2, min_samples):
        return {**result, "reason": "insufficient_complete_samples"}
    split_index = max(1, min(count - 1, int(np.floor(count * train_ratio))))
    validation_targets = candidates[split_index:]
    validation_start = clean.iloc[validation_targets[0]]["time"]
    validation_end = clean.iloc[validation_targets[-1] + output_hours - 1]["time"] + pd.Timedelta(hours=1)
    train_candidates = candidates[:split_index]
    train_targets = train_candidates[train_candidates + output_hours <= validation_targets[0]]
    train_count, validation_count = len(train_targets), len(validation_targets)
    result.update(validation_start=validation_start, validation_end=validation_end,
                  train_samples=train_count, validation_samples=validation_count,
                  purged_samples=len(train_candidates) - train_count,
                  actual_train_ratio=train_count / (train_count + validation_count),
                  actual_validation_ratio=validation_count / (train_count + validation_count),
                  split_method="chronological_window_ratio_purged_targets",
                  fit_start=clean.iloc[0]["time"], fit_end_exclusive=validation_start)
    if not train_count:
        return {**result, "reason": "insufficient_training_samples_after_purge"}
    return {**result, "eligible": True, "frame": clean}


def assess_dataset_training(dataset: Any, training_config: Any, *, zone_ids=None,
                            history_days: int = 30, horizon_days: int = 1) -> dict[str, Any]:
    """Read-only eligibility for selection; no directories, models or cache writes."""
    available = set(dataset.timeseries["zone_id"].astype(str))
    selected = sorted(available) if zone_ids is None else list(dict.fromkeys(str(zone) for zone in zone_ids))
    excluded = {str(zone): reason for zone, reason in (dataset.feature_manifest.get("excluded_sites") or {}).items()}
    statuses = {}
    for zone in selected:
        if zone in excluded or zone not in available:
            statuses[zone] = {"eligible": False, "reason": "adapter_excluded" if zone in excluded else "unknown_zone"}
            continue
        assessment = inspect_training_window(canonical_zone_frame(dataset, zone),
            history_days=history_days, horizon_days=horizon_days,
            train_ratio=training_config.train_ratio, validation_ratio=training_config.validation_ratio,
            min_history_days=training_config.min_history_days, min_samples=training_config.min_samples,
            sample_stride_hours=training_config.sample_stride_hours)
        statuses[zone] = {key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                          for key, value in assessment.items() if key != "frame"}
    return {"eligible_zone_ids": [zone for zone in selected if statuses[zone]["eligible"]], "zones": statuses}


def training_intervals(frame: pd.DataFrame) -> list[list[str]]:
    times = pd.DatetimeIndex(frame["time"]).as_unit("ns")
    boundaries = np.r_[0, np.flatnonzero(np.diff(times.asi8) != pd.Timedelta(hours=1).value) + 1, len(times)]
    return [[times[left].isoformat(), (times[right - 1] + pd.Timedelta(hours=1)).isoformat()]
            for left, right in zip(boundaries[:-1], boundaries[1:]) if right > left]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_fingerprint(
    frame: pd.DataFrame, *, data_dir: Path, zone_id: str, backend: str,
    parameters: dict[str, Any], validation_start: pd.Timestamp, validation_end: pd.Timestamp,
) -> str:
    exog = parameters.get(f"{backend}_exog_cols")
    columns = list(dict.fromkeys(["actual_kwh", "e_price", *(DEFAULT_EXOG if exog is None else exog)]))
    features = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").astype("float64")
    features = features.replace([np.inf, -np.inf], np.nan)
    features.insert(0, "time", pd.DatetimeIndex(frame["time"]).as_unit("ns").to_numpy())
    digest = hashlib.sha256(pd.util.hash_pandas_object(features, index=False).to_numpy().tobytes())
    digest.update(_json_bytes({
        "implementation": TRAINING_IMPLEMENTATION_VERSION, "dataset": dataset_path_key(data_dir),
        "zone_id": str(zone_id), "backend": backend, "parameters": parameters,
        "validation_start": validation_start.isoformat(), "validation_end": validation_end.isoformat(),
        "columns": columns,
    }))
    return digest.hexdigest()


@contextmanager
def _zone_model_lock(path: Path):
    """Wait for lengthy training even on Windows, whose LK_LOCK retries are bounded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    while handle is None:
        handle = _try_file_lock(path)
        if handle is None:
            time.sleep(0.05)
    try:
        yield
    finally:
        _release_file_lock(handle)


def validate_model_reference(reference: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reference, dict) or reference.get("schema_version") != MODEL_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing offline model reference; run the train stage")
    directory = Path(reference["model_dir"])
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Offline model manifest missing: {manifest_path}; run the train stage")
    if _file_sha256(manifest_path) != reference.get("manifest_sha256"):
        raise ValueError(f"Offline model manifest checksum mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MODEL_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported offline model manifest schema")
    for field in ("model_id", "backend", "zone_id", "parameters", "training_start", "training_end_exclusive",
                  "training_intervals", "file_checksums"):
        if manifest.get(field) != reference.get(field):
            raise ValueError(f"Offline model reference {field} does not match its manifest")
    required = {"model.pt" if manifest["backend"] == "lstm" else "model.json", "preprocess.json", "training_history.csv"}
    checksums = manifest.get("file_checksums")
    if not isinstance(checksums, dict) or not required.issubset(checksums):
        raise ValueError("Offline model manifest is missing required file checksums")
    for filename, expected in checksums.items():
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Invalid offline model file name")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Offline model file missing: {path}; run the train stage")
        if _file_sha256(path) != expected:
            raise ValueError(f"Offline model file checksum mismatch: {path}")
    return manifest


def _reference(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    keys = ("schema_version", "model_id", "backend", "zone_id", "parameters", "training_start",
            "training_end_exclusive", "training_intervals", "file_checksums")
    return {**{key: manifest[key] for key in keys}, "model_dir": str(directory.resolve()),
            "manifest_sha256": _file_sha256(directory / "manifest.json")}


def _ensure_zone_model(
    dataset: Any, data_dir: Path, zone_id: str, backend: str, training_config: Any,
    cli_overrides: dict[str, Any] | None, force_train: bool, history_days: int, horizon_days: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root = Path(dataset.cache_dir) / "weight" / backend
    folder = root / zone_key(zone_id)
    with _zone_model_lock(folder / ".training.lock"):
        assessment = inspect_training_window(
            canonical_zone_frame(dataset, zone_id), history_days=history_days, horizon_days=horizon_days,
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
            from offline_models import PREPROCESS_SCHEMA_VERSION, train_zone_model
            print(f"Training {backend} zone={zone_id}: {history_days * 24}->{horizon_days * 24} hours, "
                  f"{assessment['train_samples']} train/{assessment['validation_samples']} validation samples", flush=True)
            report = train_zone_model(backend, frame, start, end, parameters, pending)
            files = ["model.pt" if backend == "lstm" else "model.json", "preprocess.json", "training_history.csv"]
            manifest = {
                "schema_version": MODEL_CACHE_SCHEMA_VERSION, "implementation": TRAINING_IMPLEMENTATION_VERSION,
                "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
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
    history_days: int = 30, horizon_days: int = 1, data_dir: Path | None = None,
    cli_overrides: dict[str, Any] | None = None, force_train: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Prepare only explicitly selected zones for the configured H-to-P windows."""
    data_dir = Path(data_dir) if data_dir is not None else Path(dataset.cache_dir).parent
    backends = list(dict.fromkeys(normalize_forecast_model_name(name) for name in backends))
    if any(name not in TRAINABLE_BACKENDS for name in backends):
        raise ValueError("Offline training supports only lstm and xgboost")
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
        workers = training_config.lstm_workers if backend == "lstm" else training_config.xgboost_workers
        # Concurrent LSTMs require separate RNG/CUDA contexts: manual_seed and
        # dropout use process-global state. Spawn is safe after CUDA initialization.
        isolated_lstm = backend == "lstm" and workers > 1
        pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                if isolated_lstm else ThreadPoolExecutor(max_workers=workers))
        with pool as executor:
            jobs = {executor.submit(_ensure_zone_model,
                                    _zone_training_payload(dataset, zone) if isolated_lstm else dataset,
                                    data_dir, zone, backend,
                                    training_config, cli_overrides, force_train, history_days, horizon_days): zone
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
    )


def load_cached_predictor(reference: dict[str, Any]):
    validate_model_reference(reference)
    key = (str(Path(reference["model_dir"]).resolve()), reference["manifest_sha256"])
    with _PREDICTOR_LOCK:
        if key in _PREDICTORS:
            _PREDICTORS.move_to_end(key)
            return _PREDICTORS[key]
        from offline_models import load_zone_predictor
        predictor = load_zone_predictor(reference["backend"], Path(reference["model_dir"]))
        _PREDICTORS[key] = predictor
        while len(_PREDICTORS) > 32:
            _PREDICTORS.popitem(last=False)
        return predictor
