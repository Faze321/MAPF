"""Offline per-zone training, immutable weight caches, and inference-only loading."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.training import LORA_BACKENDS, TRAINABLE_BACKENDS, base_backend
from src.config.values import dataset_path_key
from src.files import _release_file_lock, _try_file_lock


MODEL_CACHE_SCHEMA_VERSION = 2
TRAINING_IMPLEMENTATION_VERSION = "selected_zone_direct_windows_v2"
FOUNDATION_BACKENDS = frozenset(LORA_BACKENDS)
FOUNDATION_REPOS = {"timesfm": "google/timesfm-2.5-200m-pytorch", "chronos": "amazon/chronos-2"}
DEFAULT_EXOG = ["T", "U", "nRAIN", "e_price", "s_price", "is_weekend", "temp_price_idx"]
_PREDICTORS: OrderedDict[tuple[str, str], Any] = OrderedDict()
_FOUNDATION_PREDICTORS: OrderedDict[tuple[str, str], Any] = OrderedDict()
_PREDICTOR_LOCK = threading.Lock()


def zone_key(zone_id: str) -> str:
    original = str(zone_id)
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", original).strip("_.")[:48] or "zone"
    return f"zone_{label}_{hashlib.sha256(original.encode('utf-8')).hexdigest()[:12]}"


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
    exog = parameters.get(f"{base_backend(backend)}_exog_cols")
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


def _required_model_files(backend: str) -> set[str]:
    common = {"preprocess.json", "training_history.csv"}
    if backend in FOUNDATION_BACKENDS:
        return common | {"adapter_config.json", "adapter_model.safetensors"}
    if backend in {"lstm", "xgboost"}:
        return common | {"model.pt" if backend == "lstm" else "model.json"}
    raise ValueError(f"Unsupported offline model backend: {backend}")


def _checked_model_files(backend: str, files: Any) -> list[str]:
    if (not isinstance(files, (list, tuple, set, dict))
            or any(not isinstance(name, str) or Path(name).name != name
                   or name in {"", ".", "..", "manifest.json"} for name in files)):
        raise ValueError("Invalid offline model file names")
    if not _required_model_files(backend).issubset(files):
        raise ValueError("Offline model manifest is missing required file checksums")
    return sorted(set(files))


def validate_model_reference(reference: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reference, dict) or reference.get("schema_version") != MODEL_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing offline model reference; run the train stage")
    if reference.get("backend") not in TRAINABLE_BACKENDS:
        raise ValueError("Offline model references require lstm, xgboost, timesfm_lora or chronos_lora")
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
    checksums = manifest.get("file_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Offline model manifest is missing required file checksums")
    _checked_model_files(manifest["backend"], checksums)
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


def load_cached_predictor(reference: dict[str, Any]):
    validate_model_reference(reference)
    key = (str(Path(reference["model_dir"]).resolve()), reference["manifest_sha256"])
    with _PREDICTOR_LOCK:
        foundation = reference["backend"] in FOUNDATION_BACKENDS
        cache = _FOUNDATION_PREDICTORS if foundation else _PREDICTORS
        limit = 2 if foundation else 32
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        # Evict before loading a large base model to bound peak retained models.
        while len(cache) >= limit:
            cache.popitem(last=False)
        if foundation:
            from src.training.lora import load_foundation_predictor
            predictor = load_foundation_predictor(reference["backend"], Path(reference["model_dir"]))
        else:
            from src.training.direct import load_zone_predictor
            predictor = load_zone_predictor(reference["backend"], Path(reference["model_dir"]))
        cache[key] = predictor
        return predictor
