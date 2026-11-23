from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.run import VALIDATION_BIAS_DAYS, validation_bias_days
from src.config.values import normalize_float_list, normalize_forecast_model_name
from src.files import atomic_write_json, process_file_lock


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent


def default_dataset_cache_dir(data_path: Path) -> Path:
    """Keep dataset caches under this checkout, independently of the process cwd."""
    resolved = Path(data_path).resolve()
    cache_root = PROJECT_ROOT / "cache"
    try:
        relative = resolved.relative_to((PROJECT_ROOT / "data").resolve())
    except ValueError:
        relative = None
    if relative is not None and relative.parts:
        return cache_root / relative
    # External datasets with identical folder names must never share weights or
    # preprocessing. Resolve aliases before hashing their source directory.
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", resolved.name).strip("_.")[:64] or "dataset"
    digest = hashlib.sha256(os.path.normcase(str(resolved)).encode("utf-8")).hexdigest()[:12]
    return cache_root / "_external" / f"{label}_{digest}"


def experiment_slug(
    *,
    zone_count: int,
    time_count: int,
    mode_count: int,
    blend_count: int,
) -> str:
    return f"{int(zone_count)}zonesx{int(time_count)}timesx{int(mode_count)}modes_{int(blend_count)}blends"


def forecast_start_slug(forecast_origin: str) -> str:
    return pd.Timestamp(forecast_origin).strftime("%Y-%m-%d_%H%M%S")


def forecast_output_dir(
    output_dir: Path, backend: str, diurnal_blend_alpha: float = 0.0, *,
    validation_bias: bool = False, validation_days: int = VALIDATION_BIAS_DAYS,
) -> Path:
    normalized = normalize_forecast_model_name(backend)
    alpha = normalize_float_list(diurnal_blend_alpha)[0]
    alpha_text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    days = validation_bias_days(validation_bias, validation_days)
    root = output_dir
    if days:
        # Switch-only runs stored seven-day results in this directory.
        root /= "validation_bias" if days == 7 else f"validation_bias_{days}d"
    return root / f"blend_{alpha_text}" / safe_filename(normalized or "forecast")


def normalize_experiment_name(value: Any, *, default: str) -> str:
    raw = str(value).strip() if value not in (None, "") else str(default).strip()
    name = safe_filename(raw)
    if not name or name in {".", ".."}:
        raise ValueError("experiment_name must contain at least one letter or number.")
    return name


def safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def dataset_folder_key(data_path: Path) -> str:
    resolved = data_path.resolve()
    digest = hashlib.sha256(os.path.normcase(str(resolved)).encode("utf-8")).hexdigest()[:10]
    return f"{safe_filename(resolved.name)}_{digest}"


def scoped_output(root: Path, adapter: str, data_path: Path) -> Path:
    """Bind a readable dataset directory beneath an already resolved experiment root."""
    adapter = str(adapter).strip().lower().replace("-", "_")
    adapter = {"urban_ev": "urbanev", "mpevdata": "mp_evdata",
               "generic": "long", "long_format": "long"}.get(adapter, adapter)
    resolved = data_path.resolve()
    normalized_path = os.path.normcase(str(resolved))
    identity = hashlib.sha256(f"{adapter}\0{normalized_path}".encode("utf-8")).hexdigest()
    registry_path = root / ".dataset_directories.json"
    with process_file_lock(root / ".dataset_directories.lock", blocking=True):
        bindings = _read_dataset_directory_bindings(registry_path)
        if identity in bindings:
            return root / bindings[identity]["directory"]

        display = {"urbanev": "UrbanEV", "charged": "CHARGED",
                   "mp_evdata": "MP-EVData"}.get(adapter, safe_filename(adapter))
        leaf = safe_filename(resolved.name)
        same_name = re.sub(r"[-_]", "", leaf).casefold() == re.sub(r"[-_]", "", display).casefold()
        base = display if same_name or not leaf else f"{display}_{leaf}"
        if not base:
            raise ValueError("Dataset output directory must contain a nonempty adapter or folder name")
        # Case-fold even on POSIX so moving an experiment to Windows stays safe.
        occupied = {entry["directory"].casefold() for entry in bindings.values()}
        occupied.update(child.name.casefold() for child in root.iterdir())
        directory = base
        digest_length = 10
        while directory.casefold() in occupied:
            if digest_length > len(identity):
                raise ValueError(f"Cannot allocate a distinct dataset output directory beneath {root}")
            directory = f"{base}_{identity[:digest_length]}"
            digest_length = min(digest_length + 10, len(identity) + 1)
        bindings[identity] = {"adapter": adapter, "data_path": normalized_path, "directory": directory}
        atomic_write_json(registry_path, {"schema_version": 1, "datasets": bindings})
        return root / directory


def _read_dataset_directory_bindings(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read dataset output registry {path}; restore the registry or use another experiment_name") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid dataset output registry: {path}")
    bindings = payload["datasets"]
    directories: set[str] = set()
    for identity, entry in bindings.items():
        if not isinstance(entry, dict) or any(not isinstance(entry.get(key), str) or not entry[key]
                                               for key in ("adapter", "data_path", "directory")):
            raise ValueError(f"Invalid dataset binding in output registry: {path}")
        directory = entry["directory"]
        expected_identity = hashlib.sha256(f"{entry['adapter']}\0{entry['data_path']}".encode("utf-8")).hexdigest()
        if identity != expected_identity or directory != safe_filename(directory) or directory.casefold() in directories:
            raise ValueError(f"Invalid or conflicting dataset binding in output registry: {path}")
        directories.add(directory.casefold())
    return bindings


def output_lock(directory: Path):
    """An OS-released process lock prevents simultaneous writers to one output root."""
    return process_file_lock(
        directory / ".mapf-run.lock", blocking=False,
        conflict_message=f"Another run is writing to {directory}; use a different experiment_name or --output-folder",
    )
