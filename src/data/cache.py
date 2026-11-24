from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.canonical import (
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    CanonicalDataset,
    DATASET_CACHE_SCHEMA_VERSION,
    PriceChangeReference,
    validate_canonical_timeseries,
    validate_static_zone_features,
)
from src.files import atomic_write_json, process_file_lock


def dataset_fingerprint_payload(
    data_dir: Path,
    sources: Iterable[Path],
    *,
    adapter: str,
    adapter_version: int,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for source in sorted((Path(path) for path in sources), key=lambda item: str(item)):
        stat = source.stat()
        with source.open("rb") as handle:
            header = handle.read(65536)
        try:
            relative = source.resolve().relative_to(data_dir.resolve())
        except ValueError:
            relative = source.resolve()
        files.append(
            {
                "path": str(relative).replace("\\", "/"),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "header_sha256": hashlib.sha256(header).hexdigest(),
            }
        )
    return {
        "schema_version": DATASET_CACHE_SCHEMA_VERSION,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "files": files,
        "mapping": mapping,
    }


def load_cached_dataset(
    *,
    dataset_cache: Path,
    artifacts: dict[str, Path],
    fingerprint: str,
) -> CanonicalDataset | None:
    if not all(path.is_file() for path in artifacts.values()):
        return None
    try:
        feature_manifest = json.loads(artifacts["feature_manifest"].read_text(encoding="utf-8"))
        if feature_manifest.get("schema_version") != DATASET_CACHE_SCHEMA_VERSION:
            return None
        if feature_manifest.get("dataset_fingerprint") != fingerprint:
            return None
        timeseries = pd.read_csv(artifacts["canonical_timeseries"], parse_dates=[CANONICAL_TIME_COLUMN],
                                 dtype={CANONICAL_ZONE_COLUMN: str})
        static = pd.read_csv(
            artifacts["static_zone_features"],
            dtype={CANONICAL_ZONE_COLUMN: str},
        )
        reference_raw = json.loads(artifacts["price_change_reference"].read_text(encoding="utf-8"))
        canonical_schema = json.loads(artifacts["canonical_schema"].read_text(encoding="utf-8"))
        if canonical_schema.get("schema_version") != DATASET_CACHE_SCHEMA_VERSION:
            return None
        if not isinstance(canonical_schema.get("columns"), list):
            return None
        poi_counts = pd.read_csv(
            artifacts["poi_zone_counts"],
            dtype={CANONICAL_ZONE_COLUMN: str},
        )
        if CANONICAL_ZONE_COLUMN not in poi_counts:
            return None
        timeseries = validate_canonical_timeseries(timeseries, allow_zero_energy_price=feature_manifest.get("allow_zero_energy_price", False))
        static = validate_static_zone_features(static, timeseries)
        reference = PriceChangeReference.from_dict(reference_raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, pd.errors.ParserError):
        return None
    return CanonicalDataset(
        timeseries=timeseries,
        static_zone_features=static,
        feature_manifest=feature_manifest,
        price_change_reference=reference,
        dataset_fingerprint=fingerprint,
        cache_dir=dataset_cache.parents[1],
        cache_keys={"dataset": fingerprint},
    )


def split_cache_key(
    dataset_fingerprint: str,
    forecast_origin: Any,
    *,
    window_hours: int,
    policy_id: str,
) -> str:
    start = pd.Timestamp(forecast_origin).isoformat()
    payload = {
        "dataset": dataset_fingerprint,
        "forecast_origin": start,
        "window_hours": int(window_hours),
        "policy_id": policy_id,
        "history_rule": "strictly_before_forecast_origin",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def cache_manifest_lock(path: Path):
    """Serialize the short read/merge/write transaction across processes."""
    return process_file_lock(path.with_suffix(".lock"), blocking=True)


def merge_cache_manifest(path: Path, update: dict[str, Any]) -> None:
    with cache_manifest_lock(path):
        _merge_cache_manifest(path, update)


def _merge_cache_manifest(path: Path, update: dict[str, Any]) -> None:
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, json.JSONDecodeError):
            current = {}
    datasets = dict(current.get("datasets") or {})
    datasets.update(update.get("datasets") or {})
    merged = {**current, **update, "datasets": datasets}
    if merged != current:
        atomic_write_json(path, merged)


def dataset_cache_manifest_entry(
    cache_root: Path,
    fingerprint: str,
    *,
    adapter: str,
    adapter_version: int,
    fingerprint_payload: dict[str, Any],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_CACHE_SCHEMA_VERSION,
        "active_dataset_fingerprint": fingerprint,
        "datasets": {
            fingerprint: {
                "adapter": adapter,
                "adapter_version": adapter_version,
                "fingerprint_inputs": fingerprint_payload,
                "artifacts": {
                    key: str(path.relative_to(cache_root)) for key, path in artifacts.items()
                },
            }
        },
    }
