from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cache import (
    dataset_cache_manifest_entry,
    dataset_fingerprint_payload,
    load_cached_dataset,
    merge_cache_manifest,
)
from src.data.canonical import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_ZONE_COLUMN,
    CanonicalDataset,
    DATASET_CACHE_SCHEMA_VERSION,
    DatasetSpec,
    build_price_change_reference,
    validate_canonical_timeseries,
    validate_static_zone_features,
)
from src.files import atomic_write_dataframe, atomic_write_json


class DatasetAdapter(ABC):
    """Adapter interface for reusable dataset ingestion and feature discovery."""

    name: str
    version: int

    @abstractmethod
    def source_files(self, spec: DatasetSpec) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def build(self, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        raise NotImplementedError

    def load(self, spec: DatasetSpec, *, force_cache: bool = False) -> CanonicalDataset:
        data_dir = Path(spec.path)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")
        cache_root = spec.resolved_cache_dir
        cache_root.mkdir(parents=True, exist_ok=True)
        sources = self.source_files(spec)
        fingerprint_payload = dataset_fingerprint_payload(
            data_dir,
            sources,
            adapter=self.name,
            adapter_version=self.version,
            mapping={
                "column_mapping": spec.column_mapping,
                "static_mapping": spec.static_mapping,
                "weather_file": spec.weather_file,
                "unit_conversions": spec.unit_conversions,
            },
        )
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        dataset_cache = cache_root / "datasets" / fingerprint
        artifacts = {
            "canonical_timeseries": dataset_cache / "canonical_timeseries.csv.gz",
            "static_zone_features": dataset_cache / "static_zone_features.csv",
            "poi_zone_counts": dataset_cache / "poi_zone_counts.csv",
            "canonical_schema": dataset_cache / "canonical_schema.json",
            "feature_manifest": dataset_cache / "feature_manifest.json",
            "price_change_reference": dataset_cache / "price_change_reference.json",
        }

        if not force_cache:
            cached = load_cached_dataset(
                dataset_cache=dataset_cache,
                artifacts=artifacts,
                fingerprint=fingerprint,
            )
            if cached is not None:
                merge_cache_manifest(
                    cache_root / "cache_manifest.json",
                    dataset_cache_manifest_entry(
                        cache_root,
                        fingerprint,
                        adapter=self.name,
                        adapter_version=self.version,
                        fingerprint_payload=fingerprint_payload,
                        artifacts=artifacts,
                    ),
                )
                return replace(cached, source_path=data_dir.resolve())

        timeseries, static, feature_manifest = self.build(spec)
        for semantic, factor in spec.unit_conversions.items():
            if semantic not in timeseries:
                raise ValueError(f"Unit conversion field does not exist: {semantic}")
            timeseries[semantic] = (
                pd.to_numeric(timeseries[semantic], errors="coerce") * float(factor)
            )
        if "temperature_price_interaction" in timeseries and "temperature" in timeseries:
            timeseries["temperature_price_interaction"] = (
                pd.to_numeric(timeseries["temperature"], errors="coerce")
                * pd.to_numeric(timeseries[CANONICAL_ENERGY_PRICE_COLUMN], errors="coerce")
            )
        timeseries = validate_canonical_timeseries(timeseries)
        static = validate_static_zone_features(static, timeseries)
        reference = build_price_change_reference(timeseries)
        feature_manifest = {
            "schema_version": DATASET_CACHE_SCHEMA_VERSION,
            "adapter": self.name,
            "adapter_version": self.version,
            "dataset_fingerprint": fingerprint,
            "unit_conversions": dict(spec.unit_conversions),
            **feature_manifest,
        }
        dataset_cache.mkdir(parents=True, exist_ok=True)
        atomic_write_dataframe(timeseries, artifacts["canonical_timeseries"], compression="gzip")
        atomic_write_dataframe(static, artifacts["static_zone_features"])
        poi_columns = [
            column
            for column in static.columns
            if column == CANONICAL_ZONE_COLUMN or column.startswith("poi_")
        ]
        atomic_write_dataframe(static[poi_columns], artifacts["poi_zone_counts"])
        atomic_write_json(
            artifacts["canonical_schema"],
            {
                "schema_version": DATASET_CACHE_SCHEMA_VERSION,
                "columns": [
                    {"name": column, "dtype": str(dtype)}
                    for column, dtype in timeseries.dtypes.items()
                ],
            },
        )
        atomic_write_json(artifacts["feature_manifest"], feature_manifest)
        atomic_write_json(artifacts["price_change_reference"], reference.to_dict())
        cache_manifest = dataset_cache_manifest_entry(
            cache_root,
            fingerprint,
            adapter=self.name,
            adapter_version=self.version,
            fingerprint_payload=fingerprint_payload,
            artifacts=artifacts,
        )
        merge_cache_manifest(cache_root / "cache_manifest.json", cache_manifest)
        return CanonicalDataset(
            timeseries=timeseries,
            static_zone_features=static,
            feature_manifest=feature_manifest,
            price_change_reference=reference,
            dataset_fingerprint=fingerprint,
            cache_dir=cache_root,
            cache_keys={"dataset": fingerprint},
            source_path=data_dir.resolve(),
        )
