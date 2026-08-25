from __future__ import annotations

import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from time_utils import normalize_datetime_series_24h


DATASET_CACHE_SCHEMA_VERSION = 1
URBAN_EV_ADAPTER_VERSION = 2
LONG_FORMAT_ADAPTER_VERSION = 2
CANONICAL_TIME_COLUMN = "timestamp"
CANONICAL_ZONE_COLUMN = "zone_id"
CANONICAL_LOAD_COLUMN = "load_kwh"
CANONICAL_ENERGY_PRICE_COLUMN = "energy_price"


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a dataset before it is converted to MAPF's canonical form."""

    path: Path
    adapter: str = "urbanev"
    weather_file: str = "weather_airport.csv"
    cache_dir: Path | None = None
    timeseries_file: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    static_file: str | None = None
    static_mapping: dict[str, str] = field(default_factory=dict)
    unit_conversions: dict[str, float] = field(default_factory=dict)

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir is not None else Path(self.path) / "cache"


@dataclass(frozen=True)
class PriceChangeReference:
    values_pct: tuple[float, ...]
    p95_pct: float | None
    observations: int
    nonzero_observations: int
    definition: str = (
        "absolute percentage change between consecutive non-overlapping 3-hour "
        "mean energy-price windows, pooled across zones"
    )

    def percentile(self, shift_pct: Any) -> float | None:
        try:
            value = abs(float(shift_pct))
        except (TypeError, ValueError):
            return None
        if not self.values_pct:
            return None
        values = np.asarray(self.values_pct, dtype=np.float64)
        return round(float(np.searchsorted(values, value, side="right") / len(values) * 100), 4)

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "definition": self.definition,
            "p95_pct": self.p95_pct,
            "observations": self.observations,
            "nonzero_observations": self.nonzero_observations,
        }
        if include_values:
            payload["values_pct"] = list(self.values_pct)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PriceChangeReference":
        raw_values = value.get("values_pct") or []
        return cls(
            values_pct=tuple(float(item) for item in raw_values),
            p95_pct=optional_float(value.get("p95_pct")),
            observations=int(value.get("observations") or len(raw_values)),
            nonzero_observations=int(
                value.get("nonzero_observations")
                or sum(float(item) > 0 for item in raw_values)
            ),
            definition=str(value.get("definition") or cls.definition),
        )


@dataclass(frozen=True)
class CanonicalDataset:
    timeseries: pd.DataFrame
    static_zone_features: pd.DataFrame
    feature_manifest: dict[str, Any]
    price_change_reference: PriceChangeReference
    dataset_fingerprint: str
    cache_dir: Path
    cache_keys: dict[str, str]

    def select_zones(self, zone_ids: Iterable[str]) -> "CanonicalDataset":
        requested = {str(zone_id) for zone_id in zone_ids}
        timeseries = self.timeseries[
            self.timeseries[CANONICAL_ZONE_COLUMN].astype(str).isin(requested)
        ].copy()
        static = self.static_zone_features[
            self.static_zone_features[CANONICAL_ZONE_COLUMN].astype(str).isin(requested)
        ].copy()
        return CanonicalDataset(
            timeseries=timeseries,
            static_zone_features=static,
            feature_manifest=dict(self.feature_manifest),
            price_change_reference=self.price_change_reference,
            dataset_fingerprint=self.dataset_fingerprint,
            cache_dir=self.cache_dir,
            cache_keys=dict(self.cache_keys),
        )

    def wide_frame(self, value_column: str, zone_ids: Iterable[str]) -> pd.DataFrame:
        zone_ids = [str(zone_id) for zone_id in zone_ids]
        if value_column not in self.timeseries:
            return pd.DataFrame(columns=["time", *zone_ids])
        selected = self.timeseries[
            self.timeseries[CANONICAL_ZONE_COLUMN].astype(str).isin(zone_ids)
        ][[CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN, value_column]].copy()
        wide = selected.pivot(
            index=CANONICAL_TIME_COLUMN,
            columns=CANONICAL_ZONE_COLUMN,
            values=value_column,
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(columns={CANONICAL_TIME_COLUMN: "time"})
        for zone_id in zone_ids:
            if zone_id not in wide:
                wide[zone_id] = np.nan
        return wide[["time", *zone_ids]].sort_values("time").reset_index(drop=True)

    def weather_frame(self) -> pd.DataFrame:
        manifest_weather = self.feature_manifest.get("weather_columns") or []
        columns = [column for column in manifest_weather if column in self.timeseries]
        if not columns:
            return pd.DataFrame({"time": sorted(self.timeseries[CANONICAL_TIME_COLUMN].unique())})
        frame = self.timeseries[[CANONICAL_TIME_COLUMN, *columns]].copy()
        frame = frame.groupby(CANONICAL_TIME_COLUMN, as_index=False)[columns].first()
        return frame.rename(columns={CANONICAL_TIME_COLUMN: "time"})


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
                return cached

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
        )


class UrbanEVDatasetAdapter(DatasetAdapter):
    name = "urbanev"
    version = URBAN_EV_ADAPTER_VERSION

    def source_files(self, spec: DatasetSpec) -> list[Path]:
        required = ["volume.csv", "e_price.csv"]
        optional = ["s_price.csv", "occupancy.csv", "inf.csv", "poi.csv", spec.weather_file]
        paths = [Path(spec.path) / name for name in required]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "UrbanEV requires load and energy-price files: "
                + ", ".join(str(path) for path in missing)
            )
        paths.extend(Path(spec.path) / name for name in optional if (Path(spec.path) / name).is_file())
        return paths

    def build(self, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        data_dir = Path(spec.path)
        load = read_wide_matrix(data_dir / "volume.csv", CANONICAL_LOAD_COLUMN)
        energy = read_wide_matrix(data_dir / "e_price.csv", CANONICAL_ENERGY_PRICE_COLUMN)
        timeseries = load.merge(
            energy,
            on=[CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN],
            how="inner",
            validate="one_to_one",
        )
        if timeseries.empty:
            raise ValueError("UrbanEV load and energy-price files have no aligned observations")
        if (pd.to_numeric(timeseries[CANONICAL_ENERGY_PRICE_COLUMN], errors="coerce") <= 0).any():
            raise ValueError("UrbanEV baseline energy price must be greater than zero")

        dynamic_columns: list[str] = []
        optional_wide = {
            "s_price.csv": "service_price",
            "occupancy.csv": "occupancy",
        }
        for filename, canonical_name in optional_wide.items():
            path = data_dir / filename
            if not path.is_file():
                continue
            values = read_wide_matrix(path, canonical_name)
            timeseries = timeseries.merge(
                values,
                on=[CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN],
                how="left",
                validate="one_to_one",
            )
            dynamic_columns.append(canonical_name)

        weather_columns: list[str] = []
        weather_path = data_dir / spec.weather_file
        if weather_path.is_file():
            weather = pd.read_csv(weather_path)
            time_column = discover_column(weather.columns, ["time", "timestamp", "datetime"], required=True)
            weather = weather.rename(columns={time_column: CANONICAL_TIME_COLUMN})
            weather[CANONICAL_TIME_COLUMN] = normalize_datetime_series_24h(
                weather[CANONICAL_TIME_COLUMN]
            )
            weather_aliases = {
                "temperature": ["temperature", "temp", "T"],
                "humidity": ["humidity", "relative_humidity", "U"],
                "rain": ["rain", "rainfall", "precipitation", "nRAIN"],
                "rain_probability": ["rain_probability", "precipitation_probability", "pop"],
            }
            rename: dict[str, str] = {}
            for canonical_name, aliases in weather_aliases.items():
                found = discover_column(weather.columns, aliases, required=False)
                if found is not None:
                    rename[found] = canonical_name
                    weather_columns.append(canonical_name)
            weather = weather.rename(columns=rename)
            keep = [CANONICAL_TIME_COLUMN, *weather_columns]
            timeseries = timeseries.merge(weather[keep], on=CANONICAL_TIME_COLUMN, how="left")
            dynamic_columns.extend(weather_columns)

        timeseries["hour"] = timeseries[CANONICAL_TIME_COLUMN].dt.hour.astype(int)
        timeseries["day_of_week"] = timeseries[CANONICAL_TIME_COLUMN].dt.dayofweek.astype(int)
        timeseries["is_weekend"] = timeseries["day_of_week"].isin([5, 6]).astype(int)
        dynamic_columns.extend(["hour", "day_of_week", "is_weekend"])
        if "temperature" in timeseries:
            timeseries["temperature_price_interaction"] = (
                pd.to_numeric(timeseries["temperature"], errors="coerce")
                * pd.to_numeric(timeseries[CANONICAL_ENERGY_PRICE_COLUMN], errors="coerce")
            )
            dynamic_columns.append("temperature_price_interaction")

        static = urban_ev_static_features(data_dir, timeseries)
        manifest = {
            "required_features": [
                CANONICAL_TIME_COLUMN,
                CANONICAL_ZONE_COLUMN,
                CANONICAL_LOAD_COLUMN,
                CANONICAL_ENERGY_PRICE_COLUMN,
            ],
            "known_future_features": [
                CANONICAL_ENERGY_PRICE_COLUMN,
                *[column for column in dynamic_columns if column != CANONICAL_LOAD_COLUMN],
            ],
            "dynamic_features": sorted(set(dynamic_columns)),
            "static_features": [
                column for column in static.columns if column != CANONICAL_ZONE_COLUMN
            ],
            "weather_columns": weather_columns,
            "missing_optional_features": [
                name
                for name in ["service_price", "occupancy", "temperature", "humidity", "rain"]
                if name not in timeseries
            ],
            "ignored_features": [],
            "row_count": int(len(timeseries)),
            "zone_count": int(timeseries[CANONICAL_ZONE_COLUMN].nunique()),
            "time_start": timeseries[CANONICAL_TIME_COLUMN].min().isoformat(),
            "time_end": timeseries[CANONICAL_TIME_COLUMN].max().isoformat(),
        }
        return timeseries, static, manifest


class LongFormatDatasetAdapter(DatasetAdapter):
    name = "long_format"
    version = LONG_FORMAT_ADAPTER_VERSION

    def source_files(self, spec: DatasetSpec) -> list[Path]:
        if not spec.timeseries_file:
            raise ValueError("Long-format datasets require data.timeseries_file")
        paths = [Path(spec.path) / spec.timeseries_file]
        if spec.static_file:
            paths.append(Path(spec.path) / spec.static_file)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Dataset source file(s) not found: " + ", ".join(map(str, missing)))
        return paths

    def build(self, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        path = Path(spec.path) / str(spec.timeseries_file)
        frame = pd.read_csv(path)
        aliases = {
            CANONICAL_TIME_COLUMN: ["timestamp", "time", "datetime", "date_time"],
            CANONICAL_ZONE_COLUMN: ["zone_id", "zone", "area_id", "item_id"],
            CANONICAL_LOAD_COLUMN: ["load_kwh", "load", "demand", "volume"],
            CANONICAL_ENERGY_PRICE_COLUMN: [
                "energy_price",
                "e_price",
                "electricity_price",
                "tariff",
            ],
        }
        rename: dict[str, str] = {}
        for semantic, candidates in aliases.items():
            explicit = spec.column_mapping.get(semantic)
            source = explicit or discover_column(frame.columns, candidates, required=True)
            if source not in frame:
                raise ValueError(f"Mapped {semantic} column does not exist: {source}")
            rename[str(source)] = semantic
        optional_aliases = {
            "temperature": ["temperature", "temp", "T"],
            "humidity": ["humidity", "relative_humidity", "U"],
            "rain": ["rain", "rainfall", "precipitation", "nRAIN"],
            "rain_probability": ["rain_probability", "precipitation_probability", "pop"],
            "occupancy": ["occupancy", "occupied", "utilization"],
            "service_price": ["service_price", "s_price", "service_fee"],
        }
        for semantic, candidates in optional_aliases.items():
            explicit = spec.column_mapping.get(semantic)
            source = explicit or discover_column(frame.columns, candidates, required=False)
            if source is not None:
                rename[str(source)] = semantic
        frame = frame.rename(columns=rename)
        frame[CANONICAL_TIME_COLUMN] = normalize_datetime_series_24h(
            frame[CANONICAL_TIME_COLUMN]
        )
        frame[CANONICAL_ZONE_COLUMN] = frame[CANONICAL_ZONE_COLUMN].astype(str)
        if (pd.to_numeric(frame[CANONICAL_ENERGY_PRICE_COLUMN], errors="coerce") <= 0).any():
            raise ValueError("Baseline energy price must be greater than zero")
        frame["hour"] = frame[CANONICAL_TIME_COLUMN].dt.hour.astype(int)
        frame["day_of_week"] = frame[CANONICAL_TIME_COLUMN].dt.dayofweek.astype(int)
        frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)

        static = pd.DataFrame(
            {CANONICAL_ZONE_COLUMN: sorted(frame[CANONICAL_ZONE_COLUMN].unique())}
        )
        if spec.static_file:
            static_raw = pd.read_csv(Path(spec.path) / spec.static_file)
            zone_source = spec.static_mapping.get(CANONICAL_ZONE_COLUMN) or discover_column(
                static_raw.columns,
                ["zone_id", "zone", "area_id", "item_id"],
                required=True,
            )
            static = static_raw.rename(columns={zone_source: CANONICAL_ZONE_COLUMN})
            static[CANONICAL_ZONE_COLUMN] = static[CANONICAL_ZONE_COLUMN].astype(str)

        weather_columns = [
            column for column in ["temperature", "humidity", "rain", "rain_probability"] if column in frame
        ]
        dynamic = [
            column
            for column in [
                CANONICAL_ENERGY_PRICE_COLUMN,
                "service_price",
                "occupancy",
                *weather_columns,
                "hour",
                "day_of_week",
                "is_weekend",
            ]
            if column in frame
        ]
        manifest = {
            "required_features": [
                CANONICAL_TIME_COLUMN,
                CANONICAL_ZONE_COLUMN,
                CANONICAL_LOAD_COLUMN,
                CANONICAL_ENERGY_PRICE_COLUMN,
            ],
            "known_future_features": dynamic,
            "dynamic_features": dynamic,
            "static_features": [column for column in static if column != CANONICAL_ZONE_COLUMN],
            "weather_columns": weather_columns,
            "missing_optional_features": [
                name
                for name in optional_aliases
                if name not in frame
            ],
            "ignored_features": [column for column in frame if column not in set(rename.values()) | {"hour", "day_of_week", "is_weekend"}],
            "row_count": int(len(frame)),
            "zone_count": int(frame[CANONICAL_ZONE_COLUMN].nunique()),
            "time_start": frame[CANONICAL_TIME_COLUMN].min().isoformat(),
            "time_end": frame[CANONICAL_TIME_COLUMN].max().isoformat(),
        }
        return frame, static, manifest


def adapter_for_spec(spec: DatasetSpec) -> DatasetAdapter:
    normalized = str(spec.adapter or "urbanev").strip().lower().replace("-", "_")
    if normalized in {"urbanev", "urban_ev"}:
        return UrbanEVDatasetAdapter()
    if normalized in {"long", "long_format", "generic"}:
        return LongFormatDatasetAdapter()
    raise ValueError(f"Unsupported dataset adapter: {spec.adapter}")


def load_canonical_dataset(spec: DatasetSpec, *, force_cache: bool = False) -> CanonicalDataset:
    return adapter_for_spec(spec).load(spec, force_cache=force_cache)


def read_wide_matrix(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = discover_column(frame.columns, ["time", "timestamp", "datetime"], required=True)
    frame = frame.rename(columns={time_column: CANONICAL_TIME_COLUMN})
    frame[CANONICAL_TIME_COLUMN] = normalize_datetime_series_24h(
        frame[CANONICAL_TIME_COLUMN]
    )
    zone_columns = [column for column in frame.columns if column != CANONICAL_TIME_COLUMN]
    if not zone_columns:
        raise ValueError(f"Wide matrix has no zone columns: {path}")
    long = frame.melt(
        id_vars=[CANONICAL_TIME_COLUMN],
        value_vars=zone_columns,
        var_name=CANONICAL_ZONE_COLUMN,
        value_name=value_name,
    )
    long[CANONICAL_ZONE_COLUMN] = long[CANONICAL_ZONE_COLUMN].astype(str)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long


def urban_ev_static_features(data_dir: Path, timeseries: pd.DataFrame) -> pd.DataFrame:
    zones = pd.DataFrame(
        {CANONICAL_ZONE_COLUMN: sorted(timeseries[CANONICAL_ZONE_COLUMN].astype(str).unique())}
    )
    path = data_dir / "inf.csv"
    if not path.is_file():
        return zones
    inf = pd.read_csv(path)
    zone_column = discover_column(inf.columns, ["TAZID", "zone_id", "zone"], required=True)
    aggregations: dict[str, tuple[str, str]] = {}
    candidates = {
        "station_count": ("station_id", "count"),
        "longitude": ("longitude", "mean"),
        "latitude": ("latitude", "mean"),
        "charge_count": ("charge_count", "sum"),
        "area": ("area", "first"),
        "perimeter": ("perimeter", "first"),
    }
    for output, (source, operation) in candidates.items():
        if source in inf:
            aggregations[output] = (source, operation)
    static = inf.groupby(zone_column).agg(**aggregations).reset_index()
    static = static.rename(columns={zone_column: CANONICAL_ZONE_COLUMN})
    static[CANONICAL_ZONE_COLUMN] = static[CANONICAL_ZONE_COLUMN].astype(str)
    if "charge_count" in static:
        static["capacity_kw_proxy"] = pd.to_numeric(static["charge_count"], errors="coerce").fillna(0) * 11.0
    static = zones.merge(static, on=CANONICAL_ZONE_COLUMN, how="left")
    poi_counts = urban_ev_poi_zone_counts(data_dir / "poi.csv", static)
    if not poi_counts.empty:
        static = static.merge(poi_counts, on=CANONICAL_ZONE_COLUMN, how="left")
        poi_columns = [column for column in static if column.startswith("poi_")]
        static[poi_columns] = static[poi_columns].fillna(0).astype(int)
    return static


def urban_ev_poi_zone_counts(poi_path: Path, zones: pd.DataFrame) -> pd.DataFrame:
    """Assign reusable POI counts to the nearest zone centroid."""

    required_zone_columns = {CANONICAL_ZONE_COLUMN, "longitude", "latitude"}
    if not poi_path.is_file() or not required_zone_columns.issubset(zones.columns):
        return pd.DataFrame(columns=[CANONICAL_ZONE_COLUMN])
    poi = pd.read_csv(poi_path)
    lon = discover_column(poi.columns, ["longitude", "lon", "lng"], required=False)
    lat = discover_column(poi.columns, ["latitude", "lat"], required=False)
    category = discover_column(
        poi.columns,
        ["primary_types", "primary_type", "category", "type"],
        required=False,
    )
    if lon is None or lat is None:
        return pd.DataFrame(columns=[CANONICAL_ZONE_COLUMN])
    centroids = zones[[CANONICAL_ZONE_COLUMN, "longitude", "latitude"]].dropna().copy()
    if centroids.empty:
        return pd.DataFrame(columns=[CANONICAL_ZONE_COLUMN])
    zone_xy = centroids[["longitude", "latitude"]].to_numpy(dtype=float)
    zone_ids = centroids[CANONICAL_ZONE_COLUMN].astype(str).to_numpy()
    assignments: list[str] = []
    poi_xy = poi[[lon, lat]].apply(pd.to_numeric, errors="coerce")
    valid = poi_xy.notna().all(axis=1)
    poi = poi.loc[valid].reset_index(drop=True)
    poi_xy = poi_xy.loc[valid].reset_index(drop=True)
    for start in range(0, len(poi_xy), 10_000):
        coordinates = poi_xy.iloc[start : start + 10_000].to_numpy(dtype=float)
        distances = ((coordinates[:, None, :] - zone_xy[None, :, :]) ** 2).sum(axis=2)
        assignments.extend(zone_ids[np.argmin(distances, axis=1)].tolist())
    if not assignments:
        return pd.DataFrame(columns=[CANONICAL_ZONE_COLUMN])
    assigned = pd.DataFrame({CANONICAL_ZONE_COLUMN: assignments})
    assigned["poi_total"] = 1
    if category is not None:
        normalized = poi[category].astype(str).str.strip().str.lower()
        assigned["poi_food"] = normalized.str.contains("food|beverage|restaurant").astype(int)
        assigned["poi_business"] = normalized.str.contains("business|residential|office").astype(int)
        assigned["poi_lifestyle"] = normalized.str.contains("lifestyle|service|leisure").astype(int)
    return assigned.groupby(CANONICAL_ZONE_COLUMN, as_index=False).sum(numeric_only=True)


def validate_canonical_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        CANONICAL_TIME_COLUMN,
        CANONICAL_ZONE_COLUMN,
        CANONICAL_LOAD_COLUMN,
        CANONICAL_ENERGY_PRICE_COLUMN,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Canonical dataset is missing required fields: " + ", ".join(sorted(missing)))
    result = frame.copy()
    result[CANONICAL_TIME_COLUMN] = normalize_datetime_series_24h(
        result[CANONICAL_TIME_COLUMN]
    )
    result[CANONICAL_ZONE_COLUMN] = result[CANONICAL_ZONE_COLUMN].astype(str)
    if result.duplicated([CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN]).any():
        raise ValueError("Canonical dataset contains duplicate timestamp/zone observations")
    result = result.sort_values([CANONICAL_ZONE_COLUMN, CANONICAL_TIME_COLUMN]).reset_index(drop=True)
    if result[CANONICAL_LOAD_COLUMN].isna().all():
        raise ValueError("Canonical dataset load is entirely missing")
    price = pd.to_numeric(result[CANONICAL_ENERGY_PRICE_COLUMN], errors="coerce")
    if price.isna().any() or (price <= 0).any():
        raise ValueError("Canonical baseline energy price must be present and greater than zero")
    return result


def validate_static_zone_features(static: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    result = static.copy()
    if CANONICAL_ZONE_COLUMN not in result:
        result = pd.DataFrame(
            {CANONICAL_ZONE_COLUMN: sorted(timeseries[CANONICAL_ZONE_COLUMN].astype(str).unique())}
        )
    result[CANONICAL_ZONE_COLUMN] = result[CANONICAL_ZONE_COLUMN].astype(str)
    result = result.drop_duplicates(CANONICAL_ZONE_COLUMN).reset_index(drop=True)
    return result


def build_price_change_reference(timeseries: pd.DataFrame) -> PriceChangeReference:
    prices = timeseries.pivot(
        index=CANONICAL_TIME_COLUMN,
        columns=CANONICAL_ZONE_COLUMN,
        values=CANONICAL_ENERGY_PRICE_COLUMN,
    ).sort_index()
    windows = prices.resample("3h", origin="start_day").mean()
    previous = windows.shift(1)
    changes = ((windows - previous) / previous).abs() * 100.0
    values = changes.where(previous > 0).stack().replace([np.inf, -np.inf], np.nan).dropna()
    sorted_values = np.sort(values.to_numpy(dtype=np.float64))
    rounded = tuple(round(float(value), 8) for value in sorted_values)
    p95 = round(float(np.quantile(sorted_values, 0.95)), 8) if len(sorted_values) else None
    return PriceChangeReference(
        values_pct=rounded,
        p95_pct=p95,
        observations=len(rounded),
        nonzero_observations=int(np.count_nonzero(sorted_values > 0)),
    )


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
        timeseries = pd.read_csv(artifacts["canonical_timeseries"], parse_dates=[CANONICAL_TIME_COLUMN])
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
        timeseries = validate_canonical_timeseries(timeseries)
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
    forecast_start: Any,
    *,
    window_hours: int,
    policy_id: str,
) -> str:
    start = pd.Timestamp(forecast_start).isoformat()
    payload = {
        "dataset": dataset_fingerprint,
        "forecast_start": start,
        "window_hours": int(window_hours),
        "policy_id": policy_id,
        "history_rule": "strictly_before_forecast_start",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def discover_column(columns: Iterable[Any], aliases: Iterable[str], *, required: bool) -> str | None:
    actual = [str(column) for column in columns]
    by_lower: dict[str, list[str]] = {}
    for column in actual:
        by_lower.setdefault(column.strip().lower(), []).append(column)
    matches: list[str] = []
    for alias in aliases:
        matches.extend(by_lower.get(str(alias).strip().lower(), []))
    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns for aliases {list(aliases)}: {matches}")
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"Required column not found; expected one of {list(aliases)}")
    return None


def atomic_write_dataframe(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp{suffix}"
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.json"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_cache_manifest(path: Path, update: dict[str, Any]) -> None:
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


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
