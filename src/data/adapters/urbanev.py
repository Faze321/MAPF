from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.adapters.base import DatasetAdapter
from src.data.canonical import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_LOAD_COLUMN,
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    DatasetSpec,
    URBAN_EV_ADAPTER_VERSION,
    discover_column,
    read_wide_matrix,
)
from src.time_utils import normalize_datetime_series_24h


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
