from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adapters.base import DatasetAdapter
from src.data.canonical import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_LOAD_COLUMN,
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    DatasetSpec,
    LONG_FORMAT_ADAPTER_VERSION,
    discover_column,
)
from src.time_utils import normalize_datetime_series_24h


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
