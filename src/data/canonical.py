from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.paths import default_dataset_cache_dir
from src.time_utils import normalize_datetime_series_24h


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
    allow_zero_energy_price: bool = False

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir is not None else default_dataset_cache_dir(self.path)


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
    source_path: Path | None = None

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
            source_path=self.source_path,
        )

    def wide_frame(self, value_column: str, zone_ids: Iterable[str]) -> pd.DataFrame:
        zone_ids = [str(zone_id) for zone_id in zone_ids]
        if value_column not in self.timeseries:
            return pd.DataFrame(columns=["time", *zone_ids])
        selected = self.timeseries[
            self.timeseries[CANONICAL_ZONE_COLUMN].astype(str).isin(zone_ids)
        ][[CANONICAL_TIME_COLUMN, CANONICAL_ZONE_COLUMN, value_column]].copy()
        wide = pivot_timeseries(selected, value_column)
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


def pivot_timeseries(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Convert canonical records to the existing time-by-zone matrix format."""
    wide = frame.pivot(
        index=CANONICAL_TIME_COLUMN, columns=CANONICAL_ZONE_COLUMN, values=value_column,
    ).reset_index().rename(columns={CANONICAL_TIME_COLUMN: "time"})
    wide.columns.name = None
    return wide


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


def valid_energy_prices(prices: pd.Series, allow_zero: bool) -> pd.Series:
    return prices.notna() & np.isfinite(prices) & (prices.ge(0) if allow_zero else prices.gt(0))


def validate_canonical_timeseries(frame: pd.DataFrame, *, allow_zero_energy_price: bool = False) -> pd.DataFrame:
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
    if not valid_energy_prices(price, allow_zero_energy_price).all():
        required = "non-negative" if allow_zero_energy_price else "greater than zero"
        raise ValueError(f"Canonical baseline energy price must be finite and {required}")
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


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def require_files(root: Path, names: list[str]) -> list[Path]:
    paths = [root / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset inputs; run python src/download_datasets.py: " + ", ".join(missing))
    return paths


def hourly_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns[0] not in {"time", "timestamp", "datetime", "Unnamed: 0"}:
        raise ValueError(f"Unrecognized time column in {path}")
    times = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    if times.isna().any() or times.duplicated().any() or not times.equals(times.dt.floor("h")):
        raise ValueError(f"Invalid hourly timestamps in {path}")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    values.index = pd.DatetimeIndex(times, name="timestamp")
    values.columns = values.columns.astype(str)
    values = values.sort_index()
    if len(values) > 1 and not (values.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"Missing hourly records in {path}; no implicit interpolation is performed")
    return values


def finalize(frame, static, *, excluded, **metadata):
    if frame.empty:
        raise ValueError(f"No usable positive-price charging sites remain: {excluded}")
    frame["hour"] = frame.timestamp.dt.hour
    frame["day_of_week"] = frame.timestamp.dt.dayofweek
    frame["is_weekend"] = (frame.day_of_week >= 5).astype(int)
    if "temperature" in frame:
        frame["temperature_price_interaction"] = frame.temperature * frame.energy_price
    required = ["timestamp", "zone_id", "load_kwh", "energy_price"]
    weather = [c for c in ("temperature", "humidity", "rain") if c in frame]
    dynamic = [c for c in frame if c not in {"timestamp", "zone_id", "load_kwh"}]
    manifest = dict(required_features=required, known_future_features=dynamic,
                    dynamic_features=dynamic, static_features=[c for c in static if c != "zone_id"],
                    weather_columns=weather, missing_optional_features=[c for c in ("temperature", "humidity", "rain", "occupancy") if c not in frame],
                    ignored_features=[], excluded_sites=excluded, row_count=len(frame),
                    zone_count=frame.zone_id.nunique(), time_start=frame.timestamp.min().isoformat(),
                    time_end=frame.timestamp.max().isoformat(), **metadata)
    return frame, static, manifest


def finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return round(number, 4)
