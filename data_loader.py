from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dataset_adapter import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_LOAD_COLUMN,
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    CanonicalDataset,
    PriceChangeReference,
    atomic_write_dataframe,
)

from load_policy import (
    HIGH_MAX_LOAD_PCT,
    LOAD_POLICY_ID,
    LOW_MAX_LOAD_PCT,
    MEDIUM_MAX_LOAD_PCT,
    load_threshold_kwh,
)
from time_utils import normalize_datetime_series_24h


POI_COLUMNS = {
    "food and beverage services": "poi_food",
    "business and residential": "poi_business",
    "lifestyle services": "poi_lifestyle",
}
LOAD_FILE = "volume.csv"


@dataclass(frozen=True)
class PipelineData:
    load: pd.DataFrame
    service_price: pd.DataFrame
    energy_price: pd.DataFrame
    occupancy: pd.DataFrame
    weather: pd.DataFrame
    profiles: pd.DataFrame
    feature_manifest: dict | None = None
    dataset_fingerprint: str | None = None
    cache_keys: dict[str, str] | None = None
    price_change_reference: PriceChangeReference | None = None
    canonical_dataset: CanonicalDataset | None = None


def read_time_matrix(path: Path, zones: Iterable[str] | None = None) -> pd.DataFrame:
    usecols = None
    if zones is not None:
        usecols = ["time", *[str(zone) for zone in zones]]
    frame = pd.read_csv(path, usecols=usecols)
    frame["time"] = normalize_datetime_series_24h(frame["time"])
    return frame


def read_weather(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = normalize_datetime_series_24h(frame["time"])
    return frame


def available_zone_ids(data_dir: Path) -> list[str]:
    header = pd.read_csv(data_dir / LOAD_FILE, nrows=0)
    return [str(col) for col in header.columns if col != "time"]


def build_zone_profiles(
    data_dir: Path,
    cache_dir: Path,
    *,
    force_cache: bool = False,
    max_poi_rows: int | None = None,
    forecast_start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_cache = cache_dir / (
        "training_zone_profiles.csv" if forecast_start is not None else "zone_profiles.csv"
    )
    if profile_cache.exists() and not force_cache and max_poi_rows is None:
        cached = pd.read_csv(profile_cache, dtype={"zone_id": str})
        if "load_source_file" in cached.columns and (cached["load_source_file"] == LOAD_FILE).all():
            deprecated_energy_price_columns = {
                "historical_min_energy_price",
                "historical_max_energy_price",
            }
            deprecated_present = deprecated_energy_price_columns.intersection(cached.columns)
            if deprecated_present:
                cached = cached.drop(columns=sorted(deprecated_present))
            price_feature_columns = {
                "mean_service_price",
                "service_price_std",
                "historical_min_service_price",
                "historical_max_service_price",
                "mean_energy_price",
                "energy_price_std",
            }
            price_features_missing = not price_feature_columns.issubset(cached.columns)
            if price_features_missing:
                service_price = read_time_matrix(data_dir / "s_price.csv")
                energy_price = read_time_matrix(data_dir / "e_price.csv")
                price_zone_ids = [str(col) for col in service_price.columns if col != "time"]
                service_price_features = compute_price_features(
                    service_price, price_zone_ids, price_type="service"
                )
                energy_price_features = compute_price_features(
                    energy_price, price_zone_ids, price_type="energy"
                )
                cached = cached.drop(
                    columns=[column for column in price_feature_columns if column in cached.columns]
                )
                cached = cached.merge(service_price_features, on="zone_id", how="left")
                cached = cached.merge(energy_price_features, on="zone_id", how="left")
            if deprecated_present or price_features_missing:
                atomic_write_dataframe(cached, profile_cache)
            return cached

    zone_ids = available_zone_ids(data_dir)
    load = read_time_matrix(data_dir / LOAD_FILE)
    service_price = read_time_matrix(data_dir / "s_price.csv")
    energy_price = read_time_matrix(data_dir / "e_price.csv")
    if forecast_start is not None:
        cutoff = pd.Timestamp(forecast_start)
        load = load[load["time"] < cutoff].copy()
        service_price = service_price[service_price["time"] < cutoff].copy()
        energy_price = energy_price[energy_price["time"] < cutoff].copy()
    station_profiles = build_station_profiles(data_dir)
    poi_counts = build_poi_zone_counts(
        data_dir,
        station_profiles,
        cache_dir,
        force_cache=force_cache,
        max_poi_rows=max_poi_rows,
    )
    load_features = compute_load_features(load, zone_ids)
    service_price_features = compute_price_features(
        service_price, zone_ids, price_type="service"
    )
    energy_price_features = compute_price_features(
        energy_price, zone_ids, price_type="energy"
    )

    profiles = (
        pd.DataFrame({"zone_id": zone_ids})
        .merge(station_profiles, on="zone_id", how="left")
        .merge(poi_counts, on="zone_id", how="left")
        .merge(load_features, on="zone_id", how="left")
        .merge(service_price_features, on="zone_id", how="left")
        .merge(energy_price_features, on="zone_id", how="left")
    )

    for col in POI_COLUMNS.values():
        profiles[col] = profiles[col].fillna(0).astype(int)
    profiles["poi_total"] = profiles[list(POI_COLUMNS.values())].sum(axis=1)
    profiles["area_sq_km"] = (profiles["area"].fillna(0) / 1_000_000).replace(0, np.nan)
    for col in POI_COLUMNS.values():
        profiles[f"{col}_density"] = profiles[col] / profiles["area_sq_km"]
    profiles["poi_total_density"] = profiles["poi_total"] / profiles["area_sq_km"]
    profiles["capacity_kw_proxy"] = profiles["charge_count"].fillna(0) * 11.0
    profiles["peak_capacity_ratio"] = safe_divide(
        profiles["peak_load_kwh"].to_numpy(), profiles["capacity_kw_proxy"].to_numpy()
    )
    profiles = profiles.replace([np.inf, -np.inf], np.nan).fillna(0)
    profiles["load_source_file"] = LOAD_FILE

    if max_poi_rows is None:
        atomic_write_dataframe(profiles, profile_cache)
    return profiles


def build_zone_profiles_from_canonical(
    dataset: CanonicalDataset,
    cache_dir: Path,
    *,
    forecast_start: pd.Timestamp | str,
    force_cache: bool = False,
) -> pd.DataFrame:
    """Build forecast-origin-safe zone profiles from canonical data."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "training_zone_profiles.csv"
    if cache_file.is_file() and not force_cache:
        try:
            cached = pd.read_csv(cache_file, dtype={"zone_id": str})
            if "profile_cutoff" in cached and (
                cached["profile_cutoff"] == pd.Timestamp(forecast_start).isoformat()
            ).all():
                return cached
        except (OSError, pd.errors.ParserError, ValueError):
            pass

    cutoff = pd.Timestamp(forecast_start)
    history = dataset.timeseries[
        dataset.timeseries[CANONICAL_TIME_COLUMN] < cutoff
    ].copy()
    if history.empty:
        raise ValueError(f"No historical observations exist before forecast_start={cutoff}")
    load = history.pivot(
        index=CANONICAL_TIME_COLUMN,
        columns=CANONICAL_ZONE_COLUMN,
        values=CANONICAL_LOAD_COLUMN,
    ).reset_index().rename(columns={CANONICAL_TIME_COLUMN: "time"})
    load.columns.name = None
    zone_ids = [str(column) for column in load.columns if column != "time"]
    load_features = compute_load_features(load, zone_ids)

    price = history.pivot(
        index=CANONICAL_TIME_COLUMN,
        columns=CANONICAL_ZONE_COLUMN,
        values=CANONICAL_ENERGY_PRICE_COLUMN,
    ).reset_index().rename(columns={CANONICAL_TIME_COLUMN: "time"})
    price.columns.name = None
    energy_features = compute_price_features(price, zone_ids, price_type="energy")

    profiles = pd.DataFrame({"zone_id": zone_ids})
    static = dataset.static_zone_features.rename(columns={CANONICAL_ZONE_COLUMN: "zone_id"}).copy()
    static["zone_id"] = static["zone_id"].astype(str)
    profiles = profiles.merge(static, on="zone_id", how="left")
    profiles = profiles.merge(load_features, on="zone_id", how="left")
    profiles = profiles.merge(energy_features, on="zone_id", how="left")

    if "service_price" in history:
        service = history.pivot(
            index=CANONICAL_TIME_COLUMN,
            columns=CANONICAL_ZONE_COLUMN,
            values="service_price",
        ).reset_index().rename(columns={CANONICAL_TIME_COLUMN: "time"})
        service.columns.name = None
        profiles = profiles.merge(
            compute_price_features(service, zone_ids, price_type="service"),
            on="zone_id",
            how="left",
        )

    profile_defaults = {
        "station_count": 0,
        "charge_count": 0,
        "capacity_kw_proxy": 0.0,
        "area": 0.0,
        "longitude": 0.0,
        "latitude": 0.0,
        "poi_food": 0,
        "poi_business": 0,
        "poi_lifestyle": 0,
        "poi_total": 0,
        "mean_service_price": 0.0,
        "service_price_std": 0.0,
        "historical_min_service_price": 0.0,
        "historical_max_service_price": 0.0,
    }
    for column, default in profile_defaults.items():
        if column not in profiles:
            profiles[column] = default
        else:
            profiles[column] = profiles[column].fillna(default)
    area_sq_km = (pd.to_numeric(profiles["area"], errors="coerce").fillna(0) / 1_000_000).replace(0, np.nan)
    for column in ("poi_food", "poi_business", "poi_lifestyle"):
        profiles[f"{column}_density"] = pd.to_numeric(
            profiles[column], errors="coerce"
        ).fillna(0) / area_sq_km
    profiles["poi_total_density"] = pd.to_numeric(
        profiles["poi_total"], errors="coerce"
    ).fillna(0) / area_sq_km
    profiles["peak_capacity_ratio"] = safe_divide(
        profiles["peak_load_kwh"].to_numpy(),
        pd.to_numeric(profiles["capacity_kw_proxy"], errors="coerce").fillna(0).to_numpy(),
    )
    profiles = profiles.replace([np.inf, -np.inf], np.nan).fillna(0)
    profiles["load_source_file"] = "canonical_dataset"
    profiles["profile_cutoff"] = cutoff.isoformat()
    atomic_write_dataframe(profiles, cache_file)
    return profiles


def build_zone_3h_load_policy_thresholds(
    data_dir: Path,
    cache_dir: Path,
    *,
    force_cache: bool = False,
    source_file: str = LOAD_FILE,
    window_hours: int = 3,
    forecast_start: pd.Timestamp | str | None = None,
    load_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        "historical_3h_load_policy.csv"
        if forecast_start is not None
        else "zone_3h_load_quantiles.csv"
    )
    if cache_file.exists() and not force_cache:
        try:
            cached = pd.read_csv(cache_file, dtype={"zone_id": str})
        except (OSError, pd.errors.ParserError, ValueError):
            cached = pd.DataFrame()
        required_columns = {
            "load_policy_id",
            "historical_min_load_3h_kwh",
            "historical_max_load_3h_kwh",
            "historical_load_range_3h_kwh",
            "low_medium_threshold_pct",
            "medium_high_threshold_pct",
            "high_extremely_high_threshold_pct",
            "load_3h_low_medium_threshold_kwh",
            "load_3h_medium_high_threshold_kwh",
            "load_3h_high_extremely_high_threshold_kwh",
        }
        source_ok = "stress_source_file" in cached.columns and (
            cached["stress_source_file"] == source_file
        ).all()
        window_ok = "stress_window_hours" in cached.columns and (
            cached["stress_window_hours"].astype(int) == int(window_hours)
        ).all()
        policy_ok = required_columns.issubset(cached.columns) and (
            (cached["load_policy_id"] == LOAD_POLICY_ID).all()
            and np.isclose(cached["low_medium_threshold_pct"], LOW_MAX_LOAD_PCT).all()
            and np.isclose(cached["medium_high_threshold_pct"], MEDIUM_MAX_LOAD_PCT).all()
            and np.isclose(
                cached["high_extremely_high_threshold_pct"], HIGH_MAX_LOAD_PCT
            ).all()
        )
        cutoff_ok = True
        if forecast_start is not None:
            cutoff_ok = "forecast_start_cutoff" in cached.columns and (
                cached["forecast_start_cutoff"] == pd.Timestamp(forecast_start).isoformat()
            ).all()
        if source_ok and window_ok and policy_ok and cutoff_ok:
            return cached

    load = load_frame.copy() if load_frame is not None else read_time_matrix(data_dir / source_file)
    if forecast_start is not None:
        load = load[load["time"] < pd.Timestamp(forecast_start)].copy()
    zone_ids = [str(col) for col in load.columns if col != "time"]
    values = load.set_index("time")[zone_ids].apply(pd.to_numeric, errors="coerce").sort_index()
    grouped = values.resample(f"{int(window_hours)}h", origin="start_day").sum(min_count=1)

    rows = []
    for zone_id in zone_ids:
        zone_values = grouped[zone_id].dropna()
        historical_min = finite_float(zone_values.min())
        historical_max = finite_float(zone_values.max())
        rows.append(
            {
                "zone_id": zone_id,
                "stress_source_file": source_file,
                "stress_window_hours": int(window_hours),
                "forecast_start_cutoff": (
                    pd.Timestamp(forecast_start).isoformat()
                    if forecast_start is not None
                    else None
                ),
                "historical_3h_windows": int(len(zone_values)),
                "load_policy_id": LOAD_POLICY_ID,
                "historical_min_load_3h_kwh": historical_min,
                "historical_max_load_3h_kwh": historical_max,
                "historical_load_range_3h_kwh": finite_float(
                    historical_max - historical_min
                ),
                "low_medium_threshold_pct": LOW_MAX_LOAD_PCT,
                "medium_high_threshold_pct": MEDIUM_MAX_LOAD_PCT,
                "high_extremely_high_threshold_pct": HIGH_MAX_LOAD_PCT,
                "load_3h_low_medium_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, LOW_MAX_LOAD_PCT)
                ),
                "load_3h_medium_high_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, MEDIUM_MAX_LOAD_PCT)
                ),
                "load_3h_high_extremely_high_threshold_kwh": finite_float(
                    load_threshold_kwh(historical_min, historical_max, HIGH_MAX_LOAD_PCT)
                ),
            }
        )
    thresholds = pd.DataFrame(rows)
    atomic_write_dataframe(thresholds, cache_file)
    return thresholds


def build_station_profiles(data_dir: Path) -> pd.DataFrame:
    inf = pd.read_csv(data_dir / "inf.csv")
    grouped = (
        inf.groupby("TAZID")
        .agg(
            station_count=("station_id", "count"),
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
            charge_count=("charge_count", "sum"),
            area=("area", "first"),
            perimeter=("perimeter", "first"),
        )
        .reset_index()
        .rename(columns={"TAZID": "zone_id"})
    )
    grouped["zone_id"] = grouped["zone_id"].astype(str)
    return grouped


def build_poi_zone_counts(
    data_dir: Path,
    station_profiles: pd.DataFrame,
    cache_dir: Path,
    *,
    force_cache: bool = False,
    max_poi_rows: int | None = None,
    chunk_size: int = 25_000,
) -> pd.DataFrame:
    cache_file = cache_dir / "poi_zone_counts.csv"
    if cache_file.exists() and not force_cache and max_poi_rows is None:
        return pd.read_csv(cache_file, dtype={"zone_id": str})

    poi = pd.read_csv(
        data_dir / "poi.csv",
        usecols=["primary_types", "longitude", "latitude"],
        nrows=max_poi_rows,
    )
    centers = station_profiles[["zone_id", "longitude", "latitude"]].copy()
    center_xy = centers[["longitude", "latitude"]].to_numpy(dtype=float)
    center_ids = centers["zone_id"].to_numpy()

    assigned_zone_ids: list[np.ndarray] = []
    lat_scale = np.cos(np.deg2rad(np.nanmean(center_xy[:, 1])))
    for start in range(0, len(poi), chunk_size):
        coords = poi.iloc[start : start + chunk_size][["longitude", "latitude"]].to_numpy(dtype=float)
        lon_delta = (coords[:, [0]] - center_xy[:, 0]) * lat_scale
        lat_delta = coords[:, [1]] - center_xy[:, 1]
        nearest = np.argmin(lon_delta * lon_delta + lat_delta * lat_delta, axis=1)
        assigned_zone_ids.append(center_ids[nearest])

    poi = poi.assign(zone_id=np.concatenate(assigned_zone_ids) if assigned_zone_ids else [])
    counts = (
        poi.groupby(["zone_id", "primary_types"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=POI_COLUMNS)
        .reset_index()
    )
    for col in POI_COLUMNS.values():
        if col not in counts:
            counts[col] = 0
    counts = counts[["zone_id", *POI_COLUMNS.values()]]
    if max_poi_rows is None:
        atomic_write_dataframe(counts, cache_file)
    return counts


def compute_load_features(load: pd.DataFrame, zone_ids: list[str]) -> pd.DataFrame:
    time = load["time"]
    values = load[zone_ids].astype(float)
    total_mean = values.mean()

    features = pd.DataFrame({"zone_id": zone_ids})
    features["mean_load_kwh"] = total_mean.to_numpy()
    features["peak_load_kwh"] = values.max().to_numpy()
    features["load_cv"] = safe_divide(values.std().to_numpy(), total_mean.to_numpy())
    features["burstiness_p99_mean"] = safe_divide(values.quantile(0.99).to_numpy(), total_mean.to_numpy())

    windows = {
        "morning_ratio": time.dt.hour.between(7, 10),
        "noon_ratio": time.dt.hour.between(11, 14),
        "evening_ratio": time.dt.hour.between(17, 22),
        "night_ratio": (time.dt.hour >= 20) | (time.dt.hour <= 6),
    }
    for name, mask in windows.items():
        features[name] = safe_divide(values.loc[mask].mean().to_numpy(), total_mean.to_numpy())

    weekend = time.dt.dayofweek >= 5
    weekday_mean = values.loc[~weekend].mean().to_numpy()
    weekend_mean = values.loc[weekend].mean().to_numpy()
    features["weekend_ratio"] = safe_divide(weekend_mean, weekday_mean)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_price_features(
    price: pd.DataFrame,
    zone_ids: list[str],
    *,
    price_type: str = "service",
) -> pd.DataFrame:
    if price_type not in {"service", "energy"}:
        raise ValueError(f"Unsupported price_type: {price_type}")
    values = price[zone_ids].astype(float).replace(0, np.nan)
    features = pd.DataFrame({"zone_id": zone_ids})
    features[f"mean_{price_type}_price"] = values.mean().fillna(0).to_numpy()
    features[f"{price_type}_price_std"] = values.std().fillna(0).to_numpy()
    if price_type == "service":
        features["historical_min_service_price"] = values.min().fillna(0).to_numpy()
        features["historical_max_service_price"] = values.max().fillna(0).to_numpy()
    return features


def load_pipeline_data(
    data_dir: Path,
    profiles: pd.DataFrame,
    selected_zone_ids: list[str],
    *,
    weather_file: str = "weather_airport.csv",
    canonical_dataset: CanonicalDataset | None = None,
) -> PipelineData:
    if canonical_dataset is not None:
        selected = canonical_dataset.select_zones(selected_zone_ids)
        load = selected.wide_frame(CANONICAL_LOAD_COLUMN, selected_zone_ids)
        energy_price = selected.wide_frame(CANONICAL_ENERGY_PRICE_COLUMN, selected_zone_ids)
        service_price = selected.wide_frame("service_price", selected_zone_ids)
        occupancy = selected.wide_frame("occupancy", selected_zone_ids)
        weather = selected.weather_frame().rename(
            columns={
                "temperature": "T",
                "humidity": "U",
                "rain": "nRAIN",
            }
        )
        return PipelineData(
            load=load,
            service_price=service_price,
            energy_price=energy_price,
            occupancy=occupancy,
            weather=weather,
            profiles=profiles,
            feature_manifest=selected.feature_manifest,
            dataset_fingerprint=selected.dataset_fingerprint,
            cache_keys=selected.cache_keys,
            price_change_reference=selected.price_change_reference,
            canonical_dataset=selected,
        )
    load = read_time_matrix(data_dir / LOAD_FILE, selected_zone_ids)
    service_price = read_time_matrix(data_dir / "s_price.csv", selected_zone_ids)
    energy_price = read_time_matrix(data_dir / "e_price.csv", selected_zone_ids)
    occupancy = read_time_matrix(data_dir / "occupancy.csv", selected_zone_ids)
    weather = read_weather(data_dir / weather_file)
    return PipelineData(
        load=load,
        service_price=service_price,
        energy_price=energy_price,
        occupancy=occupancy,
        weather=weather,
        profiles=profiles,
    )


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


def finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return round(number, 4)

