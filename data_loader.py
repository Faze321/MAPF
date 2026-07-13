from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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


def read_time_matrix(path: Path, zones: Iterable[str] | None = None) -> pd.DataFrame:
    usecols = None
    if zones is not None:
        usecols = ["time", *[str(zone) for zone in zones]]
    frame = pd.read_csv(path, usecols=usecols)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def read_weather(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
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
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_cache = cache_dir / "zone_profiles.csv"
    if profile_cache.exists() and not force_cache and max_poi_rows is None:
        cached = pd.read_csv(profile_cache, dtype={"zone_id": str})
        if "load_source_file" in cached.columns and (cached["load_source_file"] == LOAD_FILE).all():
            price_bound_columns = {
                "historical_min_service_price",
                "historical_max_service_price",
            }
            if not price_bound_columns.issubset(cached.columns):
                price = read_time_matrix(data_dir / "s_price.csv")
                price_zone_ids = [str(col) for col in price.columns if col != "time"]
                price_features = compute_price_features(price, price_zone_ids)
                update_columns = [
                    "mean_service_price",
                    "service_price_std",
                    "historical_min_service_price",
                    "historical_max_service_price",
                ]
                cached = cached.drop(
                    columns=[column for column in update_columns if column in cached.columns]
                ).merge(price_features, on="zone_id", how="left")
                cached.to_csv(profile_cache, index=False)
            return cached

    zone_ids = available_zone_ids(data_dir)
    load = read_time_matrix(data_dir / LOAD_FILE)
    price = read_time_matrix(data_dir / "s_price.csv")
    station_profiles = build_station_profiles(data_dir)
    poi_counts = build_poi_zone_counts(
        data_dir,
        station_profiles,
        cache_dir,
        force_cache=force_cache,
        max_poi_rows=max_poi_rows,
    )
    load_features = compute_load_features(load, zone_ids)
    price_features = compute_price_features(price, zone_ids)

    profiles = (
        pd.DataFrame({"zone_id": zone_ids})
        .merge(station_profiles, on="zone_id", how="left")
        .merge(poi_counts, on="zone_id", how="left")
        .merge(load_features, on="zone_id", how="left")
        .merge(price_features, on="zone_id", how="left")
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
        profiles.to_csv(profile_cache, index=False)
    return profiles


def build_zone_3h_load_quantiles(
    data_dir: Path,
    cache_dir: Path,
    *,
    force_cache: bool = False,
    source_file: str = LOAD_FILE,
    window_hours: int = 3,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "zone_3h_load_quantiles.csv"
    if cache_file.exists() and not force_cache:
        cached = pd.read_csv(cache_file, dtype={"zone_id": str})
        source_ok = "stress_source_file" in cached.columns and (cached["stress_source_file"] == source_file).all()
        window_ok = "stress_window_hours" in cached.columns and (cached["stress_window_hours"].astype(int) == int(window_hours)).all()
        if source_ok and window_ok:
            return cached

    load = read_time_matrix(data_dir / source_file)
    zone_ids = [str(col) for col in load.columns if col != "time"]
    values = load.set_index("time")[zone_ids].apply(pd.to_numeric, errors="coerce").sort_index()
    grouped = values.resample(f"{int(window_hours)}h", origin="start_day").sum(min_count=1)

    rows = []
    for zone_id in zone_ids:
        zone_values = grouped[zone_id].dropna()
        rows.append(
            {
                "zone_id": zone_id,
                "stress_source_file": source_file,
                "stress_window_hours": int(window_hours),
                "historical_3h_windows": int(len(zone_values)),
                "load_3h_q50_kwh": finite_float(zone_values.quantile(0.50)),
                "load_3h_q80_kwh": finite_float(zone_values.quantile(0.80)),
                "load_3h_q95_kwh": finite_float(zone_values.quantile(0.95)),
            }
        )
    quantiles = pd.DataFrame(rows)
    quantiles.to_csv(cache_file, index=False)
    return quantiles


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
        counts.to_csv(cache_file, index=False)
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


def compute_price_features(price: pd.DataFrame, zone_ids: list[str]) -> pd.DataFrame:
    values = price[zone_ids].astype(float).replace(0, np.nan)
    features = pd.DataFrame({"zone_id": zone_ids})
    features["mean_service_price"] = values.mean().fillna(0).to_numpy()
    features["service_price_std"] = values.std().fillna(0).to_numpy()
    features["historical_min_service_price"] = values.min().fillna(0).to_numpy()
    features["historical_max_service_price"] = values.max().fillna(0).to_numpy()
    return features


def load_pipeline_data(
    data_dir: Path,
    profiles: pd.DataFrame,
    selected_zone_ids: list[str],
    *,
    weather_file: str = "weather_airport.csv",
) -> PipelineData:
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

