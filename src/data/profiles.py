from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.control.policy import LOAD_POLICY_ID
from src.data.cache import split_cache_key
from src.data.canonical import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_LOAD_COLUMN,
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    CanonicalDataset,
    PriceChangeReference,
    pivot_timeseries,
)
from src.files import atomic_write_dataframe


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


def profiles_at_origin(dataset, cache_root: Path, origin: pd.Timestamp, *, force_cache: bool):
    """All stages use the same fingerprint, cutoff and policy for a given profile."""
    key = split_cache_key(dataset.dataset_fingerprint, origin, window_hours=3, policy_id=LOAD_POLICY_ID)
    directory = cache_root / "datasets" / dataset.dataset_fingerprint / "splits" / key
    frame = build_zone_profiles_from_canonical(dataset, directory, forecast_origin=origin, force_cache=force_cache)
    return key, directory, frame


def build_zone_profiles_from_canonical(
    dataset: CanonicalDataset,
    cache_dir: Path,
    *,
    forecast_origin: pd.Timestamp | str,
    force_cache: bool = False,
) -> pd.DataFrame:
    """Build forecast-origin-safe zone profiles from canonical data."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "training_zone_profiles.csv"
    if cache_file.is_file() and not force_cache:
        try:
            cached = pd.read_csv(cache_file, dtype={"zone_id": str})
            if "profile_cutoff" in cached and (
                cached["profile_cutoff"] == pd.Timestamp(forecast_origin).isoformat()
            ).all():
                return cached
        except (OSError, pd.errors.ParserError, ValueError):
            pass

    cutoff = pd.Timestamp(forecast_origin)
    history = dataset.timeseries[
        dataset.timeseries[CANONICAL_TIME_COLUMN] < cutoff
    ].copy()
    if history.empty:
        raise ValueError(f"No historical observations exist before forecast_origin={cutoff}")
    load = pivot_timeseries(history, CANONICAL_LOAD_COLUMN)
    zone_ids = [str(column) for column in load.columns if column != "time"]
    load_features = compute_load_features(load, zone_ids)

    price = pivot_timeseries(history, CANONICAL_ENERGY_PRICE_COLUMN)
    energy_features = compute_price_features(price, zone_ids, price_type="energy")

    profiles = pd.DataFrame({"zone_id": zone_ids})
    static = dataset.static_zone_features.rename(columns={CANONICAL_ZONE_COLUMN: "zone_id"}).copy()
    static["zone_id"] = static["zone_id"].astype(str)
    profiles = profiles.merge(static, on="zone_id", how="left")
    profiles = profiles.merge(load_features, on="zone_id", how="left")
    profiles = profiles.merge(energy_features, on="zone_id", how="left")

    if "service_price" in history:
        service = pivot_timeseries(history, "service_price")
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
    profiles: pd.DataFrame,
    selected_zone_ids: list[str],
    *,
    canonical_dataset: CanonicalDataset,
) -> PipelineData:
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


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)
