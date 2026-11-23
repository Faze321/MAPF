from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.run import normalize_zone_selection


CATEGORY_ORDER = [
    "CBD / Office",
    "Residential",
    "Transport Hub",
    "Commercial / Mall",
    "Industrial",
]

SCORE_FEATURES = {
    "poi_business_density",
    "poi_food_density",
    "poi_lifestyle_density",
    "morning_ratio",
    "noon_ratio",
    "evening_ratio",
    "night_ratio",
    "weekend_ratio",
    "mean_load_kwh",
    "peak_load_kwh",
    "charge_count",
    "burstiness_p99_mean",
    "load_cv",
}


def select_zone_categories(profiles: pd.DataFrame) -> pd.DataFrame:
    frame = validated_profiles(profiles)
    scores = pd.DataFrame({"zone_id": frame["zone_id"].astype(str)})
    scores["CBD / Office"] = (
        1.20 * winsorized_z(frame["poi_business_density"])
        + 0.90 * winsorized_z(frame["morning_ratio"])
        + 0.70 * winsorized_z(frame["noon_ratio"])
        + 0.15 * winsorized_z(frame["mean_load_kwh"])
        - 0.40 * winsorized_z(frame["night_ratio"])
    )
    scores["Residential"] = (
        1.30 * winsorized_z(frame["night_ratio"])
        + 0.80 * winsorized_z(frame["evening_ratio"])
        - 0.30 * winsorized_z(frame["morning_ratio"])
        + 0.20 * winsorized_z(frame["poi_business_density"])
        - 0.25 * winsorized_z(frame["poi_food_density"])
        - 0.25 * winsorized_z(frame["poi_lifestyle_density"])
    )
    scores["Transport Hub"] = (
        1.10 * winsorized_z(frame["charge_count"])
        + 1.00 * winsorized_z(frame["burstiness_p99_mean"])
        + 0.60 * winsorized_z(frame["load_cv"])
        + 0.25 * winsorized_z(frame["peak_load_kwh"])
    )
    scores["Commercial / Mall"] = (
        1.00 * winsorized_z(frame["poi_food_density"])
        + 1.00 * winsorized_z(frame["poi_lifestyle_density"])
        + 0.70 * winsorized_z(frame["weekend_ratio"])
        + 0.60 * winsorized_z(frame["evening_ratio"])
        + 0.15 * winsorized_z(frame["mean_load_kwh"])
    )
    scores["Industrial"] = (
        0.80 * winsorized_z(frame["charge_count"])
        + 1.00 * winsorized_z(frame["mean_load_kwh"])
        - 1.10 * winsorized_z(frame["load_cv"])
        + 0.60 * winsorized_z(1.0 - (frame["weekend_ratio"] - 1.0).abs())
        - 0.25 * winsorized_z(frame["burstiness_p99_mean"])
    )

    assignments = optimal_unique_assignment(scores)
    selected_rows = []
    by_zone = frame.set_index("zone_id", drop=False)
    for category_index, category in enumerate(CATEGORY_ORDER):
        zone_index = assignments[category_index]
        zone_id = str(scores.iloc[zone_index]["zone_id"])
        profile = by_zone.loc[zone_id].to_dict()
        selected_rows.append(
            {
                **profile,
                "category": category,
                "selection_score": float(scores.iloc[zone_index][category]),
                "selection_reason": selection_reason(category, profile),
            }
        )

    selected = pd.DataFrame(selected_rows)
    output_columns = [
        "category",
        "zone_id",
        "selection_score",
        "selection_reason",
        "longitude",
        "latitude",
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "noon_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
        "mean_energy_price",
    ]
    return selected[[column for column in output_columns if column in selected.columns]]


def filter_training_profiles(profiles: pd.DataFrame, eligible: Iterable[str], *, empty_error: str | None = None) -> pd.DataFrame:
    selected = profiles[profiles["zone_id"].astype(str).isin(eligible)]
    if selected.empty and empty_error is not None:
        raise ValueError(empty_error)
    return selected


def validated_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if len(profiles) < len(CATEGORY_ORDER):
        raise ValueError(
            f"At least {len(CATEGORY_ORDER)} zones are required for category selection; "
            f"received {len(profiles)}."
        )
    missing = sorted(({"zone_id"} | SCORE_FEATURES) - set(profiles.columns))
    if missing:
        raise ValueError(f"Zone profiles are missing required selection columns: {', '.join(missing)}")

    frame = profiles.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    if frame["zone_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["zone_id"].duplicated(), "zone_id"].unique())
        raise ValueError(f"Zone profiles contain duplicate zone ids: {', '.join(duplicates)}")
    for column in SCORE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() == 0:
            raise ValueError(f"Zone selection feature {column} contains no numeric values.")
    return frame.reset_index(drop=True)


def winsorized_z(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    median = float(np.nanmedian(arr[finite]))
    filled = np.where(finite, arr, median)
    lower, upper = np.nanpercentile(filled, [5, 95])
    clipped = np.clip(filled, lower, upper)
    std = np.nanstd(clipped)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr, dtype=float)
    return (clipped - np.nanmean(clipped)) / std


def optimal_unique_assignment(scores: pd.DataFrame) -> dict[int, int]:
    values = scores[CATEGORY_ORDER].to_numpy(dtype=float).T
    full_mask = (1 << len(CATEGORY_ORDER)) - 1
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}

    for zone_index in range(values.shape[1]):
        updated = dict(states)
        for mask, (total, assignments) in states.items():
            for category_index in range(len(CATEGORY_ORDER)):
                bit = 1 << category_index
                if mask & bit:
                    continue
                candidate_mask = mask | bit
                candidate_total = total + values[category_index, zone_index]
                current = updated.get(candidate_mask)
                if current is None or candidate_total > current[0] + 1e-12:
                    updated[candidate_mask] = (
                        candidate_total,
                        assignments + ((category_index, zone_index),),
                    )
        states = updated

    if full_mask not in states:
        raise ValueError("Unable to assign a unique zone to every category.")
    return {category_index: zone_index for category_index, zone_index in states[full_mask][1]}


def selection_reason(category: str, profile: dict) -> str:
    if category == "CBD / Office":
        return (
            "High business/residential POI density with stronger morning and noon demand signals."
        )
    if category == "Residential":
        return (
            "Strong night/evening charging with limited food/lifestyle POI dominance, "
            "used as a residential demand proxy."
        )
    if category == "Transport Hub":
        return "High charging capacity and bursty load profile, used as a hub proxy."
    if category == "Commercial / Mall":
        return "High food/lifestyle POI density with evening or weekend demand lift."
    return "Stable high base load and capacity with limited burstiness, used as an industrial proxy."


def select_experiment_zone_ids(
    profiles: pd.DataFrame, *, count: int, selection: str = "representative", seed: int = 42,
) -> list[str]:
    """Select once for all model/mode combinations without changing model RNG state."""
    if normalize_zone_selection(selection) == "representative":
        return select_representative_zone_ids(profiles, count=count)
    candidates = sorted(profiles["zone_id"].astype(str).unique())
    if not candidates:
        raise ValueError("Cannot select random zones from an empty profile table.")
    return random.Random(seed).sample(candidates, k=max(1, min(int(count), len(candidates))))


def select_representative_zone_ids(profiles: pd.DataFrame, *, count: int) -> list[str]:
    frame = profiles.copy()
    if frame.empty:
        raise ValueError("Cannot select representative zones from an empty profile table.")
    frame["zone_id"] = frame["zone_id"].astype(str)
    target = max(1, min(int(count), len(frame)))
    if len(frame) < 5 or "station_type" in frame:
        return frame.sort_values("zone_id")["zone_id"].head(target).tolist()
    selected: list[str] = []
    seen: set[str] = set()

    for zone_id in select_zone_categories(frame)["zone_id"].astype(str).tolist():
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            return selected

    ranked = frame.sort_values(["mean_load_kwh", "load_cv", "peak_capacity_ratio", "zone_id"]).reset_index(drop=True)
    if target == 1:
        candidate_indices = [len(ranked) // 2]
    else:
        candidate_indices = [
            round(idx * (len(ranked) - 1) / (target - 1))
            for idx in range(target)
        ]
    for idx in candidate_indices:
        zone_id = str(ranked.iloc[int(idx)]["zone_id"])
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            return selected

    for zone_id in ranked["zone_id"].astype(str).tolist():
        if zone_id not in seen:
            selected.append(zone_id)
            seen.add(zone_id)
        if len(selected) >= target:
            break
    return selected


def normalize_zone_ids(zone_ids: str | Iterable[str] | None) -> list[str]:
    if zone_ids is None:
        return []

    raw_values = [zone_ids] if isinstance(zone_ids, str) else list(zone_ids)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            zone_id = part.strip()
            if zone_id and zone_id not in seen:
                normalized.append(zone_id)
                seen.add(zone_id)
    return normalized


def select_requested_zones(profiles: pd.DataFrame, zone_ids: Iterable[str]) -> pd.DataFrame:
    requested = normalize_zone_ids(zone_ids)
    if not requested:
        raise ValueError("At least one zone id is required.")

    frame = profiles.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    available = set(frame["zone_id"])
    missing = [zone_id for zone_id in requested if zone_id not in available]
    if missing:
        examples = ", ".join(sorted(available)[:10])
        raise ValueError(
            f"Unknown zone id(s): {', '.join(missing)}. "
            f"Available zone id examples: {examples}"
        )

    selected = frame.set_index("zone_id", drop=False).loc[requested].reset_index(drop=True)
    selected.insert(0, "category", selected["station_type"] if "station_type" in selected else "User-selected")
    selected.insert(2, "selection_score", None)
    selected.insert(3, "selection_reason", "User-specified zone for direct validation.")

    preferred_columns = [
        "category",
        "zone_id",
        "selection_score",
        "selection_reason",
        "longitude",
        "latitude",
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "noon_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
        "mean_energy_price",
    ]
    return selected[[col for col in preferred_columns if col in selected.columns]]
