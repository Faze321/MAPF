"""Cache the historical-data checks used by random experimental site selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dataset_adapter import (
    CANONICAL_ENERGY_PRICE_COLUMN,
    CANONICAL_LOAD_COLUMN,
    CANONICAL_TIME_COLUMN,
    CANONICAL_ZONE_COLUMN,
    CanonicalDataset,
    atomic_write_dataframe,
    atomic_write_json,
    process_file_lock,
)


POLICY_VERSION = 1
CONSTANT_RTOL = 1e-9
CONSTANT_ATOL = 1e-12


def load_zone_eligibility(
    dataset: CanonicalDataset, *, forecast_starts: Iterable[str], history_days: int,
    validation_days: int, force_cache: bool = False,
) -> dict[str, Any]:
    """Read or build a versioned eligibility report without altering raw observations."""
    if isinstance(history_days, bool) or int(history_days) != history_days or history_days < 1:
        raise ValueError("history_days must be a positive integer")
    if isinstance(validation_days, bool) or int(validation_days) != validation_days or validation_days < 0:
        raise ValueError("validation_days must be a non-negative integer")
    starts = sorted({pd.Timestamp(start) for start in forecast_starts})
    if not starts or any(pd.isna(start) or start != start.floor("h") for start in starts):
        raise ValueError("forecast_starts must contain valid whole-hour timestamps")
    policy = {
        "version": POLICY_VERSION,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "forecast_starts": [start.isoformat() for start in starts],
        "history_days": int(history_days),
        "validation_days": int(validation_days),
        "variation_scope": "all_history_before_first_validation",
        "variation_cutoff": (starts[0] - pd.Timedelta(days=validation_days)).isoformat(),
        "constant_rtol": CONSTANT_RTOL,
        "constant_atol": CONSTANT_ATOL,
    }
    policy_key = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache_root = Path(dataset.cache_dir).resolve()
    folder = cache_root / "datasets" / dataset.dataset_fingerprint / "zone_selection" / policy_key
    cache_path = folder / "zone_eligibility.json"
    csv_path = folder / "zone_eligibility.csv"
    with process_file_lock(folder / "zone_eligibility.lock", blocking=True):
        report = None if force_cache else _read_report(cache_path, policy)
        if report is None:
            report = _build_report(dataset, policy)
            report.update(policy_key=policy_key, cache_path=str(cache_path), csv_path=str(csv_path))
            atomic_write_json(cache_path, report)
            _write_csv(report, csv_path)
        elif not csv_path.is_file():
            _write_csv(report, csv_path)
        _update_manifest(cache_root, report)
    return report


def _series_statistics(values: pd.Series) -> dict[str, Any]:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numbers[np.isfinite(numbers)]
    minimum = float(finite.min()) if finite.size else None
    maximum = float(finite.max()) if finite.size else None
    return {
        "observations": int(len(numbers)),
        "finite_observations": int(finite.size),
        "minimum": minimum,
        "maximum": maximum,
        "zero_observations": int(np.count_nonzero(finite == 0)),
        "constant": bool(np.isclose(maximum, minimum, rtol=CONSTANT_RTOL, atol=CONSTANT_ATOL))
        if finite.size else None,
    }


def _build_report(dataset: CanonicalDataset, policy: dict[str, Any]) -> dict[str, Any]:
    frame = dataset.timeseries[[
        CANONICAL_ZONE_COLUMN, CANONICAL_TIME_COLUMN, CANONICAL_LOAD_COLUMN,
        CANONICAL_ENERGY_PRICE_COLUMN,
    ]].copy()
    frame[CANONICAL_ZONE_COLUMN] = frame[CANONICAL_ZONE_COLUMN].astype(str)
    frame[CANONICAL_TIME_COLUMN] = pd.to_datetime(frame[CANONICAL_TIME_COLUMN], errors="coerce")
    upstream = {
        str(zone): reason for zone, reason in (dataset.feature_manifest.get("excluded_sites") or {}).items()
    }
    zone_ids = sorted(set(frame[CANONICAL_ZONE_COLUMN]) |
                      set(dataset.static_zone_features[CANONICAL_ZONE_COLUMN].astype(str)) | set(upstream))
    groups = frame.groupby(CANONICAL_ZONE_COLUMN)
    cutoff = pd.Timestamp(policy["variation_cutoff"])
    expected_hours = policy["history_days"] * 24
    zones = {}
    for zone_id in zone_ids:
        if zone_id in upstream:
            zones[zone_id] = {
                "eligible": False, "reasons": ["upstream_excluded"],
                "variation_history": None, "training_windows": [],
            }
            continue
        group = (groups.get_group(zone_id) if zone_id in groups.indices else frame.iloc[:0]).sort_values(
            CANONICAL_TIME_COLUMN
        )
        history = group[group[CANONICAL_TIME_COLUMN] < cutoff]
        load_stats = _series_statistics(history[CANONICAL_LOAD_COLUMN])
        price_stats = _series_statistics(history[CANONICAL_ENERGY_PRICE_COLUMN])
        variation_history = {
            "start": history[CANONICAL_TIME_COLUMN].min().isoformat() if len(history) else None,
            "end_exclusive": cutoff.isoformat(),
            "observed_hours": len(history),
            "load_kwh": load_stats,
            "energy_price": price_stats,
        }
        reasons = []
        if load_stats["constant"]:
            reasons.append("constant_load")
        if price_stats["constant"]:
            reasons.append("constant_energy_price")
        training_windows = []
        for origin in policy["forecast_starts"]:
            end = pd.Timestamp(origin) - pd.Timedelta(days=policy["validation_days"])
            start = end - pd.Timedelta(days=policy["history_days"])
            window = group[(group[CANONICAL_TIME_COLUMN] >= start) & (group[CANONICAL_TIME_COLUMN] < end)]
            timestamps = pd.DatetimeIndex(window[CANONICAL_TIME_COLUMN])
            expected = pd.date_range(start, periods=expected_hours, freq="h")
            complete = timestamps.equals(expected)
            load = _series_statistics(window[CANONICAL_LOAD_COLUMN])
            price = _series_statistics(window[CANONICAL_ENERGY_PRICE_COLUMN])
            window_reasons = []
            if not complete:
                window_reasons.append("incomplete_training_window")
            if load["finite_observations"] != len(window) or (
                load["minimum"] is not None and load["minimum"] < 0
            ):
                window_reasons.append("invalid_training_load")
            if price["finite_observations"] != len(window) or (
                price["minimum"] is not None and price["minimum"] <= 0
            ):
                window_reasons.append("invalid_training_energy_price")
            training_windows.append({
                "forecast_start": origin,
                "start": start.isoformat(), "end_exclusive": end.isoformat(),
                "expected_hours": expected_hours, "observed_hours": len(window),
                "complete": complete, "load_kwh": load, "energy_price": price,
                "reasons": window_reasons,
            })
            reasons.extend(reason for reason in window_reasons if reason not in reasons)
        zones[zone_id] = {
            "eligible": not reasons, "reasons": reasons,
            "variation_history": variation_history, "training_windows": training_windows,
        }
    return {
        "policy": policy,
        "eligible_zone_ids": [zone for zone, values in zones.items() if values["eligible"]],
        "excluded_zone_ids": [zone for zone, values in zones.items() if not values["eligible"]],
        "upstream_exclusions": upstream,
        "zones": zones,
    }


def _read_report(path: Path, policy: dict[str, Any]) -> dict[str, Any] | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("policy") != policy:
            return None
        zones = report["zones"]
        eligible, excluded = report["eligible_zone_ids"], report["excluded_zone_ids"]
        if not isinstance(zones, dict) or not isinstance(eligible, list) or not isinstance(excluded, list):
            return None
        if eligible != sorted(zone for zone, value in zones.items() if value["eligible"]):
            return None
        if excluded != sorted(zone for zone, value in zones.items() if not value["eligible"]):
            return None
        for value in zones.values():
            if not isinstance(value["reasons"], list) or not isinstance(value["training_windows"], list):
                return None
            if not isinstance(value["variation_history"], (dict, type(None))):
                return None
        if not all(isinstance(report[key], str) for key in ("policy_key", "cache_path", "csv_path")):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    upstream = report.get("upstream_exclusions", {})
    for zone_id, values in report["zones"].items():
        history = values["variation_history"] or {}
        load = history.get("load_kwh", {})
        price = history.get("energy_price", {})
        rows.append({
            "zone_id": zone_id,
            "eligible": values["eligible"],
            "reasons": ";".join(values["reasons"]),
            "upstream_exclusion": upstream.get(zone_id, ""),
            "load_min": load.get("minimum"), "load_max": load.get("maximum"),
            "energy_price_min": price.get("minimum"), "energy_price_max": price.get("maximum"),
            "variation_history": json.dumps(values["variation_history"], ensure_ascii=False),
            "training_windows": json.dumps(values["training_windows"], ensure_ascii=False),
        })
    atomic_write_dataframe(pd.DataFrame(rows, columns=[
        "zone_id", "eligible", "reasons", "upstream_exclusion", "load_min", "load_max",
        "energy_price_min", "energy_price_max", "variation_history", "training_windows",
    ]), path)


def _update_manifest(cache_root: Path, report: dict[str, Any]) -> None:
    path = cache_root / "zone_selection_manifest.json"
    fingerprint = report["policy"]["dataset_fingerprint"]
    policy_key = report["policy_key"]
    active_key = f"{fingerprint}/{policy_key}"
    with process_file_lock(path.with_suffix(".lock"), blocking=True):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict) or not isinstance(current.get("datasets", {}), dict):
                current = {}
        except (OSError, ValueError):
            current = {}
        datasets = dict(current.get("datasets", {}))
        dataset_entry = dict(datasets.get(fingerprint, {}))
        policies = dict(dataset_entry.get("policies", {}))
        policies[policy_key] = {
            "policy": report["policy"],
            "cache_path": str(Path(report["cache_path"]).relative_to(cache_root)),
            "csv_path": str(Path(report["csv_path"]).relative_to(cache_root)),
            "eligible_zone_ids": report["eligible_zone_ids"],
            "excluded_zone_ids": report["excluded_zone_ids"],
        }
        datasets[fingerprint] = {"latest_policy_key": policy_key, "policies": policies}
        updated = {
            "version": POLICY_VERSION,
            "active_key": active_key,
            "latest_key": active_key,
            "active_dataset_fingerprint": fingerprint,
            "active_policy_key": policy_key,
            "datasets": datasets,
        }
        if updated != current:
            atomic_write_json(path, updated)
