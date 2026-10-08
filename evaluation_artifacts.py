"""Read evaluation provenance and hourly observations without loading models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


THRESHOLD_FIELDS = (
    "historical_min_load_3h_kwh", "historical_max_load_3h_kwh",
    "low_medium_threshold_pct", "medium_high_threshold_pct",
    "load_3h_low_medium_threshold_kwh", "load_3h_medium_high_threshold_kwh",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def timestamp(value: Any) -> str | None:
    if value is None:
        return None
    # Invalid legacy identifiers are kept as identifiers, not interpreted as dates.
    text = str(value)
    if len(text) < 10:
        return text
    try:
        result = pd.Timestamp(text)
        if pd.isna(result):
            return None
        if result.tzinfo is not None:
            result = result.tz_convert("UTC")
        return result.isoformat()
    except (ValueError, TypeError):
        return text


def dataset_metadata(manifest: dict, control: dict | None = None) -> dict:
    control = control or {}
    source = manifest.get("data_source") or {}
    features = control.get("feature_manifest") or manifest.get("feature_manifest") or {}
    adapter = source.get("adapter") or features.get("adapter")
    city = features.get("city")
    fingerprint = control.get("dataset_fingerprint") or source.get("dataset_fingerprint")
    folder = str(source.get("data_dir") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    name = {"urbanev": "UrbanEV", "mp_evdata": "MP-EVData", "mp-evdata": "MP-EVData"}.get(adapter)
    if adapter == "charged":
        name = f"CHARGED_{city or folder}" if city or folder else "CHARGED"
    name = name or folder or (f"dataset_{fingerprint}" if fingerprint else "unknown")
    price_unit = features.get("price_unit")
    revenue_unit = str(price_unit).replace("/kWh", "").strip() if price_unit else "unspecified source currency"
    return {"dataset": name, "dataset_identity": str(fingerprint or source.get("data_dir") or "unknown"),
            "revenue_unit": revenue_unit, "price_unit": price_unit,
            "dataset_provenance_known": bool(fingerprint or source.get("data_dir"))}


def load_forecast_context(directory: Path) -> dict:
    manifest_path = directory / "forecaster_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    detail_dir = directory / "evaluation" / "forecast_details"
    paths = sorted(detail_dir.glob("zone_*_forecast_vs_actual.csv"))
    frames = [pd.read_csv(p, dtype={"zone_id": str}) for p in paths]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "time" in frame:
        frame["time"] = frame["time"].map(timestamp)
    metrics_path = directory / "evaluation" / "forecast_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    parameters = dict(manifest.get("forecast_parameters") or {})
    if "forecast_start" in parameters:
        parameters["forecast_start"] = timestamp(parameters["forecast_start"])
    metadata = dataset_metadata(manifest)
    for key in ("forecast_model", "diurnal_blend_alpha"):
        unique = metrics[key].dropna().unique() if key in metrics else []
        if len(unique) > 1:
            raise ValueError(f"Mixed {key} in {metrics_path}")
        metadata[key] = str(unique[0]) if len(unique) else str(parameters.get(key, "unknown"))
    metadata["forecast_origin"] = timestamp(parameters.get("forecast_start"))
    snippets_path = directory / "context_snippets.json"
    snippets = read_json(snippets_path) if snippets_path.exists() else []
    if not isinstance(snippets, list):
        snippets = []
    thresholds = {str(item["zone_id"]): {key: item.get(key) for key in THRESHOLD_FIELDS}
                  for item in snippets if isinstance(item, dict) and "zone_id" in item}
    expected_windows = {str(item["zone_id"]): {
        (timestamp(w.get("window_start")), timestamp(w.get("window_end")))
        for w in item.get("pricing_windows_3h", [])}
        for item in snippets if isinstance(item, dict) and "zone_id" in item}
    identity = signature({"dataset": metadata["dataset_identity"], "parameters": parameters}) \
        if metadata["dataset_provenance_known"] and parameters else str(directory.resolve())
    return {"directory": directory, "detail_dir": detail_dir, "frame": frame,
            "manifest": manifest, "parameters": parameters, "metadata": metadata,
            "thresholds": thresholds, "expected_windows": expected_windows, "identity": identity}


def find_forecast_context(control_path: Path, cache: dict[Path, dict]) -> dict | None:
    # Prefer the adjacent saved forecaster, rather than stale absolute paths from another host.
    for parent in list(control_path.parents)[:4]:
        candidate = parent / "forecaster"
        if (candidate / "forecaster_manifest.json").exists() or (candidate / "evaluation" / "forecast_details").is_dir():
            candidate = candidate.resolve()
            if candidate not in cache:
                cache[candidate] = load_forecast_context(candidate)
            return cache[candidate]
    return None


def pairing_metadata(control: dict, context: dict | None) -> dict:
    zones = sorted(str(z["zone_id"]) for z in control.get("zones", []))
    rows, complete = [], bool(zones)
    thresholds = (context or {}).get("thresholds", {})
    expected_hours: dict[str, set[str]] = {}
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    for zone in sorted(control.get("zones", []), key=lambda z: str(z["zone_id"])):
        zone_id = str(zone["zone_id"])
        threshold = thresholds.get(zone_id, {})
        # Older control JSON may carry the thresholds directly.
        if not threshold:
            threshold = {key: zone.get(key) for key in THRESHOLD_FIELDS}
        complete &= all(finite(threshold.get(key)) for key in
                        ("load_3h_low_medium_threshold_kwh", "load_3h_medium_high_threshold_kwh"))
        windows = zone.get("final_windows") or []
        complete &= bool(windows)
        expected = (context or {}).get("expected_windows", {}).get(zone_id)
        observed = {(timestamp(w.get("window_start")), timestamp(w.get("window_end"))) for w in windows}
        complete &= bool(expected) and observed == expected and len(observed) == len(windows)
        hours: set[str] = set()
        for start, end in expected or []:
            try:
                interval = {timestamp(t) for t in pd.date_range(start, end, freq="h")}
                if not interval or hours.intersection(interval):
                    complete = False
                hours.update(interval)
            except (ValueError, TypeError):
                complete = False
        expected_hours[zone_id] = hours
        for window in windows:
            required = ("mean_energy_price", "sum_predicted_kwh", "load_stress_level")
            complete &= all(finite(window.get(key)) for key in required[:2])
            complete &= window.get("load_stress_level") in ("Low", "Medium", "High", "Extremely High")
            rows.append([zone_id, timestamp(window.get("window_start")), timestamp(window.get("window_end")),
                         *[window.get(key) for key in required], threshold])
    rows.sort(key=lambda row: (row[0], str(row[1]), str(row[2])))
    hourly_signature = None
    if context is not None:
        frame = context["frame"]
        keys = ["zone_id", "time", "actual_kwh", "predicted_kwh"]
        if not frame.empty and all(key in frame for key in keys):
            # Equal window means can conceal different hourly price/load products.
            # Legacy files without prices still support the original comparisons;
            # their hourly revenue baseline is independently marked unknown.
            keys += [key for key in ("e_price", "e_price_valid", "price_valid") if key in frame]
            selected = frame[frame.zone_id.isin(zones)][keys].sort_values(["zone_id", "time"])
            valid = selected[["actual_kwh", "predicted_kwh"]].apply(pd.to_numeric, errors="coerce")
            finite_values = valid.map(math.isfinite) if hasattr(valid, "map") else valid.applymap(math.isfinite)
            covered = all(set(selected.loc[selected.zone_id == zone_id, "time"]) == expected_hours[zone_id]
                          for zone_id in zones)
            if (len(selected) and finite_values.to_numpy().all() and set(selected.zone_id) == set(zones)
                    and covered and not selected.duplicated(["zone_id", "time"]).any()):
                hourly_signature = signature(selected.to_dict("records"))
    parameters = (context or {}).get("parameters", {})
    return {"zone_ids": zones, "forecast_parameters": parameters,
            "baseline_signature": signature(rows) if complete else None,
            "hourly_signature": hourly_signature,
            "pairing_ready": bool(complete and hourly_signature and parameters)}
