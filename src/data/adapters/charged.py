from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.adapters.base import DatasetAdapter
from src.data.adapters.urbanev import urban_ev_poi_zone_counts
from src.data.canonical import DatasetSpec, finalize, hourly_matrix, require_files


class ChargedDatasetAdapter(DatasetAdapter):
    name = "charged"
    version = 2

    def source_files(self, spec: DatasetSpec):
        paths = require_files(spec.path, ["volume.csv", "e_price.csv"])
        return paths + [spec.path / name for name in ("s_price.csv", "weather.csv", "sites.csv", "poi.csv") if (spec.path / name).is_file()]

    def build(self, spec: DatasetSpec):
        load = hourly_matrix(spec.path / "volume.csv")
        price = hourly_matrix(spec.path / "e_price.csv")
        if not load.index.equals(price.index) or set(load.columns) != set(price.columns):
            raise ValueError("CHARGED load and price must have identical hours and site IDs")
        excluded = {}
        for site in load.columns:
            if not (np.isfinite(price[site]) & price[site].gt(0)).all():
                excluded[site] = "missing, non-finite or non-positive baseline electricity price"
            elif not (np.isfinite(load[site]) & load[site].ge(0)).all():
                excluded[site] = "missing, non-finite or negative hourly energy"
        sites = [site for site in load if site not in excluded]
        if not sites:
            raise ValueError("CHARGED has no sites with complete positive electricity prices")
        frame = load[sites].reset_index().melt(id_vars="timestamp", var_name="zone_id", value_name="load_kwh")
        energy = price[sites].reset_index().melt(id_vars="timestamp", var_name="zone_id", value_name="energy_price")
        frame = frame.merge(energy, on=["timestamp", "zone_id"], validate="one_to_one")
        if (spec.path / "s_price.csv").is_file():
            service = hourly_matrix(spec.path / "s_price.csv")
            if not service.index.equals(load.index) or not set(sites).issubset(service.columns):
                raise ValueError("CHARGED service price is not aligned with hourly load")
            service = service[sites].reset_index().melt(id_vars="timestamp", var_name="zone_id", value_name="service_price")
            if not (np.isfinite(service.service_price) & service.service_price.ge(0)).all():
                raise ValueError("Invalid CHARGED service price")
            frame = frame.merge(service, on=["timestamp", "zone_id"], validate="one_to_one")
        if (spec.path / "weather.csv").is_file():
            weather = pd.read_csv(spec.path / "weather.csv").rename(columns={"time": "timestamp", "temp": "temperature", "precip": "rain"})
            weather.timestamp = pd.to_datetime(weather.timestamp)
            keep = ["timestamp", *[c for c in ("temperature", "humidity", "rain") if c in weather]]
            frame = frame.merge(weather[keep], on="timestamp", how="left", validate="many_to_one")
        static = pd.DataFrame({"zone_id": sites})
        if (spec.path / "sites.csv").is_file():
            raw = pd.read_csv(spec.path / "sites.csv", dtype={"site_id": str, "site": str})
            identifiers = [c for c in ("site_id", "site") if c in raw]
            if len(identifiers) != 1:
                raise ValueError("CHARGED sites.csv must have one site_id or site identifier")
            raw = raw.rename(columns={identifiers[0]: "zone_id", "charger_num": "charger_count"})
            # Do not expose total_volume, total_duration or avg_power: they include future targets.
            keep = [c for c in ("zone_id", "longitude", "latitude", "charger_count", "area", "perimeter") if c in raw]
            static = static.merge(raw[keep], on="zone_id", how="left", validate="one_to_one")
            static["station_count"] = 1
            # Assign against every original site before selecting eligible sites.
            poi = urban_ev_poi_zone_counts(spec.path / "poi.csv", raw[keep])
            if not poi.empty:
                static = static.merge(poi, on="zone_id", how="left")
                cols = [c for c in static if c.startswith("poi_")]
                static[cols] = static[cols].fillna(0)
        return finalize(frame, static, excluded=excluded, load_unit="kWh per hour", price_unit="source local currency/kWh",
                        city=spec.path.name, source="https://github.com/IntelligentSystemsLab/CHARGED",
                        selection_policy="complete finite hourly energy and strictly positive electricity price; no invented tariffs",
                        excluded_static_features=["total_volume", "total_duration", "avg_power"])


def tou_schedule(tariffs: pd.DataFrame) -> pd.DataFrame:
    """Expand monthly periods; Sharp overrides the explicitly overlapping Peak.

    Boundaries are [start, end); 22:00-00:00 wraps at midnight. Other overlap
    conflicts fail, and missing hours remain missing instead of being guessed.
    """
    records = {}
    priorities = {"off-peak": 0, "shoulder": 1, "peak": 2, "sharp": 3}
    for _, row in tariffs.iterrows():
        if pd.isna(row["Month"]):
            continue
        month = int(row["Month"])
        if month not in range(1, 13):
            raise ValueError(f"Invalid tariff month: {month}")
        period = str(row["Period"])
        label = period.split("(")[0].strip().lower()
        if label not in priorities:
            raise ValueError(f"Unknown TOU period: {period}")
        spans = re.findall(r"(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})", period)
        if not spans:
            raise ValueError(f"No hours in TOU period: {period}")
        for start_h, start_m, end_h, end_m in spans:
            if start_m != "00" or end_m != "00":
                raise ValueError("Hourly MP-EVData adapter requires whole-hour tariff boundaries")
            start, end = int(start_h), int(end_h)
            if not 0 <= start < 24 or not 0 <= end <= 24:
                raise ValueError(f"Invalid TOU hour range: {period}")
            hours = range(start, end) if end > start else [*range(start, 24), *range(end)]
            values = (float(row["Electricity Price(RMB)"]), float(row["Service Price(RMB)"]))
            for hour in hours:
                key = (month, hour)
                old = records.get(key)
                if old is not None and old[0] != label and {old[0], label} != {"sharp", "peak"}:
                    raise ValueError(f"Unexpected overlapping TOU periods: {key}")
                if old is not None and old[0] == label and not np.allclose(old[1], values, equal_nan=True):
                    raise ValueError(f"Conflicting duplicate TOU tariff: {key}")
                if old is None or priorities[label] >= priorities[old[0]]:
                    records[key] = (label, values)
    return pd.DataFrame([{"month": m, "hour": h, "energy_price": v[1][0], "service_price": v[1][1]} for (m, h), v in records.items()])
