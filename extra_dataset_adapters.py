"""Native readers for the published CHARGED and MP-EVData hourly releases.

Raw inputs are never rewritten. Sites incompatible with the positive-price
control contract are excluded explicitly, with reasons retained in the manifest.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from dataset_adapter import DatasetAdapter, DatasetSpec, urban_ev_poi_zone_counts


def require_files(root: Path, names: list[str]) -> list[Path]:
    paths = [root / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset inputs; run download_datasets.py: " + ", ".join(missing))
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


class MPEVDataDatasetAdapter(DatasetAdapter):
    name = "mp_evdata"
    version = 1
    load_filename = "station-level load Profile 1h.xlsx"

    def source_files(self, spec: DatasetSpec):
        return require_files(spec.path, [self.load_filename, "price.xlsx"])

    def build(self, spec: DatasetSpec):
        frames, static_rows, excluded, trimmed = [], [], {}, {}
        with pd.ExcelFile(spec.path / self.load_filename, engine="openpyxl") as loads, pd.ExcelFile(spec.path / "price.xlsx", engine="openpyxl") as prices:
            metadata = pd.read_excel(loads, sheet_name=0).dropna(subset=["ID"]).set_index("ID")
            for site in [s for s in loads.sheet_names if re.fullmatch(r"A\d+", s)]:
                raw = pd.read_excel(loads, sheet_name=site)
                if "session_count" in raw:
                    excluded[site] = "battery swap transaction count, not charging energy"
                    continue
                if site not in prices.sheet_names:
                    excluded[site] = "no published tariff sheet"
                    continue
                raw = raw[["datetime", "power"]].dropna(how="all")
                raw["datetime"] = pd.to_datetime(raw.datetime, errors="raise")
                # The published monthly tariffs apply to 2024 only; the workbook also contains 2025 tail rows.
                trimmed[site] = int(raw.datetime.dt.year.ne(2024).sum())
                raw = raw[raw.datetime.dt.year.eq(2024)].sort_values("datetime")
                if raw.empty or raw.datetime.duplicated().any() or not raw.datetime.equals(raw.datetime.dt.floor("h")):
                    raise ValueError(f"Invalid MP-EVData timestamps: {site}")
                if not raw.datetime.diff().dropna().eq(pd.Timedelta(hours=1)).all():
                    raise ValueError(f"Missing MP-EVData hourly records: {site}")
                tariff = tou_schedule(pd.read_excel(prices, sheet_name=site))
                raw["month"], raw["hour"] = raw.datetime.dt.month, raw.datetime.dt.hour
                raw = raw.merge(tariff, on=["month", "hour"], how="left", validate="many_to_one")
                valid_price = np.isfinite(raw.energy_price) & raw.energy_price.gt(0)
                if not valid_price.all():
                    missing = raw.loc[~valid_price, ["month", "hour"]].drop_duplicates()
                    excluded[site] = f"incomplete or non-positive electricity tariff ({len(missing)} month/hour combinations)"
                    continue
                if not (np.isfinite(raw.power) & raw.power.ge(0)).all():
                    raise ValueError(f"Invalid hourly power in {site}")
                if not (np.isfinite(raw.service_price) & raw.service_price.ge(0)).all():
                    raise ValueError(f"Invalid service tariff in {site}")
                raw["zone_id"] = site
                # One-hour average kW multiplied by one hour is kWh, with unchanged numeric values.
                raw = raw.rename(columns={"datetime": "timestamp", "power": "load_kwh"})
                frames.append(raw[["timestamp", "zone_id", "load_kwh", "energy_price", "service_price"]])
                info = metadata.loc[site]
                static_rows.append({"zone_id": site, "station_type": str(info.iloc[0]),
                                    "station_count": 1, "equipment_type": str(info.iloc[4])})
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return finalize(frame, pd.DataFrame(static_rows), excluded=excluded, trimmed_outside_2024=trimmed,
                        load_unit="kWh per hour (published hourly average kW times 1h)", price_unit="CNY/kWh",
                        source="https://doi.org/10.6084/m9.figshare.29882366", tariff_year=2024,
                        selection_policy="charging energy only; complete positive published monthly TOU tariff; no imputation",
                        excluded_static_features=["Total Orders", "ambiguous merged Capacity cells"])
