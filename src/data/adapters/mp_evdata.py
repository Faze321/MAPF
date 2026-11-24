from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.adapters.base import DatasetAdapter
from src.data.adapters.charged import tou_schedule
from src.data.canonical import DatasetSpec, finalize, require_files, valid_energy_prices


class MPEVDataDatasetAdapter(DatasetAdapter):
    name = "mp_evdata"
    version = 1
    load_filename = "station-level load Profile 1h.xlsx"

    def source_files(self, spec: DatasetSpec):
        return require_files(spec.path, [self.load_filename, "price.xlsx"])

    def build(self, spec: DatasetSpec):
        frames, static_rows, excluded, trimmed = [], [], {}, {}
        required = "non-negative" if spec.allow_zero_energy_price else "positive"
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
                valid_price = valid_energy_prices(raw.energy_price, spec.allow_zero_energy_price)
                if not valid_price.all():
                    missing = raw.loc[~valid_price, ["month", "hour"]].drop_duplicates()
                    excluded[site] = f"electricity tariff must be finite and {required} ({len(missing)} invalid month/hour combinations)"
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
                        selection_policy=f"charging energy only; complete {required} published monthly TOU tariff; no imputation",
                        excluded_static_features=["Total Orders", "ambiguous merged Capacity cells"])
