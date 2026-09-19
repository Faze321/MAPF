from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from config import DataConfig, RunConfig
from dataset_adapter import DatasetSpec, load_canonical_dataset, validate_forecaster_dataset_source
from dataset_profiles import apply_dataset_profile, output_lock, scoped_output
from extra_dataset_adapters import MPEVDataDatasetAdapter, tou_schedule
from forecasting import summarize_weather
from global_forecaster import scenario_price_series
from train_datasets import dataset_job


def tariff_frame(rows):
    return pd.DataFrame(rows, columns=["Month", "Period", "Electricity Price(RMB)", "Service Price(RMB)"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def charged(self):
        folder = self.root / "LOA"
        folder.mkdir()
        times = pd.date_range("2023-04-01", periods=48, freq="h")
        pd.DataFrame({"Unnamed: 0": times, "001": np.arange(48), "002": 5}).to_csv(folder / "volume.csv", index=False)
        pd.DataFrame({"time": times, "001": 0.4, "002": 0.0}).to_csv(folder / "e_price.csv", index=False)
        pd.DataFrame({"site_id": ["001", "002"], "longitude": [1, 2], "latitude": [1, 2],
                      "total_volume": [100000, 100000], "avg_power": [999, 999]}).to_csv(folder / "sites.csv", index=False)
        return DatasetSpec(folder, adapter="charged")

    def test_charged_filters_zero_price_and_preserves_ids_and_sources(self):
        spec = self.charged()
        path = spec.path / "volume.csv"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        data = load_canonical_dataset(spec)
        self.assertEqual(set(data.timeseries.zone_id), {"001"})
        self.assertEqual(data.timeseries.load_kwh.sum(), sum(range(48)))
        self.assertIn("002", data.feature_manifest["excluded_sites"])
        self.assertNotIn("total_volume", data.static_zone_features)
        self.assertNotIn("avg_power", data.static_zone_features)
        cached = load_canonical_dataset(spec)
        self.assertEqual(set(cached.timeseries.zone_id), {"001"})
        self.assertEqual(data.dataset_fingerprint, cached.dataset_fingerprint)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_charged_rejects_misaligned_hours(self):
        spec = self.charged()
        path = spec.path / "e_price.csv"
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "identical hours"):
            load_canonical_dataset(spec)

    def test_charged_site_identifier_variant(self):
        spec = self.charged()
        path = spec.path / "sites.csv"
        pd.read_csv(path, dtype={"site_id": str}).rename(columns={"site_id": "site"}).to_csv(path, index=False)
        data = load_canonical_dataset(spec)
        self.assertEqual(data.static_zone_features.iloc[0].zone_id, "001")
        self.assertEqual(data.static_zone_features.iloc[0].longitude, 1)

    def test_tariff_peak_precedence_and_midnight(self):
        table = tariff_frame([(1, "Sharp(19:00-21:00)", 1.4, 0.4),
                              (1, "Peak(08:00-11:00,18:00-21:00)", 1.1, 0.2),
                              (1, "Off-peak(22:00-00:00,00:00-06:00)", 0.3, 0.6)])
        rates = tou_schedule(table).set_index(["month", "hour"])
        self.assertAlmostEqual(rates.loc[(1, 19), "energy_price"], 1.4)
        self.assertAlmostEqual(rates.loc[(1, 18), "energy_price"], 1.1)
        self.assertAlmostEqual(rates.loc[(1, 23), "energy_price"], 0.3)
        self.assertAlmostEqual(rates.loc[(1, 0), "energy_price"], 0.3)
        self.assertNotIn((1, 21), rates.index)

    def test_conflicting_tariffs_fail(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            tou_schedule(tariff_frame([(1, "Peak(08:00-11:00)", 1, 0), (1, "Shoulder(10:00-12:00)", 0.5, 0)]))

    def test_mp_evdata_real_workbook_layout(self):
        meta = pd.DataFrame({"Charging Prototypes": ["Taxi", "Bus", "Swap"], "ID": ["A1", "A4", "A7"],
                             "Single-pile Power": [80, 120, 60], "Total Orders": [999, 999, 999],
                             "Capacity": [5000, 3000, 1000], "Equipment Type": ["DC", "DC", "Swap"], "Price": ["TOU", "Free", "TOU"]})
        with pd.ExcelWriter(self.root / MPEVDataDatasetAdapter.load_filename) as writer:
            meta.to_excel(writer, sheet_name="metadata", index=False)
            for site, column in [("A1", "power"), ("A4", "power"), ("A7", "session_count")]:
                pd.DataFrame({"datetime": pd.date_range("2024-12-31", periods=25, freq="h"), column: 12.5}).to_excel(writer, sheet_name=site, index=False)
        with pd.ExcelWriter(self.root / "price.xlsx") as writer:
            meta.to_excel(writer, sheet_name="metadata", index=False)
            tariff_frame([(12, "Shoulder(00:00-00:00)", 0.6, 0.3)]).to_excel(writer, sheet_name="A1", index=False)
            tariff_frame([(12, "Shoulder(00:00-00:00)", np.nan, np.nan)]).to_excel(writer, sheet_name="A4", index=False)
        data = load_canonical_dataset(DatasetSpec(self.root, adapter="mp_evdata"))
        self.assertEqual(set(data.timeseries.zone_id), {"A1"})
        self.assertEqual(len(data.timeseries), 24)
        self.assertEqual(data.timeseries.load_kwh.sum(), 300)
        self.assertEqual(data.feature_manifest["trimmed_outside_2024"]["A1"], 1)
        self.assertIn("A4", data.feature_manifest["excluded_sites"])
        self.assertIn("A7", data.feature_manifest["excluded_sites"])
        self.assertNotIn("Total Orders", data.static_zone_features)

    def test_switch_drops_previous_dataset_specific_inputs(self):
        run = RunConfig(data_dir="old", forecast_start="2022-01-01", forecast_starts=["2022-01-01"],
                        zone_ids=["115"], precomputed_window_data="old.csv", forecaster_output_dir="old-model", lstm_epochs=7)
        run, data = apply_dataset_profile(run, DataConfig(cache_dir="old-cache"), "mp-evdata", None, self.root / "none.yaml")
        self.assertEqual(run.data_dir, "data/MP-EVData")
        self.assertEqual(run.forecast_start, "2024-11-01 00:00:00")
        self.assertIsNone(run.forecast_starts)
        self.assertIsNone(run.zone_ids)
        self.assertIsNone(run.forecaster_output_dir)
        self.assertIsNone(data.cache_dir)
        self.assertEqual(run.lstm_epochs, 7)

    def test_outputs_are_isolated_and_legacy_preserved(self):
        root = self.root / "output"
        self.assertEqual(scoped_output(root, "urbanev", self.root, explicit_dataset=False), root)
        paths = {scoped_output(root, name, Path(city), explicit_dataset=True)
                 for name, city in [("urbanev", "UrbanEV"), ("charged", "LOA"), ("charged", "JHB"), ("mp_evdata", "MP-EVData")]}
        self.assertEqual(len(paths), 4)

    def test_run_lock_blocks_other_process_and_releases(self):
        code = "from pathlib import Path; from dataset_profiles import output_lock; import sys\nwith output_lock(Path(sys.argv[1])): pass"
        with output_lock(self.root):
            result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Another run", result.stderr)
        result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_weather_is_not_reported_as_zero_weather(self):
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        summary = summarize_weather(pd.DataFrame({"time": times}), times.min(), times.max())
        self.assertTrue(all(value is None for value in summary.values()))

    def test_parallel_job_parsing(self):
        self.assertEqual(dataset_job("charged:loa"), ("charged", "LOA"))
        self.assertEqual(dataset_job("mp-evdata"), ("mp_evdata", None))
        with self.assertRaises(ValueError):
            dataset_job("urbanev:LOA")

    def test_forecaster_handoff_rejects_wrong_dataset_and_city(self):
        spec = DatasetSpec(self.root / "JHB", adapter="charged")
        with self.assertRaisesRegex(ValueError, "adapter"):
            validate_forecaster_dataset_source(spec, {"adapter": "mp_evdata"})
        with self.assertRaisesRegex(ValueError, "different dataset directory/city"):
            validate_forecaster_dataset_source(spec, {"adapter": "charged", "data_dir": str(self.root / "LOA")})
        validate_forecaster_dataset_source(spec, {"adapter": "charged", "data_dir": str(spec.path)})
        validate_forecaster_dataset_source(DatasetSpec(self.root, adapter="mp-evdata"), {"adapter": "mp_evdata", "data_dir": str(self.root)})

    def test_late_opening_station_only_requires_forecast_prices(self):
        times = pd.date_range("2024-11-01", periods=3, freq="h")
        schedule = pd.DataFrame({"time": times, "A10": [np.nan, 0.6, 0.7]})
        prices = scenario_price_series(schedule, "A10", timestamps=times[1:])
        self.assertEqual(prices.tolist(), [0.6, 0.7])
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            scenario_price_series(schedule, "A10", timestamps=times)


if __name__ == "__main__":
    unittest.main()
