"""Contracts for reusable, history-only random-site eligibility caches."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataset_adapter import CanonicalDataset, PriceChangeReference
from zone_eligibility import load_zone_eligibility


class ZoneEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name)

    def zone(self, zone_id="001"):
        hours = np.arange(12 * 24)
        return pd.DataFrame({
            "zone_id": zone_id,
            "timestamp": pd.date_range("2024-01-01", periods=len(hours), freq="h"),
            "load_kwh": (hours % 24).astype(float),
            "energy_price": 1.0 + (hours % 12) / 100,
        })

    def dataset(self, *frames, excluded=None):
        frame = pd.concat(frames or [self.zone()], ignore_index=True)
        return CanonicalDataset(
            timeseries=frame,
            static_zone_features=pd.DataFrame({"zone_id": frame["zone_id"].unique()}),
            feature_manifest={"excluded_sites": excluded or {}},
            price_change_reference=PriceChangeReference((), None, 0, 0),
            dataset_fingerprint="fingerprint-one",
            cache_dir=self.cache_dir,
            cache_keys={"dataset": "fingerprint-one"},
        )

    def load(self, dataset, **overrides):
        return load_zone_eligibility(dataset, **{
            "forecast_starts": ["2024-01-06", "2024-01-10"],
            "history_days": 2, "validation_days": 1, **overrides,
        })

    def test_constant_load_or_price_and_near_constant_are_excluded(self):
        valid = self.zone()
        constant_load = self.zone("002").assign(load_kwh=4.0)
        constant_price = self.zone("003").assign(energy_price=2.0)
        zero_load = self.zone("004").assign(load_kwh=0.0)
        near_load = self.zone("005").assign(load_kwh=1 + np.sin(np.arange(len(valid))) * 1e-11)
        near_price = self.zone("006").assign(energy_price=1 + np.sin(np.arange(len(valid))) * 1e-11)
        dataset = self.dataset(valid, constant_load, constant_price, zero_load, near_load, near_price)
        original = dataset.timeseries.copy(deep=True)
        report = self.load(dataset)
        self.assertEqual(report["eligible_zone_ids"], ["001"])
        self.assertEqual(report["excluded_zone_ids"], ["002", "003", "004", "005", "006"])
        for zone_id in ["002", "004", "005"]:
            self.assertIn("constant_load", report["zones"][zone_id]["reasons"])
        for zone_id in ["003", "006"]:
            self.assertIn("constant_energy_price", report["zones"][zone_id]["reasons"])
        self.assertGreater(report["zones"]["001"]["variation_history"]["load_kwh"]["zero_observations"], 0)
        pd.testing.assert_frame_equal(dataset.timeseries, original)

    def test_long_history_variation_allows_constant_short_training_window(self):
        frame = self.zone().assign(energy_price=1.0)
        frame.loc[0, "energy_price"] = 2.0
        report = self.load(self.dataset(frame))
        self.assertEqual(report["eligible_zone_ids"], ["001"])
        zone = report["zones"]["001"]
        self.assertFalse(zone["variation_history"]["energy_price"]["constant"])
        self.assertTrue(all(window["energy_price"]["constant"] for window in zone["training_windows"]))
        self.assertEqual(report["policy"]["variation_cutoff"], "2024-01-05T00:00:00")

    def test_validation_forecast_and_later_origin_changes_cannot_make_site_eligible(self):
        frame = self.zone()
        before_validation = frame["timestamp"] < pd.Timestamp("2024-01-05")
        frame.loc[before_validation, ["load_kwh", "energy_price"]] = [5.0, 1.0]
        report = self.load(self.dataset(frame))
        self.assertEqual(report["eligible_zone_ids"], [])
        self.assertEqual(report["zones"]["001"]["reasons"], ["constant_load", "constant_energy_price"])
        self.assertFalse(report["zones"]["001"]["training_windows"][1]["load_kwh"]["constant"])

    def test_every_origin_requires_all_unique_whole_hour_training_records(self):
        missing = self.zone("002")
        missing = missing[missing["timestamp"] != pd.Timestamp("2024-01-08T03:00:00")]
        duplicate = self.zone("003")
        duplicate.loc[50, "timestamp"] = duplicate.loc[49, "timestamp"]
        half_hour = self.zone("004")
        half_hour.loc[50, "timestamp"] += pd.Timedelta(minutes=30)
        report = self.load(self.dataset(self.zone(), missing, duplicate, half_hour))
        self.assertEqual(report["eligible_zone_ids"], ["001"])
        for zone_id in ["002", "003", "004"]:
            self.assertIn("incomplete_training_window", report["zones"][zone_id]["reasons"])
        windows = report["zones"]["002"]["training_windows"]
        self.assertTrue(windows[0]["complete"])
        self.assertFalse(windows[1]["complete"])

    def test_invalid_training_load_and_price_are_excluded(self):
        frames = []
        for zone_id, column, value in [
            ("nan", "load_kwh", np.nan), ("negative", "load_kwh", -1.0),
            ("infinite", "energy_price", np.inf), ("zero_price", "energy_price", 0.0),
        ]:
            frame = self.zone(zone_id)
            frame.loc[50, column] = value
            frames.append(frame)
        report = self.load(self.dataset(*frames))
        self.assertEqual(report["eligible_zone_ids"], [])
        for zone_id in ["nan", "negative"]:
            self.assertIn("invalid_training_load", report["zones"][zone_id]["reasons"])
        for zone_id in ["infinite", "zero_price"]:
            self.assertIn("invalid_training_energy_price", report["zones"][zone_id]["reasons"])

    def test_upstream_and_missing_sites_remain_visible_with_string_ids(self):
        dataset = self.dataset(excluded={"002": "non-positive baseline price"})
        dataset.static_zone_features.loc[len(dataset.static_zone_features)] = ["003"]
        report = self.load(dataset)
        self.assertEqual(report["eligible_zone_ids"], ["001"])
        self.assertEqual(report["excluded_zone_ids"], ["002", "003"])
        self.assertEqual(report["upstream_exclusions"]["002"], "non-positive baseline price")
        self.assertEqual(report["zones"]["002"]["reasons"], ["upstream_excluded"])
        self.assertIn("incomplete_training_window", report["zones"]["003"]["reasons"])
        csv = pd.read_csv(report["csv_path"], dtype={"zone_id": str})
        self.assertEqual(csv["zone_id"].tolist(), ["001", "002", "003"])

    def test_cache_hit_does_not_reanalyse_and_recovers_missing_csv(self):
        dataset = self.dataset()
        expected = self.load(dataset)
        Path(expected["csv_path"]).unlink()
        with patch("zone_eligibility._build_report", side_effect=AssertionError("cache should be reused")):
            report = self.load(dataset)
        self.assertEqual(expected, report)
        self.assertTrue(Path(report["csv_path"]).is_file())

    def test_corrupt_cache_rebuild_and_force_refresh(self):
        dataset = self.dataset()
        report = self.load(dataset)
        dataset.timeseries["energy_price"] = 1.0
        Path(report["cache_path"]).write_text("{invalid json", encoding="utf-8")
        rebuilt = self.load(dataset)
        self.assertEqual(rebuilt["eligible_zone_ids"], [])
        dataset.timeseries["energy_price"] = self.zone()["energy_price"]
        refreshed = self.load(dataset, force_cache=True)
        self.assertEqual(refreshed["eligible_zone_ids"], ["001"])

    def test_window_parameters_and_fingerprints_have_separate_manifest_entries(self):
        dataset = self.dataset()
        reports = [self.load(dataset), self.load(dataset, history_days=1),
                   self.load(dataset, forecast_starts=["2024-01-06"]),
                   self.load(replace(dataset, dataset_fingerprint="fingerprint-two"))]
        self.assertEqual(len({report["cache_path"] for report in reports}), 4)
        manifest = json.loads((self.cache_dir / "zone_selection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["datasets"]["fingerprint-one"]["policies"]), 3)
        self.assertEqual(len(manifest["datasets"]["fingerprint-two"]["policies"]), 1)
        self.assertEqual(manifest["active_dataset_fingerprint"], "fingerprint-two")
        for report in reports:
            self.assertTrue(Path(report["cache_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
