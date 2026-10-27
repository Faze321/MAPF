"""Time precision, long-running lock contention, and failed cache publication."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from config import TrainingConfig
import zone_training as manager


def training_frame():
    hours = np.arange(60 * 24)
    return pd.DataFrame({"time": pd.date_range("2024-01-01", periods=len(hours), freq="h"),
                         "actual_kwh": 20 + np.sin(hours / 6), "e_price": 0.3 + 0.1 * (hours % 24 > 12),
                         "T": 15 + np.cos(hours / 6)})


class ZoneTrainingEdgeTests(unittest.TestCase):
    def test_datetime_resolution_does_not_change_windows_intervals_or_qualification(self):
        baseline = training_frame()
        expected = manager.inspect_training_window(baseline)
        for unit in ("s", "ms", "us", "ns"):
            with self.subTest(unit=unit):
                frame = baseline.assign(time=baseline["time"].dt.as_unit(unit))
                np.testing.assert_array_equal(manager.complete_target_indices(frame, 720, 24), np.arange(720, len(frame) - 23))
                self.assertEqual(len(manager.training_intervals(frame)), 1)
                actual = manager.inspect_training_window(frame)
                self.assertTrue(actual["eligible"])
                self.assertEqual(actual["validation_start"], expected["validation_start"])
                self.assertEqual(actual["sample_count"], expected["sample_count"])

    def test_fingerprint_ignores_unused_weather_and_normalizes_time_precision(self):
        frame = training_frame()
        kwargs = dict(data_dir=Path.cwd(), zone_id="zone", backend="lstm", parameters={"lstm_exog_cols": []},
                      validation_start=frame.iloc[-336]["time"], validation_end=frame.iloc[-1]["time"] + pd.Timedelta(hours=1))
        first = manager.training_fingerprint(frame, **kwargs)
        modified = frame.assign(T=999.0, time=frame["time"].dt.as_unit("us"))
        self.assertEqual(first, manager.training_fingerprint(modified, **kwargs))
        kwargs["parameters"] = {"lstm_exog_cols": ["T"]}
        self.assertNotEqual(manager.training_fingerprint(frame, **kwargs), manager.training_fingerprint(modified, **kwargs))

    def test_long_training_lock_keeps_waiting_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            handle = Mock()
            with patch.object(manager, "_try_file_lock", side_effect=[None] * 20 + [handle]) as acquire, \
                 patch.object(manager, "_release_file_lock") as release, patch.object(manager.time, "sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    with manager._zone_model_lock(Path(directory) / "zone" / ".training.lock"):
                        raise RuntimeError("training failed")
                self.assertEqual(acquire.call_count, 21)
                self.assertEqual(sleep.call_count, 20)
                release.assert_called_once_with(handle)

    def test_nonfinite_report_never_replaces_an_existing_valid_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = training_frame().rename(columns={"time": "timestamp", "actual_kwh": "load_kwh", "e_price": "energy_price"})
            dataset = SimpleNamespace(timeseries=frame.assign(zone_id="A"), cache_dir=root / "cache",
                                      dataset_fingerprint="test", feature_manifest={})

            def fake_train(backend, frame, start, end, params, output):
                for filename in ("model.json", "preprocess.json", "training_history.csv"):
                    (output / filename).write_text("fixture", encoding="utf-8")
                return {"best_rounds": 1}

            args = (dataset, ["xgboost"], TrainingConfig(xgboost_workers=1))
            with patch("offline_models.train_zone_model", side_effect=fake_train):
                original = manager.ensure_dataset_models(*args, data_dir=root, zone_ids=["A"])["xgboost"]["A"]
            folder = Path(original["model_dir"]).parent
            before = (folder / "current.json").read_bytes()
            manifest = manager.validate_model_reference(original)
            self.assertEqual(manifest["preprocess_schema_version"], 2)

            def invalid_report(*args):
                result = fake_train(*args)
                return {**result, "invalid_metric": float("nan")}

            with patch("offline_models.train_zone_model", side_effect=invalid_report), \
                 self.assertRaisesRegex(RuntimeError, "Out of range float values"):
                manager.ensure_dataset_models(*args, data_dir=root, zone_ids=["A"], force_train=True)
            self.assertEqual((folder / "current.json").read_bytes(), before)
            self.assertFalse(list(folder.glob(".pending-*")))
            manager.validate_model_reference(original)
            index = json.loads((root / "cache" / "weight" / "xgboost" / "index.json").read_text())
            self.assertEqual(index["zones"]["A"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
