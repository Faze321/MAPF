"""Offline per-zone qualification, content identity and immutable weight caches."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import TrainingConfig
import zone_training


def zone_frame(days=60, zone="A", offset=0.0):
    times = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    position = np.arange(len(times))
    return pd.DataFrame({"timestamp": times, "zone_id": zone,
                         "load_kwh": 20 + offset + 5 * np.sin(position / 5),
                         "energy_price": .3 + .2 * (position % 24 >= 12)})


class ZoneTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = SimpleNamespace(
            cache_dir=self.root / "cache", dataset_fingerprint="dataset-v1",
            timeseries=pd.concat([zone_frame(zone="A"), zone_frame(zone="B", offset=5)], ignore_index=True),
            feature_manifest={"excluded_sites": {"excluded": "invalid tariff"}},
        )
        self.config = TrainingConfig(xgboost_workers=1, min_history_days=1, min_samples=2)

    def fake_train(self, backend, frame, validation_start, validation_end, parameters, directory):
        directory = Path(directory)
        name = "model.pt" if backend == "lstm" else "model.json"
        (directory / name).write_text(json.dumps({"backend": backend, "load_mean": float(frame.actual_kwh.mean())}))
        (directory / "preprocess.json").write_text(json.dumps(parameters))
        frame[["time", "actual_kwh"]].to_csv(directory / "training_history.csv", index=False)
        return {"best_rounds": 1, "sample_counts": {"full": len(frame)}}

    def ensure(self, **kwargs):
        return zone_training.ensure_dataset_models(self.dataset, ["lstm", "xgboost"], self.config,
                                                   data_dir=self.root, zone_ids=["A", "B"],
                                                   history_days=1, horizon_days=1, **kwargs)

    def test_default_qualification_counts_complete_720_input_24_output_windows(self):
        frame = zone_training.canonical_zone_frame(SimpleNamespace(timeseries=zone_frame(53)), "A").iloc[:1255]
        valid = zone_training.inspect_training_window(frame)
        self.assertTrue(valid["eligible"])
        self.assertEqual(valid["sample_count"], 512)
        self.assertEqual((valid["train_samples"], valid["validation_samples"], valid["purged_samples"]), (386, 103, 23))
        self.assertEqual(valid["validation_start"], frame.time.iloc[1129])
        self.assertEqual(valid["validation_end"], frame.time.iloc[-1] + pd.Timedelta(hours=1))
        self.assertFalse(zone_training.inspect_training_window(frame.iloc[:-1])["eligible"])
        self.assertEqual(zone_training.inspect_training_window(frame.iloc[:45 * 24 - 1])["reason"], "insufficient_valid_history")
        self.assertFalse(zone_training.inspect_training_window(frame, min_samples=513)["eligible"])
        self.assertEqual(zone_training.inspect_training_window(pd.concat([frame, frame.iloc[:1]]))["reason"], "duplicate_timestamps")
        self.assertEqual(zone_training.inspect_training_window(frame.assign(actual_kwh=0))["reason"], "constant_load")
        self.assertEqual(zone_training.inspect_training_window(frame.assign(e_price=.4))["reason"], "constant_energy_price")

    def test_internal_validation_and_training_windows_never_bridge_gaps(self):
        source = zone_frame(60)
        frame = zone_training.canonical_zone_frame(SimpleNamespace(timeseries=source), "A")
        frame = frame.drop(index=200).reset_index(drop=True)
        damaged = frame.copy()
        damaged.loc[damaged.index[-5:], "actual_kwh"] = np.nan
        result = zone_training.inspect_training_window(damaged, history_days=1, horizon_days=2)
        self.assertTrue(result["eligible"])
        targets = zone_training.complete_target_indices(result["frame"], 24, 48)
        times = result["frame"].time
        self.assertTrue(all(times.iloc[index + 47] - times.iloc[index - 24] == pd.Timedelta(hours=71) for index in targets))
        self.assertFalse(result["frame"].actual_kwh.isna().any())

    def test_only_selected_zone_is_trained_and_window_lengths_define_cache_identity(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as fit:
            first = zone_training.ensure_dataset_models(self.dataset, ["lstm"], self.config, zone_ids=["B"],
                                                       history_days=1, horizon_days=1)["lstm"]
            self.assertEqual(set(first), {"B"})
            self.assertEqual(fit.call_count, 1)
            self.assertAlmostEqual(fit.call_args.args[1].actual_kwh.mean(), self.dataset.timeseries.query("zone_id == 'B'").load_kwh.mean())
            second = zone_training.ensure_dataset_models(self.dataset, ["lstm"], self.config, zone_ids=["B"],
                                                        history_days=2, horizon_days=1)["lstm"]
            third = zone_training.ensure_dataset_models(self.dataset, ["lstm"], self.config, zone_ids=["B"],
                                                       history_days=1, horizon_days=2)["lstm"]
        self.assertEqual(len({mapping["B"]["model_id"] for mapping in (first, second, third)}), 3)
        self.assertFalse((self.root / "cache" / "weight" / "lstm" / zone_training.zone_key("A")).exists())

    def test_selected_zones_fit_once_and_whole_dataset_fingerprint_does_not_invalidate(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            first = self.ensure()
            self.assertEqual(train.call_count, 4)
            train.reset_mock()
            self.dataset.dataset_fingerprint = "unrelated-dataset-change"
            second = self.ensure()
            self.assertEqual(first, second)
            train.assert_not_called()
        for backend in first:
            self.assertEqual(set(first[backend]), {"A", "B"})
            index = json.loads((self.root / "cache" / "weight" / backend / "index.json").read_text())
            self.assertEqual(index["available_zone_ids"], ["A", "B"])

    def test_split_ratio_and_sample_stride_change_model_identity(self):
        configs = [self.config,
                   TrainingConfig(min_history_days=1, min_samples=2, train_ratio=.7, validation_ratio=.3),
                   TrainingConfig(min_history_days=1, min_samples=2, sample_stride_hours=2)]
        references = []
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            for config in configs:
                references.append(zone_training.ensure_dataset_models(
                    self.dataset, ["lstm"], config, zone_ids=["A"],
                    history_days=1, horizon_days=1)["lstm"]["A"])
        self.assertEqual(train.call_count, 3)
        self.assertEqual(len({ref["model_id"] for ref in references}), 3)
        for reference in references:
            zone_training.validate_model_reference(reference)

    def test_selecting_another_zone_preserves_cached_unselected_zone(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            first = zone_training.ensure_dataset_models(self.dataset, ["lstm"], self.config,
                zone_ids=["A"], history_days=1, horizon_days=1)["lstm"]["A"]
            pointer = Path(first["model_dir"]).parent / "current.json"
            before = pointer.read_bytes()
            second = zone_training.ensure_dataset_models(self.dataset, ["lstm"], self.config,
                zone_ids=["B"], history_days=1, horizon_days=1)["lstm"]
        self.assertEqual(train.call_count, 2)
        self.assertEqual(set(second), {"B"})
        self.assertEqual(pointer.read_bytes(), before)
        zone_training.validate_model_reference(first)
        index = json.loads((self.root / "cache" / "weight" / "lstm" / "index.json").read_text())
        self.assertEqual(index["available_zone_ids"], ["A", "B"])
        self.assertEqual(index["zones"]["A"]["model_id"], first["model_id"])

    def test_changing_one_zone_retrains_only_that_zones_models(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            first = self.ensure()
            train.reset_mock()
            index = self.dataset.timeseries.index[self.dataset.timeseries.zone_id == "A"][10]
            self.dataset.timeseries.loc[index, "load_kwh"] += 3
            second = self.ensure()
        self.assertEqual(train.call_count, 2)
        for backend in first:
            self.assertNotEqual(first[backend]["A"]["model_id"], second[backend]["A"]["model_id"])
            self.assertEqual(first[backend]["B"], second[backend]["B"])
            zone_training.validate_model_reference(first[backend]["A"])

    def test_force_training_creates_new_versions_without_invalidating_saved_references(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train):
            original = self.ensure()
            forced = self.ensure(force_train=True)
        for backend in original:
            for zone in original[backend]:
                self.assertNotEqual(original[backend][zone]["model_id"], forced[backend][zone]["model_id"])
                zone_training.validate_model_reference(original[backend][zone])
                zone_training.validate_model_reference(forced[backend][zone])

    def test_prepare_recovers_only_corrupt_model_and_replay_still_rejects_old_reference(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train):
            refs = self.ensure()
        reference = refs["xgboost"]["A"]
        weight = Path(reference["model_dir"]) / "model.json"
        weight.write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            zone_training.validate_model_reference(reference)
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            repaired = self.ensure()
            self.assertEqual(train.call_count, 1)
            self.assertEqual(train.call_args.args[0], "xgboost")
            train.reset_mock()
            self.assertEqual(self.ensure(), repaired)
            train.assert_not_called()
        replacement = repaired["xgboost"]["A"]
        self.assertNotEqual(reference["model_id"], replacement["model_id"])
        self.assertEqual(weight.read_text(), "corrupt")
        for backend, zones in refs.items():
            for zone, original in zones.items():
                if (backend, zone) != ("xgboost", "A"):
                    self.assertEqual(original, repaired[backend][zone])
        manifest = zone_training.validate_model_reference(replacement)
        self.assertEqual(manifest["cache_recovery"]["reason"], "recovered_invalid_cache")
        with self.assertRaisesRegex(ValueError, "checksum"):
            zone_training.load_cached_predictor(reference)

    def test_prepare_retries_missing_weights_without_overwriting_prior_directories(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train):
            refs = self.ensure()
        initial = refs["xgboost"]["A"]
        missing = Path(initial["model_dir"]) / "model.json"
        missing.unlink()
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            first = self.ensure()
            self.assertEqual(train.call_count, 1)
        repaired = first["xgboost"]["A"]
        self.assertFalse(missing.exists())
        with self.assertRaises(FileNotFoundError):
            zone_training.load_cached_predictor(initial)
        second_missing = Path(repaired["model_dir"]) / "model.json"
        second_missing.unlink()
        current_path = Path(repaired["model_dir"]).parent / "current.json"
        previous_pointer = current_path.read_bytes()
        # A failed replacement does not get swallowed as another cache miss.
        with patch("offline_models.train_zone_model", side_effect=ValueError("training rejected")) as train, \
             self.assertRaisesRegex(RuntimeError, "training rejected"):
            self.ensure()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(current_path.read_bytes(), previous_pointer)
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            second = self.ensure()
        self.assertEqual(train.call_count, 1)
        fresh = second["xgboost"]["A"]
        self.assertEqual(len({initial["model_id"], repaired["model_id"], fresh["model_id"]}), 3)
        self.assertTrue(Path(initial["model_dir"]).is_dir())
        self.assertTrue(Path(repaired["model_dir"]).is_dir())
        self.assertFalse(missing.exists())
        self.assertFalse(second_missing.exists())
        zone_training.validate_model_reference(fresh)
        index = json.loads((self.root / "cache" / "weight" / "xgboost" / "index.json").read_text())
        self.assertEqual(index["zones"]["A"]["reason"], "recovered_invalid_cache")

    def test_invalid_pointer_recovers_valid_weights_and_invalid_manifest_creates_new_version(self):
        with patch("offline_models.train_zone_model", side_effect=self.fake_train):
            refs = self.ensure()
        reference = refs["xgboost"]["A"]
        directory = Path(reference["model_dir"])
        (directory.parent / "current.json").write_text("{invalid JSON")
        with patch("offline_models.train_zone_model", side_effect=AssertionError("valid stable weights must be reused")):
            self.assertEqual(self.ensure(), refs)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text("{invalid JSON")
        with patch("offline_models.train_zone_model", side_effect=self.fake_train) as train:
            replacement = self.ensure()["xgboost"]["A"]
        self.assertEqual(train.call_count, 1)
        self.assertNotEqual(replacement["model_id"], reference["model_id"])
        self.assertEqual(manifest_path.read_text(), "{invalid JSON")
        zone_training.validate_model_reference(replacement)

    def test_zone_keys_preserve_identity_and_stay_within_weight_directory(self):
        ids = ["001", "1", "../A", "..\\A", "A/B", "A?B", "CON", "站点"]
        keys = [zone_training.zone_key(zone) for zone in ids]
        self.assertEqual(len(set(keys)), len(ids))
        for key in keys:
            self.assertEqual(Path(key).name, key)
            self.assertNotIn("/", key)
            self.assertNotIn("\\", key)
            self.assertTrue((self.root / key).resolve().is_relative_to(self.root.resolve()))

    def test_two_zones_train_real_lstm_and_xgboost_once_then_reuse_for_forecasting(self):
        import offline_models
        from forecasting import forecast_load
        config = TrainingConfig(xgboost_workers=1, min_history_days=1, min_samples=2, defaults={
            "lstm_hidden_size": 4, "lstm_epochs": 1,
            "lstm_batch_size": 128, "lstm_device": "cpu", "lstm_head_dropout": 0.0,
            "xgboost_n_estimators": 3, "xgboost_early_stopping_rounds": 2, "xgboost_n_jobs": 1,
            "xgboost_device": "cpu",
        })
        with patch("offline_models.train_zone_model", wraps=offline_models.train_zone_model) as fit:
            refs = zone_training.ensure_dataset_models(self.dataset, ["lstm", "xgboost"], config, data_dir=self.root,
                                                      zone_ids=["A", "B"], history_days=1, horizon_days=1)
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(len({ref["model_id"] for zones in refs.values() for ref in zones.values()}), 4)
        with patch("offline_models.train_zone_model", side_effect=AssertionError("saved models must not refit")):
            reused = zone_training.ensure_dataset_models(self.dataset, ["lstm", "xgboost"], config, data_dir=self.root,
                                                        zone_ids=["A", "B"], history_days=1, horizon_days=1)
            self.assertEqual(refs, reused)
            for zone in ("A", "B"):
                frame = zone_training.canonical_zone_frame(self.dataset, zone)
                origin = frame.time.iloc[-24]
                history, validation = frame.iloc[-168:-144], frame.iloc[-144:-24]
                for backend in ("lstm", "xgboost"):
                    with self.subTest(zone=zone, backend=backend):
                        result = forecast_load(history, validation, frame, origin, 24, backend=backend,
                                               model_reference=refs[backend][zone])
                        state = json.loads(json.dumps(result.attrs["reusable_forecaster_state"], allow_nan=False))
                        replay = forecast_load(history, validation, frame.assign(actual_kwh=1e10), origin, 24,
                                               backend=backend, fitted_state=state)
                        self.assertEqual(len(result), 24)
                        self.assertTrue(np.isfinite(result.predicted_kwh).all())
                        self.assertTrue((result.predicted_kwh >= 0).all())
                        pd.testing.assert_frame_equal(result, replay)


if __name__ == "__main__":
    unittest.main()
