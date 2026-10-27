"""Dataset-local training concurrency with per-zone publication protection."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import TrainingConfig
import zone_training


def dataset_at(root, zones=("A",)):
    times = pd.date_range("2025-01-01", periods=50 * 24, freq="h")
    position = np.arange(len(times))
    frames = [pd.DataFrame({"timestamp": times, "zone_id": zone,
                            "load_kwh": 20 + index + np.sin(position / 5),
                            "energy_price": .3 + .2 * (position % 24 >= 12)})
              for index, zone in enumerate(zones)]
    return SimpleNamespace(cache_dir=root / "cache", dataset_fingerprint=str(root),
                           feature_manifest={}, timeseries=pd.concat(frames, ignore_index=True))


def write_fake_model(backend, frame, start, end, parameters, directory):
    directory = Path(directory)
    model_name = "model.pt" if backend == "lstm" else "model.json"
    (directory / model_name).write_text(json.dumps({"load_mean": float(frame.actual_kwh.mean())}))
    (directory / "preprocess.json").write_text(json.dumps(parameters))
    frame[["time", "actual_kwh"]].to_csv(directory / "training_history.csv", index=False)
    return {"best_rounds": 1}


class UnlockedTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = TrainingConfig(lstm_workers=1, xgboost_workers=1,
                                     defaults={"lstm_device": "cpu", "xgboost_device": "cpu"})

    def ensure(self, dataset, backend, config=None):
        return zone_training.ensure_dataset_models(dataset, [backend], config or self.config,
                                                   data_dir=Path(dataset.cache_dir).parent,
                                                   zone_ids=sorted(dataset.timeseries.zone_id.unique()),
                                                   history_days=1, horizon_days=1)

    def test_independent_datasets_can_train_simultaneously_for_each_backend(self):
        # Three simultaneous fits exceed both former machine-wide caps (1 / 2).
        for backend in ("lstm", "xgboost"):
            with self.subTest(backend=backend):
                rendezvous = threading.Barrier(3, timeout=10)
                datasets = [dataset_at(self.root / backend / f"dataset-{index}") for index in range(3)]
                entered = set()
                guard = threading.Lock()

                def train(*args):
                    with guard:
                        entered.add(str(args[-1]))
                    rendezvous.wait()
                    return write_fake_model(*args)

                with patch("offline_models.train_zone_model", side_effect=train) as fit, ThreadPoolExecutor(max_workers=3) as pool:
                    jobs = [pool.submit(self.ensure, dataset, backend) for dataset in datasets]
                    references = [job.result(timeout=20)[backend]["A"] for job in jobs]
                self.assertEqual(fit.call_count, 3)
                self.assertEqual(len(entered), 3)
                self.assertEqual(len({ref["model_dir"] for ref in references}), 3)
                for reference in references:
                    zone_training.validate_model_reference(reference)

    def test_configured_three_xgboost_workers_really_run_three_zones_concurrently(self):
        dataset = dataset_at(self.root / "three-workers", zones=("A", "B", "C"))
        config = TrainingConfig(xgboost_workers=3, defaults={"xgboost_device": "cpu"})
        rendezvous = threading.Barrier(3, timeout=10)
        active = peak = 0
        guard = threading.Lock()

        def train(*args):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                rendezvous.wait()
                return write_fake_model(*args)
            finally:
                with guard:
                    active -= 1

        with patch("offline_models.train_zone_model", side_effect=train) as fit:
            references = self.ensure(dataset, "xgboost", config)["xgboost"]
        self.assertEqual(fit.call_count, 3)
        self.assertEqual(peak, 3)
        self.assertEqual(set(references), {"A", "B", "C"})
        index = json.loads((dataset.cache_dir / "weight" / "xgboost" / "index.json").read_text())
        self.assertEqual(index["available_zone_ids"], ["A", "B", "C"])

    def test_same_zone_contenders_serialize_and_reuse_the_first_complete_model(self):
        dataset = dataset_at(self.root / "same-zone")
        fitting, release, lock_contended = threading.Event(), threading.Event(), threading.Event()
        real_lock = zone_training._try_file_lock

        def attempt_lock(path):
            handle = real_lock(path)
            if handle is None and Path(path).name == ".training.lock":
                lock_contended.set()
            return handle

        def train(*args):
            fitting.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release the first zone fit")
            return write_fake_model(*args)

        with patch("offline_models.train_zone_model", side_effect=train) as fit, \
             patch.object(zone_training, "_try_file_lock", side_effect=attempt_lock), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.ensure, dataset, "xgboost")
            try:
                self.assertTrue(fitting.wait(timeout=10), "first fit did not start")
                second = pool.submit(self.ensure, dataset, "xgboost")
                self.assertTrue(lock_contended.wait(timeout=10), "second request did not contend on the per-zone lock")
                self.assertEqual(fit.call_count, 1)
            finally:
                release.set()
            first_result, second_result = first.result(timeout=20), second.result(timeout=20)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(first_result, second_result)
        reference = first_result["xgboost"]["A"]
        zone_training.validate_model_reference(reference)
        folder = Path(reference["model_dir"]).parent
        self.assertEqual(json.loads((folder / "current.json").read_text())["model_reference"], reference)
        self.assertFalse(list(folder.glob(".pending-*")))


if __name__ == "__main__":
    unittest.main()
