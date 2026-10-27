"""Real CUDA models can train in independently seeded spawned workers."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from config import TrainingConfig
from zone_training import _zone_training_payload, ensure_dataset_models, load_cached_predictor


class ParallelLSTMTests(unittest.TestCase):
    def test_worker_payload_contains_only_its_own_site(self):
        data = SimpleNamespace(timeseries=pd.DataFrame({"zone_id": ["A", "B"], "value": [1, 2]}),
                               cache_dir=Path("cache"), dataset_fingerprint="fixture")
        payload = _zone_training_payload(data, "A")
        self.assertEqual(payload.timeseries.zone_id.tolist(), ["A"])
        self.assertEqual(payload.dataset_fingerprint, "fixture")
        payload.timeseries.loc[0, "value"] = 99
        self.assertEqual(data.timeseries.loc[0, "value"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_two_cuda_lstm_workers_publish_independent_reusable_models(self):
        times = pd.date_range("2024-01-01", periods=45 * 24, freq="h")
        hours = np.arange(len(times))
        frames = [pd.DataFrame({"timestamp": times, "zone_id": zone,
                                "load_kwh": base + 3 * np.sin(hours / 5),
                                "energy_price": .3 + .2 * (hours % 24 >= 12)})
                  for zone, base in (("A", 10), ("B", 30))]
        config = TrainingConfig(lstm_workers=2, defaults={
            "lstm_device": "cuda", "lstm_hidden_size": 4,
            "lstm_epochs": 1, "lstm_early_stopping_patience": 1, "lstm_batch_size": 128,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = SimpleNamespace(timeseries=pd.concat(frames, ignore_index=True), cache_dir=root / "cache",
                                      dataset_fingerprint="two-zones", feature_manifest={})
            references = ensure_dataset_models(dataset, ["lstm"], config, data_dir=root,
                                                zone_ids=["A", "B"], history_days=1, horizon_days=1)["lstm"]
            self.assertEqual(set(references), {"A", "B"})
            self.assertNotEqual(references["A"]["model_id"], references["B"]["model_id"])
            means = []
            for zone, source in zip(("A", "B"), frames):
                folder = Path(references[zone]["model_dir"])
                manifest = json.loads((folder / "manifest.json").read_text())
                self.assertEqual(manifest["training_report"]["execution_device"], "cuda:0")
                self.assertFalse(manifest["training_report"]["two_phase_training"])
                state = json.loads((folder / "preprocess.json").read_text())
                means.append(state["load_mean"])
                context = source.rename(columns={"timestamp": "time", "load_kwh": "actual_kwh", "energy_price": "e_price"})
                future = pd.DataFrame({"time": pd.date_range(times[-1] + pd.Timedelta(hours=1), periods=24, freq="h"),
                                       "e_price": .5})
                result = load_cached_predictor(references[zone])(context, future)
                self.assertEqual(len(result), 24)
                self.assertTrue(np.isfinite(result.raw_predicted_kwh).all())
            self.assertAlmostEqual(means[1] - means[0], 20)


if __name__ == "__main__":
    unittest.main()
