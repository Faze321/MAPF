"""CUDA execution and saved-model replay, with a portable explicit failure path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch
import xgboost

import offline_models as models


class GPUDeviceTests(unittest.TestCase):
    def test_requested_cuda_never_silently_falls_back_to_cpu(self):
        with patch("xgboost.build_info", return_value={"USE_CUDA": False}):
            with self.assertRaisesRegex(RuntimeError, "no CUDA support"):
                models._xgboost_device("cuda")
        with patch("xgboost.build_info", return_value={"USE_CUDA": True}), \
             patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CUDA device"):
                models._xgboost_device("cuda")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                models._device("cuda")
        with patch("xgboost.build_info", return_value={"USE_CUDA": True}), \
             patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "index 1 is unavailable"):
                models._xgboost_device("cuda:1")
        self.assertEqual(models._xgboost_device("cpu"), "cpu")

    @unittest.skipUnless(torch.cuda.is_available() and xgboost.build_info().get("USE_CUDA"), "CUDA required")
    def test_real_cuda_direct_training_and_inference_after_reload(self):
        hours = np.arange(312)
        frame = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=312, freq="h"),
                              "actual_kwh": 25 + 4 * np.sin(hours / 5), "e_price": .3 + .2 * (hours % 24 >= 12)})
        training, future = frame.iloc[:288], frame.iloc[288:]
        parameters = {
            "lstm": {"input_hours": 24, "output_hours": 24, "lstm_device": "cuda", "lstm_hidden_size": 8,
                     "lstm_epochs": 2, "lstm_batch_size": 64, "lstm_early_stopping_patience": 1},
            "xgboost": {"input_hours": 24, "output_hours": 24, "xgboost_device": "cuda:0", "xgboost_n_estimators": 4,
                        "xgboost_early_stopping_rounds": 2, "xgboost_n_jobs": 1},
        }
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            for backend, settings in parameters.items():
                with self.subTest(backend=backend):
                    folder = Path(directory) / backend
                    report = models.train_zone_model(backend, training, training.time.iloc[-48],
                                                     future.time.iloc[0], settings, folder)
                    self.assertEqual(report["execution_device"], "cuda:0")
                    self.assertEqual(report["training_phases"], ["train"])
                    self.assertFalse(report["two_phase_training"])
                    state = json.loads((folder / "preprocess.json").read_text())
                    self.assertEqual(state["parameters"][backend + "_device"], settings[backend + "_device"])
                    with patch.object(models, "_fit_lstm", side_effect=AssertionError("must not fit")), \
                         patch.object(models, "_fit_xgboost", side_effect=AssertionError("must not fit")):
                        predict = models.load_zone_predictor(backend, folder)
                        first = predict(training, future)
                        reloaded = models.load_zone_predictor(backend, folder)
                        replay = reloaded(training, future.assign(actual_kwh=1e9))
                    self.assertEqual(len(first), 24)
                    self.assertTrue(np.isfinite(first.raw_predicted_kwh).all())
                    np.testing.assert_allclose(first.raw_predicted_kwh, replay.raw_predicted_kwh)
            self.assertFalse(any("falling back" in str(item.message).lower() or
                                 "no visible gpu" in str(item.message).lower() for item in emitted),
                             [str(item.message) for item in emitted])


if __name__ == "__main__":
    unittest.main()
