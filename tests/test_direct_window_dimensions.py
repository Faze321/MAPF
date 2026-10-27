"""Real default-window dimensions and best-iteration bundle bounds."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import xgboost

import offline_models as models


def complete_feature_frame(hours):
    index = np.arange(hours, dtype=float)
    time = pd.date_range("2025-01-01", periods=hours, freq="h")
    temperature = 20 + np.sin(index / 24)
    price = 0.3 + 0.1 * (index % 24 >= 12)
    return pd.DataFrame({"time": time, "actual_kwh": 30 + np.sin(index / 6) + index / 1000,
                         "T": temperature, "U": 60 + np.cos(index / 24), "nRAIN": index % 9 == 0,
                         "e_price": price, "s_price": 0.2 + index % 24 / 1000,
                         "is_weekend": (time.dayofweek >= 5).astype(float), "temp_price_idx": temperature * price})


class DirectDefaultDimensionTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_720_hour_input_supports_24_48_and_72_direct_outputs(self):
        for horizon in (24, 48, 72):
            with self.subTest(output_hours=horizon):
                frame = complete_feature_frame(720 + horizon + 2)
                params = models._parameters("lstm", {"input_hours": 720, "output_hours": horizon, "lstm_device": "cpu"})
                state = models._fit_preprocess("lstm", frame.iloc[:720], params)
                data = models._lstm_dataset(frame, np.asarray([720, 721]), state)
                history = torch.stack([data[0][0], data[1][0]])
                future = torch.stack([data[0][1], data[1][1]])
                targets = torch.stack([data[0][2], data[1][2]])
                self.assertEqual(tuple(history.shape), (2, 720, 19))
                self.assertEqual(tuple(future.shape), (2, horizon, 18))
                self.assertEqual(tuple(targets.shape), (2, horizon))
                model = models._lstm_model(state).eval()
                self.assertEqual(model.head.in_features, 32 + horizon * 18)
                with torch.inference_mode():
                    output = model(history, future)
                self.assertEqual(tuple(output.shape), (2, horizon))
                self.assertTrue(torch.isfinite(output).all())
                xgb_params = models._parameters("xgboost", {"input_hours": 720, "output_hours": horizon,
                                                            "xgboost_device": "cpu"})
                xgb_state = models._fit_preprocess("xgboost", frame.iloc[:720], xgb_params)
                features = models._xgb_features(frame, np.asarray([720, 721]), xgb_state)
                self.assertEqual(features.shape, (2, 720 + 4 + horizon * 18))
                self.assertEqual(features.dtype, np.float32)
                self.assertEqual(features.nbytes, 2 * (720 + 4 + horizon * 18) * 4)

    def test_real_720_to_24_train_save_load_smoke(self):
        device = "cuda:0" if torch.cuda.is_available() and xgboost.build_info().get("USE_CUDA") else "cpu"
        frame = complete_feature_frame(774)
        future = complete_feature_frame(798).iloc[-24:]
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("lstm", "xgboost"):
                with self.subTest(backend=backend, device=device):
                    params = {"input_hours": 720, "output_hours": 24, f"{backend}_device": device,
                              "lstm_epochs": 1, "xgboost_n_estimators": 1, "xgboost_n_jobs": 1}
                    folder = Path(directory) / backend
                    report = models.train_zone_model(backend, frame, frame.time.iloc[747],
                                                     frame.time.iloc[-1] + pd.Timedelta(hours=1), params, folder)
                    self.assertEqual(report["sample_counts"]["train"], 4)
                    self.assertEqual(report["sample_counts"]["validation"], 4)
                    self.assertEqual(report["execution_device"], device)
                    self.assertEqual(report["input_hours"], 720)
                    self.assertEqual(report["output_hours"], 24)
                    prediction = models.load_zone_predictor(backend, folder)(frame, future)
                    self.assertEqual(len(prediction), 24)
                    self.assertTrue(np.isfinite(prediction.raw_predicted_kwh).all())
                    if backend == "xgboost":
                        self.assertEqual(report["best_rounds"], [1] * 24)
                        self.assertEqual(report["actual_dimensions"]["xgboost_feature_count"], 1156)

    def test_xgboost_stale_rounds_are_sliced_and_best_iteration_cannot_overrun_reload(self):
        frame = complete_feature_frame(96)
        frame["actual_kwh"] = np.where(np.arange(96) < 72, 10.0, 20.0)
        params = {"input_hours": 8, "output_hours": 3, "xgboost_device": "cpu",
                  "xgboost_n_estimators": 8, "xgboost_early_stopping_rounds": 2, "xgboost_n_jobs": 1}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = models.train_zone_model("xgboost", frame, frame.time.iloc[72],
                                             frame.time.iloc[-1] + pd.Timedelta(hours=1), params, folder)
            self.assertEqual(report["rounds_run"], [3, 3, 3])
            self.assertEqual(report["best_rounds"], [1, 1, 1])
            bundle = json.loads((folder / "model.json").read_text())
            for native_model in bundle["models"]:
                model = xgboost.XGBRegressor()
                model.load_model(bytearray(json.dumps(native_model).encode()))
                booster = model.get_booster()
                self.assertEqual(booster.num_boosted_rounds(), 1)
                best_iteration = booster.attr("best_iteration")
                self.assertTrue(best_iteration is None or int(best_iteration) < booster.num_boosted_rounds())
            prediction = models.load_zone_predictor("xgboost", folder)(frame, complete_feature_frame(99).iloc[-3:])
            np.testing.assert_allclose(prediction.raw_predicted_kwh, 10.0)


if __name__ == "__main__":
    unittest.main()
