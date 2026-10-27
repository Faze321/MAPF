import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import offline_models as models


def frame_for_test(hours=288):
    values = np.arange(hours, dtype=float)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=hours, freq="h"),
        "actual_kwh": 30 + values * 0.02 + 4 * np.sin(values * np.pi / 12),
        "e_price": 2 + np.sin(values * np.pi / 12),
        "T": np.where(values % 7 == 0, np.nan, 20 + np.cos(values * np.pi / 12)),
        "U": np.nan,
    })


class CompleteWindowTests(unittest.TestCase):
    def test_no_padding_or_windows_across_gaps(self):
        frame = frame_for_test(12).drop(index=5).reset_index(drop=True)
        np.testing.assert_array_equal(models.complete_target_indices(frame, 3), [3, 4, 8, 9, 10])
        np.testing.assert_array_equal(models.complete_target_indices(frame.iloc[:3], 3), [])

    def test_invalid_context(self):
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                models.complete_target_indices(frame_for_test(10), value)

    def test_target_load_is_not_an_input(self):
        frame = frame_for_test()
        params = models._parameters("xgboost", {"xgboost_device": "cpu", "input_hours": 24, "output_hours": 24})
        state = models._fit_preprocess("xgboost", frame.iloc[:200], params)
        before = models._xgb_features(frame, np.asarray([200]), state)
        changed = frame.copy()
        changed.loc[200:, "actual_kwh"] = 999999
        np.testing.assert_allclose(before, models._xgb_features(changed, np.asarray([200]), state))

    def test_daily_diagnostic_refreshes_only_prior_actuals(self):
        frame = frame_for_test()
        contexts = []

        def predict(context, future):
            self.assertLess(context["time"].max(), future["time"].min())
            contexts.append(len(context))
            return pd.DataFrame({"raw_predicted_kwh": np.zeros(len(future))})

        result = models._direct_diagnostic(predict, frame, frame.iloc[240]["time"],
                                            frame.iloc[-1]["time"] + pd.Timedelta(hours=1), 1.0, 24)
        self.assertEqual(contexts, [240, 264])
        self.assertEqual(result["forecast_values"], 48)
        self.assertEqual(result["windows"], 2)

    def test_nonfinite_prediction_is_not_hidden_by_nonnegative_clip(self):
        frame = frame_for_test()
        params = models._parameters("xgboost", {"xgboost_device": "cpu", "input_hours": 24, "output_hours": 24})
        state = models._fit_preprocess("xgboost", frame, params)
        model = Mock()
        model.predict.return_value = np.asarray([np.nan])
        predict = models._predictor("xgboost", [model] * 24, state)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            predict(frame, frame_for_test(312).iloc[-24:])


class OfflineModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.frame = frame_for_test()
        cls.start = cls.frame.iloc[-48]["time"]
        cls.end = cls.frame.iloc[-1]["time"] + pd.Timedelta(hours=1)
        # A feature that appears only after the development split must stay dropped.
        cls.frame.loc[cls.frame["time"] >= cls.start, "U"] = 70.0
        cls.reports = {}
        for backend, params in {
            "lstm": dict(input_hours=24, output_hours=24, lstm_hidden_size=4, lstm_epochs=2,
                         lstm_early_stopping_patience=1, lstm_batch_size=64,
                         lstm_device="cpu", lstm_head_dropout=0.0),
            "xgboost": dict(input_hours=24, output_hours=24, xgboost_n_estimators=4, xgboost_early_stopping_rounds=2, xgboost_n_jobs=1,
                            xgboost_device="cpu"),
        }.items():
            cls.reports[backend] = models.train_zone_model(
                backend, cls.frame, cls.start, cls.end, params, Path(cls.temp.name) / backend)

    @classmethod
    def tearDownClass(cls):
        import torch
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_single_fit_keeps_training_scalers_and_records_early_stopping(self):
        for backend, report in self.reports.items():
            with self.subTest(backend=backend):
                self.assertFalse(report["full_fit_reset"])
                self.assertFalse(report["preprocessing_refitted"])
                self.assertFalse(report["two_phase_training"])
                self.assertEqual(report["training_phases"], ["train"])
                self.assertEqual(report["sample_counts"], dict(train=193, validation=25, complete_windows=241, purged_windows=23))
                self.assertEqual(report["preprocessing"]["fitted_rows"], 240)
                self.assertAlmostEqual(report["preprocessing"]["load_mean"], self.frame.iloc[:240]["actual_kwh"].mean())
                self.assertIn("U", report["dropped_exog_cols"])
                self.assertTrue(any(name.endswith("T_missing") for name in report["features"]))
                self.assertEqual(report["direct_horizon_diagnostic"]["forecast_values"], 48)
                json.dumps(report, allow_nan=False)
                folder = Path(self.temp.name) / backend
                self.assertTrue((folder / ("model.pt" if backend == "lstm" else "model.json")).is_file())
                if backend == "lstm":
                    import torch
                    weights = torch.load(folder / "model.pt", weights_only=True, map_location="cpu")
                    self.assertTrue(weights)
                    self.assertTrue(all(isinstance(value, torch.Tensor) for value in weights.values()))
                history = pd.read_csv(folder / "training_history.csv")
                self.assertEqual(set(history["phase"]), {"train"})
                self.assertTrue(np.isfinite(history["train_normalized_mae"]).all())

    def test_reload_never_fits_and_ignores_future_actuals(self):
        future = frame_for_test(312).iloc[-24:].copy()
        future["actual_kwh"] = 999999999.0
        for backend in self.reports:
            with self.subTest(backend=backend), patch.object(models, "_fit_lstm", side_effect=AssertionError("fit called")), patch.object(models, "_fit_xgboost", side_effect=AssertionError("fit called")):
                predict = models.load_zone_predictor(backend, Path(self.temp.name) / backend)
                first = predict(self.frame, future)
                future_without_actual = future.drop(columns=["actual_kwh"])
                second = predict(self.frame, future_without_actual)
                np.testing.assert_allclose(first["raw_predicted_kwh"], second["raw_predicted_kwh"])
                self.assertEqual(len(first), 24)
                self.assertTrue(np.isfinite(first["raw_predicted_kwh"]).all())
                self.assertTrue((first["raw_predicted_kwh"] >= 0).all())
                # Loading the saved model again gives the same fixed predictor.
                reloaded = models.load_zone_predictor(backend, Path(self.temp.name) / backend)
                np.testing.assert_allclose(first["raw_predicted_kwh"], reloaded(self.frame, future)["raw_predicted_kwh"])

    def test_prediction_rejects_incomplete_context_and_gaps(self):
        future = frame_for_test(312).iloc[-24:]
        for backend in self.reports:
            with self.subTest(backend=backend):
                predict = models.load_zone_predictor(backend, Path(self.temp.name) / backend)
                with self.assertRaisesRegex(ValueError, "context hours"):
                    predict(self.frame.iloc[-1:], future)
                with self.assertRaisesRegex(ValueError, "consecutive"):
                    predict(self.frame.drop(index=len(self.frame) - 2), future)
                with self.assertRaisesRegex(ValueError, "consecutive|requires exactly"):
                    predict(self.frame, future.drop(index=future.index[1]))


if __name__ == "__main__":
    unittest.main()
