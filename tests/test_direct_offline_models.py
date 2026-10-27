"""Direct H-to-P models, purged split boundaries, and best-weight persistence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

import offline_models as models


def direct_frame(hours=96):
    index = np.arange(hours, dtype=float)
    return pd.DataFrame({"time": pd.date_range("2025-01-01", periods=hours, freq="h"),
                         "actual_kwh": 30 + index * 0.05 + np.sin(index / 3),
                         "e_price": 0.3 + 0.1 * (index % 24 > 12),
                         "T": np.where(index % 5 == 0, np.nan, 20 + np.cos(index / 12)),
                         "U": np.where(index < 72, np.nan, 70.0)})


class DirectWindowFeatureTests(unittest.TestCase):
    def test_complete_windows_include_all_labels_and_restart_stride_after_gap(self):
        frame = direct_frame(20).drop(index=9).reset_index(drop=True)
        np.testing.assert_array_equal(models.complete_target_indices(frame, 3, 2, 2), [3, 5, 7, 12, 14, 16])
        np.testing.assert_array_equal(models.complete_target_indices(frame.iloc[:4], 3, 2), [])
        np.testing.assert_array_equal(models.complete_target_indices(frame.iloc[:5], 3, 2), [3])
        for kwargs in ({"horizon_hours": 0}, {"stride_hours": 0}, {"stride_hours": True}):
            with self.assertRaises(ValueError):
                models.complete_target_indices(frame, 3, **kwargs)

    def test_lstm_uses_aligned_history_and_separate_future_covariates(self):
        frame = direct_frame()
        p = models._parameters("lstm", {"input_hours": 8, "output_hours": 3, "lstm_device": "cpu"})
        state = models._fit_preprocess("lstm", frame.iloc[:72], p)
        history, future, labels = models._lstm_dataset(frame, np.asarray([40]), state)[0]
        self.assertEqual(tuple(history.shape), (8, len(state["history_feature_names"])))
        self.assertEqual(tuple(future.shape), (3, len(state["future_feature_names"])))
        np.testing.assert_allclose(history[:, 0], models._normalized_load(frame, state)[32:40], rtol=1e-6)
        np.testing.assert_allclose(history[:, 1:], models._exog_features(frame, state)[32:40])
        np.testing.assert_allclose(labels, models._normalized_load(frame, state)[40:43], rtol=1e-6)
        changed = frame.copy()
        changed.loc[40:, "actual_kwh"] = 999999
        changed_history, changed_future, _ = models._lstm_dataset(changed, np.asarray([40]), state)[0]
        np.testing.assert_array_equal(history, changed_history)
        np.testing.assert_array_equal(future, changed_future)

    def test_xgboost_uses_all_history_hours_and_future_covariates_without_future_load(self):
        frame = direct_frame()
        p = models._parameters("xgboost", {"input_hours": 8, "output_hours": 3, "xgboost_device": "cpu"})
        state = models._fit_preprocess("xgboost", frame.iloc[:72], p)
        features = models._xgb_features(frame, np.asarray([40]), state)
        np.testing.assert_allclose(features[0, :8], models._normalized_load(frame, state)[32:40][::-1], rtol=1e-6)
        self.assertEqual(state["summary_windows"], [8])
        changed = frame.copy()
        changed.loc[40:, "actual_kwh"] = 1e9
        np.testing.assert_allclose(features, models._xgb_features(changed, np.asarray([40]), state))
        changed.loc[42, "e_price"] = 99
        self.assertFalse(np.array_equal(features, models._xgb_features(changed, np.asarray([40]), state), equal_nan=True))

    def test_xgboost_outputs_share_one_input_window_without_prediction_refeeding(self):
        frame = direct_frame()
        p = models._parameters("xgboost", {"input_hours": 8, "output_hours": 3, "xgboost_device": "cpu"})
        state = models._fit_preprocess("xgboost", frame.iloc[:72], p)
        regressors = [Mock(), Mock(), Mock()]
        for step, regressor in enumerate(regressors):
            regressor.predict.return_value = np.asarray([float(step)])
        result = models._predictor("xgboost", regressors, state)(frame.iloc[:40], frame.iloc[40:43])
        self.assertEqual(len(result), 3)
        inputs = [regressor.predict.call_args.args[0] for regressor in regressors]
        for regressor in regressors:
            regressor.predict.assert_called_once()
        np.testing.assert_array_equal(inputs[0], inputs[1])
        np.testing.assert_array_equal(inputs[1], inputs[2])


class DirectModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.frame = direct_frame()
        cls.start = cls.frame.iloc[72]["time"]
        cls.end = cls.frame.iloc[-1]["time"] + pd.Timedelta(hours=1)
        cls.settings = {
            "lstm": {"input_hours": 8, "output_hours": 3, "lstm_device": "cpu", "lstm_hidden_size": 4,
                     "lstm_batch_size": 16, "lstm_epochs": 5, "lstm_min_delta": 1000,
                     "lstm_early_stopping_patience": 1, "lstm_head_dropout": 0},
            "xgboost": {"input_hours": 8, "output_hours": 3, "xgboost_device": "cpu",
                        "xgboost_n_estimators": 4, "xgboost_early_stopping_rounds": 1, "xgboost_n_jobs": 1},
        }
        cls.reports = {}
        for backend, parameters in cls.settings.items():
            fitting = "_fit_lstm" if backend == "lstm" else "_fit_xgboost"
            with patch.object(models, fitting, wraps=getattr(models, fitting)) as fit:
                cls.reports[backend] = models.train_zone_model(backend, cls.frame, cls.start, cls.end,
                                                               parameters, Path(cls.temp.name) / backend)
                if fit.call_count != 1:
                    raise AssertionError("Direct models must train once without a full-data refit")
                if not np.all(fit.call_args.args[1] + 3 <= 72):
                    raise AssertionError("Training labels cross into validation")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_single_fit_reports_purged_windows_and_training_only_scalers(self):
        for backend, report in self.reports.items():
            with self.subTest(backend=backend):
                self.assertEqual(report["training_phases"], ["train"])
                self.assertFalse(report["two_phase_training"])
                self.assertFalse(report["preprocessing_refitted"])
                self.assertEqual(report["sample_counts"], {"train": 62, "validation": 22, "complete_windows": 86, "purged_windows": 2})
                self.assertEqual(report["preprocessing"]["fitted_rows"], 72)
                self.assertAlmostEqual(report["preprocessing"]["load_mean"], self.frame.iloc[:72].actual_kwh.mean())
                self.assertIn("U", report["dropped_exog_cols"])
                self.assertEqual(report["direct_horizon_diagnostic"]["forecast_values"], 3)
                self.assertEqual(report["actual_dimensions"]["input_hours"], 8)
                self.assertEqual(report["actual_dimensions"]["output_hours"], 3)
                json.dumps(report, allow_nan=False)

    def test_best_lstm_weights_are_restored_without_refitting(self):
        report = self.reports["lstm"]
        self.assertEqual(report["best_rounds"], 1)
        self.assertEqual(report["rounds_run"], 2)
        path = Path(self.temp.name) / "lstm_first_epoch"
        settings = {**self.settings["lstm"], "lstm_epochs": 1}
        models.train_zone_model("lstm", self.frame, self.start, self.end, settings, path)
        first = torch.load(path / "model.pt", weights_only=True)
        restored = torch.load(Path(self.temp.name) / "lstm" / "model.pt", weights_only=True)
        for name in first:
            torch.testing.assert_close(first[name], restored[name], rtol=0, atol=0)

    def test_xgboost_bundle_saves_one_model_per_output_and_only_best_trees(self):
        from xgboost import XGBRegressor
        folder = Path(self.temp.name) / "xgboost"
        bundle = json.loads((folder / "model.json").read_text())
        self.assertEqual(bundle["schema_version"], 2)
        self.assertEqual(len(bundle["models"]), 3)
        self.assertEqual(len(self.reports["xgboost"]["best_rounds"]), 3)
        for payload, rounds in zip(bundle["models"], self.reports["xgboost"]["best_rounds"]):
            regressor = XGBRegressor()
            regressor.load_model(bytearray(json.dumps(payload).encode()))
            self.assertEqual(regressor.get_booster().num_boosted_rounds(), rounds)
        history = pd.read_csv(folder / "training_history.csv")
        self.assertEqual(set(history["step"]), {1, 2, 3})
        self.assertEqual(set(history["phase"]), {"train"})

    def test_fixed_horizon_reload_does_not_fit_or_read_future_actuals(self):
        for backend in self.reports:
            with self.subTest(backend=backend), patch.object(models, "_fit_lstm", side_effect=AssertionError("fit")), \
                 patch.object(models, "_fit_xgboost", side_effect=AssertionError("fit")):
                predict = models.load_zone_predictor(backend, Path(self.temp.name) / backend)
                future = direct_frame(99).iloc[-3:]
                before = predict(self.frame, future)
                after = predict(self.frame, future.assign(actual_kwh=1e12))
                pd.testing.assert_frame_equal(before, after)
                with self.assertRaisesRegex(ValueError, "exactly 3 future hours"):
                    predict(self.frame, future.iloc[:2])
                with self.assertRaisesRegex(ValueError, "complete context"):
                    predict(self.frame.iloc[-7:], future)
                self.assertTrue(np.isfinite(before.raw_predicted_kwh).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_direct_models_train_and_reload_on_cuda(self):
        import xgboost
        if not xgboost.build_info().get("USE_CUDA"):
            self.skipTest("CUDA XGBoost build required")
        for backend in self.settings:
            with self.subTest(backend=backend):
                parameters = {**self.settings[backend], f"{backend}_device": "cuda:0"}
                folder = Path(self.temp.name) / (backend + "_cuda")
                report = models.train_zone_model(backend, self.frame, self.start, self.end, parameters, folder)
                self.assertEqual(report["execution_device"], "cuda:0")
                prediction = models.load_zone_predictor(backend, folder)(self.frame, direct_frame(99).iloc[-3:])
                self.assertEqual(len(prediction), 3)
                self.assertTrue(np.isfinite(prediction.raw_predicted_kwh).all())


if __name__ == "__main__":
    unittest.main()
