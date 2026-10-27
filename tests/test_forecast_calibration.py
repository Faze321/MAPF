"""The daily calibration protocol is causal, reusable and shared by all backends."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecasting
from config import TrainingConfig
from zone_training import ensure_dataset_models


class DailyCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2024-03-01 06:00:00")
        self.history = pd.DataFrame({
            "time": pd.date_range(self.origin - pd.Timedelta(days=35), periods=720, freq="h"),
            "actual_kwh": 20.0, "e_price": 20.0,
        })
        hours = np.tile(np.arange(24, dtype=float), 5)
        days = np.repeat(np.arange(1, 6, dtype=float), 24)
        self.validation = pd.DataFrame({
            "time": pd.date_range(self.origin - pd.Timedelta(days=5), periods=120, freq="h"),
            "actual_kwh": 40 + hours - days - hours / 10,
            "e_price": 40 + hours,
        })
        self.future = pd.DataFrame({
            "time": pd.date_range(self.origin, periods=48, freq="h"),
            "actual_kwh": 99999.0, "e_price": 50 + np.tile(np.arange(24), 2),
        })

    def setup_timesfm(self, stack):
        contexts = []
        stack.enter_context(patch.object(forecasting, "load_timesfm_model", return_value=object()))

        def predict(model, load, exog_context, exog_horizon, horizon, *args):
            contexts.append(np.asarray(load).copy())
            values = np.asarray(exog_horizon[:horizon, 0], dtype=float)
            return values, values - 1, values + 1

        predictor = stack.enter_context(patch.object(forecasting, "run_timesfm_prediction", side_effect=predict))
        return predictor, contexts

    def predict(self, *, history=None, validation=None, future=None, horizon=24, **kwargs):
        return forecasting.forecast_load(
            self.history if history is None else history,
            self.validation if validation is None else validation,
            self.future if future is None else future,
            self.origin, horizon, backend="timesfm", timesfm_context_hours=720,
            timesfm_exog_cols=["e_price"], **kwargs,
        )

    def test_five_daily_predictions_average_same_offset_before_final_correction(self):
        with ExitStack() as stack:
            predictor, contexts = self.setup_timesfm(stack)
            result = self.predict()
        expected_bias = 3 + np.arange(24) / 10
        np.testing.assert_allclose(result.bias_kwh, expected_bias)
        np.testing.assert_allclose(result.predicted_kwh, 47 + np.arange(24) * 0.9)
        calibration = result.attrs["calibration"]
        self.assertEqual(calibration["method"], "daily_hour_mean_v1")
        self.assertEqual(calibration["validation_days"], 5)
        self.assertEqual(calibration["samples_per_hour"], [5] * 24)
        self.assertEqual(len(calibration["validation_predictions"]), 120)
        np.testing.assert_allclose(calibration["bias_by_hour"], expected_bias)
        self.assertEqual(predictor.call_count, 6)
        self.assertTrue(all(len(values) == 720 for values in contexts))
        np.testing.assert_array_equal(contexts[0], self.history.actual_kwh)
        for day in range(1, 6):
            known = pd.concat([self.history, self.validation.iloc[:day * 24]], ignore_index=True)
            np.testing.assert_array_equal(contexts[day], known.actual_kwh.tail(720))

    def test_all_five_backends_use_the_same_daily_protocol_and_training_interval(self):
        for backend in ("timesfm", "chronos", "lstm", "xgboost", "AR"):
            calls = []

            def predict(context, future):
                self.assertNotIn("actual_kwh", future)
                calls.append((context.copy(), future.copy()))
                return pd.DataFrame({"time": future.time.to_numpy(), "raw_predicted_kwh": future.e_price.to_numpy()})

            with self.subTest(backend=backend), patch.object(
                forecasting, "prepare_forecast_backend", return_value=(predict, {})
            ) as prepare:
                reference = ({"backend": backend, "model_id": "fixture", "parameters": {
                    "input_hours": 720, "output_hours": 48}}
                    if backend in {"lstm", "xgboost"} else None)
                result = forecasting.forecast_load(self.history, self.validation, self.future,
                                                   self.origin, 48, backend=backend, model_reference=reference)
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[0], backend)
            pd.testing.assert_frame_equal(prepare.call_args.args[1], self.history)
            self.assertEqual([len(future) for _, future in calls],
                             [48] * 6 if reference else [24] * 5 + [48])
            self.assertTrue(all(context.time.max() < future.time.min() for context, future in calls))
            np.testing.assert_allclose(result.bias_kwh, np.tile(3 + np.arange(24) / 10, 2))
            calibration = result.attrs["calibration"]
            self.assertEqual(len(calibration["validation_predictions"]), 120)
            self.assertEqual(calibration["samples_per_hour"], [5] * 24)
            # Centered residuals are [-2,-1,0,1,2] on every hourly offset.
            self.assertEqual(calibration["interval_half_width"], 2.0)
            np.testing.assert_allclose(result.q10_kwh, result.predicted_kwh - 2)
            np.testing.assert_allclose(result.q90_kwh, result.predicted_kwh + 2)

    def test_default_and_custom_zone_windows_are_exact(self):
        times = pd.date_range(self.origin - pd.Timedelta(days=40), periods=43 * 24, freq="h")
        load = pd.DataFrame({"time": times, "A": 20.0})

        def predict(context, future):
            return pd.DataFrame({"time": future.time.to_numpy(), "raw_predicted_kwh": np.full(len(future), 20.0)})

        for options, expected in (({}, (720, 120, 24)),
                                  ({"history_days": 2, "validation_days": 0, "horizon_days": 3}, (48, 0, 72)),
                                  ({"history_days": 4, "validation_days": 2, "horizon_days": 2}, (96, 48, 48))):
            with self.subTest(options=options), patch.object(
                forecasting, "prepare_forecast_backend", return_value=(predict, {})
            ), patch.object(forecasting, "forecast_load", wraps=forecasting.forecast_load) as dispatch:
                result = forecasting.forecast_zone(
                    zone_id="A", category="test", load=load, energy_price=load,
                    service_price=load, occupancy=load, weather=load[["time"]], profile={},
                    forecast_origin=self.origin, backend="AR", **options,
                )
            history, validation, _, origin, horizon = dispatch.call_args.args
            self.assertEqual((len(history), len(validation), horizon), expected)
            self.assertEqual(history.time.min(), origin - pd.Timedelta(hours=expected[0] + expected[1]))
            self.assertEqual(history.time.max(), origin - pd.Timedelta(hours=expected[1] + 1))
            self.assertEqual(len(result.hourly), horizon)
            self.assertEqual(result.hourly.time.min(), origin)
            self.assertEqual(result.hourly.time.max(), origin + pd.Timedelta(hours=horizon - 1))

    def test_ar_price_response_is_fit_only_on_training_and_reused(self):
        with patch.object(forecasting, "estimate_energy_price_response",
                          wraps=forecasting.estimate_energy_price_response) as fit:
            original = forecasting.forecast_load(self.history, self.validation, self.future,
                                                 self.origin, 24, backend="AR")
        fit.assert_called_once()
        pd.testing.assert_frame_equal(fit.call_args.args[0], self.history)
        state = json.loads(json.dumps(original.attrs["reusable_forecaster_state"], allow_nan=False))
        with patch.object(forecasting, "estimate_energy_price_response", side_effect=AssertionError("no refit")):
            replay = forecasting.forecast_load(self.history, self.validation,
                                               self.future.assign(actual_kwh=-1e10), self.origin, 24,
                                               backend="AR", fitted_state=state)
        pd.testing.assert_frame_equal(original, replay)

    def test_blend_is_applied_once_then_mean_bias_is_subtracted(self):
        for alpha in (0.25, 1.0):
            with self.subTest(alpha=alpha), ExitStack() as stack:
                self.setup_timesfm(stack)
                result = self.predict(diurnal_blend_alpha=alpha)
                raw = 50 + np.arange(24)
                val_raw = 40 + np.arange(24)
                expected_bias = (1 - alpha) * val_raw + alpha * 20 - (37 + np.arange(24) * .9)
                expected_blend = (1 - alpha) * raw + alpha * 20
                np.testing.assert_allclose(result.blended_predicted_kwh, expected_blend)
                np.testing.assert_allclose(result.bias_kwh, expected_bias)
                np.testing.assert_allclose(result.predicted_kwh, np.maximum(expected_blend - expected_bias, 0))

    def test_forecast_load_never_reads_future_actuals(self):
        with ExitStack() as stack:
            self.setup_timesfm(stack)
            original = self.predict(horizon=48)
            poisoned = self.predict(future=self.future.assign(actual_kwh=-1e12), horizon=48)
            removed = self.predict(future=self.future.drop(columns="actual_kwh"), horizon=48)
        pd.testing.assert_frame_equal(original, poisoned)
        pd.testing.assert_frame_equal(original, removed)
        np.testing.assert_array_equal(original.bias_kwh.iloc[:24], original.bias_kwh.iloc[24:])

    def test_validation_truth_affects_only_later_context_not_its_own_prediction(self):
        with ExitStack() as stack:
            _, original_contexts = self.setup_timesfm(stack)
            original = self.predict()
        altered = self.validation.copy()
        altered.loc[altered.index[:24], "actual_kwh"] += 100
        with ExitStack() as stack:
            _, altered_contexts = self.setup_timesfm(stack)
            changed = self.predict(validation=altered)
        first = original.attrs["calibration"]["validation_predictions"][:24]
        other = changed.attrs["calibration"]["validation_predictions"][:24]
        self.assertEqual([r["raw_predicted_kwh"] for r in first], [r["raw_predicted_kwh"] for r in other])
        np.testing.assert_array_equal(original_contexts[0], altered_contexts[0])
        self.assertFalse(np.array_equal(original_contexts[1], altered_contexts[1]))

    def test_saved_state_replay_skips_validation_predictions(self):
        with ExitStack() as stack:
            predictor, _ = self.setup_timesfm(stack)
            original = self.predict(diurnal_blend_alpha=.25)
            state = json.loads(json.dumps(original.attrs["reusable_forecaster_state"], allow_nan=False))
            predictor.reset_mock()
            replay = self.predict(diurnal_blend_alpha=.25, fitted_state=state)
        pd.testing.assert_frame_equal(original, replay)
        self.assertEqual(predictor.call_count, 1)

    def test_missing_or_nonfinite_validation_is_rejected(self):
        damaged = [self.validation.iloc[:-1], self.validation.drop(index=15),
                   self.validation.assign(actual_kwh=np.nan)]
        duplicate = self.validation.copy()
        duplicate.loc[1, "time"] = duplicate.loc[0, "time"]
        damaged.append(duplicate)
        for frame in damaged:
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError), ExitStack() as stack:
                self.setup_timesfm(stack)
                self.predict(validation=frame)

    def test_no_validation_has_zero_bias_and_nonnegative_clipping(self):
        history = pd.concat([self.history, self.validation], ignore_index=True).tail(720)
        with ExitStack() as stack:
            predictor, _ = self.setup_timesfm(stack)
            result = self.predict(history=history, validation=self.validation.iloc[:0])
        np.testing.assert_array_equal(result.bias_kwh, np.zeros(24))
        self.assertFalse(result.attrs["calibration"]["enabled"])
        self.assertEqual(predictor.call_count, 1)
        with ExitStack() as stack:
            self.setup_timesfm(stack)
            clipped = self.predict(future=self.future.assign(e_price=0.0))
        np.testing.assert_array_equal(clipped.predicted_kwh, np.zeros(24))


class NativeModelSmokeTests(unittest.TestCase):
    def test_small_cpu_lstm_fits_once_replays_and_has_no_future_load_leakage(self):
        import offline_models
        times = pd.date_range("2024-01-01", periods=50 * 24, freq="h")
        position = np.arange(len(times))
        frame = pd.DataFrame({"time": times, "actual_kwh": 10 + 2 * np.sin(position / 4),
                              "e_price": .3 + .2 * (position % 24 >= 12)})
        origin = frame.time.iloc[-24]
        history, validation, future = frame.iloc[-864:-144], frame.iloc[-144:-24], frame.iloc[-24:]
        parameters = dict(lstm_hidden_size=4, lstm_epochs=1, lstm_batch_size=128,
                          lstm_device="cpu", lstm_exog_cols=["e_price"], lstm_head_dropout=0.0)
        with tempfile.TemporaryDirectory() as temporary:
            dataset = SimpleNamespace(cache_dir=Path(temporary) / "cache", dataset_fingerprint="fixture", feature_manifest={},
                                      timeseries=frame.rename(columns={"time": "timestamp", "actual_kwh": "load_kwh", "e_price": "energy_price"}).assign(zone_id="A"))
            with patch("offline_models.train_zone_model", wraps=offline_models.train_zone_model) as fit:
                reference = ensure_dataset_models(dataset, ["lstm"], TrainingConfig(defaults=parameters, min_samples=2),
                                                  zone_ids=["A"], history_days=30, horizon_days=1)["lstm"]["A"]
            self.assertEqual(fit.call_count, 1)
            kwargs = dict(backend="lstm", model_reference=reference)
            with patch("offline_models.train_zone_model", side_effect=AssertionError("no refit")):
                original = forecasting.forecast_load(history, validation, future, origin, 24, **kwargs)
                state = json.loads(json.dumps(original.attrs["reusable_forecaster_state"], allow_nan=False))
                replay = forecasting.forecast_load(history, validation, future.assign(actual_kwh=1e9), origin, 24,
                                                   fitted_state=state, **kwargs)
            self.assertTrue(np.isfinite(original.predicted_kwh).all())
            pd.testing.assert_frame_equal(original, replay)


if __name__ == "__main__":
    unittest.main()
