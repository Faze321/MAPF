"""Behavioral contracts for shared configuration, accounting and forecast code."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecasting
from config import (RunConfig, normalize_agent_mode_list, normalize_float_list,
                    normalize_forecast_model_list, normalize_int_list)
from global_forecaster import ForecasterArtifact
from usage import aggregate_usage, cumulative_global_agent_usage, summarize_agent_call_usage


class ConfigAndUsageTests(unittest.TestCase):
    def test_numeric_defaults_nulls_and_explicit_zero(self):
        defaults = RunConfig()
        fields = {key: value for key, value in asdict(defaults).items()
                  if type(value) in (int, float) and key != "max_parallel_datasets"}
        for empty in (None, ""):
            parsed = RunConfig.from_mapping(dict.fromkeys(fields, empty))
            for key, expected in fields.items():
                self.assertEqual(getattr(parsed, key), expected, key)
        parsed = RunConfig.from_mapping(dict.fromkeys(fields, "0"))
        for key in fields:
            self.assertEqual(getattr(parsed, key), 0, key)

    def test_lists_keep_order_after_conversion_and_aliases(self):
        self.assertEqual(normalize_int_list("02,1;2"), [2, 1])
        self.assertEqual(normalize_float_list(["0.10000001", "0.1", "1"]), [0.1, 1.0])
        self.assertEqual(normalize_forecast_model_list("ar;LSTM,AR"), ["AR", "lstm"])
        self.assertEqual(normalize_agent_mode_list("multi-agent-discussion,multi_agent_discussion_3rounds"),
                         ["multi_agent_discussion_3rounds"])
        for normalize in (normalize_int_list, normalize_float_list, normalize_forecast_model_list):
            self.assertIsNone(normalize([]))
        with self.assertRaises(ValueError):
            normalize_float_list("1.01")

    def test_missing_usage_is_not_reported_as_complete(self):
        calls = [{"prompt_tokens": "10", "completion_tokens": 2, "total_tokens": 12},
                 {"prompt_tokens": 3, "completion_tokens": None, "total_tokens": None}]
        summary = summarize_agent_call_usage(calls)
        self.assertEqual(summary["prompt_tokens"], 13)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertFalse(summary["token_usage_complete"])
        summary["agent_invoked_zone_count"] = 1
        total = cumulative_global_agent_usage([summary, {"agent_invoked": False}])
        self.assertEqual(total["agent_invoked_round_count"], 1)
        self.assertEqual(total["agent_call_count"], 2)
        self.assertFalse(total["token_usage_complete"])
        self.assertTrue(aggregate_usage([])["token_usage_complete"])


class ForecastContracts(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2024-02-01")
        self.history = pd.DataFrame({
            "time": pd.date_range(self.origin - pd.Timedelta(hours=48), periods=48, freq="h"),
            "actual_kwh": np.arange(1.0, 49.0), "e_price": 0.4,
        })
        self.future = pd.DataFrame({
            "time": pd.date_range(self.origin, periods=7, freq="h"),
            "actual_kwh": 10000.0, "e_price": 0.4,
        })

    def backend(self, name, stack):
        def point(load, exog, horizon):
            return load[-1] + np.arange(1, horizon + 1) + exog[:horizon, 0]

        common = dict(context_hours=24, step_horizon=3, exog_cols=["e_price"],
                      diurnal_blend_alpha=0.0, roll_actuals=False)
        if name == "timesfm":
            stack.enter_context(patch.object(forecasting, "load_timesfm_model", return_value=object()))
            def predict(model, load, past_exog, exog, horizon, *args):
                values = point(load, exog, horizon)
                return values, values - 1, values + 1
            stack.enter_context(patch.object(forecasting, "run_timesfm_prediction", side_effect=predict))
            return forecasting.timesfm_forecast, {**common, "repo": "test"}, None
        if name == "chronos":
            stack.enter_context(patch.object(forecasting, "load_chronos_model", return_value=object()))
            def predict(model, load, horizon, context_hours, **kwargs):
                values = point(load, kwargs["exog_horizon"], horizon)
                return values, values - 1, values + 1
            stack.enter_context(patch.object(forecasting, "run_chronos_prediction", side_effect=predict))
            return forecasting.chronos_forecast, {**common, "repo": "test", "device": "auto"}, None
        trained = stack.enter_context(patch.object(forecasting, "train_lstm_model", return_value=object()))
        stack.enter_context(patch.object(forecasting, "serialize_lstm_bundle", return_value={"trained": True}))
        stack.enter_context(patch.object(forecasting, "deserialize_lstm_bundle", return_value=object()))
        def predict(model, load, past_exog, exog, horizon):
            return point(load, exog, horizon)
        stack.enter_context(patch.object(forecasting, "run_lstm_prediction", side_effect=predict))
        return forecasting.lstm_forecast, {
            **common, "device": "auto", "hidden_size": 8, "num_layers": 1,
            "epochs": 1, "learning_rate": 0.001, "batch_size": 8, "seed": 42,
        }, trained

    def test_chunked_forecasts_never_roll_in_future_actuals(self):
        expected = [49.4, 50.4, 51.4, 52.8, 53.8, 54.8, 56.2]
        for backend in ("timesfm", "chronos", "lstm"):
            with self.subTest(backend=backend), ExitStack() as stack:
                predict, kwargs, _ = self.backend(backend, stack)
                before = self.future.copy(deep=True)
                result = predict(self.history, self.history.iloc[:0], self.future, self.origin, 7, **kwargs)
                np.testing.assert_allclose(result.predicted_kwh, expected)
                changed = self.future.assign(actual_kwh=-123.0)
                other = predict(self.history, self.history.iloc[:0], changed, self.origin, 7, **kwargs)
                pd.testing.assert_frame_equal(result, other)
                pd.testing.assert_frame_equal(self.future, before)

    def test_fitted_state_replays_validation_bias_without_retraining(self):
        history, validation = self.history.iloc[:-4], self.history.iloc[-4:]
        for backend in ("timesfm", "chronos", "lstm"):
            with self.subTest(backend=backend), ExitStack() as stack:
                predict, kwargs, trained = self.backend(backend, stack)
                kwargs["diurnal_blend_alpha"] = 0.3
                result = predict(history, validation, self.future, self.origin, 7, **kwargs)
                state = result.attrs["reusable_forecaster_state"]
                self.assertTrue(state["calibration"]["enabled"])
                if trained:
                    trained.reset_mock()
                replay = predict(history, validation, self.future, self.origin, 7, fitted_state=state, **kwargs)
                pd.testing.assert_frame_equal(result, replay)
                if trained:
                    trained.assert_not_called()

    def test_legacy_artifact_tracks_horizon_offsets_per_zone(self):
        artifact = ForecasterArtifact(
            backend="AR", coefficients=(0, 1, 0, 0, 0, 0, 0, 0, 0), zones=("A", "B"),
            numeric_covariates=(), categorical_levels={}, history_values={"A": (10,), "B": (20,)},
            history_timestamps={}, validation_bias={"A": (1, 2), "B": (3,)}, forecast_origin=str(self.origin),
        )
        future = pd.DataFrame({"timestamp": [self.origin, self.origin, self.origin + pd.Timedelta(hours=1),
                                            self.origin + pd.Timedelta(hours=2)],
                               "zone": ["B", "A", "A", "B"], "energy_price": 0.4})
        result = artifact.predict(future).hourly
        self.assertEqual(result[result.zone == "A"].predicted_load.tolist(), [11, 13])
        self.assertEqual(result[result.zone == "B"].predicted_load.tolist(), [23, 26])
        self.assertEqual(result[result.zone == "B"].horizon_offset.tolist(), [0, 1])
        pd.testing.assert_frame_equal(result, artifact.predict(future).hourly)


if __name__ == "__main__":
    unittest.main()
