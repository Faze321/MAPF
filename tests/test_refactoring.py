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
                  if type(value) in (int, float)
                  and key not in {"max_parallel_datasets", "max_dataset_errors", "forecast_start_count", "history_days", "horizon_days"}}
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
        for normalize in (normalize_int_list, normalize_forecast_model_list):
            self.assertIsNone(normalize([]))
        with self.assertRaises(ValueError):
            normalize_float_list([])
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
            "time": pd.date_range(self.origin, periods=24, freq="h"),
            "actual_kwh": 10000.0, "e_price": 0.4,
        })

    def backend(self, name, stack, input_hours=48):
        def point(load, exog, horizon):
            return load[-1] + np.arange(1, horizon + 1) + exog[:horizon, 0]

        common = dict(backend=name, diurnal_blend_alpha=0.0)
        if name == "timesfm":
            stack.enter_context(patch.object(forecasting, "load_timesfm_model", return_value=object()))
            def predict(model, load, past_exog, exog, horizon, *args):
                values = point(load, exog, horizon)
                return values, values - 1, values + 1
            stack.enter_context(patch.object(forecasting, "run_timesfm_prediction", side_effect=predict))
            return forecasting.forecast_load, {**common, "timesfm_exog_cols": ["e_price"], "timesfm_context_hours": 24}, None
        if name == "chronos":
            stack.enter_context(patch.object(forecasting, "load_chronos_model", return_value=object()))
            def predict(model, load, horizon, context_hours, **kwargs):
                values = point(load, kwargs["exog_horizon"], horizon)
                return values, values - 1, values + 1
            stack.enter_context(patch.object(forecasting, "run_chronos_prediction", side_effect=predict))
            return forecasting.forecast_load, {**common, "chronos_exog_cols": ["e_price"], "chronos_context_hours": 24}, None
        trained = stack.enter_context(patch("offline_models.train_zone_model", side_effect=AssertionError("inference must not train")))
        def predict(context, future):
            values = point(context.actual_kwh.to_numpy(), future[["e_price"]].to_numpy(), len(future))
            return pd.DataFrame({"time": future.time.to_numpy(), "raw_predicted_kwh": values})
        stack.enter_context(patch("zone_training.load_cached_predictor", return_value=predict))
        return forecasting.forecast_load, {
            **common, "model_reference": {"backend": "lstm", "model_id": "fixture-lstm",
                                           "parameters": {"input_hours": input_hours, "output_hours": 24}},
        }, trained

    def test_forecasts_never_roll_in_future_actuals(self):
        expected = 48.4 + np.arange(1, 25)
        for backend in ("timesfm", "chronos", "lstm"):
            with self.subTest(backend=backend), ExitStack() as stack:
                predict, kwargs, _ = self.backend(backend, stack)
                before = self.future.copy(deep=True)
                result = predict(self.history, self.history.iloc[:0], self.future, self.origin, 24, **kwargs)
                np.testing.assert_allclose(result.predicted_kwh, expected)
                changed = self.future.assign(actual_kwh=-123.0)
                other = predict(self.history, self.history.iloc[:0], changed, self.origin, 24, **kwargs)
                pd.testing.assert_frame_equal(result, other)
                pd.testing.assert_frame_equal(self.future, before)

    def test_fitted_state_replays_validation_bias_without_retraining(self):
        history, validation = self.history.iloc[:-24], self.history.iloc[-24:]
        for backend in ("timesfm", "chronos", "lstm"):
            with self.subTest(backend=backend), ExitStack() as stack:
                predict, kwargs, trained = self.backend(backend, stack, input_hours=24)
                kwargs["diurnal_blend_alpha"] = 0.3
                result = predict(history, validation, self.future, self.origin, 24, **kwargs)
                state = result.attrs["reusable_forecaster_state"]
                self.assertTrue(state["calibration"]["enabled"])
                if trained:
                    trained.reset_mock()
                replay = predict(history, validation, self.future, self.origin, 24, fitted_state=state, **kwargs)
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
