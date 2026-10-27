"""Trainable forecasts must use the same per-site weather as offline training."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loader import load_pipeline_data
from dataset_adapter import CanonicalDataset
from forecasting import ForecastResult
from orchestrator import build_contexts, forecast_runtime_parameters, make_forecast_parameters
from zone_training import canonical_zone_frame


class ZoneWeatherTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2026-01-01", periods=2, freq="h")
        self.profiles = pd.DataFrame({"zone_id": ["A", "B"], "category": ["test", "test"],
                                      "selection_reason": ["fixture", "fixture"]})
        source = pd.DataFrame({"timestamp": [*times, *times], "zone_id": ["A", "A", "B", "B"],
                               "load_kwh": [1., 2., 3., 4.], "energy_price": [.3, .4, .5, .6],
                               "temperature": [10., 11., 30., np.nan],
                               "humidity": [20., 21., np.nan, 51.], "rain": [0., np.nan, 4., 5.]})
        self.dataset = CanonicalDataset(
            timeseries=source, static_zone_features=self.profiles,
            feature_manifest={"weather_columns": ["temperature", "humidity", "rain"]},
            price_change_reference=None, dataset_fingerprint="fixture", cache_dir=Path("cache"), cache_keys={},
        )
        self.pipeline = load_pipeline_data(Path("dataset"), self.profiles, ["A", "B"],
                                           canonical_dataset=self.dataset)

    def forecast_weather(self, backend, pipeline=None):
        received = {}
        def predict(**kwargs):
            received[kwargs["zone_id"]] = kwargs["weather"]
            return ForecastResult(pd.DataFrame(), {"zone_id": kwargs["zone_id"]})
        parameters = make_forecast_parameters(forecast_origin="2026-01-01", backend=backend,
                                              history_days=30, validation_days=5, horizon_days=1,
                                              diurnal_blend_alpha=0.)
        with ExitStack() as stack:
            stack.enter_context(patch("orchestrator.forecast_zone", side_effect=predict))
            stack.enter_context(patch("orchestrator.load_stress_thresholds", return_value={}))
            stack.enter_context(patch("orchestrator.build_pricing_windows_3h", return_value=[]))
            stack.enter_context(patch("orchestrator.apply_load_policy_stress", side_effect=lambda summary, *args: summary))
            stack.enter_context(patch("orchestrator.build_agent_hourly_data", return_value=[]))
            stack.enter_context(patch("orchestrator.build_hourly_averages", return_value=[]))
            build_contexts(pipeline_data=self.pipeline if pipeline is None else pipeline,
                           selected_zones=self.profiles, zone_load_thresholds=self.profiles,
                           zone_model_references={"A": {"model_id": "a"}, "B": {"model_id": "b"}},
                           **forecast_runtime_parameters(parameters))
        return received

    def test_lstm_and_xgboost_match_training_weather_without_cross_site_filling(self):
        self.assertIsNotNone(self.pipeline.canonical_dataset)
        for backend in ("lstm", "xgboost"):
            with self.subTest(backend=backend):
                weather = self.forecast_weather(backend)
                for zone in ("A", "B"):
                    expected = canonical_zone_frame(self.dataset, zone)[["time", "T", "U", "nRAIN"]]
                    pd.testing.assert_frame_equal(weather[zone], expected)
                self.assertEqual(weather["B"].loc[0, "T"], 30.)
                self.assertTrue(pd.isna(weather["B"].loc[1, "T"]))
                self.assertTrue(pd.isna(weather["B"].loc[0, "U"]))
                self.assertTrue(pd.isna(weather["A"].loc[1, "nRAIN"]))

    def test_other_backends_keep_existing_weather_input(self):
        for backend in ("timesfm", "chronos", "AR"):
            with self.subTest(backend=backend):
                weather = self.forecast_weather(backend)
                self.assertTrue(all(frame is self.pipeline.weather for frame in weather.values()))

    def test_trainable_backends_fall_back_when_canonical_data_is_unavailable(self):
        pipeline = SimpleNamespace(**{key: value for key, value in vars(self.pipeline).items()
                                      if key != "canonical_dataset"})
        weather = self.forecast_weather("xgboost", pipeline)
        self.assertTrue(all(frame is self.pipeline.weather for frame in weather.values()))


if __name__ == "__main__":
    unittest.main()
