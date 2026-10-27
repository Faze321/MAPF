"""Prediction feature parity, timestamp precision and temporal provenance."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting import build_zone_model_frame, forecast_load, forecast_zone


def raw_predict(context, future):
    return pd.DataFrame({"time": future.time.dt.as_unit("us"), "raw_predicted_kwh": 10.0})


class OfflineInferenceEdgeTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=50 * 24, freq="h"),
                                   "actual_kwh": 10 + np.sin(np.arange(50 * 24) / 5), "e_price": .5})

    def test_hourly_inference_accepts_equal_times_with_different_precision(self):
        for unit in ("s", "ms", "us", "ns"):
            frame = self.frame.assign(time=self.frame.time.dt.as_unit(unit))
            with self.subTest(unit=unit), patch("forecasting.prepare_forecast_backend", return_value=(raw_predict, {})):
                result = forecast_load(frame.iloc[:720], frame.iloc[720:840], frame,
                                       frame.time.iloc[840], 24, backend="lstm")
                self.assertEqual(len(result), 24)
                self.assertTrue(np.isfinite(result.predicted_kwh).all())

    def test_missing_temperature_preserves_missing_interaction(self):
        wide = self.frame[["time", "actual_kwh"]].rename(columns={"actual_kwh": "A"})
        missing = build_zone_model_frame(wide, wide, wide, wide, self.frame[["time"]], "A")
        self.assertTrue(missing.temp_price_idx.isna().all())
        invalid = build_zone_model_frame(wide, wide, wide, wide, self.frame[["time"]].assign(T="invalid"), "A")
        self.assertTrue(invalid.temp_price_idx.isna().all())

    def test_no_overlap_is_not_out_of_sample_when_training_uses_later_data(self):
        wide = self.frame[["time", "actual_kwh"]].rename(columns={"actual_kwh": "A"})
        origin = pd.Timestamp("2024-02-10")
        for start, end, expected in (("2024-02-11", "2024-02-13", False), ("2024-01-01", "2024-02-10", True)):
            reference = {"model_id": "fixture", "backend": "lstm", "zone_id": "A",
                         "parameters": {"input_hours": 720, "output_hours": 24},
                         "training_start": start, "training_end_exclusive": end, "training_intervals": [[start, end]]}
            with self.subTest(end=end), patch("forecasting.prepare_forecast_backend", return_value=(raw_predict, {})):
                result = forecast_zone(zone_id="A", category="test", load=wide, service_price=wide,
                                       energy_price=wide, occupancy=wide, weather=self.frame[["time"]],
                                       profile={}, forecast_origin=origin, backend="lstm", model_reference=reference)
                self.assertEqual(result.summary["training_overlap_hours"], 0)
                self.assertEqual(result.summary["evaluation_is_out_of_sample"], expected)


if __name__ == "__main__":
    unittest.main()
