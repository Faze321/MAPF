"""Reject invalid validation loads and retain calibrated uncertainty coverage."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting import forecast_load


class ForecastIntervalTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2026-02-06")
        self.history = pd.DataFrame({
            "time": pd.date_range(self.origin - pd.Timedelta(days=35), periods=720, freq="h"),
            "actual_kwh": 10.0,
        })
        self.validation = pd.DataFrame({
            "time": pd.date_range(self.origin - pd.Timedelta(days=5), periods=120, freq="h"),
            "actual_kwh": np.repeat([10.0, 12.0, 14.0, 16.0, 18.0], 24) + np.tile(np.arange(24) * .5, 5),
        })
        self.future = pd.DataFrame({"time": pd.date_range(self.origin, periods=24, freq="h")})

    def test_negative_validation_load_fails_before_backend_preparation(self):
        self.validation.loc[47, "actual_kwh"] = -1.0
        with patch("forecasting.prepare_forecast_backend") as prepare:
            with self.assertRaisesRegex(ValueError, "validation.*(non-negative|nonnegative|negative)"):
                forecast_load(self.history, self.validation, self.future, self.origin, 24, backend="AR")
            prepare.assert_not_called()

    def test_missing_quantiles_use_all_five_days_residual_interval_per_row(self):
        def prepare(*args, **kwargs):
            def predict(context, future):
                raw = 20.0 + np.arange(len(future), dtype=float)
                q10, q90 = raw - 2.0, raw + 2.0
                q10[12:], q90[12:] = np.nan, np.nan
                return pd.DataFrame({"time": future["time"].to_numpy(), "raw_predicted_kwh": raw,
                                     "q10_kwh": q10, "q90_kwh": q90})
            return predict, {}

        with patch("forecasting.prepare_forecast_backend", prepare):
            result = forecast_load(self.history, self.validation, self.future, self.origin, 24,
                                   backend="timesfm")
        calibration = result.attrs["calibration"]
        # Daily residual deviations are +4,+2,0,-2,-4; all 120 values give width 4.
        self.assertEqual(calibration["interval_half_width"], 4.0)
        self.assertEqual(calibration["samples_per_hour"], [5] * 24)
        point = 14.0 + np.arange(24) * .5
        width = np.r_[np.full(12, 2.0), np.full(12, 4.0)]
        np.testing.assert_allclose(result["predicted_kwh"], point)
        np.testing.assert_allclose(result["q50_kwh"], point)
        np.testing.assert_allclose(result["q10_kwh"], point - width)
        np.testing.assert_allclose(result["q90_kwh"], point + width)


if __name__ == "__main__":
    unittest.main()
