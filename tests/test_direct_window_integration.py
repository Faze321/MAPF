"""Real direct-window model, daily calibration, and immutable Agent replay."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import TrainingConfig
from forecasting import forecast_zone
from global_forecaster import NativeForecasterArtifact
import offline_models
import zone_training


class DirectWindowIntegrationTests(unittest.TestCase):
    def test_real_xgboost_two_day_output_five_day_bias_and_native_replay(self):
        from xgboost import XGBRegressor

        times = pd.date_range("2026-01-01", periods=16 * 24, freq="h")
        position = np.arange(len(times))
        price = .3 + .15 * ((position % 24) >= 12)
        frame = pd.DataFrame({"time": times, "actual_kwh": 20 + np.sin(position / 5) + 3 * price,
                              "e_price": price, "T": 15 + 4 * np.cos(position / 24)})
        origin = times[-48]
        config = TrainingConfig(min_history_days=1, min_samples=4, xgboost_workers=1, defaults={
            "xgboost_device": "cpu", "xgboost_n_jobs": 1, "xgboost_n_estimators": 2,
            "xgboost_early_stopping_rounds": 1, "xgboost_max_depth": 1,
            "xgboost_exog_cols": ["e_price", "temp_price_idx"],
        })

        def wide(column, source=frame):
            return source[["time", column]].rename(columns={column: "A"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = SimpleNamespace(cache_dir=root / "cache", dataset_fingerprint="direct-integration",
                feature_manifest={}, timeseries=frame.rename(columns={"time": "timestamp", "actual_kwh": "load_kwh",
                    "e_price": "energy_price", "T": "temperature"}).assign(zone_id="A"))
            with patch("offline_models.train_zone_model", wraps=offline_models.train_zone_model) as fit:
                reference = zone_training.ensure_dataset_models(dataset, ["xgboost"], config, data_dir=root,
                    zone_ids=["A"], history_days=1, horizon_days=2)["xgboost"]["A"]
            fit.assert_called_once()
            manifest = zone_training.validate_model_reference(reference)
            self.assertEqual(manifest["training_report"]["training_phases"], ["train"])
            self.assertFalse(manifest["training_report"]["full_fit_reset"])
            self.assertEqual((reference["parameters"]["input_hours"], reference["parameters"]["output_hours"]), (24, 48))
            raw_predict = zone_training.load_cached_predictor(reference)
            calls = []

            def observed_predict(context, future):
                self.assertEqual((len(context), len(future)), (24, 48))
                self.assertLess(context.time.max(), future.time.min())
                self.assertNotIn("actual_kwh", future)
                calls.append((context.copy(), future.copy()))
                return raw_predict(context, future)

            options = dict(zone_id="A", category="test", energy_price=wide("e_price"),
                service_price=wide("e_price"), occupancy=wide("e_price"), weather=frame[["time", "T"]],
                profile={}, forecast_origin=origin, history_days=1, validation_days=5, horizon_days=2,
                backend="xgboost", diurnal_blend_alpha=.25, model_reference=reference)
            with patch("zone_training.load_cached_predictor", return_value=observed_predict), \
                 patch.object(XGBRegressor, "fit", side_effect=AssertionError("Inference must never fit")), \
                 patch("offline_models.train_zone_model", side_effect=AssertionError("Inference must never retrain")):
                result = forecast_zone(load=wide("actual_kwh"), **options)
                self.assertEqual(len(calls), 6)
                self.assertEqual([future.time.iloc[0] for _, future in calls],
                                 [origin - pd.Timedelta(days=5 - day) for day in range(5)] + [origin])
                calibration = result.summary["calibration"]
                self.assertEqual(calibration["samples_per_hour"], [5] * 24)
                residuals = np.array([row["residual_kwh"] for row in calibration["validation_predictions"]]).reshape(5, 24)
                np.testing.assert_allclose(calibration["bias_by_hour"], residuals.mean(axis=0))
                np.testing.assert_allclose(result.hourly.bias_kwh, np.tile(residuals.mean(axis=0), 2))

                changed = frame.copy()
                changed.loc[changed.time >= origin, "actual_kwh"] = 1e9
                second = forecast_zone(load=wide("actual_kwh", changed), **options)
                np.testing.assert_allclose(result.hourly.predicted_kwh, second.hourly.predicted_kwh)
                self.assertEqual(result.reusable_forecaster["fitted_state"], second.reusable_forecaster["fitted_state"])

                artifact = NativeForecasterArtifact.from_forecast_result(result)
                artifact_path = root / "native.json"
                artifact_path.write_text(json.dumps(artifact.to_dict(), allow_nan=False), encoding="utf-8")
                restored = NativeForecasterArtifact.from_dict(json.loads(artifact_path.read_text(encoding="utf-8")))
                frozen_state = json.dumps(restored.fitted_state, sort_keys=True, allow_nan=False)
                calls.clear()
                replay = restored.predict(wide("e_price"))
                np.testing.assert_allclose(result.hourly.predicted_kwh, replay.hourly.predicted_load)
                self.assertEqual(len(calls), 1)  # The saved five-day bias is reused without recalibration.
                scenario = wide("e_price")
                scenario.loc[scenario.time >= origin, "A"] *= 1.5
                repriced = restored.predict(scenario)
                self.assertEqual(len(calls), 2)
                np.testing.assert_allclose(calls[-1][1].e_price, scenario.loc[scenario.time >= origin, "A"])
                np.testing.assert_allclose(calls[-1][1].temp_price_idx, calls[-1][1]["T"].to_numpy()
                                          * calls[-1][1].e_price.to_numpy())
                self.assertEqual(frozen_state, json.dumps(restored.fitted_state, sort_keys=True, allow_nan=False))
                self.assertEqual(len(repriced.hourly), 48)
                self.assertEqual(repriced.metadata["model_id"], reference["model_id"])
                zone_training.validate_model_reference(reference)


if __name__ == "__main__":
    unittest.main()
