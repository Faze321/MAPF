"""Real-model regression tests for the XGBoost forecasting and control handoff."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from config import BACKEND_PARAMETER_NAMES, RunConfig, XGBOOST_PARAMETER_DEFAULTS
from forecasting import forecast_zone
from global_forecaster import NativeForecasterArtifact
from main import main
from orchestrator import (
    FORECAST_PARAMETER_NAMES, load_forecaster_bundle, make_forecast_parameters,
    price_conditioned_parameters,
)
from xgboost_forecaster import xgboost_forecast


def sample_frame(days=12):
    times = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    rng = np.random.default_rng(42)
    price = rng.choice([0.2, 0.5, 0.9], len(times))
    return pd.DataFrame({
        "time": times, "e_price": price,
        "actual_kwh": 80 - 50 * price + 5 * np.sin(2 * np.pi * times.hour / 24),
    })


class XGBoostForecastTests(unittest.TestCase):
    def setUp(self):
        self.frame = sample_frame()
        self.origin = self.frame.time.iloc[-48]
        self.history = self.frame.iloc[:-72].copy()
        self.validation = self.frame.iloc[-72:-48].copy()
        self.future = self.frame.iloc[-48:].copy()
        self.params = dict(n_estimators=25, max_depth=3, exog_cols=["e_price"], seed=7)

    def predict(self, frame=None, **kwargs):
        return xgboost_forecast(
            self.history, self.validation, self.future if frame is None else frame,
            self.origin, 48, **{**self.params, **kwargs},
        )

    def test_forecast_has_valid_nonnegative_hourly_outputs_and_no_future_load_leakage(self):
        result = self.predict()
        changed = self.predict(self.future.assign(actual_kwh=1e9))
        removed = self.predict(self.future.drop(columns="actual_kwh"))
        pd.testing.assert_frame_equal(result, changed)
        pd.testing.assert_frame_equal(result, removed)
        self.assertEqual(len(result), 48)
        self.assertTrue(np.isfinite(result.predicted_kwh).all())
        self.assertTrue((result.predicted_kwh >= 0).all())
        self.assertTrue((result.q10_kwh <= result.predicted_kwh).all())
        self.assertTrue((result.q90_kwh >= result.predicted_kwh).all())
        self.assertTrue(result.attrs["calibration"]["enabled"])
        pd.testing.assert_series_equal(result.time, self.future.time.reset_index(drop=True))

    def test_json_replay_does_not_fit_and_uses_new_price_schedule(self):
        result = self.predict(diurnal_blend_alpha=0.2)
        state = json.loads(json.dumps(result.attrs["reusable_forecaster_state"], allow_nan=False))
        with patch.object(XGBRegressor, "fit", side_effect=AssertionError("must not refit")):
            replay = self.predict(fitted_state=state, diurnal_blend_alpha=0.2)
            low = self.predict(self.future.assign(e_price=0.2), fitted_state=state, diurnal_blend_alpha=0.2)
            high = self.predict(self.future.assign(e_price=0.9), fitted_state=state, diurnal_blend_alpha=0.2)
        pd.testing.assert_frame_equal(result, replay)
        self.assertGreater(low.predicted_kwh.mean(), high.predicted_kwh.mean() + 10)

    def test_validation_targets_calibrate_but_do_not_train_the_model(self):
        original = self.predict()
        changed = xgboost_forecast(
            self.history, self.validation.assign(actual_kwh=200), self.future,
            self.origin, 48, **self.params,
        )
        first = original.attrs["reusable_forecaster_state"]
        second = changed.attrs["reusable_forecaster_state"]
        self.assertEqual(first["model_json"], second["model_json"])
        self.assertNotEqual(first["bias_vec"], second["bias_vec"])

    def test_no_validation_and_missing_covariates(self):
        history = pd.concat([self.history, self.validation], ignore_index=True)
        result = xgboost_forecast(
            history, history.iloc[:0], self.future, self.origin, 48,
            **{**self.params, "exog_cols": ["e_price", "T", "U"]},
        )
        self.assertFalse(result.attrs["calibration"]["enabled"])
        self.assertTrue(np.isfinite(result.predicted_kwh).all())
        self.assertTrue(result.q10_kwh.isna().all())

    def test_reject_target_covariates_invalid_parameters_and_short_or_gapped_history(self):
        for params in (
            {"exog_cols": ["actual_kwh"]}, {"exog_cols": ["load"]},
            {"lags": [0]}, {"lags": []}, {"lags": [1, 1]},
            {"lags": [1.5]}, {"lags": [True]}, {"lags": [240]},
            {"n_estimators": 0}, {"learning_rate": float("nan")},
            {"colsample_bytree": 0}, {"n_jobs": 0}, {"diurnal_blend_alpha": 1.5},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.predict(**params)
        with self.assertRaisesRegex(ValueError, "consecutive hourly"):
            xgboost_forecast(self.history.iloc[:-1], self.validation, self.future, self.origin, 48, **self.params)

    def test_native_artifact_roundtrip_matches_initial_forecast(self):
        def wide(column):
            return self.frame[["time", column]].rename(columns={column: "A"})
        result = forecast_zone(
            zone_id="A", category="test", load=wide("actual_kwh"),
            energy_price=wide("e_price"), service_price=wide("e_price"),
            occupancy=wide("e_price"), weather=self.frame[["time"]], profile={},
            forecast_origin=self.origin, history_days=7, validation_days=1, horizon_days=2,
            backend="XGB", xgboost_n_estimators=25, xgboost_max_depth=3,
        )
        self.assertEqual(result.summary["backend"], "xgboost")
        self.assertEqual(result.summary["xgboost_lags"], [1, 2, 3, 24])
        artifact = NativeForecasterArtifact.from_forecast_result(result)
        restored = NativeForecasterArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))
        self.assertTrue(all("actual_kwh" not in row for row in restored.known_future))
        with patch.object(XGBRegressor, "fit", side_effect=AssertionError("must not refit")):
            replay = restored.predict(wide("e_price"))
        np.testing.assert_allclose(result.hourly.predicted_kwh, replay.hourly.predicted_load)

    def test_retired_forecaster_config_keys_fail_instead_of_loading_defaults(self):
        for key, value in (("forecast_model", "AR"), ("forecast_start", str(self.origin))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                RunConfig.from_mapping({key: value})
        self.assertTrue(set(XGBOOST_PARAMETER_DEFAULTS).issubset(BACKEND_PARAMETER_NAMES))

    def test_cli_forecaster_and_agent_only_use_same_trained_model(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="xgboost_test_", dir=scratch) as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            full_frame = sample_frame(days=36)
            origin = full_frame.time.iloc[-24]
            series = full_frame.rename(columns={
                "time": "timestamp", "actual_kwh": "load_kwh", "e_price": "energy_price",
            }).assign(zone_id="A")
            series.to_csv(data / "series.csv", index=False)
            config = root / "config.yaml"
            settings = {"agent": {}, "data": {"adapter": "long_format", "timeseries_file": "series.csv"}, "run": {
                "data_dir": str(data), "output_folder": str(root / "output"), "experiment_name": "xgb_test",
                "zones": ["A"], "forecast_starts": [str(origin)], "forecast_models": ["AR", "lstm"],
                "xgboost_n_estimators": 12, "xgboost_max_depth": 2, "xgboost_lags": [1, 24],
                "xgboost_seed": 17, "experiment_seeds": [99], "dry_run": True,
            }}
            config.write_text(yaml.safe_dump(settings), encoding="utf-8")
            output = main(["--config", str(config), "--forecast-models", "xgb", "--stage", "forecaster",
                           "--diurnal-blend-alphas", "0.2"])
            records = pd.read_csv(output["experiment_runs_csv"])
            self.assertEqual(records["backend"].tolist(), ["xgboost"])
            manifests = list(output["experiment_dir"].rglob("forecaster_manifest.json"))
            self.assertEqual(len(manifests), 1, records.to_string())
            bundle = load_forecaster_bundle(manifests[0])
            self.assertEqual(bundle.forecast_parameters["backend"], "xgboost")
            self.assertEqual(bundle.forecast_parameters["xgboost_seed"], 99)
            self.assertEqual(bundle.forecast_parameters["diurnal_blend_alpha"], 0.2)
            self.assertEqual(bundle.scenario_artifacts["A"].horizon_hours, 24)
            with patch.object(XGBRegressor, "fit", side_effect=AssertionError("must not refit")):
                agent = main(["--config", str(config), "--forecast-models", "xgboost", "--stage", "agent",
                              "--forecaster-output-dir", str(bundle.output_dir)])
            self.assertTrue(agent["control_results_json"].is_file())
            windows = pd.read_csv(agent["control_final_windows_csv"])
            self.assertEqual(len(windows), 8)


if __name__ == "__main__":
    unittest.main()
