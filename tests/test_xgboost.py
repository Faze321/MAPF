"""Real-model regression tests for the XGBoost forecasting and control handoff."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from config import RunConfig, TrainingConfig
from forecasting import forecast_load, forecast_zone
from global_forecaster import NativeForecasterArtifact
from main import main
from orchestrator import load_forecaster_bundle
from zone_training import ensure_dataset_models


def sample_frame(days=60):
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
        self.origin = self.frame.time.iloc[-72]
        self.history = self.frame.iloc[-144:-96].copy()
        self.validation = self.frame.iloc[-96:-72].copy()
        self.future = self.frame.iloc[-72:-24].copy()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.params = dict(xgboost_n_estimators=25, xgboost_max_depth=3,
                           xgboost_exog_cols=["e_price", "T", "U"], xgboost_seed=7,
                           xgboost_learning_rate=.1, xgboost_n_jobs=1, xgboost_device="cpu",
                           xgboost_early_stopping_rounds=3)
        series = self.frame.rename(columns={"time": "timestamp", "actual_kwh": "load_kwh", "e_price": "energy_price"}).assign(zone_id="A")
        dataset = SimpleNamespace(timeseries=series, cache_dir=self.root / "cache", dataset_fingerprint="fixture", feature_manifest={})
        self.reference = ensure_dataset_models(dataset, ["xgboost"], TrainingConfig(defaults=self.params), data_dir=self.root,
                                               zone_ids=["A"], history_days=2, horizon_days=2)["xgboost"]["A"]

    def predict(self, frame=None, **kwargs):
        return forecast_load(
            self.history, self.validation, self.future if frame is None else frame,
            self.origin, 48, backend="xgboost", model_reference=self.reference, **kwargs,
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
        self.assertGreater(low.predicted_kwh.mean(), high.predicted_kwh.mean() + 1)

    def test_validation_targets_calibrate_but_do_not_train_the_model(self):
        original = self.predict()
        changed = forecast_load(
            self.history, self.validation.assign(actual_kwh=200), self.future,
            self.origin, 48, backend="xgboost", model_reference=self.reference,
        )
        first = original.attrs["reusable_forecaster_state"]
        second = changed.attrs["reusable_forecaster_state"]
        self.assertEqual(first["model_reference"], second["model_reference"])
        self.assertNotEqual(first["bias_vec"], second["bias_vec"])

    def test_multiple_origins_and_alpha_values_reuse_one_offline_model(self):
        identities = []
        with patch.object(XGBRegressor, "fit", side_effect=AssertionError("forecast must not fit")):
            for origin in (self.origin, self.origin + pd.Timedelta(days=1)):
                history = self.frame[self.frame.time.between(origin - pd.Timedelta(days=7), origin - pd.Timedelta(days=5), inclusive="left")]
                validation = self.frame[self.frame.time.between(origin - pd.Timedelta(days=5), origin, inclusive="left")]
                for alpha in (0.0, .5):
                    result = forecast_load(history, validation, self.frame, origin, 48, backend="xgboost",
                                           model_reference=self.reference, diurnal_blend_alpha=alpha)
                    identities.append(result.attrs["reusable_forecaster_state"]["model_reference"]["model_id"])
        self.assertEqual(identities, [self.reference["model_id"]] * 4)

    def test_raw_predictions_are_shared_across_alpha_but_context_and_prices_invalidate(self):
        from zone_training import load_cached_predictor
        from unittest.mock import Mock
        raw_predict = Mock(wraps=load_cached_predictor(self.reference))
        history = self.frame[self.frame.time.between(self.origin - pd.Timedelta(days=7), self.origin - pd.Timedelta(days=5), inclusive="left")]
        validation = self.frame[self.frame.time.between(self.origin - pd.Timedelta(days=5), self.origin, inclusive="left")]
        cache = {}
        def predict(alpha, context=history, future=self.future):
            return forecast_load(context, validation, future, self.origin, 48, backend="xgboost",
                                 model_reference=self.reference, diurnal_blend_alpha=alpha,
                                 raw_prediction_cache=cache)
        with patch("zone_training.load_cached_predictor", return_value=raw_predict):
            first = predict(0.0)
            predict(.5)
            self.assertEqual(raw_predict.call_count, 6)
            predict(.5, future=self.future.assign(actual_kwh=1e12))
            self.assertEqual(raw_predict.call_count, 6)
            predict(.5, future=self.future.assign(e_price=.95))
            # The last validation origin also consumes the first formal day's prices
            # because every direct model call predicts the complete 48-hour output.
            self.assertEqual(raw_predict.call_count, 8)
            changed = history.copy()
            changed.loc[changed.index[-1], "actual_kwh"] += 4
            predict(.5, context=changed)
            self.assertGreater(raw_predict.call_count, 8)
            replay = predict(0.0)
        pd.testing.assert_frame_equal(first, replay)

    def test_no_validation_and_missing_covariates(self):
        history = pd.concat([self.history, self.validation], ignore_index=True).tail(48)
        result = forecast_load(
            history, history.iloc[:0], self.future, self.origin, 48,
            backend="xgboost", model_reference=self.reference,
        )
        self.assertFalse(result.attrs["calibration"]["enabled"])
        self.assertTrue(np.isfinite(result.predicted_kwh).all())
        self.assertTrue(result.q10_kwh.isna().all())

    def test_reject_target_covariates_invalid_parameters_and_short_or_gapped_history(self):
        for params in (
            {"xgboost_exog_cols": ["actual_kwh"]}, {"xgboost_exog_cols": ["load"]},
            {"xgboost_lags": [0]}, {"xgboost_lags": []}, {"xgboost_lags": [1, 1]},
            {"xgboost_lags": [1.5]}, {"xgboost_lags": [True]},
            {"xgboost_n_estimators": 0}, {"xgboost_learning_rate": float("nan")},
            {"xgboost_colsample_bytree": 0}, {"xgboost_n_jobs": 0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                TrainingConfig(defaults=params)
        with self.assertRaises(ValueError):
            self.predict(diurnal_blend_alpha=1.5)
        with self.assertRaisesRegex(ValueError, "consecutive hourly"):
            forecast_load(self.history.iloc[:-1], self.validation, self.future, self.origin, 48,
                          backend="xgboost", model_reference=self.reference)
        with self.assertRaisesRegex(ValueError, "model|offline|reference"):
            forecast_load(self.history, self.validation, self.future, self.origin, 48, backend="xgboost")

    def test_native_artifact_roundtrip_matches_initial_forecast(self):
        def wide(column):
            return self.frame[["time", column]].rename(columns={column: "A"})
        result = forecast_zone(
            zone_id="A", category="test", load=wide("actual_kwh"),
            energy_price=wide("e_price"), service_price=wide("e_price"),
            occupancy=wide("e_price"), weather=self.frame[["time"]], profile={},
            forecast_origin=self.origin, history_days=2, validation_days=1, horizon_days=2,
            backend="XGB", model_reference=self.reference,
        )
        self.assertEqual(result.summary["backend"], "xgboost")
        self.assertEqual(result.summary["model_id"], self.reference["model_id"])
        self.assertEqual(result.summary["training_overlap_hours"], 48)
        self.assertFalse(result.summary["evaluation_is_out_of_sample"])
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
        with self.assertRaises(ValueError):
            RunConfig.from_mapping({"xgboost_max_depth": 3})

    def test_cli_forecaster_and_agent_only_use_same_trained_model(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="xgboost_test_", dir=scratch) as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            full_frame = sample_frame(days=60)
            origin = full_frame.time.iloc[-24]
            series = full_frame.rename(columns={
                "time": "timestamp", "actual_kwh": "load_kwh", "e_price": "energy_price",
            }).assign(zone_id="A")
            series.to_csv(data / "series.csv", index=False)
            config = root / "config.yaml"
            settings = {"agent": {}, "data": {"adapter": "long_format", "timeseries_file": "series.csv"},
                        "training": {"defaults": {"xgboost_n_estimators": 12, "xgboost_max_depth": 2,
                                                   "xgboost_seed": 17,
                                                   "xgboost_n_jobs": 1, "xgboost_device": "cpu",
                                                   "xgboost_early_stopping_rounds": 3}}, "run": {
                "data_dir": str(data), "output_folder": str(root / "output"), "experiment_name": "xgb_test",
                "zones": ["A"], "forecast_starts": [str(origin)], "forecast_models": ["AR", "lstm"],
                "experiment_seeds": [99], "dry_run": True,
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
            self.assertNotIn("xgboost_seed", bundle.forecast_parameters)
            self.assertEqual(bundle.manifest["model_provenance"]["A"]["model_parameters"]["xgboost_seed"], 17)
            self.assertEqual(bundle.forecast_parameters["diurnal_blend_alpha"], 0.2)
            self.assertEqual(bundle.scenario_artifacts["A"].horizon_hours, 24)
            with patch.object(XGBRegressor, "fit", side_effect=AssertionError("must not refit")):
                agent = main(["--config", str(config), "--forecast-models", "xgboost", "--stage", "agent",
                              "--forecaster-output-dir", str(bundle.output_dir)])
            self.assertTrue(agent["control_results_json"].is_file())
            windows = pd.read_csv(agent["control_final_windows_csv"])
            self.assertEqual(len(windows), 8)
            control = json.loads(agent["control_results_json"].read_text(encoding="utf-8"))
            self.assertEqual(control["model_ids"], bundle.manifest["model_ids"])
            model_id = bundle.manifest["model_ids"]["A"]
            self.assertEqual(set(windows.model_id), {model_id})
            self.assertEqual(control["zones"][0]["model_id"], model_id)
            self.assertEqual(control["zones"][0]["training_overlap_hours"], 24)
            self.assertFalse(control["zones"][0]["evaluation_is_out_of_sample"])


if __name__ == "__main__":
    unittest.main()
