"""New result names and offline-only historical provenance normalization."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluate import evaluate_agent, evaluate_directory
from evaluation_artifacts import (
    canonical_forecast_parameters,
    load_forecast_context,
    normalize_evaluation_columns,
    normalize_evaluation_fields,
    pairing_metadata,
    signature,
)
from prompts import agent_safe_context, compact_economist_context


class ForecastOutputMigrationTests(unittest.TestCase):
    def make_model_forecast(self, root, model_id):
        details = root / "evaluation" / "forecast_details"
        details.mkdir(parents=True)
        (root / "forecaster_manifest.json").write_text(json.dumps({
            "data_source": {"dataset_fingerprint": "fixture"},
            "forecast_parameters": {"backend": "xgboost", "forecast_origin": "2026-01-01",
                                    "diurnal_blend_alpha": 0.0},
            "model_ids": {"001": model_id},
            "model_provenance": {"001": {"model_id": model_id, "model_training_start": "2025-01-01",
                                         "model_training_end_exclusive": "2026-01-01T01:00:00"}},
        }))
        pd.DataFrame({"zone_id": ["001"], "model_id": [model_id]}).to_csv(root / "evaluation" / "forecast_metrics.csv", index=False)
        pd.DataFrame({"zone_id": ["001"] * 2, "time": ["2026-01-01", "2026-01-01T01:00:00"],
                      "actual_kwh": [1.0, 2.0], "predicted_kwh": [1.0, 2.0], "model_id": [model_id] * 2,
                      "training_period_overlap": [True, False]}).to_csv(details / "zone_001_forecast_vs_actual.csv", index=False)

    def test_saved_model_version_changes_forecast_identity_and_never_pools_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            self.make_model_forecast(first, "trained-version-one")
            self.make_model_forecast(second, "trained-version-two")
            one, two = load_forecast_context(first), load_forecast_context(second)
            self.assertNotEqual(one["identity"], two["identity"])
            self.assertEqual(one["parameters"]["model_ids"], {"001": "trained-version-one"})
            report = evaluate_directory(root, section="forecaster")
            rows = report["tables"]["forecast_by_dataset"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["n"] == 2 for row in rows))
            self.assertTrue(all(row["training_overlap_sample_count"] == 1 for row in rows))
            self.assertTrue(all(row["evaluation_is_out_of_sample"] is False for row in rows))

    def test_inconsistent_per_zone_model_ids_fail_instead_of_assuming_same_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_model_forecast(root, "original-model")
            pd.DataFrame({"zone_id": ["001"], "model_id": ["different-model"]}).to_csv(root / "evaluation" / "forecast_metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting model_id"):
                load_forecast_context(root)

    def test_zero_training_overlap_does_not_establish_temporal_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_model_forecast(root, "saved-model")
            csv = root / "evaluation" / "forecast_details" / "zone_001_forecast_vs_actual.csv"
            frame = pd.read_csv(csv, dtype={"zone_id": str})
            frame["training_period_overlap"] = False
            frame.to_csv(csv, index=False)
            manifest_path = root / "forecaster_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for end, expected in (("2026-01-02", False), ("2026-01-01", True), (None, None)):
                manifest["model_provenance"]["001"]["model_training_end_exclusive"] = end
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(training_end=end):
                    result = evaluate_directory(root, section="forecaster")["forecaster"]["runs"][0]
                    self.assertEqual(result["training_overlap_sample_count"], 0)
                    self.assertIs(result["evaluation_is_out_of_sample"], expected)

    def test_control_pairing_rejects_model_versions_that_conflict_with_saved_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_model_forecast(root, "saved-model")
            context = load_forecast_context(root)
            start, end = "2026-01-01T00:00:00", "2026-01-01T01:00:00"
            context["thresholds"] = {"001": {"load_3h_low_medium_threshold_kwh": 1,
                                               "load_3h_medium_high_threshold_kwh": 10}}
            context["expected_windows"] = {"001": {(start, end)}}
            control = {"model_ids": {"001": "saved-model"}, "zones": [
                {"zone_id": "001", "model_id": "saved-model", "final_windows": [
                    {"window_start": start, "window_end": end, "mean_energy_price": .5,
                     "sum_predicted_kwh": 3, "load_stress_level": "Medium"}]}]}
            self.assertTrue(pairing_metadata(control, context)["pairing_ready"])
            for location in ("top", "zone"):
                changed = deepcopy(control)
                if location == "top":
                    changed["model_ids"]["001"] = "different-model"
                else:
                    changed["zones"][0]["model_id"] = "different-model"
                with self.subTest(location=location):
                    self.assertFalse(pairing_metadata(changed, context)["pairing_ready"])

    def test_old_names_normalize_only_at_offline_boundary(self):
        old = {"forecast_start": "2026-01-01", "forecast_model": "AR",
               "forecast_parameters": {"forecast_model": "AR"},
               "forecast_starts": ["2026-01-01"], "forecast_models": ["AR"]}
        before = deepcopy(old)
        normalized = normalize_evaluation_fields(old)
        self.assertEqual(old, before)
        self.assertEqual(normalized["backend"], "AR")
        self.assertEqual(normalized["forecast_origin"], "2026-01-01")
        self.assertNotIn("forecast_model", normalized["forecast_parameters"])
        self.assertEqual(normalized["forecast_models"], ["AR"])
        frame = normalize_evaluation_columns(pd.DataFrame([old]))
        self.assertIn("backend", frame)
        self.assertNotIn("forecast_model", frame)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_evaluation_fields({"forecast_model": "AR", "backend": "lstm"})

    def test_identity_uses_resolved_backend_parameters(self):
        raw = {"backend": "xgboost", "forecast_origin": "2026-01-01",
               "history_days": 30, "validation_days": 5, "horizon_days": 1,
               "diurnal_blend_alpha": 0.1001, "calibration_method": "daily_hour_mean_v1",
               "xgboost_max_depth": 4, "lstm_hidden_size": 64,
               "forecast_starts": ["2026-01-01"], "forecast_start_count": 2,
               "forecast_models": ["xgboost", "lstm"],
               "diurnal_blend_alphas": [0.1001, 0.1002],
               "xgboost_diurnal_blend_alphas": [0.1001],
               "lstm_diurnal_blend_alphas": [0.5]}
        expected = canonical_forecast_parameters(raw)
        changed_candidates = deepcopy(raw)
        changed_candidates.update(lstm_hidden_size=128, forecast_start_count=9,
                                  forecast_models=["xgboost"], diurnal_blend_alphas=[1.0],
                                  xgboost_diurnal_blend_alphas=[0.1001, 0.9])
        self.assertEqual(expected, canonical_forecast_parameters(changed_candidates))
        for key, value in (("xgboost_max_depth", 6), ("diurnal_blend_alpha", 0.1002),
                           ("calibration_method", "legacy_unknown")):
            self.assertNotEqual(signature(expected), signature(canonical_forecast_parameters({**raw, key: value})))
        legacy = canonical_forecast_parameters({"forecast_model": "AR", "diurnal_blend_alpha": .5,
                                                "ar_diurnal_blend_alpha": 0.0, "lstm_hidden_size": 9})
        self.assertEqual(legacy["diurnal_blend_alpha"], 0.0)
        self.assertEqual(legacy["calibration_method"], "legacy_unknown")
        self.assertNotIn("lstm_hidden_size", legacy)
        self.assertEqual(canonical_forecast_parameters({}), {})

    def test_metrics_supply_resolved_legacy_alpha_for_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evaluation").mkdir()
            (root / "forecaster_manifest.json").write_text(json.dumps({
                "data_source": {"dataset_fingerprint": "fixture"},
                "forecast_parameters": {"forecast_model": "AR", "forecast_start": "2026-01-01"},
            }))
            metrics = root / "evaluation" / "forecast_metrics.csv"
            pd.DataFrame({"forecast_model": ["AR"], "diurnal_blend_alpha": [.1001]}).to_csv(metrics, index=False)
            first = load_forecast_context(root)
            self.assertEqual(first["parameters"]["diurnal_blend_alpha"], .1001)
            self.assertEqual(first["parameters"]["calibration_method"], "legacy_unknown")
            pd.DataFrame({"backend": ["AR"], "diurnal_blend_alpha": [.1002]}).to_csv(metrics, index=False)
            self.assertNotEqual(first["identity"], load_forecast_context(root)["identity"])

    def test_schema_four_and_historical_results_emit_new_fields(self):
        for schema in (2, 3, 4):
            payload = {"schema_version": schema, "status": "fail", "zones": [],
                       "agent_mode": "single_agent_full_retry"}
            if schema == 4:
                payload.update(backend="AR", forecast_origin="2026-01-01")
            else:
                payload.update(forecast_model="AR", forecast_start="2026-01-01")
            result = evaluate_agent(payload)
            self.assertEqual(result["backend"], "AR")
            self.assertEqual(result["forecast_origin"], "2026-01-01")
            self.assertNotIn("forecast_model", result)
            self.assertNotIn("forecast_start", result)

    def test_prompt_excludes_calibration_audit_and_legacy_names(self):
        context = {"backend": "xgboost", "forecast_origin": "2026-01-01",
                   "forecast_model": "old", "forecast_start": "old",
                   "calibration": {"bias_by_hour": [1.0] * 24, "validation_predictions": [
                       {"actual_kwh": 12.0, "residual_kwh": 2.0}]},
                   "validation_predictions": [{"actual_kwh": 12.0}]}
        safe = agent_safe_context(context)
        self.assertEqual(safe, {"backend": "xgboost", "forecast_origin": "2026-01-01"})
        self.assertEqual(compact_economist_context(context), {"forecast_origin": "2026-01-01"})


if __name__ == "__main__":
    unittest.main()
