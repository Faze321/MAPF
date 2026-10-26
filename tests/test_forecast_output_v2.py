"""New result names and offline-only historical provenance normalization."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluate import evaluate_agent
from evaluation_artifacts import (
    canonical_forecast_parameters,
    load_forecast_context,
    normalize_evaluation_columns,
    normalize_evaluation_fields,
    signature,
)
from prompts import agent_safe_context, compact_economist_context


class ForecastOutputMigrationTests(unittest.TestCase):
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
