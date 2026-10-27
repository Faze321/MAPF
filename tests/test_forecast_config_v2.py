"""Resolved forecast configuration and independent blend-matrix contracts."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from config import (
    BACKEND_PARAMETER_NAMES, BACKEND_BLEND_PARAMETER_NAMES, RunConfig,
    normalize_float_list, resolve_backend_blend_alphas,
)
from dataset_adapter import DatasetSpec
from main import build_parser, main
from orchestrator import (
    forecast_output_dir, forecast_runtime_parameters, make_forecast_parameters,
    run_experiment_matrix,
)


class ForecastConfigV2Tests(unittest.TestCase):
    def test_defaults_and_retired_settings(self):
        run = RunConfig.from_mapping({})
        self.assertEqual((run.history_days, run.validation_days, run.horizon_days), (30, 5, 1))
        self.assertNotIn("forecast_start", asdict(run))
        self.assertNotIn("forecast_model", asdict(run))
        for name in ("forecast_start", "forecast_model", "diurnal_blend_alpha",
                     "lstm_step_horizon", "lstm_roll_actuals", "lstm_diurnal_blend_alpha",
                     "roll_actuals", "step_horizon"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "removed"):
                RunConfig.from_mapping({name: None})
        self.assertFalse(set(BACKEND_BLEND_PARAMETER_NAMES).intersection(BACKEND_PARAMETER_NAMES))

    def test_alpha_lists_validate_normalize_and_override(self):
        self.assertEqual(normalize_float_list("0.2,0.1,0.2000001"), [0.2, 0.1])
        self.assertEqual(normalize_float_list(0.5), [0.5])
        for value in ([], "", float("nan"), float("inf"), -0.1, 1.1, True, [False]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_float_list(value)
        self.assertEqual(resolve_backend_blend_alphas("lstm", [0.1, 0.2], {"lstm_diurnal_blend_alphas": [0.6]}), [0.6])
        self.assertEqual(resolve_backend_blend_alphas("AR", [0.1, 0.2], {"ar_diurnal_blend_alphas": None}), [0.1, 0.2])
        self.assertEqual(resolve_backend_blend_alphas("xgboost", None, {}), [0.0])

    def test_parser_does_not_accept_retired_flags_as_abbreviations(self):
        parser = build_parser()
        for option in ("--forecast-model", "--forecast-start", "--diurnal-blend-alpha"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([option, "0"])
        args = parser.parse_args(["--forecast-models", "lstm", "AR", "--lstm-diurnal-blend-alphas", "0.2", "0.8"])
        self.assertEqual(args.lstm_diurnal_blend_alphas, ["0.2", "0.8"])

    def test_provenance_contains_active_backend_only(self):
        values = {"forecast_origin": "2024-02-01", "backend": "lstm", "history_days": 30,
                  "validation_days": 5, "horizon_days": 1, "diurnal_blend_alpha": 0.3,
                  "model_ids": {"A": "saved-lstm-model"}}
        parameters = make_forecast_parameters(**values)
        self.assertEqual(parameters["model_ids"], {"A": "saved-lstm-model"})
        self.assertEqual(parameters["calibration_method"], "daily_hour_mean_v1")
        self.assertNotIn("xgboost_seed", parameters)
        self.assertNotIn("forecast_start", parameters)
        with self.assertRaisesRegex(ValueError, "scalar"):
            make_forecast_parameters(**{**values, "diurnal_blend_alpha": [0.2, 0.4]})
        with self.assertRaisesRegex(ValueError, "calibration"):
            make_forecast_parameters(**{**values, "calibration_method": "first_day_only"})
        expanded = forecast_runtime_parameters(parameters)
        self.assertNotIn("xgboost_seed", expanded)
        self.assertNotIn("calibration_method", expanded)
        self.assertNotEqual(forecast_output_dir(Path("root"), "lstm", 0.1001),
                            forecast_output_dir(Path("root"), "lstm", 0.1002))
        self.assertIn("blend_0", forecast_output_dir(Path("root"), "AR").parts)


class ForecastMatrixV2Tests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="forecast_config_v2_", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for patcher in (patch("orchestrator.load_canonical_dataset", return_value=object()),
                        patch("orchestrator.eligible_training_zone_ids", return_value=["A"]),
                        patch("orchestrator.prepare_offline_models", return_value={"lstm": {"A": {"model_id": "fixture"}}})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_matrix_sums_per_model_alpha_counts_and_reuses_same_paths(self):
        common = dict(data_dir=self.root / "data", output_dir=self.root / "output",
                      forecast_starts=["2024-02-01", "2024-02-02"], forecast_models=["lstm", "AR"],
                      zone_ids=["A"], experiment_seeds=[11, 12],
                      agent_modes=["multi_agent_full_retry", "single_agent_full_retry"],
                      diurnal_blend_alphas=[0.1, 0.2], lstm_diurnal_blend_alphas=[0.7])
        created = set()
        def save(**kwargs):
            folder = forecast_output_dir(kwargs["output_dir"], kwargs["backend"], kwargs["diurnal_blend_alpha"]) / "forecaster"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "forecaster_manifest.json").write_text("{}", encoding="utf-8")
            created.add(folder)
            return {}
        with patch("orchestrator.run_pipeline", side_effect=save) as run:
            output = run_experiment_matrix(pipeline_stage="forecaster", **common)
        self.assertEqual(run.call_count, 2 * 2 * (1 + 2))
        self.assertTrue(all(call.kwargs["diurnal_blend_alpha"] == 0.7
                            for call in run.call_args_list if call.kwargs["backend"] == "lstm"))
        records = pd.read_csv(output["experiment_runs_csv"])
        self.assertIn("backend", records)
        self.assertIn("forecast_origin", records)
        self.assertNotIn("forecast_model", records)
        with patch("orchestrator.run_pipeline", return_value={}) as replay:
            run_experiment_matrix(pipeline_stage="agent", **common)
        self.assertEqual(replay.call_count, 2 * 2 * 2 * (1 + 2))
        self.assertEqual({call.kwargs["forecaster_output_dir"] for call in replay.call_args_list}, created)
        self.assertTrue(all("blend_" in str(call.kwargs["agent_output_dir"]) for call in replay.call_args_list))
        created.clear()
        with patch("orchestrator.run_pipeline", side_effect=save) as full:
            run_experiment_matrix(pipeline_stage="full", **common)
        self.assertEqual(full.call_count, 24)
        with patch("orchestrator.run_pipeline", return_value={}) as replay:
            run_experiment_matrix(pipeline_stage="agent", **common)
        self.assertEqual({call.kwargs["forecaster_output_dir"] for call in replay.call_args_list}, created)

    def test_cli_independent_alphas_override_global_and_trigger_matrix(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump({"agent": {}, "run": {
            "data_dir": str(self.root), "output_folder": str(self.root / "output"),
            "forecast_models": ["lstm"], "forecast_starts": ["2024-02-01"],
            "diurnal_blend_alphas": [0.1, 0.2], "lstm_diurnal_blend_alphas": [0.5],
        }}), encoding="utf-8")
        spec = DatasetSpec(self.root, adapter="long_format")
        with patch("main.resolve_dataset_spec", return_value=spec), \
                patch("main.run_pipeline", return_value={}) as run, redirect_stdout(io.StringIO()):
            main(["--config", str(config_path), "--diurnal-blend-alphas", "0.8", "0.9"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["diurnal_blend_alpha"], 0.5)
        self.assertEqual(run.call_args.kwargs["forecast_origin"], "2024-02-01")
        with patch("main.resolve_dataset_spec", return_value=spec), \
                patch("main.run_experiment_matrix", return_value={}) as matrix, redirect_stdout(io.StringIO()):
            main(["--config", str(config_path), "--lstm-diurnal-blend-alphas", "0.3", "0.7"])
        self.assertEqual(matrix.call_args.kwargs["lstm_diurnal_blend_alphas"], [0.3, 0.7])


if __name__ == "__main__":
    unittest.main()
