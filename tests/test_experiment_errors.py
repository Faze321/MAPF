"""Experiment exceptions count per dataset up to a configurable failure limit."""
from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agents import AgentStageError
from config import RunConfig
from orchestrator import run_experiment_matrix


class ExperimentErrorConfigTests(unittest.TestCase):
    def test_default_and_custom_limits(self):
        self.assertEqual(RunConfig().max_dataset_errors, 3)
        self.assertEqual(RunConfig.from_mapping({}).max_dataset_errors, 3)
        for limit in (1, 3, 5):
            with self.subTest(limit=limit):
                self.assertEqual(RunConfig.from_mapping({"max_dataset_errors": limit}).max_dataset_errors, limit)

    def test_invalid_limits_are_rejected(self):
        for limit in (0, -1, True, False, 1.5, "3", None, "", [], {}):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "run.max_dataset_errors"):
                RunConfig.from_mapping({"max_dataset_errors": limit})


class ExperimentErrorTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        temporary_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="experiment_errors_", dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Qualification is tested separately; these fixtures start at the matrix's
        # experiment boundary with two eligible Zones and already prepared weights.
        for patcher in (
            patch("orchestrator.load_canonical_dataset", return_value=object()),
            patch("orchestrator.eligible_training_zone_ids", return_value=["1029", "741"]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_matrix(self, dataset="UrbanEV", **overrides):
        arguments = {
            "data_dir": self.root / "data" / dataset,
            "output_dir": self.root / "output" / dataset,
            "forecast_starts": ["2023-01-01"],
            "forecast_models": ["lstm"],
            # These tests isolate experiment error accounting after offline preparation.
            "zone_model_references": {},
            "zone_ids": ["1029", "741"],
            "agent_modes": ["multi_agent_economist_retry", "single_agent_price_retry"],
            "diurnal_blend_alphas": [0.0, 0.5],
        }
        arguments.update(overrides)
        return run_experiment_matrix(**arguments)

    def read_records(self, dataset="UrbanEV"):
        return pd.read_csv(self.root / "output" / dataset / "experiment_runs.csv")

    def test_two_errors_are_recorded_and_later_combinations_and_summaries_complete(self):
        forecast_path = self.root / "metrics.csv"
        pd.DataFrame([{"zone_id": "741", "MAE": 1.5}]).to_csv(forecast_path, index=False)
        success = {"forecast_metrics_csv": forecast_path}
        repair_error = AgentStageError(
            stage="agent.economist_repair",
            zone_id="1029",
            agent="Market Economist Repair",
            original=TimeoutError("Request timed out."),
        )
        log = io.StringIO()
        with patch("orchestrator.run_pipeline", side_effect=[
            repair_error, success, ValueError("invalid result"), success,
        ]) as pipeline, redirect_stderr(log):
            outputs = self.run_matrix()

        self.assertEqual(pipeline.call_count, 4)
        records = self.read_records()
        self.assertEqual(records["status"].tolist(), ["failed", "success", "failed", "success"])
        self.assertEqual(records["dataset_error_count"].tolist(), [1, 1, 2, 2])
        self.assertEqual(records.loc[0, "error_code"], "MAPF-E231")
        self.assertEqual(records.loc[0, "error_stage"], "agent.economist_repair")
        self.assertEqual(int(records.loc[0, "error_zone_id"]), 1029)
        self.assertEqual(records.loc[0, "error_agent"], "Market Economist Repair")
        self.assertEqual(records.loc[0, "error_type"], "TimeoutError")
        self.assertEqual(records.loc[0, "error_message"], "Request timed out.")
        self.assertTrue(records["completed_at"].notna().all())
        for index, count in ((0, 1), (2, 2)):
            trace = Path(records.loc[index, "traceback_file"]).read_text(encoding="utf-8")
            self.assertIn("TRACEBACK:", trace)
            self.assertIn(f"dataset_error_count: {count}", trace)
        self.assertIn("(1/3)", log.getvalue())
        self.assertIn("(2/3)", log.getvalue())
        self.assertEqual(log.getvalue().count("Skipping this combination"), 2)
        metrics = pd.read_csv(outputs["experiment_forecast_metrics_csv"])
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics["diurnal_blend_alpha"].tolist(), [0.5, 0.5])
        self.assertTrue(outputs["experiment_forecast_summary_csv"].is_file())

    def test_third_nonconsecutive_error_stops_all_remaining_matrix_dimensions(self):
        errors = [RuntimeError(f"failure {index}") for index in range(1, 4)]
        outcomes = [{} for _ in range(32)]
        for position, error in zip((0, 17, 25), errors):
            outcomes[position] = error
        log = io.StringIO()
        with patch("orchestrator.run_pipeline", side_effect=outcomes) as pipeline, \
                redirect_stderr(log), self.assertRaises(RuntimeError) as raised:
            self.run_matrix(
                forecast_starts=["2023-01-01", "2023-01-02"],
                forecast_models=["lstm", "AR"],
                experiment_seeds=[41, 42],
            )

        self.assertIs(raised.exception, errors[-1])
        self.assertEqual(pipeline.call_count, 26)
        records = self.read_records()
        self.assertEqual(len(records), 26)
        self.assertEqual(records["dataset_error_count"].tolist(), [1] * 17 + [2] * 8 + [3])
        self.assertEqual((records["status"] == "failed").sum(), 3)
        self.assertEqual(records.iloc[-1]["status"], "failed")
        self.assertEqual(records.iloc[-1]["error_message"], "failure 3")
        self.assertTrue(Path(records.iloc[-1]["traceback_file"]).is_file())
        self.assertEqual(len(list((self.root / "output" / "UrbanEV" / "errors").glob("*.txt"))), 3)
        self.assertIn("(3/3)", log.getvalue())
        self.assertIn("Stopping this dataset: error limit reached.", log.getvalue())

    def test_consecutive_errors_stop_at_three_and_do_not_leak_to_another_dataset(self):
        outcomes = [TimeoutError("timeout") for _ in range(3)] + [{}, {}, {}, {}]
        with patch("orchestrator.run_pipeline", side_effect=outcomes) as pipeline, \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(TimeoutError):
                self.run_matrix()
            self.assertEqual(pipeline.call_count, 3)
            self.run_matrix(dataset="MP-EVData")

        self.assertEqual(pipeline.call_count, 7)
        self.assertEqual(self.read_records()["dataset_error_count"].tolist(), [1, 2, 3])
        other_records = self.read_records("MP-EVData")
        self.assertEqual(other_records["status"].tolist(), ["success"] * 4)
        self.assertEqual(other_records["dataset_error_count"].tolist(), [0] * 4)

    def test_all_failed_combinations_below_limit_still_finish_the_matrix(self):
        with patch("orchestrator.run_pipeline", side_effect=RuntimeError("unavailable")) as pipeline, \
                redirect_stderr(io.StringIO()):
            outputs = self.run_matrix(diurnal_blend_alphas=[0.5])

        self.assertEqual(pipeline.call_count, 2)
        records = pd.read_csv(outputs["experiment_runs_csv"])
        self.assertEqual(records["status"].tolist(), ["failed", "failed"])
        self.assertEqual(records["dataset_error_count"].tolist(), [1, 2])

    def test_keyboard_interrupt_is_not_skipped_or_counted(self):
        with patch("orchestrator.run_pipeline", side_effect=KeyboardInterrupt) as pipeline, \
                redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.run_matrix()

        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(self.read_records()["dataset_error_count"].tolist(), [0])

    def test_custom_limit_controls_when_the_dataset_stops(self):
        for limit in (1, 4):
            log = io.StringIO()
            with self.subTest(limit=limit), \
                    patch("orchestrator.run_pipeline", side_effect=RuntimeError("unavailable")) as pipeline, \
                    redirect_stderr(log), self.assertRaises(RuntimeError):
                self.run_matrix(
                    dataset=f"limit_{limit}", max_dataset_errors=limit,
                    diurnal_blend_alphas=[0.0, 0.5, 1.0],
                )
            self.assertEqual(pipeline.call_count, limit)
            records = self.read_records(f"limit_{limit}")
            self.assertEqual(records["dataset_error_count"].tolist(), list(range(1, limit + 1)))
            self.assertIn(f"({limit}/{limit})", log.getvalue())
            self.assertEqual(log.getvalue().count("Skipping this combination"), limit - 1)
            self.assertIn("Stopping this dataset: error limit reached.", log.getvalue())

    def test_invalid_matrix_limit_fails_before_any_pipeline_runs(self):
        for limit in (0, -1, True, None, "3", 1.5):
            with self.subTest(limit=limit), patch("orchestrator.run_pipeline") as pipeline, \
                    self.assertRaisesRegex(ValueError, "run.max_dataset_errors"):
                self.run_matrix(max_dataset_errors=limit)
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
