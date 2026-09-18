import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluate import evaluate_agent, evaluate_directory, evaluate_forecaster, summarize_agents, write_evaluation


def usage(calls=3, complete=True):
    return {"agent_call_count": calls, "prompt_tokens": 20, "completion_tokens": 10,
            "total_tokens": 30, "token_usage_complete": complete}


def window(index, success):
    return {"window_start": str(index), "window_end": str(index + 1), "control_success": success}


def control():
    return {
        "schema_version": 3, "forecast_model": "AR", "agent_mode": "single_agent_full_retry",
        "status": "fail", "attempts_used": 3, "agent_cumulative_usage": usage(),
        "zones": [{
            "zone_id": "001", "status": "fail", "attempts_used": 3,
            "agent_cumulative_usage": usage(),
            "attempt_trace": [
                {"attempt": 2, "control_status": "success", "windows": [window(i, True) for i in range(4)]},
                {"attempt": 1, "control_status": "fail", "windows": [window(i, v) for i, v in enumerate([True, False, True, False])]},
            ],
            "final_windows": [window(i, v) for i, v in enumerate([True, True, False, False])],
        }],
    }


class ForecastEvaluationTests(unittest.TestCase):
    def test_pooled_metrics_not_zone_average(self):
        frame = pd.DataFrame({"actual_kwh": [1, 3, 6], "predicted_kwh": [2, 3, 3]})
        metrics = evaluate_forecaster(frame)
        self.assertAlmostEqual(metrics["MAE"], 4 / 3)
        self.assertAlmostEqual(metrics["RMSE"], math.sqrt(10 / 3))
        self.assertAlmostEqual(metrics["RAE"], 0.75)
        self.assertAlmostEqual(metrics["MAPE_pct"], 50)
        self.assertAlmostEqual(metrics["WAPE_pct"], 40)

    def test_nonfinite_zero_and_constant_actuals(self):
        frame = pd.DataFrame({"actual_kwh": [0, 0, float("inf"), "bad", 2],
                              "predicted_kwh": [1, 0, 1, 1, float("nan")]})
        metrics = evaluate_forecaster(frame)
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["n_excluded"], 3)
        self.assertEqual(metrics["MAE"], 0.5)
        for key in ("RAE", "MAPE_pct", "WAPE_pct"):
            self.assertIsNone(metrics[key])
        json.dumps(metrics, allow_nan=False)
        empty = evaluate_forecaster(frame.iloc[0:0])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["MAE"])


class AgentEvaluationTests(unittest.TestCase):
    def test_transitions_match_window_identity_and_first_proposal(self):
        payload = control()
        payload["zones"][0]["final_windows"].reverse()
        result = evaluate_agent(payload)
        summary = result["summary"]
        for name in ("retained", "gained", "lost", "never"):
            self.assertEqual(summary[f"window_{name}_count"], 1)
        self.assertEqual(summary["window_first_success_count"], 2)
        self.assertEqual(summary["window_final_success_count"], 2)
        self.assertEqual(summary["run_final_success_count"], 0)
        self.assertEqual(summary["mean_calls_per_run"], 3)
        self.assertEqual(summary["total_tokens"], 30)  # Not global + Zone totals.
        self.assertEqual(summary["mean_total_tokens_per_call"], 10)

    def test_incomplete_usage_not_reported_as_full_average(self):
        payload = control()
        payload["agent_cumulative_usage"] = usage(5, False)
        summary = summarize_agents([evaluate_agent(control()), evaluate_agent(payload)])
        self.assertEqual(summary["known_total_tokens"], 60)
        self.assertIsNone(summary["total_tokens"])
        self.assertIsNone(summary["mean_total_tokens_per_call"])
        self.assertEqual(summary["mean_calls_per_run"], 4)
        self.assertEqual(summary["incomplete_usage_run_count"], 1)

    def test_missing_first_round_is_unknown_not_failure(self):
        payload = control()
        payload["zones"][0]["attempt_trace"] = []
        payload.pop("agent_cumulative_usage")
        summary = evaluate_agent(payload)["summary"]
        self.assertIsNone(summary["run_first_success_rate_pct"])
        self.assertEqual(summary["window_unknown_transition_count"], 4)
        self.assertEqual(summary["window_gained_count"], 0)
        self.assertIsNone(summary["mean_calls_per_run"])

    def test_successful_early_stop(self):
        payload = control()
        payload["status"] = "success"
        payload["attempts_used"] = 1
        zone = payload["zones"][0]
        zone.update(status="success", attempts_used=1)
        zone["attempt_trace"] = [{"attempt": 1, "control_status": "success",
                                  "windows": [window(0, True)]}]
        zone["final_windows"] = [window(0, True)]
        summary = evaluate_agent(payload)["summary"]
        self.assertEqual(summary["run_retained_count"], 1)
        self.assertEqual(summary["mean_attempts_per_run"], 1)

    def test_analyzer_is_optional_and_receives_independent_payload(self):
        payload = control()
        original = copy.deepcopy(payload)

        def analyzer(request):
            request["control_result"]["zones"].clear()
            request["quantitative_analysis"]["zones"].clear()
            return {"assessment": "example"}

        result = evaluate_agent(payload, analyzer=analyzer)
        self.assertEqual(payload, original)
        self.assertEqual(len(result["zones"]), 1)
        self.assertEqual(result["llm_analysis"]["status"], "completed")
        self.assertEqual(evaluate_agent(payload)["llm_analysis"]["status"], "not_configured")

    def test_duplicate_window_rejected_and_schema_two_supported(self):
        payload = control()
        payload["schema_version"] = 2
        evaluate_agent(payload)
        payload["zones"][0]["final_windows"].append(window(0, True))
        with self.assertRaises(ValueError):
            evaluate_agent(payload)


class DirectoryEvaluationTests(unittest.TestCase):
    def test_recursive_discovery_deduplicates_forecast_copies_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("mode_a", "mode_b"):
                forecaster = root / mode / "forecaster"
                details = forecaster / "evaluation" / "forecast_details"
                details.mkdir(parents=True)
                pd.DataFrame({"zone_id": ["001", "001"], "time": ["t1", "t2"],
                              "actual_kwh": [1, 3], "predicted_kwh": [2, 3]}).to_csv(
                    details / "zone_001_forecast_vs_actual.csv", index=False)
                pd.DataFrame({"forecast_model": ["AR"], "diurnal_blend_alpha": [0.7]}).to_csv(
                    details.parent / "forecast_metrics.csv", index=False)
                (forecaster / "forecaster_manifest.json").write_text(json.dumps({
                    "data_source": {"dataset_fingerprint": "abc"}, "zone_ids": ["001"],
                    "forecast_parameters": {"forecast_model": "AR", "forecast_start": "t1"},
                }))
                (root / mode / "control_results.json").write_text(json.dumps(control()))
            report = evaluate_directory(root)
            self.assertEqual(len(report["forecaster"]["runs"]), 2)
            pooled = report["forecaster"]["global_by_model"][0]
            self.assertEqual(pooled["n"], 2)
            self.assertEqual(pooled["duplicate_sample_count"], 2)
            self.assertEqual(report["agent"]["overall"]["run_count"], 2)
            output = write_evaluation(report, root / "analysis")
            json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(list(output.parent.glob("*.csv"))), 4)
            self.assertEqual(evaluate_directory(root), report)  # Outputs are not rediscovered.
            self.assertFalse(evaluate_directory(root, section="agent")["forecaster"]["runs"])

    def test_empty_directory_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                evaluate_directory(temporary)


if __name__ == "__main__":
    unittest.main()
