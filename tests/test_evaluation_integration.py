"""Offline end-to-end contracts for the reusable analysis entry points."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from evaluate import evaluate_agent, evaluate_directory, evaluate_experiments, write_evaluation


def saved_run(root, mode="single", price=3, cache=True, forecast_delta=0):
    run = root / "arbitrary" / mode / "AR"
    forecast = run / "forecaster"
    details = forecast / "evaluation" / "forecast_details"
    details.mkdir(parents=True)
    times = pd.date_range("2026-01-01", periods=6, freq="h")
    pd.DataFrame({"zone_id": ["001"] * 6, "time": times,
                  "actual_kwh": [10, 10, 10, 20, 20, 20],
                  "predicted_kwh": [10 + forecast_delta] * 3 + [20] * 3}).to_csv(
        details / "zone_001_forecast_vs_actual.csv", index=False)
    pd.DataFrame({"forecast_model": ["AR"], "diurnal_blend_alpha": [.5]}).to_csv(
        details.parent / "forecast_metrics.csv", index=False)
    features = {"adapter": "mp_evdata", "price_unit": "CNY/kWh"}
    manifest = {"data_source": {"adapter": "mp_evdata", "dataset_fingerprint": "fixture"},
                "feature_manifest": features, "zone_ids": ["001"],
                "forecast_parameters": {"forecast_start": "2026-01-01 00:00:00", "forecast_model": "AR", "horizon_days": .25}}
    (forecast / "forecaster_manifest.json").write_text(json.dumps(manifest))
    windows = [{"window_start": str(times[i * 3]), "window_end": str(times[i * 3 + 2]),
                "mean_energy_price": 2, "sum_predicted_kwh": 30 * (i + 1),
                "load_stress_level": "Low" if i == 0 else "Medium", "proposed_energy_price": price,
                "price_conditioned_baseline_load_kwh": 40 if i == 0 else 60,
                "price_conditioned_load_stress_level": "Medium", "control_success": True, "price_valid": True}
               for i in range(2)]
    snippets = [{"zone_id": "001", "load_3h_low_medium_threshold_kwh": 35,
                 "load_3h_medium_high_threshold_kwh": 80, "pricing_windows_3h": windows}]
    (forecast / "context_snippets.json").write_text(json.dumps(snippets))
    calls = []
    rounds = []
    for r, prompt, completion in ((1, 10, 2), (2, 20, 3)):
        u = {"agent_call_count": 1, "prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion, "token_usage_complete": True}
        if cache:
            u["cached_tokens"] = prompt // 2
        rounds.append({"attempt": r, **u})
        calls.append({"attempt": r, "zone_id": "001", "stage": "single" if r == 1 else "schema_repair",
                      "agent": "single", "provider_attempt_count": 1, **u})
    total = {"agent_call_count": 2, "prompt_tokens": 30, "completion_tokens": 5,
             "total_tokens": 35, "token_usage_complete": True}
    if cache:
        total["cached_tokens"] = 15
    control = {"schema_version": 3, "forecast_model": "AR", "agent_mode": mode,
               "forecast_origin": "2026-01-01T00:00:00", "dataset_fingerprint": "fixture",
               "feature_manifest": features, "attempt_limit": 2, "attempts_used": 2, "status": "success",
               "agent_cumulative_usage": total, "agent_round_usage": rounds, "agent_step_token_usage": calls,
               "zones": [{"zone_id": "001", "status": "success", "attempts_used": 2,
                          "agent_cumulative_usage": total, "final_windows": windows,
                          "attempt_trace": [{"attempt": r, "control_status": "success", "windows": copy.deepcopy(windows),
                                             "agent_usage": rounds[r-1], "agent_call_usage": [calls[r-1]]}
                                            for r in (1, 2)]}]}
    agent = run / "agent"
    agent.mkdir()
    path = agent / "control_results.json"
    path.write_text(json.dumps(control))
    return path


def add_hourly_prices(root, prices=(1, 2, 3, 3, 2, 1)):
    """Keep window price means/load totals fixed while hourly products differ."""
    path = next(root.rglob("zone_*_forecast_vs_actual.csv"))
    frame = pd.read_csv(path, dtype={"zone_id": str})
    frame["predicted_kwh"] = [5, 10, 15, 10, 20, 30]
    frame["e_price"] = prices
    frame.to_csv(path, index=False)
    return frame


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_tables_report_and_json_agree_and_sources_remain_unchanged(self):
        path = saved_run(self.root)
        before = path.read_bytes()
        report = evaluate_directory(self.root)
        self.assertEqual(len(report["comparison"]["common_run_ids"]), 1)
        revenue = next(r for r in report["tables"]["revenue_by_dataset_model_mode"] if r["phase"] == "final")
        self.assertEqual(revenue["baseline_revenue"], 180)
        self.assertEqual(revenue["revenue"], 300)
        self.assertEqual(revenue["revenue_change"], 120)
        usage = report["tables"]["usage_by_mode"][0]
        self.assertEqual(usage["cached_tokens"], 15)
        self.assertEqual(usage["cache_hit_ratio"], .5)
        self.assertEqual(usage["mean_total_tokens_per_call"], 17.5)
        self.assertEqual(usage["median_total_tokens_per_call"], 17.5)
        target = write_evaluation(report, self.root / "analysis")
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), report)
        frame = pd.read_csv(target.parent / "revenue_by_dataset_model_mode.csv")
        self.assertEqual(frame.loc[frame.phase == "final", "revenue"].iloc[0], 300)
        markdown = (target.parent / "evaluation_report_zh.md").read_text(encoding="utf-8")
        self.assertIn("300", markdown)
        self.assertIn("cached_tokens", markdown)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(evaluate_directory(self.root), report)

    def test_two_revenue_baselines_rounds_final_and_export_agree(self):
        path = saved_run(self.root)
        frame = add_hourly_prices(self.root)
        direct = evaluate_agent(json.loads(path.read_text()), baseline_hourly=frame)
        self.assertEqual(direct["outcome_summary"]["baseline_hourly_revenue"], 170)
        payloads = []
        report = evaluate_directory(self.root, section="agent", analyzer=lambda p: payloads.append(p) or {})
        self.assertEqual(payloads[0]["quantitative_analysis"]["outcome_summary"]["baseline_hourly_revenue"], 170)
        rows = report["tables"]["revenue_by_dataset_model_mode"]
        for row in rows:
            self.assertEqual(row["baseline_revenue"], 180)
            self.assertEqual(row["baseline_hourly_revenue"], 170)
            self.assertTrue(row["baseline_hourly_revenue_complete"])
            self.assertEqual(row["baseline_hourly_revenue_coverage_pct"], 100)
            expected = 180 if row["phase"] == "baseline" else 300
            self.assertEqual(row["revenue"], expected)
            self.assertEqual(row["revenue_change"], expected - 180)
            self.assertEqual(row["revenue_change_vs_hourly"], expected - 170)
            self.assertAlmostEqual(row["revenue_change_vs_hourly_pct"], 100 * (expected - 170) / 170)
        target = write_evaluation(report, self.root / "analysis")
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), report)
        csv = pd.read_csv(target.parent / "revenue_by_dataset_model_mode.csv")
        final = csv[csv.phase == "final"].iloc[0]
        self.assertEqual(final.baseline_hourly_revenue, 170)
        self.assertEqual(final.revenue_change_vs_hourly, 130)
        markdown = (target.parent / "evaluation_report_zh.md").read_text(encoding="utf-8")
        self.assertIn("baseline_hourly_revenue", markdown)
        self.assertIn("revenue_change_vs_hourly", markdown)
        self.assertIn("| 180 | 170 | 300 | 120 | 66.6667 | 130 | 76.4706 |", markdown)

    def test_missing_hourly_price_preserves_original_baseline(self):
        saved_run(self.root)
        report = evaluate_directory(self.root)
        row = next(r for r in report["tables"]["revenue_by_dataset_model_mode"] if r["phase"] == "final")
        self.assertEqual(row["baseline_revenue"], 180)
        self.assertEqual(row["revenue_change"], 120)
        self.assertIsNone(row["baseline_hourly_revenue"])
        self.assertIsNone(row["known_baseline_hourly_revenue"])
        self.assertIsNone(row["revenue_change_vs_hourly"])
        self.assertEqual(row["baseline_hourly_revenue_coverage_pct"], 0)

    def test_partial_hourly_baseline_exports_coverage_not_full_revenue(self):
        saved_run(self.root)
        add_hourly_prices(self.root, prices=(None, 2, 3, 3, 2, 1))
        report = evaluate_directory(self.root)
        row = next(r for r in report["tables"]["revenue_by_dataset_model_mode"] if r["phase"] == "final")
        self.assertIsNone(row["baseline_hourly_revenue"])
        self.assertEqual(row["known_baseline_hourly_revenue"], 100)
        self.assertEqual(row["baseline_hourly_revenue_coverage_pct"], 50)
        self.assertEqual(row["baseline_hourly_revenue_known_count"], 1)
        self.assertIsNone(row["revenue_change_vs_hourly"])
        self.assertEqual(row["known_revenue_change_vs_hourly"], 80)
        target = write_evaluation(report, self.root / "analysis")
        markdown = (target.parent / "evaluation_report_zh.md").read_text(encoding="utf-8")
        self.assertIn("收益数据覆盖", markdown)
        self.assertIn("known_baseline_hourly_revenue", markdown)

    def test_hourly_price_profile_is_required_to_match_between_batches(self):
        left, right = self.root / "left", self.root / "right"
        saved_run(left, price=3)
        saved_run(right, price=4)
        add_hourly_prices(left)
        add_hourly_prices(right)
        report = evaluate_experiments({"27B": left, "9B": right})
        pair = report["comparison"]["paired_summary"][0]
        self.assertEqual(pair["left"]["baseline_hourly_revenue"], 170)
        self.assertEqual(pair["right"]["baseline_hourly_revenue"], 170)
        self.assertEqual(pair["delta"]["revenue_change_vs_hourly"], 100)
        # Same window mean, different hourly energy price profile and baseline.
        add_hourly_prices(right, prices=(3, 2, 1, 3, 2, 1))
        report = evaluate_experiments({"27B": left, "9B": right})
        self.assertFalse(report["comparison"]["pairs"])
        self.assertTrue(any("hourly_signature" in r.get("mismatched_fields", []) for r in report["comparison"]["audit"]))

    def test_multi_batch_pairing_and_forecasts_copied_between_modes(self):
        left, right = self.root / "left", self.root / "right"
        saved_run(left, "single", 3)
        saved_run(left, "multi", 3)
        saved_run(right, "single", 4)
        report = evaluate_experiments({"27B": left, "9B": right})
        self.assertEqual(len(report["comparison"]["pairs"]), 1)
        self.assertEqual(len(report["comparison"]["common_run_ids"]), 3)
        pair = report["comparison"]["paired_summary"][0]
        self.assertEqual(pair["delta"]["revenue"], 100)
        self.assertEqual(pair["delta"]["total_tokens"], 0)
        self.assertEqual([r["n"] for r in report["tables"]["ppt_forecast"]], [6, 6])
        self.assertTrue(any(r["status"] == "missing" for r in report["comparison"]["audit"]))

    def test_inconsistent_hourly_baseline_is_not_paired(self):
        left, right = self.root / "left", self.root / "right"
        saved_run(left)
        saved_run(right, forecast_delta=1)
        report = evaluate_experiments({"a": left, "b": right})
        self.assertFalse(report["comparison"]["pairs"])
        self.assertTrue(any("hourly_signature" in r.get("mismatched_fields", []) for r in report["comparison"]["audit"]))

    def test_missing_final_window_or_threshold_rejects_fair_cohort(self):
        path = saved_run(self.root)
        raw = json.loads(path.read_text())
        raw["zones"][0]["final_windows"].pop()
        path.write_text(json.dumps(raw))
        self.assertFalse(evaluate_directory(self.root)["comparison"]["common_run_ids"])

    def test_nonfinite_baseline_unknown_final_and_truncated_hourly_reject_pairing(self):
        path = saved_run(self.root)
        original = json.loads(path.read_text())
        for change in ("baseline", "final"):
            raw = copy.deepcopy(original)
            for window in raw["zones"][0]["final_windows"]:
                if change == "baseline":
                    window["mean_energy_price"] = float("nan")
                else:
                    window.pop("price_conditioned_load_stress_level")
            path.write_text(json.dumps(raw))
            with self.subTest(change=change):
                self.assertFalse(evaluate_directory(self.root)["comparison"]["common_run_ids"])
        path.write_text(json.dumps(original))
        hourly = next(self.root.rglob("zone_*_forecast_vs_actual.csv"))
        pd.read_csv(hourly).iloc[:1].to_csv(hourly, index=False)
        self.assertFalse(evaluate_directory(self.root)["comparison"]["common_run_ids"])

    def test_nested_inputs_cannot_compare_a_result_with_itself(self):
        saved_run(self.root)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_experiments({"parent": self.root, "child": self.root / "arbitrary"})

    def test_changing_section_removes_only_obsolete_evaluator_tables(self):
        saved_run(self.root)
        output = self.root / "analysis"
        write_evaluation(evaluate_directory(self.root), output)
        (output / "user_notes.csv").write_text("keep")
        self.assertTrue((output / "ppt_forecast.csv").exists())
        write_evaluation(evaluate_directory(self.root, section="agent"), output)
        self.assertFalse((output / "ppt_forecast.csv").exists())
        self.assertEqual((output / "user_notes.csv").read_text(), "keep")

    def test_legacy_cache_is_unknown_in_outputs(self):
        saved_run(self.root, cache=False)
        report = evaluate_directory(self.root)
        usage = report["tables"]["usage_by_mode"][0]
        self.assertIsNone(usage["cached_tokens"])
        self.assertFalse(usage["cache_usage_complete"])
        self.assertEqual(usage["cache_reported_call_count"], 0)
        self.assertIsNone(report["tables"]["ppt_tokens"][0]["cached_tokens"])

    def test_section_and_analyzer_keep_old_contract(self):
        saved_run(self.root)
        calls = []
        report = evaluate_directory(self.root, section="agent", analyzer=lambda payload: calls.append(payload) or {"ok": True})
        self.assertEqual(len(calls), 1)
        self.assertFalse(report["forecaster"]["runs"])
        forecast = evaluate_directory(self.root, section="forecaster")
        self.assertFalse(forecast["agent"]["runs"])
        self.assertEqual(forecast["tables"]["ppt_forecast"][0]["n"], 6)

    def test_batch_cli_requires_explicit_output(self):
        result = subprocess.run([sys.executable, "-B", "evaluate.py", "--batch", f"label={self.root}"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--batch requires --output", result.stderr)


if __name__ == "__main__":
    unittest.main()
