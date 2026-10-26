from __future__ import annotations

from copy import deepcopy
import unittest

from evaluation_comparison import build_comparisons


def run(batch="27B", mode="single", *, origin="2026-01-01", suffix="", dataset="UrbanEV", **overrides):
    revenue = 60
    result = {
        "batch": batch, "dataset": dataset, "dataset_identity": dataset,
        "backend": "lstm", "agent_mode": mode, "forecast_origin": origin,
        "run_id": f"{batch}/{dataset}/{origin}/{mode}{suffix}",
        "forecast_parameters": {"horizon": 24, "alpha": 0.5}, "zone_ids": ["A", "B"],
        "baseline_signature": "baseline-1", "hourly_signature": "hourly-1", "pairing_ready": True,
        "revenue_unit": "CNY",
        "outcome_summary": {
            "window_count": 2, "medium_count": 1, "medium_known_count": 2, "medium_unknown_count": 0,
            "zone_count": 2, "zone_complete_count": 2, "zone_unknown_count": 0, "zone_all_medium_count": 1,
            "run_count": 1, "run_complete_count": 1, "run_unknown_count": 0, "run_all_medium_count": 0,
            "revenue": revenue, "known_revenue": revenue, "revenue_complete": True,
            "revenue_known_count": 2, "revenue_unknown_count": 0,
            "baseline_revenue": 50, "known_baseline_revenue": 50, "baseline_revenue_complete": True,
            "baseline_revenue_known_count": 2, "baseline_revenue_unknown_count": 0,
        },
        "usage": {
            "agent_call_count": 2, "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "token_usage_complete": True, "cached_tokens": 60, "known_cached_tokens": 60,
            "cache_reported_call_count": 2, "cache_usage_complete": True, "cache_hit_ratio": 0.6,
        },
    }
    result.update(overrides)
    return result


def hourly_baseline(record, baseline, *, known=None, known_count=2, comparable_count=2, known_change=None):
    complete = baseline is not None
    revenue = record["outcome_summary"]["revenue"]
    record["outcome_summary"].update({
        "baseline_hourly_revenue": baseline,
        "known_baseline_hourly_revenue": baseline if complete else known,
        "baseline_hourly_revenue_complete": complete,
        "baseline_hourly_revenue_known_count": known_count,
        "baseline_hourly_revenue_unknown_count": 2 - known_count,
        "baseline_hourly_revenue_coverage_pct": known_count / 2 * 100,
        "revenue_hourly_comparable_count": comparable_count,
        "known_revenue_change_vs_hourly": revenue - baseline if complete else known_change,
        "revenue_change_vs_hourly": revenue - baseline if complete else None,
        "revenue_change_vs_hourly_pct": (revenue - baseline) / baseline * 100 if complete and baseline else None,
    })
    return record


class EvaluationComparisonTests(unittest.TestCase):
    def test_common_cohort_uses_all_observed_modes_and_preserves_inputs(self):
        runs = [run(mode=mode) for mode in ("single", "multi", "debate")]
        runs += [run(mode=mode, origin="2026-02-01") for mode in ("single", "multi")]
        before = deepcopy(runs)
        result = build_comparisons(runs)
        self.assertEqual(result["common_run_ids"], sorted(row["run_id"] for row in runs[:3]))
        self.assertEqual(result["audit"][0]["missing_modes"], ["debate"])
        self.assertEqual(before, runs)

    def test_parameters_and_baselines_and_hourly_samples_must_match(self):
        for field, value in (
            ("forecast_parameters", {"horizon": 48}),
            ("baseline_signature", "different-price-or-threshold"),
            ("hourly_signature", "different-hourly-actual"),
            ("revenue_unit", "EUR"),
        ):
            with self.subTest(field=field):
                result = build_comparisons([run(mode="single"), run(mode="multi", **{field: value})])
                self.assertFalse(result["common_run_ids"])
                self.assertEqual(result["audit"][0]["status"], "inconsistent")
                self.assertIn(field, result["audit"][0]["mismatched_fields"])

    def test_missing_evidence_cannot_match_even_if_both_missing(self):
        for field, value in (("baseline_signature", None), ("hourly_signature", ""), ("forecast_parameters", {}), ("pairing_ready", False)):
            with self.subTest(field=field):
                result = build_comparisons([run("9B", **{field: value}), run("27B", **{field: value})])
                self.assertFalse(result["pairs"])
                self.assertFalse(result["common_run_ids"])
                self.assertTrue(all(item["status"] == "unknown" for item in result["audit"]))

    def test_zone_order_does_not_change_identity_but_partial_zones_do(self):
        same = build_comparisons([run("9B"), run("27B", zone_ids=["B", "A"])])
        self.assertEqual(len(same["pairs"]), 1)
        partial = build_comparisons([run("9B"), run("27B", zone_ids=["A"])])
        self.assertFalse(partial["pairs"])
        self.assertEqual(len([row for row in partial["audit"] if row["scope"] == "cross_batch"]), 2)

    def test_duplicates_are_ambiguous_not_cartesian_pairs(self):
        result = build_comparisons([run("9B"), run("9B", suffix="-copy"), run("27B")])
        self.assertFalse(result["pairs"])
        self.assertEqual(len(result["paired_summary"]), 0)
        cross = [row for row in result["audit"] if row["scope"] == "cross_batch"]
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["status"], "ambiguous")
        self.assertEqual(len(cross[0]["right_run_ids"]), 2)  # Lexicographic batch order.

    def test_duplicate_global_run_ids_raise_instead_of_overwriting(self):
        with self.assertRaisesRegex(ValueError, "Duplicate comparison run_id"):
            build_comparisons([run(), run()])

    def test_arbitrary_number_of_batches_and_only_equal_modes(self):
        runs = [run(batch, mode=mode) for batch in ("large", "small", "medium") for mode in ("single", "multi")]
        result = build_comparisons(runs)
        self.assertEqual(len(result["pairs"]), 6)
        self.assertEqual(len(result["paired_summary"]), 6)
        self.assertFalse(result["audit"])
        self.assertEqual(set(row["agent_mode"] for row in result["pairs"]), {"single", "multi"})

    def test_missing_modes_not_imputed_and_unmatched_runs_excluded_from_totals(self):
        runs = [run("27B"), run("9B"), run("27B", mode="multi"), run("9B", origin="later")]
        result = build_comparisons(runs)
        self.assertEqual(len(result["pairs"]), 1)
        summary = result["paired_summary"][0]
        self.assertEqual(summary["left"]["total_tokens"], 110)
        self.assertEqual(summary["right"]["total_tokens"], 110)
        self.assertEqual(summary["pair_count"], 1)
        self.assertTrue(any(row["status"] == "missing" for row in result["audit"]))

    def test_summary_pools_cache_and_revenue_and_provides_directed_deltas(self):
        runs = [run(batch, origin=origin) for batch in ("27B", "9B") for origin in ("a", "b")]
        runs[1]["usage"].update(prompt_tokens=300, total_tokens=310, cached_tokens=30, known_cached_tokens=30, cache_hit_ratio=0.1)
        runs[2]["outcome_summary"].update(medium_count=2, revenue=80, known_revenue=80)
        result = build_comparisons(runs)
        summary = result["paired_summary"][0]
        self.assertEqual(summary["left"]["cached_tokens"], 90)
        self.assertAlmostEqual(summary["left"]["cache_hit_ratio"], 90 / 400)
        self.assertEqual(summary["left"]["mean_cached_tokens_per_call"], 22.5)
        self.assertEqual(summary["left"]["revenue"], 120)
        self.assertEqual(summary["right"]["revenue"], 140)
        self.assertEqual(summary["delta"]["revenue"], 20)
        self.assertEqual(summary["delta"]["medium_count"], 1)
        self.assertEqual(summary["delta"]["total_tokens"], -200)
        self.assertEqual(summary["delta_direction"], "right_minus_left")

    def test_unknown_cache_and_partial_tokens_are_not_zero_or_full_totals(self):
        a, b = run("27B"), run("9B")
        b["usage"].update(token_usage_complete=False, completion_tokens=None, total_tokens=None, cached_tokens=None,
                          cache_usage_complete=False, cache_reported_call_count=1, known_cached_tokens=30)
        summary = build_comparisons([a, b])["paired_summary"][0]
        right = summary["right"]
        self.assertIsNone(right["total_tokens"])
        self.assertIsNone(right["cached_tokens"])
        self.assertEqual(right["known_cached_tokens"], 30)
        self.assertEqual(right["cache_reported_call_count"], 1)
        self.assertIsNone(right["cache_hit_ratio"])
        self.assertIsNone(summary["delta"]["cached_tokens"])
        self.assertEqual(right["known_prompt_tokens"], 100)

    def test_incomplete_outcomes_and_revenue_have_no_misleading_delta(self):
        a, b = run("27B"), run("9B")
        b["outcome_summary"].update(medium_unknown_count=1, medium_known_count=1,
                                    revenue_complete=False, revenue=None, known_revenue=30, revenue_unknown_count=1)
        summary = build_comparisons([a, b])["paired_summary"][0]
        self.assertIsNone(summary["delta"]["medium_count"])
        self.assertIsNone(summary["right"]["revenue"])
        self.assertEqual(summary["right"]["known_revenue"], 30)
        self.assertIsNone(summary["delta"]["revenue"])

    def test_datasets_and_units_never_pool(self):
        runs = [run(batch, dataset=dataset, revenue_unit=unit) for batch in ("27B", "9B") for dataset, unit in (("UrbanEV", "CNY"), ("CHARGED", "ZAR"))]
        summaries = build_comparisons(runs)["paired_summary"]
        self.assertEqual(len(summaries), 2)
        self.assertEqual({row["revenue_unit"] for row in summaries}, {"CNY", "ZAR"})
        self.assertTrue(all(row["left"]["revenue"] == 60 for row in summaries))

    def test_one_unknown_unit_disables_revenue_delta_but_keeps_other_metrics(self):
        result = build_comparisons([run("27B"), run("9B", revenue_unit=None)])
        summary = result["paired_summary"][0]
        self.assertEqual(len(result["pairs"]), 1)
        self.assertIsNone(summary["delta"]["revenue"])
        self.assertEqual(summary["delta"]["total_tokens"], 0)

    def test_invalid_identity_is_audited(self):
        for field, value in (("dataset_identity", None), ("zone_ids", []), ("zone_ids", ["A", "A"]), ("forecast_origin", None), ("agent_mode", None)):
            with self.subTest(field=field):
                result = build_comparisons([run(**{field: value})])
                self.assertFalse(result["common_run_ids"])
                self.assertEqual(result["audit"][0]["scope"], "identity")

    def test_empty_runs_return_empty_collections(self):
        self.assertEqual(build_comparisons([]), {"common_run_ids": [], "pairs": [], "audit": [], "paired_summary": []})

    def test_hourly_baseline_pools_totals_and_computes_weighted_changes(self):
        runs = [hourly_baseline(run(batch, origin=origin), baseline)
                for batch in ("27B", "9B") for origin, baseline in (("a", 40), ("b", 80))]
        for item, revenue in zip(runs[2:], (90, 70)):
            item["outcome_summary"].update(revenue=revenue, known_revenue=revenue)
            hourly_baseline(item, item["outcome_summary"]["baseline_hourly_revenue"])
        summary = build_comparisons(runs)["paired_summary"][0]
        self.assertEqual(summary["left"]["baseline_hourly_revenue"], 120)
        self.assertEqual(summary["left"]["known_baseline_hourly_revenue"], 120)
        self.assertTrue(summary["left"]["baseline_hourly_revenue_complete"])
        self.assertEqual(summary["left"]["baseline_hourly_revenue_known_count"], 4)
        self.assertEqual(summary["left"]["baseline_hourly_revenue_unknown_count"], 0)
        self.assertEqual(summary["left"]["baseline_hourly_revenue_coverage_pct"], 100)
        self.assertEqual(summary["left"]["revenue_hourly_comparable_count"], 4)
        self.assertEqual(summary["left"]["revenue_change_vs_hourly"], 0)
        self.assertEqual(summary["left"]["known_revenue_change_vs_hourly"], 0)
        self.assertEqual(summary["left"]["revenue_change_vs_hourly_pct"], 0)
        self.assertEqual(summary["right"]["revenue_change_vs_hourly"], 40)
        self.assertAlmostEqual(summary["right"]["revenue_change_vs_hourly_pct"], 100 / 3)
        self.assertEqual(summary["delta"]["baseline_hourly_revenue"], 0)
        self.assertEqual(summary["delta"]["revenue_change_vs_hourly"], 40)
        self.assertAlmostEqual(summary["delta"]["revenue_change_vs_hourly_pct"], 100 / 3)
        self.assertEqual(summary["left"]["baseline_revenue"], 100)
        self.assertEqual(summary["left"]["revenue_change"], 20)  # Existing window-mean baseline stays independent.

    def test_missing_hourly_baseline_is_unknown_and_old_metrics_stay_valid(self):
        summary = build_comparisons([run("27B"), run("9B")])["paired_summary"][0]
        for side in ("left", "right"):
            self.assertFalse(summary[side]["baseline_hourly_revenue_complete"])
            for field in ("baseline_hourly_revenue", "known_baseline_hourly_revenue",
                          "baseline_hourly_revenue_known_count", "baseline_hourly_revenue_coverage_pct",
                          "revenue_hourly_comparable_count", "known_revenue_change_vs_hourly",
                          "revenue_change_vs_hourly", "revenue_change_vs_hourly_pct"):
                self.assertIsNone(summary[side][field], field)
            self.assertEqual(summary[side]["revenue_change"], 10)
        self.assertIsNone(summary["delta"]["baseline_hourly_revenue"])
        self.assertIsNone(summary["delta"]["revenue_change_vs_hourly"])

    def test_partial_hourly_baseline_preserves_known_amounts_and_window_coverage(self):
        runs = [hourly_baseline(run(batch, origin=origin), 40)
                for batch in ("27B", "9B") for origin in ("a", "b")]
        hourly_baseline(runs[3], None, known=15, known_count=1, comparable_count=1, known_change=10)
        summary = build_comparisons(runs)["paired_summary"][0]
        partial = summary["right"]
        self.assertIsNone(partial["baseline_hourly_revenue"])
        self.assertEqual(partial["known_baseline_hourly_revenue"], 55)
        self.assertFalse(partial["baseline_hourly_revenue_complete"])
        self.assertEqual(partial["baseline_hourly_revenue_known_count"], 3)
        self.assertEqual(partial["baseline_hourly_revenue_unknown_count"], 1)
        self.assertEqual(partial["baseline_hourly_revenue_coverage_pct"], 75)
        self.assertEqual(partial["revenue_hourly_comparable_count"], 3)
        self.assertEqual(partial["known_revenue_change_vs_hourly"], 30)
        self.assertIsNone(partial["revenue_change_vs_hourly"])
        self.assertIsNone(partial["revenue_change_vs_hourly_pct"])
        self.assertIsNone(summary["delta"]["baseline_hourly_revenue"])
        self.assertIsNone(summary["delta"]["revenue_change_vs_hourly"])

    def test_explicit_missing_hourly_has_zero_coverage_without_zero_revenue(self):
        runs = [hourly_baseline(run(batch), None, known_count=0, comparable_count=0) for batch in ("27B", "9B")]
        side = build_comparisons(runs)["paired_summary"][0]["left"]
        self.assertEqual(side["baseline_hourly_revenue_known_count"], 0)
        self.assertEqual(side["baseline_hourly_revenue_unknown_count"], 2)
        self.assertEqual(side["baseline_hourly_revenue_coverage_pct"], 0)
        self.assertEqual(side["revenue_hourly_comparable_count"], 0)
        self.assertIsNone(side["known_baseline_hourly_revenue"])
        self.assertIsNone(side["known_revenue_change_vs_hourly"])

    def test_zero_hourly_baseline_has_known_change_but_unknown_percentage(self):
        runs = [hourly_baseline(run(batch), 0) for batch in ("27B", "9B")]
        summary = build_comparisons(runs)["paired_summary"][0]
        self.assertEqual(summary["left"]["baseline_hourly_revenue"], 0)
        self.assertEqual(summary["left"]["known_revenue_change_vs_hourly"], 60)
        self.assertEqual(summary["left"]["revenue_change_vs_hourly"], 60)
        self.assertIsNone(summary["left"]["revenue_change_vs_hourly_pct"])
        self.assertEqual(summary["delta"]["revenue_change_vs_hourly"], 0)
        self.assertIsNone(summary["delta"]["revenue_change_vs_hourly_pct"])

    def test_unit_guard_applies_to_hourly_baseline_and_change_deltas(self):
        runs = [hourly_baseline(run("27B"), 40), hourly_baseline(run("9B", revenue_unit=None), 40)]
        summary = build_comparisons(runs)["paired_summary"][0]
        self.assertEqual(summary["left"]["baseline_hourly_revenue"], 40)
        self.assertEqual(summary["right"]["revenue_change_vs_hourly"], 20)
        for field in ("revenue", "revenue_change", "baseline_hourly_revenue",
                      "revenue_change_vs_hourly", "revenue_change_vs_hourly_pct"):
            self.assertIsNone(summary["delta"][field], field)
        self.assertEqual(summary["delta"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
