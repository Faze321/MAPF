"""Offline evaluation accounting: no provider clients or forecast models."""
import copy
import json
import unittest

from evaluation_usage import build_usage_tables, extract_usage_details, summarize_usage


def call(prompt=100, completion=10, cached=80, **extra):
    return {"zone_id": "001", "attempt": 1, "stage": "agent.discussion.round_2.grid",
            "agent": "Grid", "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "cached_tokens": cached,
            "token_usage_complete": True, **extra}


def aggregate(calls):
    summary = summarize_usage(calls)
    return {key: summary[key] for key in (
        "agent_call_count", "prompt_tokens", "completion_tokens", "total_tokens", "token_usage_complete",
        "cached_tokens", "known_cached_tokens", "cache_reported_call_count", "cache_usage_complete", "cache_hit_ratio")}


def control(calls=None):
    calls = calls if calls is not None else [call()]
    usage = aggregate(calls)
    return {"schema_version": 3, "status": "success", "attempts_used": 1, "attempt_limit": 3,
            "agent_step_token_usage": copy.deepcopy(calls), "agent_round_usage": [{"attempt": 1, **usage}],
            "agent_cumulative_usage": usage,
            "zones": [{"zone_id": "001", "status": "success", "attempts_used": 1,
                       "agent_cumulative_usage": usage, "agent_call_usage": copy.deepcopy(calls),
                       "attempt_trace": [{"attempt": 1, "agent_usage": usage, "agent_call_usage": copy.deepcopy(calls)}]}]}


class UsageSummaryTests(unittest.TestCase):
    def test_pooled_cache_ratio_and_call_distributions(self):
        result = summarize_usage([call(10, 0, 1), call(90, 20, 90)])
        self.assertEqual(result["cached_tokens"], 91)
        self.assertEqual(result["cache_hit_ratio"], .91)
        self.assertEqual(result["mean_cached_tokens_per_call"], 45.5)
        self.assertEqual(result["total_tokens"], 120)
        self.assertEqual(result["mean_total_tokens_per_call"], 60)
        self.assertEqual(result["median_total_tokens_per_call"], 60)
        self.assertEqual(result["p95_total_tokens_per_call"], 105)

    def test_missing_tokens_keep_known_subtotal_and_distribution(self):
        result = summarize_usage([call(), {**call(), "prompt_tokens": None}])
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["total_tokens"])
        self.assertIsNone(result["mean_total_tokens_per_call"])
        self.assertIsNone(result["median_total_tokens_per_call"])
        self.assertEqual(result["known_prompt_tokens"], 100)
        self.assertEqual(result["known_prompt_tokens_call_count"], 1)
        self.assertEqual(result["known_total_tokens"], 220)
        self.assertEqual(result["known_median_total_tokens_per_call"], 110)

    def test_cache_missing_is_distinct_from_reported_zero(self):
        missing = summarize_usage([call(cached=None)])
        zero = summarize_usage([call(cached=0)])
        self.assertIsNone(missing["cached_tokens"])
        self.assertIsNone(missing["cache_hit_ratio"])
        self.assertEqual(missing["cache_reported_call_count"], 0)
        self.assertEqual(zero["cached_tokens"], 0)
        self.assertEqual(zero["cache_hit_ratio"], 0)
        self.assertEqual(zero["cache_reported_call_count"], 1)

    def test_partial_cache_keeps_known_coverage(self):
        result = summarize_usage([call(), call(cached=None)])
        self.assertIsNone(result["cached_tokens"])
        self.assertIsNone(result["mean_cached_tokens_per_call"])
        self.assertIsNone(result["cache_hit_ratio"])
        self.assertEqual(result["known_cached_tokens"], 80)
        self.assertEqual(result["cache_reported_call_fraction"], .5)

    def test_aggregate_rows_cannot_invent_call_distribution(self):
        result = summarize_usage([aggregate([call(), call(200, 20, 100)])], aggregated=True)
        self.assertEqual(result["agent_call_count"], 2)
        self.assertEqual(result["mean_total_tokens_per_call"], 165)
        self.assertIsNone(result["median_total_tokens_per_call"])
        self.assertIsNone(result["p95_total_tokens_per_call"])
        self.assertIsNone(result["known_total_tokens_call_count"])
        self.assertFalse(result["call_distribution_available"])

    def test_explicit_zero_rows_do_not_hide_valid_cache_ratio(self):
        result = summarize_usage([aggregate([call()]), {"agent_invoked": False}], aggregated=True)
        self.assertEqual(result["total_tokens"], 110)
        self.assertEqual(result["cached_tokens"], 80)
        self.assertEqual(result["cache_hit_ratio"], .8)
        self.assertIsNone(summarize_usage([], aggregated=True)["cache_hit_ratio"])

    def test_missing_aggregate_is_not_zero(self):
        result = summarize_usage([aggregate([call()]), {}], aggregated=True)
        self.assertIsNone(result["agent_call_count"])
        self.assertEqual(result["known_agent_call_count"], 1)
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["known_total_tokens"], 110)
        self.assertIsNone(result["cached_tokens"])

    def test_provider_retry_preserves_known_tokens_but_never_fabricates_calls(self):
        result = summarize_usage([call(provider_attempt_count=2)])
        self.assertEqual(result["agent_call_count"], 1)
        self.assertFalse(result["token_usage_complete"])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["known_total_tokens"], 110)
        self.assertFalse(result["cache_usage_complete"])
        self.assertEqual(result["known_cached_tokens"], 80)
        self.assertIsNone(result["cached_tokens"])


class UsageExtractionTests(unittest.TestCase):
    def test_authoritative_steps_and_global_rounds_are_not_double_counted(self):
        value = control()
        value["zones"][0]["agent_call_usage"] *= 3
        value["zones"][0]["attempt_trace"][0]["agent_call_usage"] *= 2
        details = extract_usage_details(value)
        self.assertEqual(len(details["calls"]), 1)
        self.assertEqual(details["calls"][0]["discussion_round"], 2)
        self.assertEqual(details["calls"][0]["zone_id"], "001")
        self.assertEqual(details["rounds"][0]["total_tokens"], 110)
        self.assertEqual([r["attempt"] for r in details["rounds"]], [1, 2, 3])
        self.assertFalse(any(a["status"] == "mismatch" for a in details["audit"]))

    def test_legacy_trace_fallback_and_explicit_success_zeros(self):
        value = control()
        value.pop("agent_step_token_usage")
        value.pop("agent_round_usage")
        value["schema_version"] = 2
        details = extract_usage_details(value)
        self.assertEqual(len(details["calls"]), 1)
        self.assertEqual(details["calls"][0]["usage_source"], "attempt_trace.agent_call_usage")
        self.assertEqual(details["rounds"][0]["usage_source"], "zone_attempt_usage")
        self.assertTrue(details["rounds"][1]["carried_forward"])
        self.assertEqual(details["rounds"][1]["total_tokens"], 0)
        self.assertEqual(details["rounds"][1]["cached_tokens"], 0)

    def test_missing_middle_and_failure_tail_are_unknown_not_zero(self):
        value = control()
        value.update(status="fail", attempts_used=3, attempt_limit=4)
        value["zones"][0].update(status="fail", attempts_used=3)
        value["agent_round_usage"].append({"attempt": 3, **aggregate([call()])})
        rounds = extract_usage_details(value)["rounds"]
        self.assertEqual(len(rounds), 4)
        self.assertIsNone(rounds[1]["total_tokens"])
        self.assertIsNone(rounds[3]["total_tokens"])
        self.assertFalse(rounds[1]["token_usage_complete"])

    def test_partial_zone_round_cannot_claim_global_complete(self):
        value = control()
        value.pop("agent_round_usage")
        value["zones"].append({"zone_id": "002", "status": "fail", "attempts_used": 3, "attempt_trace": []})
        row = extract_usage_details(value)["rounds"][0]
        self.assertIsNone(row["agent_call_count"])
        self.assertFalse(row["token_usage_complete"])
        self.assertEqual(row["total_tokens"], 110)
        summary = summarize_usage([row], aggregated=True)
        self.assertEqual(summary["known_agent_call_count"], 1)
        self.assertEqual(summary["known_cached_tokens"], 80)
        self.assertEqual(summary["cache_reported_call_count"], 1)
        self.assertIsNone(summary["cached_tokens"])

    def test_audit_reports_conflict_without_rewriting_authority(self):
        value = control()
        value["agent_round_usage"][0]["total_tokens"] = 999
        details = extract_usage_details(value)
        self.assertEqual(details["rounds"][0]["total_tokens"], 999)
        mismatch = [a for a in details["audit"] if a["check"] == "calls_vs_round" and a["status"] == "mismatch"]
        self.assertEqual(mismatch[0]["mismatches"][0]["field"], "total_tokens")

    def test_repair_flag_and_retry_incompleteness_propagate(self):
        details = extract_usage_details(control([call(stage="agent.single_agent.schema_repair", provider_attempt_count=3)]))
        row = details["calls"][0]
        self.assertTrue(row["is_repair"])
        self.assertIsNone(row["discussion_round"])
        self.assertEqual(row["provider_attempt_count"], 3)
        self.assertFalse(row["token_usage_complete"])
        self.assertFalse(details["rounds"][0]["cache_usage_complete"])
        self.assertEqual(len(details["calls"]), 1)

    def test_no_round_identity_is_not_guessed(self):
        value = control()
        value["agent_step_token_usage"][0].pop("attempt")
        details = extract_usage_details(value)
        self.assertIsNone(details["calls"][0]["attempt"])
        self.assertTrue(any(a["check"] == "call_round_identity" for a in details["audit"]))


class UsageTableTests(unittest.TestCase):
    def annotate(self, details, run):
        for kind in ("calls", "rounds"):
            for row in details[kind]:
                row.update(batch="27B", dataset="UrbanEV", backend="lstm", agent_mode="single", run_id=run)
        return details

    def test_round_mean_uses_runs_and_includes_success_zero(self):
        first = self.annotate(extract_usage_details(control()), "r1")
        second_payload = control([call(200, 20, 100)])
        second_payload["agent_round_usage"].append({"attempt": 2, **aggregate([call(200, 20, 100)])})
        second_payload["agent_step_token_usage"].append(call(200, 20, 100, attempt=2))
        second_payload["agent_cumulative_usage"] = aggregate(second_payload["agent_step_token_usage"])
        second_payload.update(attempts_used=2)
        second = self.annotate(extract_usage_details(second_payload), "r2")
        tables = build_usage_tables(first["calls"] + second["calls"], first["rounds"] + second["rounds"])
        row = next(r for r in tables["tokens_by_mode_round"] if r["attempt"] == 2)
        self.assertEqual(row["run_count"], 2)
        self.assertEqual(row["total_tokens"], 220)
        self.assertEqual(row["mean_total_tokens_per_run"], 110)
        self.assertEqual(row["mean_total_tokens_per_call"], 220)
        self.assertEqual(row["median_total_tokens_per_call"], 220)
        self.assertEqual(row["cached_tokens"], 100)
        self.assertEqual(row["mean_cached_tokens_per_run"], 50)
        self.assertEqual(tables["usage_by_batch"][0]["total_tokens"], 550)
        json.dumps(tables, allow_nan=False)

    def test_missing_round_disables_full_group_mean_but_keeps_known(self):
        detail = self.annotate(extract_usage_details(control()), "r1")
        detail["rounds"][1] = {**detail["rounds"][1], "agent_call_count": None,
                                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                                "token_usage_complete": False, "cache_usage_complete": False}
        tables = build_usage_tables(detail["calls"], detail["rounds"])
        row = tables["usage_by_mode"][0]
        self.assertIsNone(row["mean_total_tokens_per_run"])
        self.assertIsNone(row["total_tokens"])
        self.assertEqual(row["known_total_tokens"], 110)
        self.assertIsNone(row["median_total_tokens_per_call"])

    def test_absent_calls_keep_authoritative_mean_but_not_distribution(self):
        payload = control([call(), call(200, 20, 100)])
        payload.pop("agent_step_token_usage")
        payload["zones"][0].pop("agent_call_usage")
        payload["zones"][0]["attempt_trace"][0].pop("agent_call_usage")
        detail = self.annotate(extract_usage_details(payload), "r1")
        self.assertEqual(detail["calls"], [])
        row = build_usage_tables(detail["calls"], detail["rounds"])["usage_by_mode"][0]
        self.assertEqual(row["agent_call_count"], 2)
        self.assertEqual(row["mean_total_tokens_per_call"], 165)
        self.assertEqual(row["total_tokens"], 330)
        self.assertEqual(row["recorded_call_count"], 0)
        self.assertFalse(row["call_details_complete"])
        self.assertIsNone(row["median_total_tokens_per_call"])
        self.assertIsNone(row["p95_total_tokens_per_call"])

    def test_partial_calls_do_not_claim_complete_role_or_zone_totals(self):
        payload = control([call(), call(200, 20, 100)])
        payload["agent_step_token_usage"] = payload["agent_step_token_usage"][:1]
        detail = self.annotate(extract_usage_details(payload), "r1")
        tables = build_usage_tables(detail["calls"], detail["rounds"])
        mode = tables["usage_by_mode"][0]
        self.assertEqual(mode["agent_call_count"], 2)
        self.assertEqual(mode["mean_total_tokens_per_call"], 165)
        self.assertFalse(mode["call_details_complete"])
        self.assertEqual(mode["recorded_call_count"], 1)
        self.assertEqual(mode["known_mean_total_tokens_per_call"], 110)
        for name in ("usage_by_zone", "usage_by_stage", "usage_by_agent", "usage_by_discussion_round", "usage_by_repair"):
            with self.subTest(name=name):
                row = tables[name][0]
                self.assertIsNone(row["agent_call_count"])
                self.assertEqual(row["known_agent_call_count"], 1)
                self.assertIsNone(row["total_tokens"])
                self.assertIsNone(row["mean_total_tokens_per_call"])
                self.assertIsNone(row["mean_cached_tokens_per_call"])
                self.assertIsNone(row["cache_hit_ratio"])
                self.assertEqual(row["known_total_tokens"], 110)
                self.assertEqual(row["known_cached_tokens"], 80)
                self.assertEqual(row["known_mean_total_tokens_per_call"], 110)
                self.assertFalse(row["call_details_complete"])

    def test_missing_calls_and_rounds_do_not_allocate_cumulative_to_guessed_round(self):
        payload = control()
        payload.pop("agent_step_token_usage")
        payload.pop("agent_round_usage")
        payload["zones"][0].pop("agent_call_usage")
        payload["zones"][0]["attempt_trace"] = []
        detail = self.annotate(extract_usage_details(payload), "r1")
        row = build_usage_tables(detail["calls"], detail["rounds"])["usage_by_mode"][0]
        self.assertIsNone(row["agent_call_count"])
        self.assertIsNone(row["total_tokens"])
        self.assertIsNone(row["mean_total_tokens_per_call"])
        self.assertFalse(row["call_details_complete"])
        self.assertEqual(payload["agent_cumulative_usage"]["total_tokens"], 110)

    def test_mode_summary_pools_datasets_models_and_preserves_detailed_table(self):
        first = self.annotate(extract_usage_details(control()), "r1")
        second = self.annotate(extract_usage_details(control([call(200, 20, 100)])), "r2")
        for kind in ("calls", "rounds"):
            for row in second[kind]:
                row.update(dataset="MP-EVData", backend="timesfm")
        tables = build_usage_tables(first["calls"] + second["calls"], first["rounds"] + second["rounds"])
        self.assertEqual(len(tables["usage_by_mode"]), 1)
        self.assertEqual(len(tables["usage_by_dataset_model_mode"]), 2)
        self.assertEqual(tables["usage_by_mode"][0]["mean_total_tokens_per_call"], 165)
        self.assertEqual(tables["usage_by_mode"][0]["cached_tokens"], 180)
        self.assertAlmostEqual(tables["usage_by_mode"][0]["cache_hit_ratio"], .6)

    def test_derived_partial_rounds_do_not_become_full_by_self_consistency(self):
        payload = control([call(), call(200, 20, 100)])
        payload.pop("agent_round_usage")
        payload["zones"] = []
        payload["agent_step_token_usage"] = payload["agent_step_token_usage"][:1]
        detail = self.annotate(extract_usage_details(payload), "r1")
        row = build_usage_tables(detail["calls"], detail["rounds"])["usage_by_mode"][0]
        self.assertIsNone(row["agent_call_count"])
        self.assertIsNone(row["total_tokens"])
        self.assertEqual(row["known_total_tokens"], 110)
        self.assertFalse(row["call_details_complete"])


if __name__ == "__main__":
    unittest.main()
