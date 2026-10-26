"""Offline cache accounting from provider calls through reports and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluate import evaluate_agent, summarize_agents
from reporting import (
    normalized_attempt_usage, normalized_zone_cumulative_usage,
    token_only_call_usage, token_only_cumulative_usage, token_only_round_usage,
    write_control_outputs,
)
from usage import (
    CACHE_USAGE_FIELDS, aggregate_usage, cache_usage_fields,
    cumulative_global_agent_usage, summarize_agent_call_usage,
    summarize_cache_usage, token_total_summary,
)


def call(prompt: int = 100, cached: int | None = 80) -> dict:
    return {"stage": "discussion.round_1.grid", "agent": "Grid",
            "prompt_tokens": prompt, "completion_tokens": 10,
            "total_tokens": prompt + 10, "token_usage_complete": True,
            "cached_tokens": cached}


def control(usage: dict) -> dict:
    return {"schema_version": 3, "status": "success", "attempts_used": 1,
            "agent_cumulative_usage": usage, "zones": [{
                "zone_id": "A", "status": "success", "attempts_used": 1,
                "agent_cumulative_usage": usage, "attempt_trace": [],
                "final_windows": [],
            }]}


class CacheUsageTests(unittest.TestCase):
    def test_reported_zero_differs_from_missing_statistics(self):
        zero = cache_usage_fields(call(cached=0))
        missing = cache_usage_fields(call(cached=None))
        self.assertEqual(zero["cached_tokens"], 0)
        self.assertEqual(zero["cache_hit_ratio"], 0)
        self.assertTrue(zero["cache_usage_complete"])
        self.assertIsNone(missing["cached_tokens"])
        self.assertIsNone(missing["cache_hit_ratio"])
        self.assertFalse(missing["cache_usage_complete"])
        self.assertEqual(missing["cache_reported_call_count"], 0)

    def test_invalid_cache_counts_are_unknown(self):
        for cached in (-1, 101, 1.5, True, False, "50", float("nan"), float("inf")):
            with self.subTest(cached=cached):
                result = cache_usage_fields(call(cached=cached))
                self.assertIsNone(result["cached_tokens"])
                self.assertIsNone(result["cache_hit_ratio"])
                self.assertFalse(result["cache_usage_complete"])
                self.assertEqual(result["known_cached_tokens"], 0)
                self.assertEqual(result["cache_reported_call_count"], 0)

    def test_missing_prompt_count_cannot_validate_cache_count(self):
        for prompt in (None, -1, True, "100"):
            result = cache_usage_fields({"prompt_tokens": prompt, "cached_tokens": 0})
            self.assertIsNone(result["cached_tokens"])
            self.assertIsNone(result["cache_hit_ratio"])

    def test_partial_totals_remain_partial_after_multiple_aggregation_levels(self):
        partial = summarize_agent_call_usage([call(), call(cached=None)])
        total = aggregate_usage([partial, summarize_agent_call_usage([call(50, 20)])])
        final = token_total_summary(cumulative_global_agent_usage([total]))
        self.assertEqual(final["prompt_tokens"], 250)
        self.assertEqual(final["known_cached_tokens"], 100)
        self.assertEqual(final["cache_reported_call_count"], 2)
        self.assertIsNone(final["cached_tokens"])
        self.assertIsNone(final["cache_hit_ratio"])
        self.assertFalse(final["cache_usage_complete"])
        self.assertTrue(final["token_usage_complete"])

    def test_ratio_pools_input_tokens_instead_of_averaging_call_ratios(self):
        result = aggregate_usage([
            summarize_agent_call_usage([call(10, 1)]),
            summarize_agent_call_usage([call(90, 90)]),
        ])
        self.assertEqual(result["cached_tokens"], 91)
        self.assertAlmostEqual(result["cache_hit_ratio"], .91)
        self.assertEqual(result["cache_reported_call_count"], 2)

    def test_provider_retry_incompleteness_keeps_known_cache_count(self):
        value = {**call(), **cache_usage_fields(call()),
                 "provider_attempt_count": 2, "cache_usage_complete": False}
        normalized = cache_usage_fields(value)
        self.assertIsNone(normalized["cached_tokens"])
        self.assertIsNone(normalized["cache_hit_ratio"])
        self.assertEqual(normalized["known_cached_tokens"], 80)
        self.assertEqual(cache_usage_fields({**value, **normalized}), normalized)
        total = summarize_agent_call_usage([{**value, **normalized}])
        self.assertEqual(total["known_cached_tokens"], 80)
        self.assertFalse(total["cache_usage_complete"])

    def test_no_calls_and_empty_prompts_have_no_division_by_zero(self):
        for value in (summarize_cache_usage([]), aggregate_usage([]),
                      cache_usage_fields({"agent_call_count": 0}),
                      cache_usage_fields(call(0, 0))):
            self.assertEqual(value["cached_tokens"], 0)
            self.assertTrue(value["cache_usage_complete"])
            self.assertIsNone(value["cache_hit_ratio"])
            json.dumps(value, allow_nan=False)

    def test_frozen_rows_without_prompt_counts_do_not_hide_pooled_cache_ratio(self):
        real = summarize_agent_call_usage([call()])
        for frozen in ({"agent_call_count": 0}, {"agent_invoked": False}):
            with self.subTest(frozen=frozen):
                result = summarize_cache_usage([real, frozen], aggregated=True)
                self.assertEqual(result["cached_tokens"], 80)
                self.assertEqual(result["cache_reported_call_count"], 1)
                self.assertTrue(result["cache_usage_complete"])
                self.assertEqual(result["cache_hit_ratio"], .8)
        missing = summarize_cache_usage([real, {"agent_call_count": 1}], aggregated=True)
        self.assertFalse(missing["cache_usage_complete"])
        self.assertIsNone(missing["cache_hit_ratio"])

    def test_invalid_partial_known_count_cannot_keep_reported_call_coverage(self):
        result = cache_usage_fields({
            "agent_call_count": 2, "prompt_tokens": 100, "cached_tokens": None,
            "known_cached_tokens": 101, "cache_reported_call_count": 1,
            "cache_usage_complete": False,
        })
        self.assertIsNone(result["cached_tokens"])
        self.assertEqual(result["known_cached_tokens"], 0)
        self.assertEqual(result["cache_reported_call_count"], 0)
        self.assertFalse(result["cache_usage_complete"])

    def test_cache_accounting_does_not_depend_on_completion_usage(self):
        result = summarize_agent_call_usage([
            {**call(), "completion_tokens": None, "total_tokens": None,
             "token_usage_complete": False},
        ])
        self.assertFalse(result["token_usage_complete"])
        self.assertTrue(result["cache_usage_complete"])
        self.assertEqual(result["cached_tokens"], 80)

    def test_report_whitelists_keep_cache_fields_and_drop_timing(self):
        value = {**summarize_agent_call_usage([call()]), "elapsed_seconds": 4,
                 "provider_response": "private raw payload"}
        for normalize in (token_only_call_usage, token_only_round_usage, token_only_cumulative_usage):
            with self.subTest(normalize=normalize.__name__):
                result = normalize(value)
                self.assertEqual(result["cached_tokens"], 80)
                self.assertEqual(result["cache_hit_ratio"], .8)
                self.assertNotIn("elapsed_seconds", result)
                self.assertNotIn("provider_response", result)

    def test_reporting_derives_old_attempt_cache_fields_from_calls(self):
        attempt = {"attempt": 1, "agent_call_usage": [call()]}
        result = normalized_attempt_usage(attempt)
        self.assertEqual(result["cached_tokens"], 80)
        zone = normalized_zone_cumulative_usage({}, [attempt])
        self.assertEqual(zone["cache_hit_ratio"], .8)

    def test_authoritative_incomplete_summary_is_not_replaced_by_complete_calls(self):
        usage = summarize_agent_call_usage([call(), call(cached=None)])
        result = normalized_attempt_usage({
            "attempt": 1, "agent_usage": usage, "agent_call_usage": [call()],
        })
        self.assertFalse(result["cache_usage_complete"])
        self.assertEqual(result["known_cached_tokens"], 80)
        self.assertIsNone(result["cached_tokens"])

    def test_unrecorded_calls_make_legacy_attempt_cache_unknown(self):
        result = normalized_attempt_usage({
            "attempt": 1, "agent_usage": {"agent_call_count": 2},
            "agent_call_usage": [call()],
        })
        self.assertIsNone(result["cached_tokens"])
        self.assertFalse(result["cache_usage_complete"])
        self.assertEqual(result["known_cached_tokens"], 80)

    def test_reporting_exports_step_attempt_zone_global_and_compact_totals(self):
        report = {"zone_id": "A", "category": "test", "control_status": "success",
                  "attempts_used": 1, "agent_call_usage": [call()],
                  "control_attempt_trace": [{"attempt": 1, "agent_call_usage": [call()]}]}
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("control.json", "final.csv", "trace.csv",
                     "attempt.csv", "step.csv", "summary.csv")]
            write_control_outputs(*paths, [report], backend="AR", agent_mode="test",
                                  manifest=None, control_agent_usage={"agent_cumulative_usage": {}})
            result = json.loads(paths[0].read_text(encoding="utf-8"))
            for usage in (result["agent_round_usage"][0], result["agent_cumulative_usage"],
                          result["agent_token_totals"], result["zones"][0]["agent_cumulative_usage"],
                          result["zones"][0]["agent_token_totals"],
                          result["zones"][0]["agent_call_usage"][0]):
                self.assertEqual(usage["cached_tokens"], 80)
                self.assertEqual(usage["cache_hit_ratio"], .8)
            steps = pd.read_csv(paths[4])
            attempts = pd.read_csv(paths[3])
            self.assertEqual(len(steps), 1)
            self.assertEqual(len(attempts), 2)  # Independent global and Zone views.
            for frame in (steps, attempts):
                self.assertTrue(set(CACHE_USAGE_FIELDS).issubset(frame.columns))
                self.assertTrue((frame.cached_tokens == 80).all())
                self.assertTrue((frame.cache_hit_ratio == .8).all())

    def test_evaluation_uses_authoritative_global_counts_without_double_counting(self):
        run = evaluate_agent(control(summarize_agent_call_usage([call()])))
        self.assertEqual(run["summary"]["cached_tokens"], 80)
        self.assertEqual(run["summary"]["cache_reported_call_count"], 1)
        self.assertEqual(run["summary"]["cache_hit_ratio"], .8)
        second = evaluate_agent(control(summarize_agent_call_usage([call(50, 10)])))
        summary = summarize_agents([run, second])
        self.assertEqual(summary["cached_tokens"], 90)
        self.assertEqual(summary["cache_hit_ratio"], .6)

    def test_evaluation_of_legacy_or_partial_statistics_remains_unknown(self):
        known = evaluate_agent(control(summarize_agent_call_usage([call()])))
        legacy_usage = {"agent_call_count": 1, "prompt_tokens": 100,
                        "completion_tokens": 10, "total_tokens": 110, "token_usage_complete": True}
        legacy = evaluate_agent(control(legacy_usage))
        result = summarize_agents([known, legacy])
        self.assertEqual(result["known_cached_tokens"], 80)
        self.assertIsNone(result["cached_tokens"])
        self.assertIsNone(result["cache_hit_ratio"])
        self.assertEqual(result["incomplete_cache_usage_run_count"], 1)
        self.assertEqual(result["total_tokens"], 220)
        missing = control({})
        missing.pop("agent_cumulative_usage")
        self.assertFalse(evaluate_agent(missing)["summary"]["cache_usage_complete"])


if __name__ == "__main__":
    unittest.main()
