"""Offline native Anthropic usage contracts; no provider requests are made."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from anthropic_usage import anthropic_response_token_usage
from usage import cache_usage_fields, summarize_agent_call_usage


def native_usage(**changes):
    return {"input_tokens": 20, "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 50, "output_tokens": 10, **changes}


def normalized(**changes):
    return anthropic_response_token_usage({"usage": native_usage(**changes)})


class AnthropicUsageTests(unittest.TestCase):
    def test_input_includes_cache_reads_and_writes_once(self):
        result = normalized()
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 10)
        self.assertEqual(result["total_tokens"], 110)
        self.assertEqual(result["cached_tokens"], 50)
        self.assertEqual(result["known_cached_tokens"], 50)
        self.assertEqual(result["cache_hit_ratio"], .5)
        self.assertEqual(result["cache_reported_call_count"], 1)
        self.assertTrue(result["cache_usage_complete"])

    def test_sdk_objects_and_dicts_are_equivalent(self):
        expected = normalized()
        self.assertEqual(anthropic_response_token_usage(SimpleNamespace(usage=SimpleNamespace(**native_usage()))), expected)
        from anthropic.types import Usage
        self.assertEqual(anthropic_response_token_usage(SimpleNamespace(usage=Usage(**native_usage()))), expected)

    def test_cache_writes_do_not_count_as_cache_hits(self):
        result = normalized(cache_creation_input_tokens=80, cache_read_input_tokens=0)
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["cached_tokens"], 0)
        self.assertEqual(result["cache_hit_ratio"], 0)
        self.assertTrue(result["cache_usage_complete"])

    def test_zero_is_distinct_from_unknown_including_zero_input(self):
        zero = normalized(input_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=0)
        self.assertEqual(zero["prompt_tokens"], 0)
        self.assertEqual(zero["total_tokens"], 0)
        self.assertEqual(zero["cached_tokens"], 0)
        self.assertIsNone(zero["cache_hit_ratio"])
        self.assertTrue(zero["cache_usage_complete"])
        missing = normalized(cache_read_input_tokens=None)
        self.assertIsNone(missing["prompt_tokens"])
        self.assertIsNone(missing["total_tokens"])
        self.assertIsNone(missing["cached_tokens"])
        self.assertFalse(missing["cache_usage_complete"])

    def test_missing_usage_is_unknown_and_json_serializable(self):
        for response in ({}, {"usage": None}, None, SimpleNamespace()):
            with self.subTest(response=response):
                result = anthropic_response_token_usage(response)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_hit_ratio"):
                    self.assertIsNone(result[field], field)
                self.assertEqual(result["cache_reported_call_count"], 0)
                self.assertFalse(result["cache_usage_complete"])
                json.dumps(result, allow_nan=False)

    def test_missing_each_input_component_prevents_partial_total(self):
        for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            with self.subTest(field=field):
                usage = native_usage()
                usage.pop(field)
                result = anthropic_response_token_usage({"usage": usage})
                self.assertIsNone(result["prompt_tokens"])
                self.assertIsNone(result["total_tokens"])
                self.assertEqual(result["completion_tokens"], 10)
                self.assertIsNone(result["cached_tokens"])
                self.assertIsNone(result["cache_hit_ratio"])
                self.assertFalse(result["cache_usage_complete"])
                self.assertEqual(result["cache_reported_call_count"], 0)
                if field != "cache_read_input_tokens":
                    self.assertEqual(result["cache_read_input_tokens"], 50)

    def test_invalid_components_are_not_coerced_or_counted(self):
        for field in native_usage():
            for value in (-1, True, False, 1.0, 1.5, "10", float("nan"), float("inf"), [], {}):
                with self.subTest(field=field, value=value):
                    result = normalized(**{field: value})
                    self.assertIsNone(result["total_tokens"])
                    if field == "output_tokens":
                        self.assertIsNone(result["completion_tokens"])
                        self.assertEqual(result["prompt_tokens"], 100)
                        self.assertEqual(result["cached_tokens"], 50)
                    else:
                        self.assertIsNone(result["prompt_tokens"])
                        self.assertIsNone(result["cached_tokens"])
                        self.assertFalse(result["cache_usage_complete"])
                    json.dumps(result, allow_nan=False)

    def test_missing_output_does_not_hide_known_cache_statistics(self):
        result = normalized(output_tokens=None)
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["cached_tokens"], 50)
        self.assertTrue(result["cache_usage_complete"])

    def test_native_breakdown_and_incompatible_totals_are_not_added(self):
        result = normalized(cache_creation={"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20},
                            total_tokens=9999, prompt_tokens=999, completion_tokens=1000,
                            prompt_tokens_details={"cached_tokens": 999})
        self.assertEqual(result, normalized())

    def test_shared_aggregation_uses_weighted_prompt_denominator(self):
        a = normalized()
        b = normalized(input_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=9, output_tokens=2)
        summary = summarize_agent_call_usage([a, b])
        self.assertEqual(summary["prompt_tokens"], 110)
        self.assertEqual(summary["completion_tokens"], 12)
        self.assertEqual(summary["total_tokens"], 122)
        self.assertEqual(summary["cached_tokens"], 59)
        self.assertAlmostEqual(summary["cache_hit_ratio"], 59 / 110)
        self.assertEqual(summary["cache_reported_call_count"], 2)

    def test_existing_retry_incomplete_normalization_keeps_known_hits(self):
        result = normalized()
        result["cache_usage_complete"] = False
        retry = cache_usage_fields(result)
        self.assertIsNone(retry["cached_tokens"])
        self.assertIsNone(retry["cache_hit_ratio"])
        self.assertFalse(retry["cache_usage_complete"])
        self.assertEqual(retry["known_cached_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
