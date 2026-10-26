"""Offline contracts for deterministic prompts and reusable cache prefixes."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import prompts


def sample_context():
    return {
        "zone_id": "区域甲",
        "category": "residential",
        "forecast_horizon_days": 2,
        "forecast_horizon_hours": 48,
        "forecast_origin": "2024-02-01 00:00:00",
        "forecast_end": "2024-02-03 00:00:00",
        "forecast_total_kwh": 42.0,
        "price": {"energy": {"upper": 0.8, "lower": 0.2}, "service": 0.1},
        "instructions": ["保留预测", "仅调整电价"],
        "hourly_forecast": [
            {"time": "2024-02-01 00:00:00", "predicted_kwh": 3.0, "e_price": 0.3},
            {"time": "2024-02-01 01:00:00", "predicted_kwh": 2.0, "e_price": 0.4},
        ],
        "hourly_averages": {
            "mean_predicted_kwh": 2.5,
            "hourly_shape": {"night": [3.0, 2.0], "day": [1.0, 4.0]},
        },
        "pricing_windows_3h": [
            {
                "window_start": "2024-02-01 00:00:00",
                "window_end": "2024-02-01 03:00:00",
                "sum_predicted_kwh": 9.0,
                "mean_energy_price": 0.3,
                "load_stress_level": "Medium",
            },
            {
                "window_start": "2024-02-01 03:00:00",
                "window_end": "2024-02-01 06:00:00",
                "sum_predicted_kwh": 12.0,
                "mean_energy_price": 0.4,
                "load_stress_level": "High",
            },
        ],
        "retry_feedback": [{
            "attempt": 1,
            "window_start": "2024-02-01 03:00:00",
            "window_end": "2024-02-01 06:00:00",
            "previous_energy_price": 0.4,
            "reforecast_load_kwh": 12.0,
            "failure_reason": "load remains High",
        }],
    }


def sample_reports():
    return (
        {"reasoning_summary": "电网", "window_assessments": [{"stress": "High", "load": 12}]},
        {"confidence": 0.8, "demand_drivers": [{"weather": "rain", "occupancy": 0.5}]},
        {"price_rationale": "转移充电", "price_change_windows_3h": [{"price": 0.4, "shift": 10}]},
        {"conclusion_summary": "review", "key_decisions": [{"action": "hold", "zone": "甲"}],
         "disagreements": ["elasticity", "price"]},
    )


def reverse_object_order(value):
    """Change every object's insertion order without reordering arrays."""
    if isinstance(value, dict):
        return {key: reverse_object_order(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reverse_object_order(item) for item in value]
    return value


def render_all(context=None, reports=None, *, discussion_round=1, errors=None):
    context = sample_context() if context is None else context
    grid, behavior, previous, handoff = sample_reports() if reports is None else reports
    discussion = dict(discussion_round=discussion_round, previous_exchange=handoff)
    return {
        "grid": prompts.grid_prompt(context),
        "behavior": prompts.behavior_prompt(context, grid),
        "economist": prompts.economist_prompt(context, grid, behavior),
        "single": prompts.single_agent_prompt(context),
        "discussion_grid": prompts.discussion_grid_prompt(context, **discussion),
        "discussion_behavior": prompts.discussion_behavior_prompt(context, grid, **discussion),
        "discussion_economist": prompts.discussion_economist_prompt(context, grid, behavior, **discussion),
        "repair": prompts.repair_economist_prompt(
            context, grid, behavior, previous, ["missing price"] if errors is None else errors),
        "retry": prompts.price_retry_prompt(context, grid, behavior, previous),
        "single_retry": prompts.price_retry_prompt(context, grid, behavior, previous, single_agent=True),
    }


def section_json(prompt, label="Context"):
    return json.JSONDecoder().raw_decode(prompt.split(label + ":\n", 1)[1].lstrip())[0]


class PromptCachingTests(unittest.TestCase):
    def test_all_prompts_ignore_nested_object_insertion_order(self):
        normal = render_all()
        reordered = render_all(
            reverse_object_order(sample_context()),
            tuple(reverse_object_order(report) for report in sample_reports()),
        )
        for name in normal:
            with self.subTest(prompt=name):
                self.assertEqual(normal[name], reordered[name])

    def test_serialization_is_compact_and_keeps_unicode_and_array_order(self):
        value = {"z": [{"b": "区域甲", "a": 1}, {"d": 2, "c": 3}], "a": "雨"}
        serialized = prompts.prompt_json(value)
        self.assertEqual(serialized, '{"a":"雨","z":[{"a":1,"b":"区域甲"},{"c":3,"d":2}]}')
        self.assertEqual(json.loads(serialized), value)
        for name, prompt in render_all().items():
            with self.subTest(prompt=name):
                context = section_json(prompt)
                self.assertEqual(context["zone_id"], "区域甲")
                self.assertEqual([window["window_start"] for window in context["pricing_windows_3h"]],
                                 ["2024-02-01 00:00:00", "2024-02-01 03:00:00"])

    def test_context_serialization_handles_empty_and_missing_feedback(self):
        for context in ({}, {"retry_feedback": []}, {"zone_id": "甲"},
                        {"retry_feedback": None, "zone_id": "甲"}):
            with self.subTest(context=context):
                self.assertEqual(json.loads(prompts.context_json(context)), context)

    def test_all_prompts_are_identical_across_hash_seeds(self):
        script = (
            "import hashlib, json, sys; "
            "sys.path.insert(0, 'tests'); "
            "from test_prompt_caching import render_all; "
            "value = json.dumps(render_all(), ensure_ascii=False, sort_keys=True); "
            "print(hashlib.sha256(value.encode('utf-8')).hexdigest())"
        )
        digests = []
        for seed in ("1", "42", "123456"):
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True, text=True, check=True, timeout=20,
            )
            digests.append(result.stdout.strip())
        self.assertEqual(len(set(digests)), 1, digests)
        self.assertEqual(len(digests[0]), 64)

    def test_forecast_horizon_is_context_data_and_does_not_change_static_prefix(self):
        short = sample_context()
        long = deepcopy(short)
        long.update(forecast_horizon_days=7, forecast_horizon_hours=168)
        longer_prompts = render_all(long)
        for name, prompt in render_all(short).items():
            with self.subTest(prompt=name):
                other = longer_prompts[name]
                self.assertEqual(prompt.split("Context:\n", 1)[0], other.split("Context:\n", 1)[0])
                self.assertEqual(section_json(prompt)["forecast_horizon_days"], 2)
                self.assertEqual(section_json(other)["forecast_horizon_days"], 7)

    def test_updated_forecasts_reach_every_prompt_without_freezing_context(self):
        context = sample_context()
        changed = deepcopy(context)
        changed["forecast_total_kwh"] = 87.0
        changed["pricing_windows_3h"][0]["sum_predicted_kwh"] = 22.0
        normal, updated = render_all(context), render_all(changed)
        for name, prompt in normal.items():
            with self.subTest(prompt=name):
                other = updated[name]
                self.assertNotEqual(prompt, other)
                self.assertEqual(prompt.split("Context:\n", 1)[0], other.split("Context:\n", 1)[0])
                self.assertEqual(section_json(other)["forecast_total_kwh"], 87.0)
                self.assertEqual(section_json(other)["pricing_windows_3h"][0]["sum_predicted_kwh"], 22.0)

    def test_discussion_variations_follow_context_and_current_reports(self):
        first = render_all(discussion_round=1)
        reports = list(sample_reports())
        reports[-1] = {"conclusion_summary": "revised", "key_decisions": ["raise price"],
                       "disagreements": []}
        last = render_all(reports=reports, discussion_round=3)
        for name in ("discussion_grid", "discussion_behavior", "discussion_economist"):
            with self.subTest(prompt=name):
                prompt, other = first[name], last[name]
                prefix = prompt.split("Discussion round:\n", 1)[0]
                self.assertEqual(prefix, other.split("Discussion round:\n", 1)[0])
                self.assertEqual(section_json(prompt, "Discussion round"), 1)
                self.assertEqual(section_json(other, "Discussion round"), 3)
                self.assertEqual(section_json(other, "Previous round compact handoff"), reports[-1])
                self.assertLess(prompt.index("Context:\n"), prompt.index("Discussion round:\n"))
                if name != "discussion_grid":
                    self.assertLess(prompt.index("Grid report:\n"), prompt.index("Discussion round:\n"))
                if name == "discussion_economist":
                    self.assertLess(prompt.index("Behaviour report:\n"), prompt.index("Discussion round:\n"))
                static_rules = prompt.split("Context:\n", 1)[0]
                for field in ("agreements", "disagreements", "revisions_from_prior_round",
                              "message_to_other_agents", "conclusion_summary", "key_decisions"):
                    self.assertIn(field, static_rules)

    def test_discussion_without_previous_exchange_keeps_fixed_rules(self):
        context = sample_context()
        first = prompts.discussion_grid_prompt(context, discussion_round=1, previous_exchange=None)
        later = prompts.discussion_grid_prompt(context, discussion_round=2, previous_exchange=sample_reports()[-1])
        self.assertEqual(first.split("Discussion round:\n", 1)[0], later.split("Discussion round:\n", 1)[0])
        handoff = section_json(first, "Previous round compact handoff")
        self.assertIn("No previous exchange", handoff["status"])

    def test_repair_errors_only_change_tail(self):
        first = render_all(errors=["missing price"])["repair"]
        later = render_all(errors=["wrong count", "missing action_label"])["repair"]
        self.assertEqual(first.split("Validation errors:\n", 1)[0], later.split("Validation errors:\n", 1)[0])
        self.assertEqual(section_json(later, "Validation errors"), ["wrong count", "missing action_label"])
        self.assertLess(first.index("Behaviour report:\n"), first.index("Validation errors:\n"))
        self.assertLess(first.index("Context:\n"), first.index("Expected window count:\n"))
        self.assertEqual(section_json(first, "Expected window count"), 2)
        instructions = first.split("Context:\n", 1)[0]
        self.assertIn("exactly as many items as context.pricing_windows_3h", instructions)
        self.assertIn("same order", instructions)

    def test_repair_expected_count_is_dynamic_after_static_rules(self):
        context = sample_context()
        one_window = deepcopy(context)
        one_window["pricing_windows_3h"] = one_window["pricing_windows_3h"][:1]
        first, later = render_all(context)["repair"], render_all(one_window)["repair"]
        self.assertEqual(first.split("Context:\n", 1)[0], later.split("Context:\n", 1)[0])
        self.assertEqual(section_json(later, "Expected window count"), 1)

    def test_previous_report_changes_do_not_change_context_or_current_reports(self):
        reports = list(sample_reports())
        changed_reports = deepcopy(reports)
        changed_reports[2]["price_rationale"] = "新的建议"
        first, later = render_all(reports=reports), render_all(reports=changed_reports)
        for name in ("repair", "retry", "single_retry"):
            with self.subTest(prompt=name):
                label = "Previous invalid response" if name == "repair" else "Previous report"
                self.assertEqual(first[name].split(label + ":\n", 1)[0],
                                 later[name].split(label + ":\n", 1)[0])
                self.assertEqual(section_json(later[name], label)["price_rationale"], "新的建议")

    def test_retry_feedback_is_last_context_field_and_reuses_earlier_prefix(self):
        context = sample_context()
        changed = deepcopy(context)
        changed["retry_feedback"][0].update(attempt=2, previous_energy_price=0.7,
                                            failure_reason="load remains Low")
        normal, retried = render_all(context), render_all(changed)
        for name, prompt in normal.items():
            with self.subTest(prompt=name):
                other = retried[name]
                self.assertEqual(prompt.split(',"retry_feedback":', 1)[0],
                                 other.split(',"retry_feedback":', 1)[0])
                parsed = section_json(other)
                self.assertEqual(list(parsed)[-1], "retry_feedback")
                self.assertEqual(parsed["retry_feedback"][0]["attempt"], 2)
                self.assertEqual(parsed["retry_feedback"][0]["previous_energy_price"], 0.7)
                self.assertEqual(parsed["retry_feedback"][0]["failure_reason"], "load remains Low")

    def test_future_actuals_and_evaluation_fields_stay_out_of_context(self):
        context = sample_context()
        context.update(actual_future_load="LEAK_TOP_LEVEL", evaluation_labels="LEAK_LABEL")
        context["hourly_forecast"][0].update(actual_kwh="LEAK_HOURLY", error="LEAK_ERROR")
        context["hourly_averages"].update(mean_actual_kwh="LEAK_AVERAGE", mean_abs_pct_error="LEAK_AVG_ERROR")
        context["pricing_windows_3h"][0].update(actual_load_kwh="LEAK_WINDOW", stress_correct="LEAK_STRESS")
        context["retry_feedback"][0].update(actual_load_kwh="LEAK_RETRY", evaluation="LEAK_EVALUATION")
        original = deepcopy(context)
        for name, prompt in render_all(context).items():
            with self.subTest(prompt=name):
                self.assertNotIn("LEAK_", prompt)
                self.assertEqual(section_json(prompt)["pricing_windows_3h"][0]["sum_predicted_kwh"], 9.0)
        self.assertEqual(context, original)

    def test_existing_report_sections_and_business_rules_are_preserved(self):
        rendered = render_all()
        grid, behavior, previous, _ = sample_reports()
        for name in ("behavior", "economist", "discussion_behavior", "discussion_economist",
                     "repair", "retry", "single_retry"):
            with self.subTest(prompt=name):
                self.assertEqual(section_json(rendered[name], "Grid report"), grid)
        for name in ("economist", "discussion_economist", "repair", "retry", "single_retry"):
            with self.subTest(prompt=name):
                self.assertEqual(section_json(rendered[name], "Behaviour report"), behavior)
        for name in ("repair", "retry", "single_retry"):
            label = "Previous invalid response" if name == "repair" else "Previous report"
            self.assertEqual(section_json(rendered[name], label), previous)
        for name in ("grid", "single", "discussion_grid"):
            static = rendered[name].split("Context:\n", 1)[0]
            for expected in ("35%", "80%", "90%", "Low", "Medium", "High", "Extremely High",
                             "reasoning_summary", "window_assessments", "confidence"):
                self.assertIn(expected, static, name)
        for name in ("economist", "single", "discussion_economist", "repair", "retry", "single_retry"):
            static = rendered[name].split("Context:\n", 1)[0]
            for key in ("reasoning_summary", "suggested_price_shift_pct", "action_label", "price_rationale",
                        "price_change_windows_3h", "window_start", "window_end", "proposed_energy_price"):
                self.assertIn(key, static, name)
        for name in ("retry", "single_retry"):
            static = " ".join(rendered[name].split("Context:\n", 1)[0].split())
            self.assertIn("non-failed window's previous price unchanged", static)
            self.assertIn("non-negative", static)
            self.assertIn("original order", static)


if __name__ == "__main__":
    unittest.main()
