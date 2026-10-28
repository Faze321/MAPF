"""Offline schema contracts for the prompts used by native Agent requests."""
from __future__ import annotations

import json
import unittest

from agent_output_schemas import agent_output_schema

try:
    import jsonschema
except ImportError:
    jsonschema = None


PRICING_KEYS = {
    "reasoning_summary", "suggested_price_shift_pct", "action_label",
    "price_rationale", "price_change_windows_3h",
}
GRID_KEYS = {"reasoning_summary", "adjustment_needed", "confidence", "window_assessments"}
BEHAVIOR_KEYS = {"reasoning_summary", "demand_drivers", "elasticity_factor", "window_elasticities", "confidence"}
DISCUSSION_KEYS = {"agreements", "disagreements", "revisions_from_prior_round", "message_to_other_agents"}
BASE_STAGES = (
    "agent.grid", "agent.behavior", "agent.economist", "agent.economist_repair",
    "agent.single_agent", "agent.single_agent.schema_repair",
    "agent.economist_retry", "agent.economist_retry.schema_repair",
    "agent.single_agent_price_retry", "agent.single_agent_price_retry.schema_repair",
)
DISCUSSION_STAGES = tuple(
    f"agent.discussion.round_{round_number}.{role}"
    for round_number in (1, 3, 25)
    for role in ("grid", "behavior", "economist", "economist.schema_repair")
)


class AgentOutputSchemaTests(unittest.TestCase):
    def test_known_stage_schemas_are_recursively_closed_and_provider_compatible(self):
        def check(node):
            self.assertIsInstance(node, dict)
            self.assertTrue("type" in node or "anyOf" in node)
            self.assertFalse(set(node) & {"$ref", "minimum", "maximum", "minItems", "maxItems", "pattern", "format"})
            if node.get("type") == "object":
                self.assertIs(node["additionalProperties"], False)
                self.assertTrue(node["properties"])
                self.assertEqual(set(node["required"]), set(node["properties"]))
                self.assertEqual(len(node["required"]), len(set(node["required"])))
                for child in node["properties"].values():
                    check(child)
            if node.get("type") == "array":
                check(node["items"])
            for alternative in node.get("anyOf", []):
                check(alternative)

        for stage in BASE_STAGES + DISCUSSION_STAGES:
            with self.subTest(stage=stage):
                schema = agent_output_schema(stage)
                check(schema)
                self.assertEqual(json.loads(json.dumps(schema)), schema)
                if jsonschema is not None:
                    jsonschema.Draft202012Validator.check_schema(schema)

    def test_role_required_fields_match_prompt_contracts(self):
        for stage, expected in (
            ("agent.grid", GRID_KEYS),
            ("agent.behavior", BEHAVIOR_KEYS),
            ("agent.economist", PRICING_KEYS),
            ("agent.single_agent", PRICING_KEYS | BEHAVIOR_KEYS | {"window_assessments"}),
        ):
            with self.subTest(stage=stage):
                schema = agent_output_schema(stage)
                self.assertEqual(set(schema["properties"]), expected)
                self.assertEqual(set(schema["required"]), expected)
        self.assertNotIn("adjustment_needed", agent_output_schema("agent.single_agent")["required"])

    def test_all_repairs_and_price_retries_require_only_pricing(self):
        pricing = agent_output_schema("agent.economist")
        stages = (
            "agent.economist_repair", "agent.single_agent.schema_repair",
            "agent.economist_retry", "agent.economist_retry.schema_repair",
            "agent.single_agent_price_retry", "agent.single_agent_price_retry.schema_repair",
            "agent.discussion.round_3.economist.schema_repair",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertEqual(agent_output_schema(stage), pricing)

    def test_discussion_extras_are_only_in_nonrepair_discussion_and_round_stable(self):
        for role in ("grid", "behavior", "economist"):
            with self.subTest(role=role):
                base = agent_output_schema(f"agent.{role}")
                first = agent_output_schema(f"agent.discussion.round_1.{role}")
                later = agent_output_schema(f"agent.discussion.round_25.{role}")
                self.assertEqual(first, later)
                self.assertEqual(set(first["required"]), set(base["required"]) | DISCUSSION_KEYS)
                for key in ("agreements", "disagreements", "revisions_from_prior_round"):
                    self.assertEqual(first["properties"][key]["items"], {"type": "string"})
                self.assertEqual(first["properties"]["message_to_other_agents"], {"type": "string"})
                self.assertFalse(DISCUSSION_KEYS & set(base["properties"]))

    def test_window_fields_and_confidence_preserve_business_meanings(self):
        fields = agent_output_schema("agent.single_agent")["properties"]
        price = fields["price_change_windows_3h"]["items"]
        self.assertEqual(set(price["required"]), {
            "window_start", "window_end", "suggested_price_shift_pct", "proposed_energy_price", "action_label", "price_rationale",
        })
        assessment = fields["window_assessments"]["items"]
        self.assertEqual(set(assessment["required"]), {
            "window_start", "window_end", "predicted_load_kwh", "load_range_position_pct",
            "grid_stress_level", "adjustment_needed", "reasoning_summary",
        })
        self.assertEqual(assessment["properties"]["grid_stress_level"]["enum"], ["Low", "Medium", "High", "Extremely High"])
        self.assertEqual({item["type"] for item in fields["confidence"]["anyOf"]}, {"string", "number"})
        self.assertEqual({item["type"] for item in assessment["properties"]["load_range_position_pct"]["anyOf"]}, {"number", "null"})
        self.assertEqual(fields["demand_drivers"]["items"], {"type": "string"})
        self.assertEqual(set(fields["window_elasticities"]["items"]["required"]), {"window_start", "window_end", "elasticity_factor"})

    def test_results_are_independent_and_nested_mutation_does_not_change_later_requests(self):
        expected = agent_output_schema("agent.single_agent")
        changed = agent_output_schema("agent.single_agent")
        changed["required"].clear()
        changed["properties"]["confidence"]["anyOf"][0]["type"] = "boolean"
        changed["properties"]["window_assessments"]["items"]["properties"]["grid_stress_level"]["enum"].append("Invalid")
        self.assertEqual(agent_output_schema("agent.single_agent"), expected)
        self.assertEqual(agent_output_schema("agent.grid")["properties"]["confidence"]["anyOf"][0]["type"], "string")

    def test_unknown_stages_never_guess_a_generic_or_wrong_schema(self):
        for stage in ("", "grid", "agent.unknown", "agent.unknown.schema_repair", "agent.discussion.round_0.grid",
                      "agent.discussion.round_1.single_agent", "agent.discussion.round_1.economist.extra", None):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, "Unknown Agent output stage"):
                    agent_output_schema(stage)

    @unittest.skipIf(jsonschema is None, "jsonschema is optional and is not installed")
    def test_canonical_single_and_discussion_samples_validate_when_jsonschema_available(self):
        sample = {
            "reasoning_summary": "Night demand is low.", "confidence": 0.8,
            "window_assessments": [{
                "window_start": "2023-01-01 00:00:00", "window_end": "2023-01-01 02:00:00",
                "predicted_load_kwh": 30, "load_range_position_pct": None,
                "grid_stress_level": "Low", "adjustment_needed": False, "reasoning_summary": "Use the forecast.",
            }],
            "demand_drivers": ["Low overnight occupancy"], "elasticity_factor": 0.5,
            "window_elasticities": [{
                "window_start": "2023-01-01 00:00:00", "window_end": "2023-01-01 02:00:00", "elasticity_factor": 0.5,
            }],
            "suggested_price_shift_pct": -10, "action_label": "reduce", "price_rationale": "Encourage charging.",
            "price_change_windows_3h": [{
                "window_start": "2023-01-01 00:00:00", "window_end": "2023-01-01 02:00:00",
                "suggested_price_shift_pct": -10, "proposed_energy_price": 0.9,
                "action_label": "reduce", "price_rationale": "Encourage charging.",
            }],
        }
        jsonschema.validate(sample, agent_output_schema("agent.single_agent"))
        pricing = {key: value for key, value in sample.items() if key in PRICING_KEYS}
        jsonschema.validate(pricing, agent_output_schema("agent.single_agent.schema_repair"))
        discussion = {key: value for key, value in sample.items() if key in BEHAVIOR_KEYS}
        discussion.update(confidence="medium", agreements=["Use baseline loads"], disagreements=[],
                          revisions_from_prior_round=[], message_to_other_agents="Reduce overnight prices.")
        jsonschema.validate(discussion, agent_output_schema("agent.discussion.round_2.behavior"))
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({**sample, "unknown": True}, agent_output_schema("agent.single_agent"))


if __name__ == "__main__":
    unittest.main()
