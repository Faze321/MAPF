"""Offline native structured-output conversion and Agent-stage dispatch tests."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents import AgentChatClient, AgentStageError, complete_agent_json
from anthropic_backend import anthropic_request
from test_anthropic_client import SIMPLE_OUTPUT_SCHEMA, config, native_response


def bounded_schema():
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "price": {"type": "number", "minimum": 0, "maximum": 100},
            "load_band": {"type": "string", "enum": ["Low", "Medium", "High", "Extremely High"]},
        },
        "required": ["price", "load_band"],
    }


class AnthropicSchemaConversionTests(unittest.TestCase):
    def test_json_object_becomes_native_format_with_effort_and_cache(self):
        source = bounded_schema()
        before = copy.deepcopy(source)
        request = anthropic_request(
            config(reasoning_effort="high", anthropic_cache_control={"type": "ephemeral", "ttl": "5m"}),
            "prompt", system="system", response_format={"type": "json_object"}, output_schema=source,
        )
        self.assertEqual(source, before)
        self.assertNotIn("response_format", request)
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"]["effort"], "high")
        self.assertEqual(request["cache_control"], {"type": "ephemeral", "ttl": "5m"})
        native = request["output_config"]["format"]
        self.assertEqual(native["type"], "json_schema")
        self.assertFalse(native["schema"]["additionalProperties"])
        self.assertEqual(set(native["schema"]["properties"]), {"price", "load_band"})
        self.assertEqual(native["schema"]["properties"]["load_band"]["enum"], before["properties"]["load_band"]["enum"])
        # Anthropic's supported JSON schema subset retains bounds as descriptive
        # guidance; the client must use SDK transformation instead of sending
        # unsupported minimum/maximum keywords to the API.
        price = native["schema"]["properties"]["price"]
        self.assertNotIn("minimum", price)
        self.assertNotIn("maximum", price)
        self.assertIn("100", price.get("description", ""))
        json.dumps(request, allow_nan=False)

    def test_openai_json_schema_envelope_is_unwrapped_without_mutation(self):
        fmt = {"type": "json_schema", "json_schema": {
            "name": "pricing", "strict": True, "schema": bounded_schema(),
        }}
        original = copy.deepcopy(fmt)
        request = anthropic_request(config(), "prompt", system="system", response_format=fmt)
        self.assertEqual(fmt, original)
        converted = request["output_config"]["format"]
        self.assertEqual(converted["type"], "json_schema")
        self.assertEqual(set(converted), {"type", "schema"})
        self.assertEqual(converted["schema"]["required"], ["price", "load_band"])
        self.assertNotIn("response_format", request)
        self.assertNotIn("effort", request["output_config"])

    def test_json_object_cannot_fall_back_to_unconstrained_or_empty_schema(self):
        for schema in (None, {}, {"type": "object"}, {"type": "object", "properties": {}},
                       {"type": "array", "items": {"type": "string"}}):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                anthropic_request(config(), "arbitrary prompt", system="system",
                                  response_format={"type": "json_object"}, output_schema=schema)


class AnthropicStageSchemaRequestTests(unittest.IsolatedAsyncioTestCase):
    def client(self, **settings):
        sdk = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=native_response())), close=AsyncMock())
        with patch("anthropic.AsyncAnthropic", return_value=sdk):
            return AgentChatClient(config(**settings)), sdk

    async def request_for_stage(self, stage, *, windows=1, client=None, sdk=None):
        owns_client = client is None
        if owns_client:
            client, sdk = self.client()
        try:
            await complete_agent_json(client, "prompt", context={"zone_id": "A1", "pricing_windows_3h": [{}] * windows},
                                      temperature=.2, stage=stage, agent="test")
            return sdk.messages.create.call_args.kwargs
        finally:
            if owns_client:
                await client.aclose()

    async def test_direct_native_call_requires_schema_before_http(self):
        client, sdk = self.client()
        async with client:
            with self.assertRaisesRegex(ValueError, "schema"):
                await client.complete_json("arbitrary prompt", temperature=.2)
        sdk.messages.create.assert_not_awaited()

    async def test_direct_json_schema_native_call_merges_format_and_effort(self):
        client, sdk = self.client(reasoning_effort="low")
        async with client:
            await client.complete_json("prompt", temperature=.2, response_format={
                "type": "json_schema", "json_schema": {"name": "summary", "strict": True, "schema": SIMPLE_OUTPUT_SCHEMA},
            })
        request = sdk.messages.create.call_args.kwargs
        self.assertEqual(request["output_config"], {
            "effort": "low", "format": {"type": "json_schema", "schema": SIMPLE_OUTPUT_SCHEMA},
        })
        self.assertNotIn("response_format", request)

    async def test_every_agent_stage_and_repair_sends_meaningful_closed_schema(self):
        pricing = {"reasoning_summary", "suggested_price_shift_pct", "action_label", "price_rationale", "price_change_windows_3h"}
        stage_fields = {
            "agent.grid": {"reasoning_summary", "adjustment_needed", "confidence", "window_assessments"},
            "agent.behavior": {"reasoning_summary", "demand_drivers", "elasticity_factor", "window_elasticities", "confidence"},
            "agent.economist": pricing,
            "agent.economist_repair": pricing,
            "agent.single_agent": pricing | {"window_assessments", "demand_drivers", "elasticity_factor", "window_elasticities", "confidence"},
            "agent.single_agent.schema_repair": pricing,
            "agent.economist_retry": pricing,
            "agent.economist_retry.schema_repair": pricing,
            "agent.single_agent_price_retry": pricing,
            "agent.single_agent_price_retry.schema_repair": pricing,
            "agent.discussion.round_1.grid": {"window_assessments", "agreements", "disagreements", "message_to_other_agents"},
            "agent.discussion.round_2.behavior": {"window_elasticities", "agreements", "disagreements", "message_to_other_agents"},
            "agent.discussion.round_3.economist": pricing | {"agreements", "disagreements", "message_to_other_agents"},
            "agent.discussion.round_3.economist.schema_repair": pricing,
        }
        client, sdk = self.client()
        async with client:
            for stage, required in stage_fields.items():
                with self.subTest(stage=stage):
                    request = await self.request_for_stage(stage, client=client, sdk=sdk)
                    self.assertNotIn("response_format", request)
                    output = request["output_config"]["format"]
                    self.assertEqual(output["type"], "json_schema")
                    schema = output["schema"]
                    self.assertEqual(schema["type"], "object")
                    self.assertFalse(schema["additionalProperties"])
                    self.assertTrue(required.issubset(schema["properties"]))
                    self.assertTrue(required.issubset(schema["required"]))
                    if "price_change_windows_3h" in required:
                        window = schema["properties"]["price_change_windows_3h"]["items"]
                        self.assertFalse(window["additionalProperties"])
                        self.assertTrue({"window_start", "window_end", "proposed_energy_price"}.issubset(window["required"]))
                    if stage == "agent.grid":
                        assessment = schema["properties"]["window_assessments"]["items"]
                        self.assertEqual(assessment["properties"]["grid_stress_level"]["enum"], ["Low", "Medium", "High", "Extremely High"])
        self.assertEqual(sdk.messages.create.await_count, len(stage_fields))

    async def test_schema_is_stable_across_window_counts_and_discussion_numbers(self):
        client, sdk = self.client()
        async with client:
            first = await self.request_for_stage("agent.single_agent", windows=1, client=client, sdk=sdk)
            many = await self.request_for_stage("agent.single_agent", windows=16, client=client, sdk=sdk)
            self.assertEqual(first["output_config"]["format"], many["output_config"]["format"])
            for role in ("grid", "behavior", "economist"):
                first = await self.request_for_stage(f"agent.discussion.round_1.{role}", client=client, sdk=sdk)
                later = await self.request_for_stage(f"agent.discussion.round_9.{role}", windows=16, client=client, sdk=sdk)
                self.assertEqual(first["output_config"]["format"], later["output_config"]["format"])

    async def test_unknown_native_stage_is_rejected_before_http(self):
        client, sdk = self.client()
        async with client:
            with self.assertRaises(AgentStageError) as caught:
                await complete_agent_json(client, "prompt", context={}, temperature=.2, stage="agent.typo", agent="test")
        self.assertIsInstance(caught.exception.original, ValueError)
        sdk.messages.create.assert_not_awaited()

    async def test_custom_legacy_client_signature_is_not_changed(self):
        class LegacyClient:
            async def complete_json(self, prompt, *, temperature):
                return {"reasoning_summary": "custom"}

        result = await complete_agent_json(LegacyClient(), "prompt", context={}, temperature=.2,
                                           stage="custom.stage", agent="custom")
        self.assertEqual(result.content, {"reasoning_summary": "custom"})

    async def test_non_native_json_object_behavior_is_preserved(self):
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value={
            "choices": [{"message": {"content": '{"reasoning_summary":"ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }))), close=AsyncMock())
        with patch("openai.AsyncOpenAI", return_value=sdk):
            client = AgentChatClient(config(backend="openai_compatible"))
        async with client:
            await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
            await complete_agent_json(client, "prompt", context={}, temperature=.2, stage="custom.stage", agent="test")
        for call in sdk.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["response_format"], {"type": "json_object"})
            self.assertNotIn("output_config", call.kwargs)


if __name__ == "__main__":
    unittest.main()
