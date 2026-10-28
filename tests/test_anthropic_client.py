"""Native Anthropic request/response contracts using only offline SDK doubles."""
from __future__ import annotations

from contextlib import asynccontextmanager
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2

from agents import AGENT_COMPLETION_USAGE_KEY, AgentChatClient, ProviderResponseDecodeError, complete_agent_json
from anthropic_backend import anthropic_base_url, anthropic_message_content, anthropic_request
from config import AgentConfig
from orchestrator import build_zone_attempt_agent_usage, cumulative_zone_agent_usage
from prompts import SYSTEM_MESSAGE


SIMPLE_OUTPUT_SCHEMA = {
    "type": "object", "properties": {"reasoning_summary": {"type": "string"}},
    "required": ["reasoning_summary"], "additionalProperties": False,
}


def native_response(**changes):
    return {
        "id": "msg_offline", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{"type": "text", "text": '{"reasoning_summary":"ok"}'}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 20, "cache_creation_input_tokens": 30,
                  "cache_read_input_tokens": 50, "output_tokens": 10},
        **changes,
    }


def config(**changes):
    return AgentConfig(**{
        "backend": "anthropic", "api_key": "offline-key", "base_url": "https://api.anthropic.com",
        "model": "claude-test", "provider_json_retry_backoff_seconds": 0, **changes,
    })


def status_error(status, **headers):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://native.invalid/v1/messages"), headers=headers)
    return anthropic.APIStatusError("offline error", response=response, body={"error": {"message": "offline error"}})


class AnthropicRequestTests(unittest.TestCase):
    def test_effort_maps_to_native_controls_without_openai_parameters(self):
        for effort in (None, "none", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                request = anthropic_request(config(reasoning_effort=effort), "same prompt", system="system")
                self.assertEqual(request["messages"], [{"role": "user", "content": "same prompt"}])
                self.assertEqual(request["system"], "system")
                self.assertEqual(request["max_tokens"], 8192)
                self.assertNotIn("response_format", request)
                self.assertNotIn("temperature", request)
                self.assertNotIn("reasoning_effort", request)
                self.assertNotIn("chat_template_kwargs", request)
                if effort is None:
                    self.assertNotIn("thinking", request)
                    self.assertNotIn("output_config", request)
                elif effort == "none":
                    self.assertEqual(request["thinking"], {"type": "disabled"})
                    self.assertNotIn("output_config", request)
                else:
                    self.assertEqual(request["thinking"], {"type": "adaptive"})
                    self.assertEqual(request["output_config"], {"effort": effort})

    def test_explicit_thinking_cache_and_output_budget_are_copied(self):
        for thinking, effort in (({"type": "disabled"}, "high"),
                                 ({"type": "adaptive"}, "none"),
                                 ({"type": "enabled", "budget_tokens": 2048}, None)):
            with self.subTest(thinking=thinking, effort=effort):
                settings = config(reasoning_effort=effort, anthropic_thinking=thinking,
                                  anthropic_cache_control={"type": "ephemeral", "ttl": "1h"}, max_tokens=4096)
                before = copy.deepcopy(settings.anthropic_thinking)
                request = anthropic_request(settings, "prompt", system="system")
                self.assertEqual(request["thinking"], before)
                self.assertEqual(request["max_tokens"], 4096)
                self.assertEqual(request["cache_control"], {"type": "ephemeral", "ttl": "1h"})
                request["thinking"]["type"] = "changed"
                request["cache_control"]["ttl"] = "5m"
                self.assertEqual(settings.anthropic_thinking, before)
                self.assertEqual(thinking, before)
                self.assertEqual(settings.anthropic_cache_control["ttl"], "1h")

    def test_base_url_normalizes_only_version_suffix_preserving_gateway(self):
        for given, expected in (
            ("https://native.invalid", "https://native.invalid"),
            ("https://native.invalid/", "https://native.invalid"),
            ("https://native.invalid/v1/", "https://native.invalid"),
            ("https://native.invalid/gateway/v1", "https://native.invalid/gateway"),
            ("https://native.invalid/gateway", "https://native.invalid/gateway"),
        ):
            with self.subTest(given=given):
                self.assertEqual(anthropic_base_url(given), expected)
        with self.assertRaisesRegex(ValueError, "messages endpoint"):
            anthropic_base_url("https://native.invalid/v1/messages")

    def test_text_blocks_only_support_both_native_dicts_and_sdk_objects(self):
        blocks = [{"type": "thinking", "thinking": '{"wrong":true}'},
                  {"type": "redacted_thinking", "data": "redacted"},
                  {"type": "text", "text": '{"answer":'},
                  {"type": "text", "text": '42}'}]
        for content in (blocks, [SimpleNamespace(**item) for item in blocks]):
            self.assertEqual(anthropic_message_content(native_response(content=content), requested_model="claude-test"), '{"answer":42}')

    def test_incomplete_and_non_final_responses_are_not_accepted_as_json(self):
        for stop in ("max_tokens", "refusal", "tool_use", "pause_turn"):
            with self.subTest(stop=stop), self.assertRaisesRegex(RuntimeError, stop):
                anthropic_message_content(native_response(stop_reason=stop), requested_model="claude-test")
        for content in ([], None, [{"type": "thinking", "thinking": "not final"}],
                        [{"type": "text", "text": "  "}], [{"type": "text", "text": 1}]):
            with self.subTest(content=content), self.assertRaisesRegex(RuntimeError, "no final text"):
                anthropic_message_content(native_response(content=content), requested_model="claude-test")


class AnthropicClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, **settings):
        sdk = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=native_response())), close=AsyncMock())
        with patch("anthropic.AsyncAnthropic", return_value=sdk) as factory, \
                patch("openai.AsyncOpenAI", side_effect=AssertionError("OpenAI must not be used")):
            client = AgentChatClient(config(**settings))
        return client, sdk, factory

    async def test_native_sdk_dispatch_default_budget_and_shutdown(self):
        client, sdk, factory = self.client(base_url="https://native.invalid/v1", timeout_seconds=42)
        self.assertEqual(factory.call_args.kwargs, {
            "api_key": "offline-key", "base_url": "https://native.invalid", "timeout": 42, "max_retries": 0,
        })
        async with client:
            payload = await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
        await client.aclose()
        sdk.close.assert_awaited_once()
        request = sdk.messages.create.call_args.kwargs
        self.assertEqual(request["system"], SYSTEM_MESSAGE)
        self.assertEqual(request["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(request["max_tokens"], 8192)
        self.assertNotIn("temperature", request)
        self.assertNotIn("response_format", request)
        usage = payload[AGENT_COMPLETION_USAGE_KEY]
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]), (100, 10, 110))
        self.assertEqual(usage["cached_tokens"], 50)
        self.assertEqual(usage["cache_hit_ratio"], .5)
        self.assertEqual(usage["provider_attempt_count"], 1)
        self.assertTrue(usage["token_usage_complete"])

    async def test_actual_sdk_serialization_headers_paths_and_native_thinking_blocks(self):
        original_factory = anthropic.AsyncAnthropic
        for base_url, expected_path in (("https://native.invalid", "/v1/messages"),
                                        ("https://native.invalid/v1/", "/v1/messages"),
                                        ("https://native.invalid/gateway/v1", "/gateway/v1/messages")):
            with self.subTest(base_url=base_url):
                requests = []
                clients = []

                async def handle(request):
                    requests.append(request)
                    return httpx2.Response(200, json=native_response(content=[
                        {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
                        {"type": "redacted_thinking", "data": "redacted"},
                        {"type": "text", "text": '{"reasoning_summary":"ok"}'},
                    ]))

                def make_client(**options):
                    transport_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
                    clients.append(transport_client)
                    return original_factory(http_client=transport_client, **options)

                with patch("anthropic.AsyncAnthropic", side_effect=make_client) as factory, \
                        patch("openai.AsyncOpenAI", side_effect=AssertionError("OpenAI must not be used")):
                    client = AgentChatClient(config(base_url=base_url, reasoning_effort="high",
                                                    anthropic_cache_control={"type": "ephemeral"}))
                async with client:
                    payload = await client.complete_json("fixed prompt", temperature=.7, output_schema=SIMPLE_OUTPUT_SCHEMA)
                self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
                self.assertTrue(clients[0].is_closed)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].url.path, expected_path)
                self.assertEqual(requests[0].headers["authorization"], "Bearer offline-key")
                self.assertIn("anthropic-version", requests[0].headers)
                body = json.loads(requests[0].content)
                self.assertEqual(body["system"], SYSTEM_MESSAGE)
                self.assertEqual(body["messages"], [{"role": "user", "content": "fixed prompt"}])
                self.assertEqual(body["thinking"], {"type": "adaptive"})
                self.assertEqual(body["output_config"]["effort"], "high")
                self.assertEqual(body["output_config"]["format"], {"type": "json_schema", "schema": SIMPLE_OUTPUT_SCHEMA})
                self.assertEqual(body["cache_control"], {"type": "ephemeral"})
                for absent in ("response_format", "temperature", "reasoning_effort", "extra_body"):
                    self.assertNotIn(absent, body)
                self.assertEqual(payload["reasoning_summary"], "ok")
                self.assertNotIn("thinking", payload)
                self.assertEqual(payload[AGENT_COMPLETION_USAGE_KEY]["cached_tokens"], 50)

    async def test_native_usage_survives_existing_agent_attempt_and_cumulative_layers(self):
        client, _, _ = self.client()
        async with client:
            result = await complete_agent_json(client, "prompt", context={"zone_id": "A1"},
                                               temperature=.2, stage="agent.grid", agent="Grid")
        self.assertNotIn(AGENT_COMPLETION_USAGE_KEY, result.content)
        attempt_usage, calls = build_zone_attempt_agent_usage(
            [result.usage], attempt=1, proposal_phase="initial", triggered_by_attempt=None)
        cumulative, cumulative_calls = cumulative_zone_agent_usage([
            {"agent_usage": attempt_usage, "agent_call_usage": calls}])
        self.assertEqual(cumulative["prompt_tokens"], 100)
        self.assertEqual(cumulative["total_tokens"], 110)
        self.assertEqual(cumulative["cached_tokens"], 50)
        self.assertEqual(cumulative["cache_hit_ratio"], .5)
        self.assertEqual(cumulative_calls[0]["cached_tokens"], 50)

    async def test_partial_native_usage_stays_incomplete(self):
        client, sdk, _ = self.client()
        sdk.messages.create.return_value = native_response(usage={"input_tokens": 20, "output_tokens": 10})
        async with client:
            payload = await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
        usage = payload[AGENT_COMPLETION_USAGE_KEY]
        self.assertIsNone(usage["prompt_tokens"])
        self.assertIsNone(usage["total_tokens"])
        self.assertIsNone(usage["cached_tokens"])
        self.assertFalse(usage["token_usage_complete"])
        self.assertFalse(usage["cache_usage_complete"])

    async def test_manual_retries_cover_decode_rate_server_connection_and_timeout(self):
        request = httpx2.Request("POST", "https://native.invalid/v1/messages")
        failures = [json.JSONDecodeError("truncated", "", 0),
                    *(status_error(status) for status in (408, 409, 429, 500, 503)),
                    anthropic.APIConnectionError(request=request), anthropic.APITimeoutError(request=request)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__, status=getattr(failure, "status_code", None)):
                client, sdk, _ = self.client(reasoning_effort="low")
                sdk.messages.create.side_effect = [failure, native_response()]
                async with client:
                    payload = await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
                self.assertEqual(sdk.messages.create.await_count, 2)
                self.assertEqual(sdk.messages.create.call_args_list[0], sdk.messages.create.call_args_list[1])
                usage = payload[AGENT_COMPLETION_USAGE_KEY]
                self.assertEqual(usage["provider_attempt_count"], 2)
                self.assertFalse(usage["token_usage_complete"])
                self.assertFalse(usage["cache_usage_complete"])
                self.assertIsNone(usage["cached_tokens"])
                self.assertEqual(usage["known_cached_tokens"], 50)

    async def test_permanent_errors_and_server_no_retry_directive_are_not_retried(self):
        for failure in (status_error(400), status_error(401), status_error(429, **{"x-should-retry": "false"})):
            with self.subTest(status=failure.status_code):
                client, sdk, _ = self.client()
                sdk.messages.create.side_effect = failure
                with self.assertRaises(anthropic.APIStatusError):
                    async with client:
                        await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
                self.assertEqual(sdk.messages.create.await_count, 1)
                sdk.close.assert_awaited_once()

    async def test_explicit_retry_directive_and_exhaustion_respect_attempt_budget(self):
        client, sdk, _ = self.client(provider_json_retries=1)
        sdk.messages.create.side_effect = [status_error(400, **{"x-should-retry": "true"}), native_response()]
        async with client:
            payload = await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
        self.assertEqual(payload[AGENT_COMPLETION_USAGE_KEY]["provider_attempt_count"], 2)
        for failure, expected_error in ((status_error(503), anthropic.APIStatusError),
                                        (json.JSONDecodeError("truncated", "", 0), ProviderResponseDecodeError)):
            with self.subTest(error=type(failure).__name__):
                client, sdk, _ = self.client(provider_json_retries=1)
                sdk.messages.create.side_effect = failure
                with self.assertRaises(expected_error):
                    async with client:
                        await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
                self.assertEqual(sdk.messages.create.await_count, 2)

    async def test_shared_limiter_wraps_every_retry_and_no_key_blocks_client_creation(self):
        events = []

        class Limiter:
            @asynccontextmanager
            async def slot(self):
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")

        with patch("agents.SharedRequestLimiter", return_value=Limiter()):
            client, sdk, _ = self.client(max_concurrent_requests_total=2)

        async def create(**request):
            self.assertEqual(events[-1], "enter")
            events.append("request")
            if events.count("request") == 1:
                raise status_error(429)
            return native_response()

        sdk.messages.create.side_effect = create
        async with client:
            await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
        self.assertEqual(events, ["enter", "request", "exit", "enter", "request", "exit"])
        with patch("anthropic.AsyncAnthropic") as factory:
            with self.assertRaisesRegex(ValueError, "api_key"):
                AgentChatClient(config(api_key=None))
            factory.assert_not_called()

    async def test_truncated_or_refused_completion_fails_without_provider_retry(self):
        for stop in ("max_tokens", "refusal", "tool_use", "pause_turn"):
            with self.subTest(stop=stop):
                client, sdk, _ = self.client()
                sdk.messages.create.return_value = native_response(stop_reason=stop)
                with self.assertRaisesRegex(RuntimeError, stop):
                    async with client:
                        await client.complete_json("prompt", temperature=.2, output_schema=SIMPLE_OUTPUT_SCHEMA)
                self.assertEqual(sdk.messages.create.await_count, 1)


if __name__ == "__main__":
    unittest.main()
