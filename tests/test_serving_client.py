"""Offline request/usage integration for local serving and existing cloud APIs."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents import (AGENT_COMPLETION_USAGE_KEY, AgentChatClient, agent_call_usage_record,
                    complete_agent_json, request_extra_body, response_token_usage)
from config import AgentConfig
from orchestrator import build_zone_attempt_agent_usage, cumulative_zone_agent_usage


def response(cached=60):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"reasoning_summary":"ok"}'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


class ServingClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, **settings):
        config = AgentConfig(**{
            "api_key": None, "base_url": "http://localhost:8000/v1", "model": "served-qwen",
            "backend": "vllm", "provider_json_retry_backoff_seconds": 0, **settings,
        })
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response()))),
                              close=AsyncMock())
        with patch("openai.AsyncOpenAI", return_value=sdk) as factory:
            client = AgentChatClient(config)
        return client, sdk, factory

    async def test_local_backends_send_selected_template_options_and_output_budget(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                client, sdk, factory = self.client(
                    backend=backend, chat_template_kwargs={"enable_thinking": False}, max_tokens=4096)
                async with client:
                    payload = await client.complete_json("fixed prompt", temperature=0.2)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.kwargs["api_key"], "EMPTY")
                request = sdk.chat.completions.create.call_args.kwargs
                self.assertEqual(request["extra_body"], {
                    "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
                self.assertEqual(request["max_tokens"], 4096)
                self.assertEqual(request["response_format"], {"type": "json_object"})
                self.assertEqual(request["messages"][-1]["content"], "fixed prompt")
                self.assertEqual(payload["reasoning_summary"], "ok")
                self.assertEqual(payload[AGENT_COMPLETION_USAGE_KEY]["cached_tokens"], 60)
                self.assertEqual(payload[AGENT_COMPLETION_USAGE_KEY]["cache_hit_ratio"], 0.6)
                sdk.close.assert_awaited_once()

    async def test_local_backends_forward_effort_and_default_thinking_switch(self):
        for backend in ("vllm", "sglang"):
            for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(backend=backend, effort=effort):
                    client, sdk, _ = self.client(backend=backend, reasoning_effort=effort)
                    async with client:
                        await client.complete_json("prompt", temperature=0.2)
                    request = sdk.chat.completions.create.call_args.kwargs
                    self.assertEqual(request["extra_body"], {
                        "reasoning_effort": effort,
                        "chat_template_kwargs": {"enable_thinking": effort != "none"},
                    })
                    self.assertNotIn("max_tokens", request)

    def test_explicit_thinking_switch_wins_without_mutating_template_options(self):
        for backend in ("vllm", "sglang"):
            for effort, flag in (("none", True), ("high", False)):
                with self.subTest(backend=backend, effort=effort, flag=flag):
                    options = {"enable_thinking": flag, "nested": {"value": 1}}
                    config = AgentConfig(api_key=None, base_url="http://localhost/v1", model="local",
                                         backend=backend, reasoning_effort=effort, chat_template_kwargs=options)
                    body = request_extra_body(config)
                    self.assertEqual(body, {"reasoning_effort": effort, "chat_template_kwargs": options})
                    body["chat_template_kwargs"]["nested"]["value"] = 2
                    body["chat_template_kwargs"]["enable_thinking"] = not flag
                    self.assertEqual(options, {"enable_thinking": flag, "nested": {"value": 1}})
                    self.assertEqual(config.chat_template_kwargs, options)

    def test_local_server_defaults_do_not_inject_reasoning_options(self):
        for backend in ("vllm", "sglang"):
            for effort in (None, "", "   "):
                for options in ({}, {"enable_thinking": True, "custom": "selected"}):
                    with self.subTest(backend=backend, effort=effort, options=options):
                        config = AgentConfig(api_key=None, base_url="http://localhost/v1", model="local",
                                             backend=backend, reasoning_effort=effort,
                                             chat_template_kwargs=options)
                        expected = {"chat_template_kwargs": options} if options else None
                        self.assertEqual(request_extra_body(config), expected)

    async def test_local_extensions_are_top_level_on_http_wire_and_cache_is_preserved(self):
        import httpx
        from openai import AsyncOpenAI

        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                received = []

                async def handle(request):
                    received.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        "id": "offline-test", "object": "chat.completion", "created": 0,
                        "model": "served-qwen",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": '{"reasoning_summary":"ok"}'}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                                  "prompt_tokens_details": {"cached_tokens": 60}},
                    })

                sdk = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
                config = AgentConfig(api_key=None, base_url="http://localhost:8000/v1", model="served-qwen",
                                     backend=backend, reasoning_effort="high")
                with patch("openai.AsyncOpenAI", return_value=sdk):
                    client = AgentChatClient(config)
                async with client:
                    result = await client.complete_json("prompt", temperature=0.2)
                self.assertEqual(len(received), 1)
                self.assertEqual(received[0]["reasoning_effort"], "high")
                self.assertEqual(received[0]["chat_template_kwargs"], {"enable_thinking": True})
                self.assertNotIn("extra_body", received[0])
                self.assertNotIn("reasoning", received[0])
                self.assertEqual(result[AGENT_COMPLETION_USAGE_KEY]["cached_tokens"], 60)
                self.assertEqual(result[AGENT_COMPLETION_USAGE_KEY]["cache_hit_ratio"], 0.6)

    async def test_auto_cloud_keeps_openrouter_reasoning_and_existing_token_budget_default(self):
        client, sdk, _ = self.client(api_key="test-only", backend="auto",
                                     base_url="https://openrouter.ai/api/v1", reasoning_effort="none")
        async with client:
            await client.complete_json("prompt", temperature=0.2)
        request = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"reasoning": {"effort": "none"}})
        self.assertNotIn("max_tokens", request)

    async def test_auto_generic_endpoint_omits_vendor_reasoning(self):
        client, sdk, _ = self.client(api_key="test-only", backend="auto")
        async with client:
            await client.complete_json("prompt", temperature=0.2)
        self.assertNotIn("extra_body", sdk.chat.completions.create.call_args.kwargs)

    def test_generic_backend_is_not_guessed_and_cloud_envelope_is_unchanged(self):
        for backend in ("auto", "openai_compatible"):
            for options in ({}, {"enable_thinking": False}):
                with self.subTest(backend=backend, options=options):
                    config = AgentConfig(api_key="test-only", base_url="http://localhost:8000/v1",
                                         model="vllm/served-qwen", backend=backend,
                                         reasoning_effort="high", chat_template_kwargs=options)
                    self.assertEqual(request_extra_body(config),
                                     {"chat_template_kwargs": options} if options else None)
        cloud = AgentConfig(api_key="test-only", base_url="https://openrouter.ai/api/v1",
                            model="qwen/qwen3", reasoning_effort="high")
        self.assertEqual(request_extra_body(cloud), {"reasoning": {"effort": "high"}})
        llama = AgentConfig(api_key="test-only", base_url="https://openrouter.ai/api/v1",
                            model="meta-llama/llama-3.3-70b-instruct", reasoning_effort="high")
        self.assertIsNone(request_extra_body(llama))

    def test_extensions_do_not_mutate_config_and_legacy_llama_omits_reasoning(self):
        config = AgentConfig(api_key="test-only", base_url="https://openrouter.ai/api/v1",
                             model="meta-llama/llama-3.3-70b-instruct")
        self.assertIsNone(request_extra_body(config))
        local = AgentConfig(api_key=None, base_url="http://localhost/v1", model="local", backend="sglang",
                            chat_template_kwargs={"enable_thinking": False, "nested": {"value": 1}})
        body = request_extra_body(local)
        body["chat_template_kwargs"]["nested"]["value"] = 2
        self.assertEqual(local.chat_template_kwargs["nested"]["value"], 1)

    def test_generic_and_cloud_without_credentials_still_fail(self):
        for backend in ("auto", "openrouter", "openai_compatible"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "api_key"):
                self.client(backend=backend)

    async def test_shared_slot_wraps_each_attempt_and_provider_retry_marks_usage_incomplete(self):
        events = []

        class Limiter:
            @asynccontextmanager
            async def slot(self):
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")

        with patch("agents.SharedRequestLimiter", return_value=Limiter()) as limiter_factory:
            client, sdk, _ = self.client(max_concurrent_requests_total=2)
        limiter_factory.assert_called_once_with("http://localhost:8000/v1", 2)

        async def create(**request):
            self.assertEqual(events[-1], "enter")
            events.append("request")
            if events.count("request") == 1:
                raise json.JSONDecodeError("truncated provider JSON", "", 0)
            return response()

        sdk.chat.completions.create.side_effect = create
        async with client:
            payload = await client.complete_json("prompt", temperature=0.2)
        self.assertEqual(events, ["enter", "request", "exit", "enter", "request", "exit"])
        usage = payload[AGENT_COMPLETION_USAGE_KEY]
        self.assertEqual(usage["provider_attempt_count"], 2)
        self.assertFalse(usage["token_usage_complete"])
        self.assertFalse(usage["cache_usage_complete"])
        self.assertIsNone(usage["cached_tokens"])
        self.assertEqual(usage["known_cached_tokens"], 60)
        self.assertIsNone(usage["cache_hit_ratio"])
        requests = sdk.chat.completions.create.call_args_list
        self.assertEqual(requests[0], requests[1])

    async def test_usage_survives_agent_call_attempt_and_cumulative_summary(self):
        client, _, _ = self.client()
        async with client:
            result = await complete_agent_json(client, "prompt", context={"zone_id": "A1"},
                                               temperature=0.2, stage="agent.grid", agent="Grid")
        self.assertNotIn(AGENT_COMPLETION_USAGE_KEY, result.content)
        self.assertEqual(result.usage["cached_tokens"], 60)
        attempt_usage, calls = build_zone_attempt_agent_usage(
            [result.usage], attempt=1, proposal_phase="initial", triggered_by_attempt=None)
        cumulative, cumulative_calls = cumulative_zone_agent_usage([
            {"agent_usage": attempt_usage, "agent_call_usage": calls}])
        self.assertEqual(cumulative["cached_tokens"], 60)
        self.assertEqual(cumulative["cache_hit_ratio"], 0.6)
        self.assertEqual(cumulative_calls[0]["cached_tokens"], 60)

    async def test_reasoning_settings_are_preserved_across_provider_retries(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                client, sdk, _ = self.client(backend=backend, reasoning_effort="high",
                                              chat_template_kwargs={"enable_thinking": False})
                sdk.chat.completions.create.side_effect = [
                    json.JSONDecodeError("truncated provider JSON", "", 0), response()]
                async with client:
                    result = await client.complete_json("prompt", temperature=0.2)
                requests = sdk.chat.completions.create.call_args_list
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(requests[1].kwargs["extra_body"], {
                    "reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}})
                usage = result[AGENT_COMPLETION_USAGE_KEY]
                self.assertEqual(usage["known_cached_tokens"], 60)
                self.assertFalse(usage["cache_usage_complete"])


class ResponseCacheUsageTests(unittest.TestCase):
    def test_dict_and_sdk_response_usage_are_supported(self):
        as_dict = {"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                              "prompt_tokens_details": {"cached_tokens": 60}}}
        self.assertEqual(response_token_usage(as_dict), response_token_usage(response()))

    def test_absent_zero_and_invalid_cache_counts_are_distinct(self):
        for cached in (None, -1, 101, True, 0.5, "60"):
            with self.subTest(cached=cached):
                result = response_token_usage(response(cached))
                self.assertIsNone(result["cached_tokens"])
                self.assertFalse(result["cache_usage_complete"])
                record = agent_call_usage_record(stage="test", agent="test", zone_id="A1", token_usage=result)
                self.assertIsNone(record["cached_tokens"])
        zero = response_token_usage(response(0))
        self.assertEqual(zero["cached_tokens"], 0)
        self.assertEqual(zero["cache_hit_ratio"], 0)
        self.assertTrue(zero["cache_usage_complete"])


if __name__ == "__main__":
    unittest.main()
