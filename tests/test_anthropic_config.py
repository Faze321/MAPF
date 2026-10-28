"""Native Anthropic configuration contracts, without any provider calls."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from config import AgentConfig, DEFAULT_ANTHROPIC_MAX_TOKENS


class AnthropicConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.yaml"
        self.native = AgentConfig(api_key="test-key", base_url="https://api.anthropic.com",
                                  model="claude-sonnet-4-6", backend="anthropic")

    def load(self, agent, **kwargs):
        self.path.write_text(yaml.safe_dump({"agent": agent}), encoding="utf-8")
        return AgentConfig.from_file(self.path, **kwargs)

    def test_native_defaults_and_credentials_remain_explicit(self):
        config = self.load({"backend": "anthropic"})
        self.assertEqual(config.resolved_backend, "anthropic")
        self.assertEqual(config.base_url, "https://api.anthropic.com")
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertEqual(config.reasoning_effort, "none")
        self.assertIsNone(config.client_api_key)
        self.assertIsNone(config.max_tokens)
        self.assertIsNone(config.anthropic_thinking)
        self.assertIsNone(config.anthropic_cache_control)
        self.assertEqual(DEFAULT_ANTHROPIC_MAX_TOKENS, 8192)
        for blank in (None, "", " "):
            config = self.load({"backend": "anthropic", "base_url": blank, "model": blank})
            self.assertEqual(config.base_url, "https://api.anthropic.com")
            self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertEqual(replace(config, api_key="native-key").client_api_key, "native-key")

    def test_auto_does_not_guess_native_api_from_model_or_host(self):
        for url in ("https://api.anthropic.com", "https://api.anthropic.com/v1", "http://localhost:8000"):
            config = replace(self.native, backend="auto", base_url=url)
            self.assertEqual(config.resolved_backend, "openai_compatible")

    def test_native_effort_normalization_and_empty_values(self):
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                value = f" {effort.upper()} "
                self.assertEqual(replace(self.native, reasoning_effort=value).reasoning_effort, effort)
                self.assertEqual(self.load({"backend": "anthropic", "reasoning_effort": value}).reasoning_effort, effort)
        for value in (None, "", " "):
            self.assertIsNone(replace(self.native, reasoning_effort=value).reasoning_effort)
            self.assertIsNone(self.load({"backend": "anthropic", "reasoning_effort": value}).reasoning_effort)

    def test_invalid_native_effort_rejected_before_request(self):
        for invalid in (True, False, 1, .5, {}, [], "minimal", "auto", "unsupported"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                    replace(self.native, reasoning_effort=invalid)
                with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                    self.load({"backend": "anthropic", "reasoning_effort": invalid})

    def test_profile_inheritance_overrides_and_clearing(self):
        shared = {"backend": "anthropic", "api_key": "parent", "base_url": "https://proxy.example/anthropic",
                  "model": "parent-claude", "reasoning_effort": "high", "max_tokens": 16000,
                  "anthropic_thinking": {"type": "adaptive"},
                  "anthropic_cache_control": {"type": "ephemeral", "ttl": "1h"},
                  "multi_agent": {"model": "multi-claude"},
                  "single_agent": {"model": "single-claude", "reasoning_effort": None,
                                   "anthropic_thinking": None, "anthropic_cache_control": None}}
        multi = self.load(shared, agent_mode="multi_agent_full_retry")
        single = AgentConfig.from_file(self.path, agent_mode="single_agent_full_retry")
        self.assertEqual(multi.model, "multi-claude")
        self.assertEqual(multi.base_url, "https://proxy.example/anthropic")
        self.assertEqual(multi.reasoning_effort, "high")
        self.assertEqual(multi.anthropic_thinking, {"type": "adaptive"})
        self.assertEqual(multi.anthropic_cache_control, {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(single.model, "single-claude")
        self.assertEqual(single.max_tokens, 16000)
        self.assertIsNone(single.reasoning_effort)
        self.assertIsNone(single.anthropic_thinking)
        self.assertIsNone(single.anthropic_cache_control)
        self.assertEqual(AgentConfig.from_file(self.path, model="cli-claude").model, "cli-claude")

    def test_environment_expansion_uses_only_selected_profile(self):
        with patch.dict(os.environ, {"MAPF_TEST_BACKEND": "anthropic", "MAPF_TEST_EFFORT": " HIGH ",
                                     "MAPF_TEST_CACHE_TTL": "5m", "MAPF_TEST_THINKING": "adaptive"}):
            config = self.load({"backend": "${MAPF_TEST_BACKEND}",
                                "reasoning_effort": "${MAPF_TEST_EFFORT}",
                                "anthropic_thinking": {"type": "${MAPF_TEST_THINKING}"},
                                "anthropic_cache_control": {"type": "ephemeral", "ttl": "${MAPF_TEST_CACHE_TTL}"},
                                "single_agent": {"api_key": "${MAPF_MISSING_INACTIVE_SECRET}"}})
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(config.anthropic_thinking, {"type": "adaptive"})
        self.assertEqual(config.anthropic_cache_control, {"type": "ephemeral", "ttl": "5m"})

    def test_explicit_thinking_shapes_and_override_are_preserved(self):
        for value in ({"type": "disabled"}, {"type": "adaptive"}, {"type": "enabled", "budget_tokens": 1024}):
            with self.subTest(value=value):
                config = replace(self.native, anthropic_thinking=value, reasoning_effort="none")
                self.assertEqual(config.anthropic_thinking, value)
                self.assertEqual(config.reasoning_effort, "none")
        for invalid in ({}, [], True, "adaptive", {"type": None}, {"type": []},
                        {"type": "invalid"}, {"type": "adaptive", "budget_tokens": 1024},
                        {"type": "disabled", "extra": 1}, {"type": "enabled"},
                        {"type": "enabled", "budget_tokens": 1024, "extra": 1}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "anthropic_thinking"):
                replace(self.native, anthropic_thinking=invalid)

    def test_manual_thinking_budget_validation_and_output_cap(self):
        for invalid in (True, False, "1024", 1024.0, None, -1, 0, 1023, 8192, 9000):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "budget_tokens"):
                replace(self.native, anthropic_thinking={"type": "enabled", "budget_tokens": invalid})
        config = replace(self.native, max_tokens=16000, anthropic_thinking={"type": "enabled", "budget_tokens": 8192})
        self.assertEqual(config.anthropic_thinking["budget_tokens"], 8192)
        with self.assertRaisesRegex(ValueError, "budget_tokens"):
            replace(self.native, max_tokens=1024, anthropic_thinking={"type": "enabled", "budget_tokens": 1024})
        # Provider-specific fields do not change a different backend's output cap.
        config = replace(self.native, backend="openrouter", max_tokens=100,
                         anthropic_thinking={"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(config.max_tokens, 100)

    def test_cache_control_shapes_and_validation(self):
        for value in ({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "5m"},
                      {"type": "ephemeral", "ttl": "1h"}):
            self.assertEqual(replace(self.native, anthropic_cache_control=value).anthropic_cache_control, value)
        for invalid in ({}, [], True, "ephemeral", {"type": []}, {"type": "wrong"},
                        {"type": "ephemeral", "ttl": "1d"}, {"type": "ephemeral", "ttl": None},
                        {"type": "ephemeral", "ttl": []}, {"type": "ephemeral", "extra": 1}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "anthropic_cache_control"):
                replace(self.native, anthropic_cache_control=invalid)

    def test_native_rejects_local_template_options_instead_of_ignoring_them(self):
        with self.assertRaisesRegex(ValueError, "chat_template_kwargs.*anthropic_thinking"):
            replace(self.native, chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(replace(self.native, chat_template_kwargs={}).chat_template_kwargs, {})
        self.assertEqual(replace(self.native, chat_template_kwargs=None).chat_template_kwargs, {})

    def test_native_dictionaries_are_copied_and_validated_from_yaml(self):
        thinking, cache = {"type": "enabled", "budget_tokens": 1024}, {"type": "ephemeral", "ttl": "5m"}
        config = self.load({"backend": "anthropic", "anthropic_thinking": thinking,
                            "anthropic_cache_control": cache})
        direct = replace(self.native, anthropic_thinking=thinking, anthropic_cache_control=cache)
        thinking["budget_tokens"] = 4096
        cache["ttl"] = "1h"
        for item in (config, direct):
            self.assertEqual(item.anthropic_thinking["budget_tokens"], 1024)
            self.assertEqual(item.anthropic_cache_control["ttl"], "5m")
        with self.assertRaisesRegex(ValueError, "budget_tokens"):
            self.load({"backend": "anthropic", "anthropic_thinking": {"type": "enabled", "budget_tokens": 1}})
        with self.assertRaisesRegex(ValueError, "anthropic_cache_control"):
            self.load({"backend": "anthropic", "anthropic_cache_control": {"type": "permanent"}})

    def test_existing_cloud_defaults_are_unchanged(self):
        for backend in ("auto", "openrouter", "openai_compatible"):
            config = self.load({"backend": backend, "reasoning_effort": None})
            self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(config.model, "meta-llama/llama-3.1-8b-instruct")
            self.assertEqual(config.reasoning_effort, "none")
            self.assertIsNone(config.max_tokens)


if __name__ == "__main__":
    unittest.main()
