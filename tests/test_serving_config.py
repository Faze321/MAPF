"""Offline configuration contracts for OpenRouter and local inference servers."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from config import AgentConfig


class ServingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test")
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "config.yaml"

    def load(self, agent, **kwargs):
        self.path.write_text(yaml.safe_dump({"agent": agent}), encoding="utf-8")
        return AgentConfig.from_file(self.path, **kwargs)

    def test_legacy_flat_config_keeps_defaults(self):
        parsed = self.load({"api_key": "legacy", "model": "legacy-model", "single_agent_model": "single-model"})
        self.assertEqual(parsed.resolved_backend, "openrouter")
        self.assertEqual(parsed.client_api_key, "legacy")
        self.assertEqual(parsed.chat_template_kwargs, {})
        self.assertIsNone(parsed.max_tokens)
        self.assertIsNone(parsed.max_concurrent_requests_total)
        single = AgentConfig.from_file(self.path, agent_mode="single_agent_price_retry")
        self.assertEqual(single.model, "single-model")

    def test_defaults_allow_offline_loading_without_credentials(self):
        parsed = AgentConfig.from_file(self.path, required=False)
        self.assertIsNone(parsed.client_api_key)
        self.assertEqual(parsed.backend, "auto")
        self.assertEqual(parsed.resolved_backend, "openrouter")

    def test_auto_uses_hostname_not_model_port_or_substring(self):
        for base_url, expected in (
            ("https://openrouter.ai/api/v1", "openrouter"),
            ("https://api.OPENROUTER.ai:8443/v1", "openrouter"),
            ("http://localhost:8000/v1", "openai_compatible"),
            ("http://localhost:30000/v1", "openai_compatible"),
            ("https://openrouter.ai.evil.example/v1", "openai_compatible"),
            ("https://notopenrouter.ai/v1", "openai_compatible"),
            ("https://openrouter.ai@other.example/v1", "openai_compatible"),
        ):
            with self.subTest(base_url=base_url):
                self.assertEqual(replace(self.config, base_url=base_url, model="vllm/sglang").resolved_backend, expected)

    def test_explicit_backend_wins_and_only_local_backends_supply_placeholder_key(self):
        for backend in ("auto", "openrouter", "openai_compatible", "vllm", "sglang"):
            with self.subTest(backend=backend):
                parsed = replace(self.config, backend=backend, api_key=None, base_url="http://localhost:8000/v1")
                expected_key = "EMPTY" if backend in {"vllm", "sglang"} else None
                self.assertEqual(parsed.client_api_key, expected_key)
                self.assertEqual(parsed.resolved_backend, "openai_compatible" if backend == "auto" else backend)
                self.assertEqual(replace(parsed, api_key="server-key").client_api_key, "server-key")

    def test_selected_profile_inherits_parent_serving_settings(self):
        parsed = self.load({
            "api_key": None, "backend": "vllm", "base_url": "http://localhost:8000/v1",
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096,
            "max_concurrent_requests_total": 3,
            "multi_agent": {"model": "multi"},
            "single_agent": {"model": "single", "max_tokens": 8192},
        }, agent_mode="single_agent_full_retry")
        self.assertEqual(parsed.model, "single")
        self.assertEqual(parsed.backend, "vllm")
        self.assertEqual(parsed.client_api_key, "EMPTY")
        self.assertEqual(parsed.chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(parsed.max_tokens, 8192)
        self.assertEqual(parsed.max_concurrent_requests_total, 3)

    def test_profile_can_replace_or_clear_parent_optional_fields(self):
        parsed = self.load({
            "backend": "vllm", "chat_template_kwargs": {"enable_thinking": False, "extra": 1},
            "max_tokens": 4096, "max_concurrent_requests_total": 8,
            "single_agent": {"backend": "sglang", "chat_template_kwargs": {"enable_thinking": True},
                             "max_tokens": None, "max_concurrent_requests_total": None},
        }, agent_mode="single_agent_price_retry")
        self.assertEqual(parsed.backend, "sglang")
        self.assertEqual(parsed.chat_template_kwargs, {"enable_thinking": True})
        self.assertIsNone(parsed.max_tokens)
        self.assertIsNone(parsed.max_concurrent_requests_total)
        self.assertEqual(self.load({"chat_template_kwargs": None}).chat_template_kwargs, {})

    def test_environment_expands_selected_profile_and_template_strings_only(self):
        with patch.dict(os.environ, {"MAPF_TEST_BACKEND": "vllm", "MAPF_TEST_MODEL": "served-model",
                                     "MAPF_TEST_URL": "http://localhost:8000/v1", "MAPF_TEST_LABEL": "test"}, clear=True):
            parsed = self.load({
                "backend": "${MAPF_TEST_BACKEND}",
                "multi_agent": {"model": "${MAPF_TEST_MODEL}", "base_url": "${MAPF_TEST_URL}",
                                "chat_template_kwargs": {"label": "${MAPF_TEST_LABEL}", "enable_thinking": False}},
                "single_agent": {"api_key": "${UNSET_INACTIVE_API_KEY}"},
            })
        self.assertEqual(parsed.backend, "vllm")
        self.assertEqual(parsed.model, "served-model")
        self.assertEqual(parsed.chat_template_kwargs, {"label": "test", "enable_thinking": False})

    def test_new_counts_reject_coerced_values_directly_and_from_yaml(self):
        for field in ("max_tokens", "max_concurrent_requests_total"):
            self.assertEqual(getattr(replace(self.config, **{field: 1}), field), 1)
            for invalid in (False, True, 0, -1, 1.5, 3.0, "3", "", float("inf"), float("nan")):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(self.config, **{field: invalid})
                    with self.assertRaisesRegex(ValueError, field):
                        self.load({field: invalid})

    def test_backend_validation_rejects_unknown_or_nonstring_values(self):
        for invalid in (None, True, 1, "", "VLLM", "unsupported", [], {}):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "backend"):
                    replace(self.config, backend=invalid)
                with self.assertRaisesRegex(ValueError, "backend"):
                    self.load({"backend": invalid})

    def test_local_reasoning_efforts_are_normalized_and_validated(self):
        for backend in ("vllm", "sglang"):
            for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(backend=backend, effort=effort):
                    source = f" {effort.upper()} "
                    self.assertEqual(replace(self.config, backend=backend, reasoning_effort=source).reasoning_effort, effort)
                    self.assertEqual(self.load({"backend": backend, "reasoning_effort": source}).reasoning_effort, effort)
            for invalid in (True, False, 1, 0.5, [], {}, "false", "auto", "unsupported"):
                with self.subTest(backend=backend, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                        replace(self.config, backend=backend, reasoning_effort=invalid)
                    with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                        self.load({"backend": backend, "reasoning_effort": invalid})

    def test_local_none_disables_but_null_or_blank_leaves_server_default(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                self.assertEqual(self.load({"backend": backend}).reasoning_effort, "none")
                self.assertEqual(self.load({"backend": backend, "reasoning_effort": "none"}).reasoning_effort, "none")
                for empty in (None, "", " "):
                    self.assertIsNone(self.load({"backend": backend, "reasoning_effort": empty}).reasoning_effort)
                    self.assertIsNone(replace(self.config, backend=backend, reasoning_effort=empty).reasoning_effort)
        for backend in ("auto", "openrouter", "openai_compatible"):
            for empty in (None, ""):
                self.assertEqual(self.load({"backend": backend, "reasoning_effort": empty}).reasoning_effort, "none")

    def test_local_reasoning_profiles_inherit_override_and_clear_independently(self):
        with patch.dict(os.environ, {"MAPF_TEST_REASONING": " HIGH "}):
            settings = {"backend": "vllm", "reasoning_effort": "${MAPF_TEST_REASONING}",
                        "single_agent": {"backend": "sglang", "reasoning_effort": "low"}}
            multi = self.load(settings, agent_mode="multi_agent_full_retry")
            single = AgentConfig.from_file(self.path, agent_mode="single_agent_full_retry")
            self.assertEqual(multi.reasoning_effort, "high")
            self.assertEqual(single.reasoning_effort, "low")
            settings["single_agent"]["reasoning_effort"] = None
            self.assertIsNone(self.load(settings, agent_mode="single_agent_price_retry").reasoning_effort)
            self.assertEqual(AgentConfig.from_file(self.path, agent_mode="multi_agent_full_retry").reasoning_effort, "high")

    def test_explicit_template_switch_is_preserved_for_client_precedence(self):
        for backend in ("vllm", "sglang"):
            config = self.load({"backend": backend, "reasoning_effort": "high",
                                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096})
            self.assertEqual(config.reasoning_effort, "high")
            self.assertEqual(config.chat_template_kwargs, {"enable_thinking": False})
            self.assertEqual(config.max_tokens, 4096)

    def test_template_validation_rejects_invalid_json_or_thinking_values(self):
        circular = {}
        circular["self"] = circular
        for invalid in ([], "{}", {1: "value"}, {"nested": {2: "value"}}, {"obj": object()},
                        {"number": float("nan")}, {"nested": [float("inf")]}, circular,
                        {"enable_thinking": "false"}, {"enable_thinking": 0}, {"enable_thinking": None}):
            with self.subTest(invalid_type=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, "chat_template_kwargs"):
                    replace(self.config, chat_template_kwargs=invalid)

    def test_template_kwargs_are_deep_copied_and_defaults_not_shared(self):
        original = {"enable_thinking": False, "nested": {"items": [1, 2]}}
        parsed = replace(self.config, chat_template_kwargs=original)
        original["nested"]["items"].append(3)
        original["enable_thinking"] = True
        self.assertEqual(parsed.chat_template_kwargs, {"enable_thinking": False, "nested": {"items": [1, 2]}})
        a, b = AgentConfig.default(), AgentConfig.default()
        a.chat_template_kwargs["different"] = True
        self.assertEqual(b.chat_template_kwargs, {})

    def test_existing_numeric_settings_keep_valid_legacy_strings(self):
        parsed = self.load({"timeout_seconds": "12.5", "max_concurrent_requests": "3",
                            "provider_json_retries": "0", "provider_json_retry_backoff_seconds": "0"})
        self.assertEqual(parsed.timeout_seconds, 12.5)
        self.assertEqual(parsed.max_concurrent_requests, 3)
        self.assertEqual(parsed.provider_json_retries, 0)
        self.assertEqual(parsed.provider_json_retry_backoff_seconds, 0.0)

    def test_existing_numeric_settings_reject_invalid_values(self):
        invalid_values = {
            "timeout_seconds": (True, 0, -1, float("nan"), float("inf")),
            "max_concurrent_requests": (True, 0, -1, 1.5, float("inf")),
            "provider_json_retries": (True, -1, 1.5, float("inf")),
            "provider_json_retry_backoff_seconds": (True, -1, float("nan"), float("inf")),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(self.config, **{field: value})
                    with self.assertRaisesRegex(ValueError, field):
                        self.load({field: value})


if __name__ == "__main__":
    unittest.main()
