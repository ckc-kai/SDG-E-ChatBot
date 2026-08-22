from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from generation.providers.base import ProviderError
from generation.providers.deepseek import DeepSeekProvider, DeepSeekUsage
from generation.providers.ollama import ANSWER_SCHEMA


class FakeDeepSeekTransport:
    def __init__(
        self,
        response: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"ok","cited_chunk_ids":["1"],'
                            '"insufficient_context":false,'
                            '"answered_requirements":[],"missing_requirements":[]}'
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 40,
                "prompt_cache_miss_tokens": 60,
            },
        }
        self.error = error
        self.call = None

    def post_json(self, url, payload, headers, timeout_seconds):
        self.call = (url, payload, headers, timeout_seconds)
        if self.error:
            raise self.error
        return self.response


class DeepSeekProviderTests(unittest.TestCase):
    def test_sends_non_thinking_request_and_records_usage(self) -> None:
        transport = FakeDeepSeekTransport()
        provider = DeepSeekProvider(transport, "secret")

        raw = provider.generate("grounded prompt")

        url, payload, headers, timeout = transport.call
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["max_tokens"], 1_500)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("include_reasoning", payload)
        self.assertNotIn("seed", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(timeout, 180)
        self.assertIn('"answer":"ok"', raw)
        self.assertEqual(provider.last_usage.input_tokens, 100)
        self.assertEqual(provider.last_usage.output_tokens, 20)
        self.assertEqual(provider.last_usage.cache_hit_tokens, 40)
        self.assertEqual(provider.last_usage.cache_miss_tokens, 60)
        self.assertIsInstance(provider.last_usage.latency_ms, int)

    def test_structured_generation_uses_json_object_mode(self) -> None:
        transport = FakeDeepSeekTransport()
        provider = DeepSeekProvider(transport, "secret")

        provider.generate_structured("return JSON", ANSWER_SCHEMA)

        _, payload, _, _ = transport.call
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        rendered_prompt = payload["messages"][0]["content"]
        self.assertIn("JSON Schema", rendered_prompt)
        self.assertIn('"cited_chunk_ids"', rendered_prompt)

    def test_environment_requires_key_and_has_comparison_defaults(self) -> None:
        with self.assertRaisesRegex(ProviderError, "DEEPSEEK_API_KEY is required"):
            DeepSeekProvider.from_env(environ={}, transport=FakeDeepSeekTransport())

        provider = DeepSeekProvider.from_env(
            environ={"DEEPSEEK_API_KEY": "secret"},
            transport=FakeDeepSeekTransport(),
        )
        self.assertEqual(provider.model_id, "deepseek/deepseek-v4-flash")
        self.assertEqual(provider.context_tokens, 1_000_000)
        self.assertEqual(provider.prompt_token_budget, 6_500)
        self.assertEqual(provider.max_tokens, 1_500)
        self.assertEqual(provider.thinking, "disabled")

    def test_generation_trace_marks_seed_unsupported(self) -> None:
        response = {
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp-ds",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"answer":"ok"}'},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        with tempfile.TemporaryDirectory() as trace_dir:
            provider = DeepSeekProvider(
                FakeDeepSeekTransport(response=response),
                "secret",
                trace_dir=trace_dir,
            )
            raw = provider.generate("trace deepseek")
            trace_file = next(Path(trace_dir).glob("model-*.json"))
            record = json.loads(trace_file.read_text(encoding="utf-8"))
        self.assertIsNone(record["seed"])
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(
            record["response_metadata"]["system_fingerprint"], "fp-ds"
        )

    def test_environment_overrides_are_applied(self) -> None:
        provider = DeepSeekProvider.from_env(
            environ={
                "DEEPSEEK_API_KEY": "secret",
                "DEEPSEEK_MODEL": "deepseek-v4-pro",
                "DEEPSEEK_MAX_TOKENS": "900",
                "DEEPSEEK_PROMPT_TOKEN_BUDGET": "12000",
                "DEEPSEEK_TEMPERATURE": "1.3",
                "DEEPSEEK_THINKING": "enabled",
            },
            transport=FakeDeepSeekTransport(),
        )
        self.assertEqual(provider.model, "deepseek-v4-pro")
        self.assertEqual(provider.max_tokens, 900)
        self.assertEqual(provider.prompt_token_budget, 12_000)
        self.assertEqual(provider.temperature, 1.3)
        self.assertEqual(provider.thinking, "enabled")

    def test_invalid_configuration_is_controlled(self) -> None:
        with self.assertRaisesRegex(ProviderError, "Invalid DeepSeek configuration"):
            DeepSeekProvider.from_env(
                environ={
                    "DEEPSEEK_API_KEY": "secret",
                    "DEEPSEEK_THINKING": "sometimes",
                },
                transport=FakeDeepSeekTransport(),
            )

    def test_transport_error_is_preserved(self) -> None:
        provider = DeepSeekProvider(
            FakeDeepSeekTransport(error=ProviderError("DeepSeek unavailable")),
            "secret",
        )
        with self.assertRaisesRegex(ProviderError, "unavailable"):
            provider.generate("prompt")


if __name__ == "__main__":
    unittest.main()
