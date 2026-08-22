from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from generation.providers.base import ProviderError
from generation.providers.ollama import ANSWER_SCHEMA, OllamaProvider, OllamaUsage


class FakeOllamaTransport:
    def __init__(
        self,
        response: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "message": {
                "role": "assistant",
                "content": (
                    '{"answer":"Supported","cited_chunk_ids":["1"],'
                    '"insufficient_context":false}'
                ),
            },
            "prompt_eval_count": 80,
            "eval_count": 15,
            "total_duration": 250_000_000,
        }
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], float]] = []

    def post_json(
        self, url: str, payload: Mapping[str, Any], timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        if self.error:
            raise self.error
        return self.response


class OllamaProviderTests(unittest.TestCase):
    def test_generate_uses_chat_schema_and_records_usage(self) -> None:
        transport = FakeOllamaTransport()
        provider = OllamaProvider(
            transport,
            "qwen3:4b",
            max_tokens=300,
            temperature=0,
            timeout_seconds=30,
        )

        raw = provider.generate("grounded prompt")

        self.assertIn('"answer":"Supported"', raw)
        url, payload, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "grounded prompt"}])
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["format"], ANSWER_SCHEMA)
        self.assertEqual(
            payload["options"],
            {"num_predict": 300, "num_ctx": 4096, "temperature": 0, "seed": 42},
        )
        self.assertEqual(timeout, 30)
        self.assertEqual(provider.model_id, "ollama/qwen3:4b")
        self.assertEqual(provider.last_usage, OllamaUsage(80, 15, 250))
        self.assertEqual(provider.last_raw_text, raw)
        self.assertEqual(provider.last_request_payload, payload)
        self.assertEqual(provider.capabilities.context_window, 4096)
        self.assertEqual(provider.capabilities.prompt_token_budget, 3796)

    def test_warmup_loads_model_without_requesting_an_answer(self) -> None:
        transport = FakeOllamaTransport(response={"done": True})
        provider = OllamaProvider(transport, "qwen3:4b", keep_alive="45m")

        provider.warmup()

        url, payload, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(
            payload,
            {
                "model": "qwen3:4b",
                "prompt": "",
                "stream": False,
                "keep_alive": "45m",
            },
        )
        self.assertEqual(timeout, provider.timeout_seconds)

    def test_from_env_uses_injected_transport(self) -> None:
        provider = OllamaProvider.from_env(
            environ={
                "OLLAMA_MODEL": "gemma3:4b",
                "OLLAMA_BASE_URL": "http://localhost:11434/",
                "OLLAMA_MAX_TOKENS": "250",
                "OLLAMA_TEMPERATURE": "0",
                "OLLAMA_TIMEOUT_SECONDS": "45",
                "OLLAMA_CONTEXT_TOKENS": "8192",
                "OLLAMA_TOKEN_SAFETY_FACTOR": "1.2",
                "OLLAMA_KEEP_ALIVE": "20m",
                "MODEL_SEED": "17",
            },
            transport=FakeOllamaTransport(),
        )
        self.assertEqual(provider.model, "gemma3:4b")
        self.assertEqual(provider.base_url, "http://localhost:11434")
        self.assertEqual(provider.max_tokens, 250)
        self.assertEqual(provider.timeout_seconds, 45)
        self.assertEqual(provider.context_tokens, 8192)
        self.assertEqual(provider.token_safety_factor, 1.2)
        self.assertEqual(provider.keep_alive, "20m")
        self.assertEqual(provider.seed, 17)

    def test_generation_writes_full_model_trace(self) -> None:
        with tempfile.TemporaryDirectory() as trace_dir:
            provider = OllamaProvider(
                FakeOllamaTransport(),
                "qwen3:4b",
                trace_dir=trace_dir,
            )
            raw = provider.generate("trace this prompt")

            trace_file = next(Path(trace_dir).glob("model-*.json"))
            record = json.loads(trace_file.read_text(encoding="utf-8"))
        self.assertEqual(record["model_id"], "ollama/qwen3:4b")
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["prompt"], "trace this prompt")
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(record["usage"]["output_tokens"], 15)
        self.assertEqual(record["outcome"], "success")

    def test_from_env_requires_model(self) -> None:
        with self.assertRaisesRegex(ProviderError, "OLLAMA_MODEL is required"):
            OllamaProvider.from_env(environ={}, transport=FakeOllamaTransport())

    def test_invalid_configuration_is_controlled(self) -> None:
        with self.assertRaisesRegex(ProviderError, "Invalid Ollama configuration"):
            OllamaProvider.from_env(
                environ={"OLLAMA_MODEL": "model", "OLLAMA_MAX_TOKENS": "many"},
                transport=FakeOllamaTransport(),
            )

    def test_missing_response_text_is_rejected(self) -> None:
        provider = OllamaProvider(
            FakeOllamaTransport(response={"message": {"content": ""}}), "model"
        )
        with self.assertRaisesRegex(ProviderError, "contains no text"):
            provider.generate("prompt")

    def test_transport_provider_error_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as trace_dir:
            provider = OllamaProvider(
                FakeOllamaTransport(error=ProviderError("Ollama request failed")),
                "model",
                trace_dir=trace_dir,
            )
            with self.assertRaisesRegex(ProviderError, "Ollama request failed"):
                provider.generate("prompt")
            trace_file = next(Path(trace_dir).glob("model-*.json"))
            record = json.loads(trace_file.read_text(encoding="utf-8"))
        self.assertEqual(record["outcome"], "error")
        self.assertEqual(record["error"]["type"], "ProviderError")


if __name__ == "__main__":
    unittest.main()
