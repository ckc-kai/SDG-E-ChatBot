from __future__ import annotations

import json
import tempfile
import unittest
import httpx
from pathlib import Path
from unittest.mock import MagicMock, patch

from generation.providers.base import ProviderError
from generation.providers.capabilities import RateLimitState
from generation.providers.groq import HttpxGroqTransport, GroqProvider, GroqUsage


class FakeGroqTransport:
    def __init__(self, response=None, error=None, metadata=None, response_headers=None):
        self.response = response or {
            "choices": [{"message": {"content": '{"answer":"ok","cited_chunk_ids":["1"],"insufficient_context":false}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "x_groq": {"total_time": 0.25},
        }
        self.error = error
        self.metadata = metadata or {
            "id": "openai/gpt-oss-120b",
            "context_window": 131_072,
            "max_completion_tokens": 65_536,
        }
        self.last_response_headers = response_headers or {}
        self.call = None
        self.get_call = None

    def post_json(self, url, payload, headers, timeout_seconds):
        self.call = (url, payload, headers, timeout_seconds)
        if self.error:
            raise self.error
        return self.response

    def get_json(self, url, headers, timeout_seconds):
        self.get_call = (url, headers, timeout_seconds)
        return self.metadata


class GroqProviderTests(unittest.TestCase):
    def test_sends_request_and_records_usage(self):
        transport = FakeGroqTransport()
        provider = GroqProvider(transport, "secret", "openai/gpt-oss-120b")
        raw = provider.generate("grounded prompt")
        url, payload, headers, _ = transport.call
        self.assertEqual(url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertFalse(payload["include_reasoning"])
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(headers["User-Agent"], "SDGE-ChatBot/0.1")
        self.assertIn('"answer":"ok"', raw)
        self.assertEqual(provider.last_usage, GroqUsage(100, 20, 250))
        self.assertIn("openai%2Fgpt-oss-120b", transport.get_call[0])

    def test_environment_requires_key_and_has_safe_defaults(self):
        with self.assertRaisesRegex(ProviderError, "GROQ_API_KEY is required"):
            GroqProvider.from_env(environ={}, transport=FakeGroqTransport())
        provider = GroqProvider.from_env(
            environ={"GROQ_API_KEY": "secret"}, transport=FakeGroqTransport()
        )
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual(provider.context_tokens, 131_072)
        self.assertEqual(provider.prompt_token_budget, 6_500)
        self.assertEqual(provider.temperature, 0)
        self.assertEqual(provider.reasoning_effort, "low")
        self.assertEqual(provider.seed, 42)

    def test_generation_writes_seed_and_provider_metadata_to_trace(self):
        response = {
            "id": "call-1",
            "model": "openai/gpt-oss-120b",
            "system_fingerprint": "fp-test",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"answer":"ok"}'},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        with tempfile.TemporaryDirectory() as trace_dir:
            provider = GroqProvider(
                FakeGroqTransport(response=response),
                "secret",
                trace_dir=trace_dir,
            )
            raw = provider.generate("trace groq")
            trace_file = next(Path(trace_dir).glob("model-*.json"))
            record = json.loads(trace_file.read_text(encoding="utf-8"))
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(
            record["response_metadata"]["system_fingerprint"], "fp-test"
        )
        self.assertEqual(record["response_metadata"]["finish_reason"], "stop")

    def test_environment_can_override_operational_prompt_budget(self):
        provider = GroqProvider.from_env(
            environ={
                "GROQ_API_KEY": "secret",
                "GROQ_PROMPT_TOKEN_BUDGET": "16000",
            },
            transport=FakeGroqTransport(),
        )
        self.assertEqual(provider.context_tokens, 131_072)
        self.assertEqual(provider.prompt_token_budget, 16_000)

    def test_records_rate_limit_headers(self):
        transport = FakeGroqTransport(
            response_headers={
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "6123",
                "x-ratelimit-remaining-requests": "999",
                "x-ratelimit-reset-tokens": "7.5s",
            }
        )
        provider = GroqProvider(transport, "secret")
        provider.generate("prompt")
        self.assertEqual(provider.last_rate_limit.token_limit, 8000)
        self.assertEqual(provider.last_rate_limit.remaining_tokens, 6123)
        self.assertEqual(provider.last_rate_limit.remaining_requests, 999)
        self.assertEqual(provider.last_rate_limit.token_reset, "7.5s")
        self.assertEqual(provider.available_prompt_token_budget(), 5623)

    def test_first_request_uses_6500_fallback_before_quota_is_known(self):
        provider = GroqProvider(FakeGroqTransport(), "secret")
        provider.refresh_capabilities()
        self.assertEqual(provider.capabilities.prompt_token_budget, 130_572)
        self.assertEqual(provider.acquire_prompt_token_budget(20_000), 6_500)

    def test_known_paid_quota_allows_larger_prompt(self):
        provider = GroqProvider(FakeGroqTransport(), "secret", max_tokens=1_500)
        provider.refresh_capabilities()
        provider.last_rate_limit = RateLimitState(
            token_limit=250_000,
            remaining_tokens=250_000,
            token_reset="10s",
            captured_at_monotonic=100.0,
        )
        with patch("generation.providers.capabilities.time.monotonic", return_value=100.0):
            self.assertEqual(
                provider.acquire_prompt_token_budget(30_000),
                129_572,
            )

    def test_waits_for_reset_when_request_fits_restored_tpm(self):
        provider = GroqProvider(FakeGroqTransport(), "secret", max_tokens=1_500)
        provider.refresh_capabilities()
        provider.last_rate_limit = RateLimitState(
            token_limit=8_000,
            remaining_tokens=2_000,
            token_reset="7.5s",
            captured_at_monotonic=100.0,
        )
        with patch("generation.providers.capabilities.time.monotonic", return_value=102.0):
            with patch("generation.providers.groq.time.sleep") as sleep:
                budget = provider.acquire_prompt_token_budget(5_000)
        self.assertEqual(budget, 6_500)
        sleep.assert_called_once_with(5.5)

    def test_oversized_request_is_capped_by_full_tpm_without_waiting(self):
        provider = GroqProvider(FakeGroqTransport(), "secret", max_tokens=1_500)
        provider.refresh_capabilities()
        provider.last_rate_limit = RateLimitState(
            token_limit=8_000,
            remaining_tokens=8_000,
            token_reset="7.5s",
            captured_at_monotonic=100.0,
        )
        with patch("generation.providers.capabilities.time.monotonic", return_value=100.0):
            with patch("generation.providers.groq.time.sleep") as sleep:
                budget = provider.acquire_prompt_token_budget(20_000)
        self.assertEqual(budget, 6_500)
        sleep.assert_not_called()

    def test_does_not_block_past_user_facing_wait_limit(self):
        provider = GroqProvider(FakeGroqTransport(), "secret", max_tokens=1_500)
        provider.refresh_capabilities()
        provider.last_rate_limit = RateLimitState(
            token_limit=8_000,
            remaining_tokens=2_000,
            token_reset="30s",
            captured_at_monotonic=100.0,
        )
        with patch("generation.providers.capabilities.time.monotonic", return_value=100.0):
            with patch("generation.providers.groq.time.sleep") as sleep:
                with self.assertRaisesRegex(
                    ProviderError, "token quota is temporarily exhausted"
                ):
                    provider.acquire_prompt_token_budget(5_000)
        sleep.assert_not_called()

    def test_expired_rate_limit_window_no_longer_restricts_budget(self):
        state = RateLimitState(
            remaining_tokens=100,
            token_reset="7.5s",
            captured_at_monotonic=100.0,
        )
        with patch("generation.providers.capabilities.time.monotonic", return_value=108.0):
            self.assertIsNone(state.available_tokens())

    def test_transport_error_is_preserved(self):
        provider = GroqProvider(
            FakeGroqTransport(error=ProviderError("Groq free-tier rate limit reached")),
            "secret",
        )
        with self.assertRaisesRegex(ProviderError, "free-tier rate limit"):
            provider.generate("prompt")

    def test_http_error_exposes_only_safe_machine_code(self):
        request = httpx.Request("POST", "https://api.groq.com")
        response = httpx.Response(
            403,
            request=request,
            json={"error": {"message": "private detail", "code": "blocked"}},
        )
        client = MagicMock()
        client.post.return_value = response
        with patch("time.sleep"):
            with self.assertRaisesRegex(ProviderError, r"HTTP 403 \(blocked\)") as caught:
                HttpxGroqTransport(client).post_json(
                    "https://api.groq.com", {}, {"Authorization": "Bearer secret"}, 1
                )
        self.assertNotIn("private detail", str(caught.exception))
        self.assertEqual(client.post.call_count, 1)

    def test_connect_error_retries_once_but_http_errors_do_not(self):
        request = httpx.Request("POST", "https://api.groq.com")
        response = httpx.Response(200, request=request, json={"ok": True})
        client = MagicMock()
        client.post.side_effect = [httpx.ConnectError("tls"), response]
        with patch("time.sleep") as sleep:
            result = HttpxGroqTransport(client).post_json(
                "https://api.groq.com", {}, {}, 1
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.post.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
