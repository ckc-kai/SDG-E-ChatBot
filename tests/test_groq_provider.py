from __future__ import annotations

import unittest
import httpx
from unittest.mock import MagicMock, patch

from generation.providers.base import ProviderError
from generation.providers.groq import HttpxGroqTransport, GroqProvider, GroqUsage


class FakeGroqTransport:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "choices": [{"message": {"content": '{"answer":"ok","cited_chunk_ids":["1"],"insufficient_context":false}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "x_groq": {"total_time": 0.25},
        }
        self.error = error
        self.call = None

    def post_json(self, url, payload, headers, timeout_seconds):
        self.call = (url, payload, headers, timeout_seconds)
        if self.error:
            raise self.error
        return self.response


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
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(headers["User-Agent"], "SDGE-ChatBot/0.1")
        self.assertIn('"answer":"ok"', raw)
        self.assertEqual(provider.last_usage, GroqUsage(100, 20, 250))

    def test_environment_requires_key_and_has_safe_defaults(self):
        with self.assertRaisesRegex(ProviderError, "GROQ_API_KEY is required"):
            GroqProvider.from_env(environ={}, transport=FakeGroqTransport())
        provider = GroqProvider.from_env(
            environ={"GROQ_API_KEY": "secret"}, transport=FakeGroqTransport()
        )
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual(provider.context_tokens, 8192)
        self.assertEqual(provider.temperature, 0)
        self.assertEqual(provider.reasoning_effort, "low")

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
