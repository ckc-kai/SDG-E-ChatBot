"""A transient provider failure must not lose the whole request."""

from __future__ import annotations

import unittest

import httpx

from generation.providers.base import ProviderError, TransientProviderError
from generation.providers.groq import GroqProvider


def _response(status: int, code: str | None) -> httpx.Response:
    body = {"error": {"code": code}} if code else {}
    return httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/x"),
    )


class _ScriptedTransport:
    """Fails with the given errors, then returns one good completion."""

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def post_json(self, url, payload, headers, timeout_seconds):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _provider(transport, **kwargs):
    return GroqProvider(
        transport, api_key="k", model="openai/gpt-oss-120b",
        retry_backoff_seconds=0, **kwargs
    )


class GroqRetryTests(unittest.TestCase):
    def test_an_empty_completion_is_retried_and_recovers(self):
        """json_validate_failed is one bad sample, not a bad request."""
        transport = _ScriptedTransport(
            [TransientProviderError("Groq request failed with HTTP 400 "
                                    "(json_validate_failed)")]
        )

        result = _provider(transport).generate("prompt")

        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(transport.calls, 2)

    def test_retries_are_bounded_and_the_last_error_surfaces(self):
        transport = _ScriptedTransport(
            [TransientProviderError("boom") for _ in range(5)]
        )

        with self.assertRaises(TransientProviderError):
            _provider(transport, max_attempts=3).generate("prompt")

        self.assertEqual(transport.calls, 3)

    def test_a_non_transient_failure_is_not_retried(self):
        """Retrying a rejected credential wastes time and hides the cause."""
        transport = _ScriptedTransport([ProviderError("invalid api key")])

        with self.assertRaises(ProviderError):
            _provider(transport).generate("prompt")

        self.assertEqual(transport.calls, 1)

    def test_invalid_retry_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            _provider(_ScriptedTransport([]), max_attempts=0)


class TransportClassificationTests(unittest.TestCase):
    """The transport decides what is worth retrying; assert that boundary."""

    def _raise(self, status, code):
        from generation.providers.groq import HttpxGroqTransport

        class _Client:
            def post(self, *args, **kwargs):
                return _response(status, code)

        transport = HttpxGroqTransport(client=_Client())
        return transport.post_json("https://api.groq.com/v1/x", {}, {}, 1.0)

    def test_json_validate_failed_is_transient(self):
        with self.assertRaises(TransientProviderError):
            self._raise(400, "json_validate_failed")

    def test_server_errors_are_transient(self):
        with self.assertRaises(TransientProviderError):
            self._raise(503, None)

    def test_rate_limiting_is_transient(self):
        with self.assertRaises(TransientProviderError):
            self._raise(429, None)

    def test_other_client_errors_are_permanent(self):
        with self.assertRaises(ProviderError) as caught:
            self._raise(401, "invalid_api_key")
        self.assertNotIsInstance(caught.exception, TransientProviderError)


if __name__ == "__main__":
    unittest.main()
