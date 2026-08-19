"""Groq chat-completions provider with schema-constrained JSON output."""

from __future__ import annotations

import json
import os
import time
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from generation.providers.base import ProviderError, TransientProviderError
from generation.providers.ollama import ANSWER_SCHEMA

import httpx


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_TOKEN_SAFETY_FACTOR = 1.2
DEFAULT_REASONING_EFFORT = "low"
# Three attempts clears an isolated bad sample without masking a real outage.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


class GroqTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class HttpxGroqTransport:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client()

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        try:
            response = self._post_with_connect_retry(
                url, payload, headers, timeout_seconds
            )
            response.raise_for_status()
            decoded = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            error_code = _safe_http_error_code(exc.response)
            suffix = f" ({error_code})" if error_code else ""
            if status == 429:
                raise TransientProviderError(
                    f"Groq rate limit reached{suffix}"
                ) from exc
            # json_validate_failed means the model returned content that did
            # not satisfy the response schema -- commonly an empty completion
            # when reasoning consumed the budget. It is a property of one
            # sampled generation, not of the request, so the same request can
            # succeed on a retry.
            if status >= 500 or error_code == "json_validate_failed":
                raise TransientProviderError(
                    f"Groq request failed with HTTP {status}{suffix}"
                ) from exc
            raise ProviderError(
                f"Groq request failed with HTTP {status}{suffix}"
            ) from exc
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            raise TransientProviderError("Groq request failed") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("Groq returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderError("Groq response must be a JSON object")
        return decoded

    def _post_with_connect_retry(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> httpx.Response:
        for attempt in range(2):
            try:
                return self.client.post(
                    url,
                    json=dict(payload),
                    headers=dict(headers),
                    timeout=timeout_seconds,
                )
            except httpx.ConnectError:
                if attempt == 1:
                    raise
                time.sleep(1)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class GroqUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class GroqProvider:
    def __init__(
        self,
        transport: GroqTransport,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip() or max_tokens <= 0 or timeout_seconds <= 0 or context_tokens <= 0:
            raise ValueError("invalid Groq model or numeric configuration")
        if not 0 <= temperature <= 1 or token_safety_factor < 1:
            raise ValueError("invalid Groq temperature or token safety factor")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("invalid Groq reasoning effort")
        if max_attempts < 1 or retry_backoff_seconds < 0:
            raise ValueError("invalid Groq retry configuration")
        normalized_url = base_url.rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Groq base_url must be an HTTPS URL")
        self.transport = transport
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.model_id = f"groq/{self.model}"
        self.base_url = normalized_url
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.token_safety_factor = token_safety_factor
        self.reasoning_effort = reasoning_effort
        self.last_usage: GroqUsage | None = None
        self.last_request_payload: Mapping[str, Any] | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        transport: GroqTransport | None = None,
    ) -> "GroqProvider":
        values = os.environ if environ is None else environ
        key = values.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ProviderError("GROQ_API_KEY is required")
        try:
            return cls(
                transport or HttpxGroqTransport(),
                key,
                values.get("GROQ_MODEL", DEFAULT_MODEL),
                base_url=values.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
                max_tokens=int(values.get("GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                max_attempts=int(
                    values.get("GROQ_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
                ),
                temperature=float(values.get("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE)),
                timeout_seconds=float(values.get("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
                context_tokens=int(values.get("GROQ_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)),
                token_safety_factor=float(values.get("GROQ_TOKEN_SAFETY_FACTOR", DEFAULT_TOKEN_SAFETY_FACTOR)),
                reasoning_effort=values.get(
                    "GROQ_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
                ).strip().casefold(),
            )
        except ValueError as exc:
            raise ProviderError("Invalid Groq configuration") from exc

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, None)

    def generate_structured(self, prompt: str, schema: Mapping[str, Any]) -> str:
        return self._generate(prompt, schema)

    def _generate(self, prompt: str, schema: Mapping[str, Any] | None) -> str:
        """Send one request, retrying only failures a retry can actually clear.

        A single transient failure used to abort a whole evaluation run and, in
        service, to fail a user's question outright.
        """
        last_error: TransientProviderError | None = None
        for attempt in range(self.max_attempts):
            if attempt:
                time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
            try:
                return self._generate_once(prompt, schema)
            except TransientProviderError as error:
                last_error = error
        raise last_error

    def _generate_once(self, prompt: str, schema: Mapping[str, Any] | None) -> str:
        if not prompt.strip():
            raise ProviderError("Groq prompt must not be empty")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "include_reasoning": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_response", "schema": dict(schema)},
            }
        self.last_request_payload = payload
        response = self.transport.post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SDGE-ChatBot/0.1",
            },
            self.timeout_seconds,
        )
        self.last_usage = _extract_usage(response)
        return _extract_text(response)


def _extract_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Groq response is missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("Groq response contains no text")
    return content.strip()


def _extract_usage(response: Mapping[str, Any]) -> GroqUsage:
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    timing = response.get("x_groq")
    timing = timing if isinstance(timing, Mapping) else {}
    total_time = timing.get("total_time")
    latency = round(total_time * 1000) if isinstance(total_time, (int, float)) else None
    return GroqUsage(
        input_tokens=_optional_int(usage.get("prompt_tokens")),
        output_tokens=_optional_int(usage.get("completion_tokens")),
        latency_ms=latency,
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_http_error_code(response: httpx.Response) -> str | None:
    """Extract only Groq's non-secret machine-readable error code."""
    try:
        decoded = response.json()
        error = decoded.get("error") if isinstance(decoded, Mapping) else None
        code = error.get("code") if isinstance(error, Mapping) else None
    except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(code, str):
        return None
    safe = "".join(char for char in code if char.isalnum() or char in "_-.")
    return safe[:80] or None
