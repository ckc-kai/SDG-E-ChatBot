"""Groq chat-completions provider with schema-constrained JSON output."""

from __future__ import annotations

import json
import os
import time
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from generation.providers.base import ProviderError
from generation.providers.capabilities import (
    ModelCapabilities,
    RateLimitState,
    rate_limit_state_from_headers,
)
from generation.providers.ollama import ANSWER_SCHEMA
from generation.providers.tracing import (
    DEFAULT_MODEL_SEED,
    ModelCallTrace,
    configured_seed,
    configured_trace_dir,
)

import httpx


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOKENS = 131_072
DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 65_536
# Conservative first-request budget. After Groq returns rate-limit headers,
# later requests use the selected model and account's discovered limits.
DEFAULT_PROMPT_TOKEN_BUDGET = 6_500
DEFAULT_TOKEN_SAFETY_FACTOR = 1.2
DEFAULT_REASONING_EFFORT = "low"
# User-facing requests may wait for a short provider reset, but should not
# remain blocked for a long quota window.
DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 15.0


class GroqTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class HttpxGroqTransport:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client()
        self.last_response_headers: Mapping[str, str] = {}

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
            self.last_response_headers = dict(response.headers)
            response.raise_for_status()
            decoded = response.json()
        except httpx.HTTPStatusError as exc:
            error_code = _safe_http_error_code(exc.response)
            suffix = f" ({error_code})" if error_code else ""
            if exc.response.status_code == 429:
                raise ProviderError(
                    "Groq free-tier rate limit reached; no automatic retry was made"
                    f"{suffix}"
                ) from exc
            raise ProviderError(
                f"Groq request failed with HTTP {exc.response.status_code}{suffix}"
            ) from exc
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            raise ProviderError("Groq request failed") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("Groq returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderError("Groq response must be a JSON object")
        return decoded

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        try:
            response = self.client.get(url, headers=dict(headers), timeout=timeout_seconds)
            self.last_response_headers = dict(response.headers)
            response.raise_for_status()
            decoded = response.json()
        except (httpx.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ProviderError("Groq model metadata request failed") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderError("Groq model metadata must be a JSON object")
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
        prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
        token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_rate_limit_wait_seconds: float = DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS,
        seed: int = DEFAULT_MODEL_SEED,
        trace_dir: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip() or max_tokens <= 0 or timeout_seconds <= 0 or context_tokens <= 0:
            raise ValueError("invalid Groq model or numeric configuration")
        if prompt_token_budget <= 0 or prompt_token_budget + max_tokens > context_tokens:
            raise ValueError("invalid Groq prompt token budget")
        if not 0 <= temperature <= 1 or token_safety_factor < 1:
            raise ValueError("invalid Groq temperature or token safety factor")
        if max_rate_limit_wait_seconds < 0:
            raise ValueError("invalid Groq rate-limit wait")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("invalid Groq reasoning effort")
        if seed < 0:
            raise ValueError("Groq seed must not be negative")
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
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.prompt_token_budget = prompt_token_budget
        self.initial_prompt_token_budget = prompt_token_budget
        self.token_safety_factor = token_safety_factor
        self.reasoning_effort = reasoning_effort
        self.max_rate_limit_wait_seconds = max_rate_limit_wait_seconds
        self.seed = seed
        self.trace_dir = trace_dir
        self.last_usage: GroqUsage | None = None
        self.last_rate_limit = RateLimitState()
        self.last_request_payload: Mapping[str, Any] | None = None
        self.last_raw_text: str | None = None
        self.capabilities = ModelCapabilities(
            context_window=context_tokens,
            max_output_tokens=DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
            requested_output_tokens=max_tokens,
            prompt_token_budget=prompt_token_budget,
        )
        self._capabilities_discovered = False

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
                temperature=float(values.get("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE)),
                timeout_seconds=float(values.get("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
                context_tokens=int(values.get("GROQ_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)),
                prompt_token_budget=int(
                    values.get(
                        "GROQ_PROMPT_TOKEN_BUDGET",
                        DEFAULT_PROMPT_TOKEN_BUDGET,
                    )
                ),
                token_safety_factor=float(values.get("GROQ_TOKEN_SAFETY_FACTOR", DEFAULT_TOKEN_SAFETY_FACTOR)),
                reasoning_effort=values.get(
                    "GROQ_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
                ).strip().casefold(),
                max_rate_limit_wait_seconds=float(
                    values.get(
                        "GROQ_MAX_RATE_LIMIT_WAIT_SECONDS",
                        DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS,
                    )
                ),
                seed=configured_seed(values, "groq"),
                trace_dir=configured_trace_dir(values),
            )
        except ValueError as exc:
            raise ProviderError("Invalid Groq configuration") from exc

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, None)

    def generate_structured(self, prompt: str, schema: Mapping[str, Any]) -> str:
        return self._generate(prompt, schema)

    def _generate(self, prompt: str, schema: Mapping[str, Any] | None) -> str:
        if not prompt.strip():
            raise ProviderError("Groq prompt must not be empty")
        self.refresh_capabilities()
        available_requests = self.last_rate_limit.available_requests()
        if available_requests is not None and available_requests <= 0:
            raise ProviderError("Groq request quota is temporarily exhausted")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "include_reasoning": False,
            "seed": self.seed,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_response", "schema": dict(schema)},
            }
        self.last_request_payload = payload
        self.last_usage = None
        self.last_raw_text = None
        with ModelCallTrace(
            model_id=self.model_id,
            prompt=prompt,
            request_payload=payload,
            trace_dir=self.trace_dir,
            seed=self.seed,
            schema=schema,
        ) as trace:
            try:
                response = self.transport.post_json(
                    f"{self.base_url}/chat/completions",
                    payload,
                    self._headers(),
                    self.timeout_seconds,
                )
            finally:
                self._capture_rate_limits()
            trace.capture_response(response)
            self.last_usage = _extract_usage(response)
            self.last_raw_text = _extract_text(response)
            trace.succeed(
                raw_output=self.last_raw_text,
                usage=self.last_usage,
            )
            return self.last_raw_text

    def refresh_capabilities(self) -> ModelCapabilities:
        """Discover model limits once, retaining configured fallbacks on failure."""
        if self._capabilities_discovered:
            return self.capabilities
        self._capabilities_discovered = True
        get_json = getattr(self.transport, "get_json", None)
        if not callable(get_json):
            return self.capabilities
        try:
            metadata = get_json(
                f"{self.base_url}/models/{quote(self.model, safe='')}",
                self._headers(),
                self.timeout_seconds,
            )
            context_window = _optional_int(metadata.get("context_window"))
            max_completion = _optional_int(metadata.get("max_completion_tokens"))
            if context_window is None or max_completion is None:
                return self.capabilities
            requested_output = min(self.max_tokens, max_completion)
            prompt_budget = context_window - requested_output
            self.context_tokens = context_window
            self.max_tokens = requested_output
            self.prompt_token_budget = prompt_budget
            self.capabilities = ModelCapabilities(
                context_window=context_window,
                max_output_tokens=max_completion,
                requested_output_tokens=requested_output,
                prompt_token_budget=prompt_budget,
            )
        except (ProviderError, ValueError, TypeError):
            pass
        return self.capabilities

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SDGE-ChatBot/0.1",
        }

    def _capture_rate_limits(self) -> None:
        headers = getattr(self.transport, "last_response_headers", {})
        if isinstance(headers, Mapping):
            self.last_rate_limit = rate_limit_state_from_headers(headers)

    def available_prompt_token_budget(self) -> int:
        """Return the immediate budget without waiting for quota recovery."""
        remaining = self.last_rate_limit.available_tokens()
        if remaining is None:
            return min(
                self.initial_prompt_token_budget,
                self.capabilities.prompt_token_budget,
            )
        return min(
            self.capabilities.prompt_token_budget,
            max(1, remaining - self.max_tokens),
        )

    def acquire_prompt_token_budget(self, required_tokens: int) -> int:
        """Wait for a temporary TPM reset, then return the safe input budget.

        The first request uses the configured 6,500-token fallback because no
        account quota headers exist yet. Once headers are known, a request that
        fits the full TPM window waits for recovery instead of discarding good
        chunks. Evidence larger than the model/account window is left to the
        ranked prompt selector to trim.
        """
        if required_tokens <= 0:
            raise ValueError("required_tokens must be positive")
        state = self.last_rate_limit
        if state.token_limit is None:
            return min(
                self.initial_prompt_token_budget,
                self.capabilities.prompt_token_budget,
            )

        restored_budget = min(
            self.capabilities.prompt_token_budget,
            max(1, state.token_limit - self.max_tokens),
        )
        needed = min(required_tokens, restored_budget)
        remaining = state.available_tokens()
        if remaining is None or max(0, remaining - self.max_tokens) >= needed:
            return restored_budget

        wait_seconds = state.seconds_until_token_reset()
        if (
            wait_seconds is None
            or wait_seconds > self.max_rate_limit_wait_seconds
        ):
            raise ProviderError("Groq token quota is temporarily exhausted")
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        return restored_budget


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
