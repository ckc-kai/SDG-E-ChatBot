"""DeepSeek chat-completions provider with JSON-object output."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from generation.providers.base import ProviderError
from generation.providers.capabilities import ModelCapabilities
from generation.providers.tracing import ModelCallTrace, configured_trace_dir


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 1_500
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_CONTEXT_TOKENS = 1_000_000
DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 384_000
# Keep the first DeepSeek comparison aligned with the existing Groq run. The
# model supports a much larger window, but retrieval quality should be compared
# with the same evidence budget before deliberately increasing this value.
DEFAULT_PROMPT_TOKEN_BUDGET = 6_500
DEFAULT_TOKEN_SAFETY_FACTOR = 1.2
DEFAULT_THINKING = "disabled"


class DeepSeekTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class HttpxDeepSeekTransport:
    """Small HTTP boundary so provider behavior can be tested without an API call."""

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
            code = _safe_http_error_code(exc.response)
            suffix = f" ({code})" if code else ""
            if exc.response.status_code == 429:
                raise ProviderError(
                    f"DeepSeek rate limit reached{suffix}"
                ) from exc
            if exc.response.status_code == 402:
                raise ProviderError(
                    f"DeepSeek account balance is insufficient{suffix}"
                ) from exc
            raise ProviderError(
                f"DeepSeek request failed with HTTP "
                f"{exc.response.status_code}{suffix}"
            ) from exc
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            raise ProviderError("DeepSeek request failed") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("DeepSeek returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderError("DeepSeek response must be a JSON object")
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
class DeepSeekUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    latency_ms: int | None = None


class DeepSeekProvider:
    def __init__(
        self,
        transport: DeepSeekTransport,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        model_max_output_tokens: int = DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
        prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
        token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
        thinking: str = DEFAULT_THINKING,
        trace_dir: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if (
            not model.strip()
            or max_tokens <= 0
            or timeout_seconds <= 0
            or context_tokens <= 0
            or model_max_output_tokens <= 0
        ):
            raise ValueError("invalid DeepSeek model or numeric configuration")
        if max_tokens > model_max_output_tokens:
            raise ValueError("requested output exceeds DeepSeek model limit")
        if prompt_token_budget <= 0 or prompt_token_budget + max_tokens > context_tokens:
            raise ValueError("invalid DeepSeek prompt token budget")
        if not 0 <= temperature <= 2 or token_safety_factor < 1:
            raise ValueError("invalid DeepSeek temperature or token safety factor")
        normalized_thinking = thinking.strip().casefold()
        if normalized_thinking not in {"enabled", "disabled"}:
            raise ValueError("DeepSeek thinking must be enabled or disabled")
        normalized_url = base_url.rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("DeepSeek base_url must be an HTTPS URL")

        self.transport = transport
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.model_id = f"deepseek/{self.model}"
        self.base_url = normalized_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.prompt_token_budget = prompt_token_budget
        self.token_safety_factor = token_safety_factor
        self.thinking = normalized_thinking
        self.trace_dir = trace_dir
        self.last_usage: DeepSeekUsage | None = None
        self.last_request_payload: Mapping[str, Any] | None = None
        self.last_raw_text: str | None = None
        self.capabilities = ModelCapabilities(
            context_window=context_tokens,
            max_output_tokens=model_max_output_tokens,
            requested_output_tokens=max_tokens,
            prompt_token_budget=prompt_token_budget,
        )

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        transport: DeepSeekTransport | None = None,
    ) -> "DeepSeekProvider":
        values = os.environ if environ is None else environ
        key = values.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ProviderError("DEEPSEEK_API_KEY is required")
        try:
            return cls(
                transport or HttpxDeepSeekTransport(),
                key,
                values.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                base_url=values.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
                max_tokens=int(
                    values.get("DEEPSEEK_MAX_TOKENS", DEFAULT_MAX_TOKENS)
                ),
                temperature=float(
                    values.get("DEEPSEEK_TEMPERATURE", DEFAULT_TEMPERATURE)
                ),
                timeout_seconds=float(
                    values.get(
                        "DEEPSEEK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
                    )
                ),
                context_tokens=int(
                    values.get("DEEPSEEK_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)
                ),
                model_max_output_tokens=int(
                    values.get(
                        "DEEPSEEK_MODEL_MAX_OUTPUT_TOKENS",
                        DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
                    )
                ),
                prompt_token_budget=int(
                    values.get(
                        "DEEPSEEK_PROMPT_TOKEN_BUDGET",
                        DEFAULT_PROMPT_TOKEN_BUDGET,
                    )
                ),
                token_safety_factor=float(
                    values.get(
                        "DEEPSEEK_TOKEN_SAFETY_FACTOR",
                        DEFAULT_TOKEN_SAFETY_FACTOR,
                    )
                ),
                thinking=values.get("DEEPSEEK_THINKING", DEFAULT_THINKING),
                trace_dir=configured_trace_dir(values),
            )
        except ValueError as exc:
            raise ProviderError("Invalid DeepSeek configuration") from exc

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, structured=False, schema=None)

    def generate_structured(
        self, prompt: str, schema: Mapping[str, Any]
    ) -> str:
        # DeepSeek supports JSON-object mode, but not the json_schema payload
        # used by Groq. Put the exact contract in the prompt so JSON-object
        # mode cannot silently rename fields (for example ``excel`` instead of
        # ``need_excel``). Callers still validate the returned object locally.
        schema_prompt = (
            f"{prompt}\n\nYour response MUST conform exactly to this JSON Schema. "
            "Use only the property names defined by the schema and return no "
            "additional keys:\n"
            f"{json.dumps(dict(schema), ensure_ascii=False, separators=(',', ':'))}"
        )
        return self._generate(schema_prompt, structured=True, schema=schema)

    def _generate(
        self,
        prompt: str,
        *,
        structured: bool,
        schema: Mapping[str, Any] | None,
    ) -> str:
        if not prompt.strip():
            raise ProviderError("DeepSeek prompt must not be empty")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": self.thinking},
        }
        if structured:
            payload["response_format"] = {"type": "json_object"}
        self.last_request_payload = payload
        self.last_usage = None
        self.last_raw_text = None
        with ModelCallTrace(
            model_id=self.model_id,
            prompt=prompt,
            request_payload=payload,
            trace_dir=self.trace_dir,
            seed=None,
            schema=schema,
        ) as trace:
            started = time.perf_counter()
            response = self.transport.post_json(
                f"{self.base_url}/chat/completions",
                payload,
                self._headers(),
                self.timeout_seconds,
            )
            trace.capture_response(response)
            latency_ms = round((time.perf_counter() - started) * 1000)
            self.last_usage = _extract_usage(response, latency_ms=latency_ms)
            self.last_raw_text = _extract_text(response)
            trace.succeed(
                raw_output=self.last_raw_text,
                usage=self.last_usage,
            )
            return self.last_raw_text

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SDGE-ChatBot/0.1",
        }


def _extract_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("DeepSeek response is missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("DeepSeek response contains no text")
    return content.strip()


def _extract_usage(
    response: Mapping[str, Any], *, latency_ms: int
) -> DeepSeekUsage:
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return DeepSeekUsage(
        input_tokens=_optional_int(usage.get("prompt_tokens")),
        output_tokens=_optional_int(usage.get("completion_tokens")),
        cache_hit_tokens=_optional_int(usage.get("prompt_cache_hit_tokens")),
        cache_miss_tokens=_optional_int(usage.get("prompt_cache_miss_tokens")),
        latency_ms=latency_ms,
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_http_error_code(response: httpx.Response) -> str | None:
    """Extract only a non-secret, machine-readable DeepSeek error code."""
    try:
        decoded = response.json()
        error = decoded.get("error") if isinstance(decoded, Mapping) else None
        code = error.get("code") if isinstance(error, Mapping) else None
    except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(code, str):
        return None
    safe = "".join(char for char in code if char.isalnum() or char in "_-." )
    return safe[:80] or None
