"""Local Ollama chat provider with schema-constrained JSON output."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from generation.providers.base import ProviderError


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_TOKEN_SAFETY_FACTOR = 1.25
DEFAULT_KEEP_ALIVE = "30m"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "insufficient_context": {"type": "boolean"},
    },
    "required": ["answer", "cited_chunk_ids", "insufficient_context"],
    "additionalProperties": False,
}


class OllamaTransport(Protocol):
    """Small injectable HTTP boundary used by the provider."""

    def post_json(
        self, url: str, payload: Mapping[str, Any], timeout_seconds: float
    ) -> Mapping[str, Any]:
        ...


class UrllibOllamaTransport:
    """Use Python's standard library so Ollama adds no Python dependency."""

    def post_json(
        self, url: str, payload: Mapping[str, Any], timeout_seconds: float
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("Ollama request failed") from exc
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("Ollama returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ProviderError("Ollama response must be a JSON object")
        return decoded


@dataclass(frozen=True)
class OllamaUsage:
    """Local inference metadata retained for evaluation, not public output."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class OllamaProvider:
    """Generate a Task 3 model answer through Ollama's local chat API."""

    def __init__(
        self,
        transport: OllamaTransport,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if token_safety_factor < 1:
            raise ValueError("token_safety_factor must be at least 1")
        if not keep_alive.strip():
            raise ValueError("keep_alive must not be empty")
        normalized_url = base_url.rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) URL")

        self.transport = transport
        self.model = model.strip()
        self.model_id = f"ollama/{self.model}"
        self.base_url = normalized_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.context_tokens = context_tokens
        self.token_safety_factor = token_safety_factor
        self.keep_alive = keep_alive.strip()
        self.last_usage: OllamaUsage | None = None
        self.last_request_payload: Mapping[str, Any] | None = None
        self.last_raw_text: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        transport: OllamaTransport | None = None,
    ) -> OllamaProvider:
        values = os.environ if environ is None else environ
        model = values.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise ProviderError("OLLAMA_MODEL is required")
        base_url = values.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).strip()
        try:
            max_tokens = int(values.get("OLLAMA_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
            temperature = float(
                values.get("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE))
            )
            timeout_seconds = float(
                values.get("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
            )
            context_tokens = int(
                values.get("OLLAMA_CONTEXT_TOKENS", str(DEFAULT_CONTEXT_TOKENS))
            )
            token_safety_factor = float(
                values.get(
                    "OLLAMA_TOKEN_SAFETY_FACTOR",
                    str(DEFAULT_TOKEN_SAFETY_FACTOR),
                )
            )
            keep_alive = values.get("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
            return cls(
                transport or UrllibOllamaTransport(),
                model,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                context_tokens=context_tokens,
                token_safety_factor=token_safety_factor,
                keep_alive=keep_alive,
            )
        except ValueError as exc:
            raise ProviderError("Invalid Ollama configuration") from exc

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ProviderError("Ollama prompt must not be empty")
        self.last_usage = None
        self.last_raw_text = None
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "format": ANSWER_SCHEMA,
            "options": {
                "num_predict": self.max_tokens,
                "num_ctx": self.context_tokens,
                "temperature": self.temperature,
            },
        }
        self.last_request_payload = payload
        response = self.transport.post_json(
            f"{self.base_url}/api/chat",
            payload,
            self.timeout_seconds,
        )
        self.last_usage = _extract_usage(response)
        self.last_raw_text = _extract_text(response)
        return self.last_raw_text

    def warmup(self) -> None:
        """Load the configured model and keep it resident without answering."""
        payload = {
            "model": self.model,
            "prompt": "",
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        self.transport.post_json(
            f"{self.base_url}/api/generate",
            payload,
            self.timeout_seconds,
        )


def _extract_text(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise ProviderError("Ollama response is missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("Ollama response contains no text")
    return content.strip()


def _extract_usage(response: Mapping[str, Any]) -> OllamaUsage:
    duration_ns = _optional_int(response.get("total_duration"))
    return OllamaUsage(
        input_tokens=_optional_int(response.get("prompt_eval_count")),
        output_tokens=_optional_int(response.get("eval_count")),
        latency_ms=round(duration_ns / 1_000_000) if duration_ns is not None else None,
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
