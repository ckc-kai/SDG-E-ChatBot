"""Amazon Bedrock Converse provider.

The runtime client is injectable so unit tests never need AWS credentials,
network access, or paid model calls.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from generation.providers.base import ProviderError


DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.0


class BedrockRuntimeClient(Protocol):
    """Small subset of the boto3 Bedrock Runtime client used by Task 3."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class BedrockUsage:
    """Usage metadata kept inside Task 3 for evaluation and cost tracking."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None


class BedrockProvider:
    """Generate raw model text through the model-neutral Converse API."""

    def __init__(
        self,
        client: BedrockRuntimeClient,
        model_id: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_usage: BedrockUsage | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        client: BedrockRuntimeClient | None = None,
    ) -> BedrockProvider:
        """Create a provider from non-secret environment configuration.

        boto3 uses its normal credential chain. Credentials are deliberately
        not read or stored by this class.
        """
        values = os.environ if environ is None else environ
        model_id = values.get("BEDROCK_MODEL_ID", "").strip()
        if not model_id:
            raise ProviderError("BEDROCK_MODEL_ID is required")
        region = values.get("AWS_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION
        try:
            max_tokens = int(values.get("BEDROCK_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
            temperature = float(
                values.get("BEDROCK_TEMPERATURE", str(DEFAULT_TEMPERATURE))
            )
        except ValueError as exc:
            raise ProviderError("Invalid Bedrock inference configuration") from exc

        runtime_client = client or _create_boto3_client(region)
        return cls(
            runtime_client,
            model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ProviderError("Bedrock prompt must not be empty")
        self.last_usage = None
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as exc:
            raise ProviderError("Amazon Bedrock Converse request failed") from exc

        self.last_usage = _extract_usage(response)
        return _extract_text(response)


def _create_boto3_client(region: str) -> BedrockRuntimeClient:
    try:
        boto3 = importlib.import_module("boto3")
    except ImportError as exc:
        raise ProviderError(
            "boto3 is required for live Bedrock calls; inject a client for tests"
        ) from exc
    try:
        return boto3.client("bedrock-runtime", region_name=region)
    except Exception as exc:
        raise ProviderError("Could not create the Amazon Bedrock runtime client") from exc


def _extract_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ProviderError("Bedrock response is missing output message content") from exc
    if not isinstance(content, list):
        raise ProviderError("Bedrock response content must be a list")
    parts = [
        block["text"]
        for block in content
        if isinstance(block, Mapping)
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    if not parts:
        raise ProviderError("Bedrock response contains no text")
    return "".join(parts).strip()


def _extract_usage(response: Mapping[str, Any]) -> BedrockUsage:
    usage = response.get("usage", {})
    metrics = response.get("metrics", {})
    if not isinstance(usage, Mapping):
        usage = {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    return BedrockUsage(
        input_tokens=_optional_int(usage.get("inputTokens")),
        output_tokens=_optional_int(usage.get("outputTokens")),
        total_tokens=_optional_int(usage.get("totalTokens")),
        latency_ms=_optional_int(metrics.get("latencyMs")),
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
