"""Environment-based provider selection for Task 4 and command-line entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping

from generation.providers.base import ModelProvider, ProviderError
from generation.providers.bedrock import BedrockProvider
from generation.providers.groq import GroqProvider
from generation.providers.ollama import OllamaProvider


def create_provider_from_env(
    provider_name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelProvider:
    """Create the configured real model provider without making a model call."""
    values = os.environ if environ is None else environ
    selected = (provider_name or values.get("TASK3_PROVIDER", "")).strip().casefold()
    if selected == "ollama":
        return OllamaProvider.from_env(environ=values)
    if selected == "bedrock":
        return BedrockProvider.from_env(environ=values)
    if selected == "groq":
        return GroqProvider.from_env(environ=values)
    if not selected:
        raise ProviderError("TASK3_PROVIDER is required")
    raise ProviderError(f"Unsupported TASK3_PROVIDER: {selected}")
