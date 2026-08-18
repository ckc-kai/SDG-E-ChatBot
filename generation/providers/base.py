"""Provider boundary for future Bedrock integration."""

from __future__ import annotations

from typing import Protocol

from generation.providers.capabilities import ModelCapabilities


class ProviderError(RuntimeError):
    """A model-provider failure safe for AnswerService to handle."""


class ModelProvider(Protocol):
    model_id: str
    capabilities: ModelCapabilities

    def generate(self, prompt: str) -> str:
        """Return the model's raw text response."""
        ...
