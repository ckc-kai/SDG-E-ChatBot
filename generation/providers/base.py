"""Provider boundary for future Bedrock integration."""

from __future__ import annotations

from typing import Protocol


class ProviderError(RuntimeError):
    """A model-provider failure safe for AnswerService to handle."""


class ModelProvider(Protocol):
    model_id: str

    def generate(self, prompt: str) -> str:
        """Return the model's raw text response."""
        ...
