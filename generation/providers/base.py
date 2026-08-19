"""Provider boundary for future Bedrock integration."""

from __future__ import annotations

from typing import Protocol


class ProviderError(RuntimeError):
    """A model-provider failure safe for AnswerService to handle."""


class TransientProviderError(ProviderError):
    """A provider failure that a bounded retry of the same request may clear.

    Kept distinct from ``ProviderError`` so a caller can retry the recoverable
    cases without also retrying a malformed request or a bad credential.
    """


class ModelProvider(Protocol):
    model_id: str

    def generate(self, prompt: str) -> str:
        """Return the model's raw text response."""
        ...
