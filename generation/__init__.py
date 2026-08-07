"""Task 3 grounded answer generation."""

from generation.adapters import adapt_ranked_results
from generation.providers import (
    BedrockProvider,
    BedrockUsage,
    ModelProvider,
    OllamaProvider,
    OllamaUsage,
    ProviderError,
    RecordingScriptedMockProvider,
    create_provider_from_env,
)
from generation.schemas import AnswerRequest, AnswerResponse, Chunk, ChunkMetadata, ErrorResponse
from generation.service import AnswerService

__all__ = [
    "AnswerRequest",
    "AnswerResponse",
    "AnswerService",
    "BedrockProvider",
    "BedrockUsage",
    "Chunk",
    "ChunkMetadata",
    "ErrorResponse",
    "ModelProvider",
    "OllamaProvider",
    "OllamaUsage",
    "ProviderError",
    "RecordingScriptedMockProvider",
    "adapt_ranked_results",
    "create_provider_from_env",
]
