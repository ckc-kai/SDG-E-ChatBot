"""Task 3 grounded answer generation."""

from generation.adapters import adapt_ranked_results
from generation.providers import ModelProvider, ProviderError, RecordingScriptedMockProvider
from generation.schemas import AnswerRequest, AnswerResponse, Chunk, ChunkMetadata, ErrorResponse
from generation.service import AnswerService

__all__ = [
    "AnswerRequest",
    "AnswerResponse",
    "AnswerService",
    "Chunk",
    "ChunkMetadata",
    "ErrorResponse",
    "ModelProvider",
    "ProviderError",
    "RecordingScriptedMockProvider",
    "adapt_ranked_results",
]
