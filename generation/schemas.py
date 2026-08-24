"""Framework-neutral Task 3 data contracts.

Task 4 may wrap these dataclasses in Pydantic models. Keeping the core contract
free of FastAPI/Pydantic lets Task 3 run before the backend dependencies land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkMetadata:
    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    revision: str | None = None
    sub_document: str | None = None
    breadcrumb: str | None = None
    section_number: str | None = None
    content_type: str | None = None
    chunk_index: int | None = None
    token_count: int | None = None
    distance: float | None = None
    rerank_score: float | None = None
    contributing_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_id: str
    content: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if not self.content.strip():
            raise ValueError("chunk content must not be empty")


@dataclass(frozen=True)
class AnswerRequest:
    request_id: str
    question: str
    chunks: tuple[Chunk, ...]
    # Set only on a refusal review. The first pass reported insufficient
    # context; this states what the record actually holds, so the second pass
    # answers the supported part instead of the whole question or nothing.
    # Absent by default, and an absent field leaves the rendered prompt
    # byte-identical to before this existed.
    evidence_notice: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique within a request")


@dataclass(frozen=True)
class ModelAnswer:
    """The only fields the model is allowed to decide."""

    answer: str
    cited_chunk_ids: tuple[str, ...]
    insufficient_context: bool
    answered_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source_pdf: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    breadcrumb: str | None = None
    contributing_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerTimings:
    """Internal performance data; never included in the public response."""

    adapter_ms: int = 0
    prompt_build_ms: int = 0
    model_call_ms: int = 0
    model_reported_ms: int | None = None
    parse_ms: int = 0
    citation_validation_ms: int = 0
    answer_service_total_ms: int = 0
    generation_total_ms: int = 0
    model_input_tokens: int | None = None
    model_output_tokens: int | None = None


@dataclass(frozen=True)
class AnswerResponse:
    request_id: str
    answer: str
    cited_chunk_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    insufficient_context: bool
    model_id: str
    latency_ms: int
    warnings: tuple[str, ...] = ()
    timings: AnswerTimings | None = None
    answered_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable structure for Task 4."""
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """Return the minimal response contract exposed by Task 4."""
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "citations": [asdict(citation) for citation in self.citations],
            "insufficient_context": self.insufficient_context,
        }


@dataclass(frozen=True)
class ErrorResponse:
    request_id: str
    error: str = "answer_generation_failed"
    timings: AnswerTimings | None = None

    def to_public_dict(self) -> dict[str, str]:
        return {"request_id": self.request_id, "error": self.error}
