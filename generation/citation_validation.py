"""Validate model-selected IDs and hydrate trusted citation metadata."""

from __future__ import annotations

from generation.schemas import AnswerRequest, Citation, ModelAnswer


class CitationValidationError(ValueError):
    """Raised when a supported answer has no valid citation."""


def validate_and_hydrate_citations(
    request: AnswerRequest, model_answer: ModelAnswer
) -> tuple[tuple[str, ...], tuple[Citation, ...], tuple[str, ...]]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in request.chunks}
    valid_ids: list[str] = []
    warnings: list[str] = []

    for chunk_id in model_answer.cited_chunk_ids:
        if chunk_id not in chunks_by_id:
            warnings.append(f"Model cited unknown chunk_id: {chunk_id}")
            continue
        if chunk_id not in valid_ids:
            valid_ids.append(chunk_id)

    if not model_answer.insufficient_context and not valid_ids:
        details = "; ".join(warnings + ["Answer has no valid supporting citation"])
        raise CitationValidationError(details)

    citations = []
    for chunk_id in valid_ids:
        chunk = chunks_by_id[chunk_id]
        metadata = chunk.metadata
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                source_pdf=metadata.source_file,
                page_start=metadata.page_start,
                page_end=metadata.page_end,
                sheet=metadata.sheet,
                row_start=metadata.row_start,
                row_end=metadata.row_end,
                breadcrumb=metadata.breadcrumb,
                contributing_sources=metadata.contributing_sources,
            )
        )
    return tuple(valid_ids), tuple(citations), tuple(warnings)
