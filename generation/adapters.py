"""Adapters from Task 2 retrieval results to the Task 3 contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from generation.schemas import Chunk, ChunkMetadata


_MISSING = object()


def _read(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is _MISSING:
        raise ValueError(f"Task 2 result is missing required field {name!r}")
    return default


def adapt_ranked_result(result: Any) -> Chunk:
    """Adapt one Task 2 ``RankedResult`` (or an equivalent mapping).

    The adapter matches ``origin/ckc_dev`` without importing it: the outer
    object has ``query_object`` and ``rerank_score``; the query object has
    ``chunk_id``, ``source_pdf``, ``content`` and retrieval metadata.
    """
    query_object = _read(result, "query_object", result)
    source_file = str(_read(query_object, "source_pdf"))
    chunk_id = str(_read(query_object, "chunk_id"))
    metadata = ChunkMetadata(
        source_file=source_file,
        page_start=_read(query_object, "page_start", None),
        page_end=_read(query_object, "page_end", None),
        sub_document=_read(query_object, "sub_document", None),
        breadcrumb=_read(query_object, "breadcrumb", None),
        section_number=_read(query_object, "section_number", None),
        content_type=_read(query_object, "content_type", None),
        chunk_index=_read(query_object, "chunk_index", None),
        token_count=_read(query_object, "token_count", None),
        distance=_read(query_object, "distance", None),
        rerank_score=_read(result, "rerank_score", None),
    )
    return Chunk(
        # Task 2 does not return documents.id. Filename is the least-lossy
        # source identifier available today and remains separately copied as
        # citation metadata.
        source_id=source_file,
        chunk_id=chunk_id,
        content=str(_read(query_object, "content")),
        metadata=metadata,
    )


def adapt_ranked_results(results: Iterable[Any]) -> tuple[Chunk, ...]:
    return tuple(adapt_ranked_result(result) for result in results)
