"""
Role: Wraps retrieval.query.retrieve() and translates its
output (QueryObject / RankedResult) into source schema
"""
import logging

from retrieval.query import retrieve
from retrieval.utils import connect_db

from models.schemas import Source

logger = logging.getLogger(__name__)

# Snippets are truncated to this length for display...
# do we need the full content here ? 
SNIPPET_MAX_CHARS = 500


class RetrievalService:
    def __init__(self):
        self._conn = connect_db()

    def retrieve(
        self,
        question: str,
        embedding_mode: str | None = None,
        rewrite_mode: str | None = None,
    ) -> list[Source]:
        kwargs = {}
        if embedding_mode is not None:
            kwargs["embedding_mode"] = embedding_mode
        if rewrite_mode is not None:
            kwargs["rewrite_mode"] = rewrite_mode

        ranked_results = retrieve(question, self._conn, **kwargs)
        return [self._to_source(r) for r in ranked_results]

    def _to_source(self, ranked_result) -> Source:
        qo = ranked_result.query_object
        return Source(
            doc_id=qo.chunk_id,
            source_pdf=qo.source_pdf,
            breadcrumb=qo.breadcrumb,
            section_number=qo.section_number,
            page_start=qo.page_start,
            page_end=qo.page_end,
            content_type=qo.content_type,
            snippet=qo.content[:SNIPPET_MAX_CHARS],
            caption=qo.caption,
            object_key=qo.object_key,
            rerank_score=ranked_result.rerank_score,
        )