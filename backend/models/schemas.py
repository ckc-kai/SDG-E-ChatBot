"""
Role: Pydantic request/response shapes for the API. 
- data-shape definitions
- validate incoming requests automatically (rejects malformed input with 422 error ) 
- auto-generate interactive docs at /docs.

Field names/types derived directly from QueryObject/RankedResult dataclasses in retrieval/query.py
"""
from pydantic import BaseModel


class Filters(BaseModel):

    content_type: str | None = None   # narrative | table | figure
    section_number: str | None = None
    page: int | None = None


class AskRequest(BaseModel):
    question: str
    filters: Filters | None = None
    embedding_mode: str | None = None   # raw | contextual | hybrid
    rewrite_mode: str | None = None     # auto | off | always


class Source(BaseModel):
    doc_id: int                # QueryObject.chunk_id
    source_pdf: str
    breadcrumb: str
    section_number: str | None
    page_start: int
    page_end: int
    content_type: str          # narrative | table | figure
    snippet: str                # QueryObject.content, truncated for display
    caption: str | None
    object_key: str | None      # present for figures -- an S3 reference
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    page_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None