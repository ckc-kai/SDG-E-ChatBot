"""Public FastAPI request and response models.

The response mirrors Task 3's minimal public contract.  Retrieval scores and
full chunk text stay inside the backend; only validated citations are exposed
to the browser.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Filters(BaseModel):
    content_type: Literal["narrative", "table", "figure", "excel_card"] | None = None
    content_types: list[
        Literal["narrative", "table", "figure", "excel_card"]
    ] | None = Field(default=None, min_length=1)
    section_number: str | None = None
    page: int | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=False)

    request_id: str | None = None
    question: str = Field(min_length=1)
    filters: Filters | None = None
    embedding_mode: Literal["raw", "contextual", "hybrid"] | None = None
    rewrite_mode: Literal["auto", "off", "always"] | None = None


class Citation(BaseModel):
    chunk_id: str
    source_pdf: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    breadcrumb: str | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    cited_chunk_ids: list[str]
    citations: list[Citation]
    insufficient_context: bool


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    page_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]


class ErrorResponse(BaseModel):
    request_id: str
    error: str
