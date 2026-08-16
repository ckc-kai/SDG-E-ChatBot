
from pydantic import BaseModel


class Filters(BaseModel):
    content_types: list[str] | None = None   # e.g. ["narrative", "table"]
    content_type: str | None = None           # legacy single-type form


class AskRequest(BaseModel):
    question: str
    filters: Filters | None = None
    # "off" disables automatic question-planning, "always" forces it,
    # omitted/"auto" plans only genuinely complex questions.
    rewrite_mode: str | None = None


class Citation(BaseModel):
    chunk_id: str
    source_pdf: str
    page_start: int   # zero-based; page_start + 1 = real first PDF page
    page_end: int      # exclusive
    breadcrumb: str
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    cited_chunk_ids: list[str]
    citations: list[Citation]
    insufficient_context: bool


class AskErrorResponse(BaseModel):
    request_id: str
    error: str


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    page_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]