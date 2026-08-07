"""
Role: defines GET /api/documents 
- a browsable list of ingested documents, independent of the Q&A flow
- queries the `documents` table directly 

KNOWN GAP: page_count has no obvious source column in what's been shared
so far
"""
from fastapi import APIRouter

from models.schemas import DocumentMetadata, DocumentsResponse
from retrieval.utils import connect_db

router = APIRouter()


@router.get("/api/documents", response_model=DocumentsResponse)
def list_documents():
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename FROM documents")
            rows = cur.fetchall()
    finally:
        conn.close()

    return DocumentsResponse(
        documents=[
            DocumentMetadata(
                doc_id=str(row[0]),
                title=row[1],
                # TODO: no page_count column confirmed yet 
                page_count=0,
            )
            for row in rows
        ]
    )