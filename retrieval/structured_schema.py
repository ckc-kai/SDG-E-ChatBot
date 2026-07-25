"""PostgreSQL schema for structured (table/figure) chunks.

Additive companion to `retrieval.setup_db`: the narrative `chunks` table and its
constraints are intentionally left untouched. Structured content lives in its
own `structured_chunks` table so text retrieval keeps its existing behavior and
unique keys, while structured retrieval can be enabled, re-extracted, or dropped
independently.

Tables
------
structured_chunks
    One row per embedded table/figure piece. Mirrors the narrative `chunks`
    retrieval columns (both raw and contextual vectors) and adds:
      - content_type:     'table' | 'figure'
      - structured_data:  exact table grid as JSONB (tables only)
      - image_path:       cropped figure PNG relative to the project root
                          (figures only)
      - caption:          the element's own caption text, when present
    Stable key: (document_id, content_type, page_start, chunk_index) -- the
    same shape the eval harness uses for narrative chunks, with content_type
    added because a table piece and a figure piece can share a page.

structured_ingest_state
    Per-document skip/refresh bookkeeping for the structured pass, independent
    of the narrative ingest's content_hash logic. `extractor_signature` encodes
    the extractor version and options so changing them triggers re-extraction
    even when the PDF bytes are unchanged.
"""

from __future__ import annotations

import psycopg2

from retrieval.utils import DB_CONFIG, load_config

EMBEDDING_DIMENSIONS = load_config()["local"]["embedding"]["dimensions"]

STRUCTURED_CONTENT_TYPES = ("table", "figure")


def setup_structured_database() -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS structured_chunks (
                    id serial PRIMARY KEY,
                    document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    sub_document text,
                    breadcrumb text,
                    section_number text,
                    page_start int NOT NULL,
                    page_end int NOT NULL,
                    chunk_index int NOT NULL,
                    content_type text NOT NULL
                        CHECK (content_type IN ('table', 'figure')),
                    content text NOT NULL,
                    caption text,
                    structured_data jsonb,
                    image_path text,
                    token_count int NOT NULL,
                    embedding_model text NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                    contextual_embedding_model text,
                    contextual_embedding_recipe text,
                    contextual_embedding vector({EMBEDDING_DIMENSIONS}),
                    extractor text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (document_id, content_type, page_start, chunk_index)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_ingest_state (
                    document_id int PRIMARY KEY
                        REFERENCES documents(id) ON DELETE CASCADE,
                    content_hash text NOT NULL,
                    extractor_signature text NOT NULL,
                    table_chunks int NOT NULL,
                    figure_chunks int NOT NULL,
                    ingested_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS structured_chunks_embedding_hnsw_idx
                ON structured_chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS structured_chunks_contextual_embedding_hnsw_idx
                ON structured_chunks USING hnsw (contextual_embedding vector_cosine_ops)
                WHERE contextual_embedding IS NOT NULL;
                """
            )
        conn.commit()
    finally:
        conn.close()
        print("Structured schema ready")


if __name__ == "__main__":
    setup_structured_database()
