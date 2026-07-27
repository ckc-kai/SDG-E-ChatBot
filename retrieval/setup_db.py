"""Unified PostgreSQL/pgvector schema for narrative, table, and figure chunks."""

from __future__ import annotations

import argparse

import psycopg2

from retrieval.utils import database_config, embedding_config

CONTENT_TYPES = ("narrative", "table", "figure")


def setup_database(conn=None) -> None:
    """Create the unified schema on an empty database.

    Existing pre-unification tables are deliberately not mutated here. A clear
    compatibility check tells the operator to perform the documented clean
    schema reset before re-ingesting.
    """
    owns_connection = conn is None
    connection = conn or psycopg2.connect(**database_config())
    dimensions = int(embedding_config()["dimensions"])
    try:
        with connection.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id serial PRIMARY KEY,
                    filename text UNIQUE NOT NULL,
                    page_count int NOT NULL,
                    content_hash text NOT NULL,
                    ingest_signature text NOT NULL,
                    chunk_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
                    ingested_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id serial PRIMARY KEY,
                    document_id int NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    sub_document text,
                    breadcrumb text NOT NULL,
                    section_number text,
                    page_start int NOT NULL,
                    page_end int NOT NULL,
                    chunk_index int NOT NULL,
                    content_type text NOT NULL
                        CHECK (content_type IN ('narrative', 'table', 'figure')),
                    content text NOT NULL,
                    retrieval_hint text,
                    caption text,
                    structured_data jsonb,
                    object_key text,
                    media_type text,
                    token_count int NOT NULL,
                    content_hash text NOT NULL,
                    embedding_provider text NOT NULL,
                    embedding_model text NOT NULL,
                    embedding vector({dimensions}) NOT NULL,
                    contextual_embedding_model text NOT NULL,
                    contextual_embedding_recipe text NOT NULL,
                    contextual_embedding vector({dimensions}) NOT NULL,
                    extractor text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (
                        document_id, content_type, page_start, chunk_index
                    )
                );
                """
            )
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'chunks'
                """
            )
            columns = {row[0] for row in cur.fetchall()}
            required = {
                "retrieval_hint",
                "caption",
                "structured_data",
                "object_key",
                "content_hash",
                "embedding_provider",
                "extractor",
            }
            missing = sorted(required - columns)
            if missing:
                raise RuntimeError(
                    "Legacy chunks schema detected (missing "
                    + ", ".join(missing)
                    + "). Run `python -m retrieval.setup_db --reset --yes`, "
                    "then perform a clean re-ingest."
                )
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'chunks'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            )
            vector_type = cur.fetchone()
            expected_type = f"vector({dimensions})"
            if not vector_type or vector_type[0] != expected_type:
                actual = vector_type[0] if vector_type else "missing"
                raise RuntimeError(
                    f"Embedding schema is {actual}, but YAML config requires "
                    f"{expected_type}. Reset the schema and re-ingest after "
                    "changing embedding dimensions."
                )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_contextual_embedding_hnsw_idx
                ON chunks USING hnsw (contextual_embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_search_gin_idx
                ON chunks USING gin (
                    (
                        setweight(
                            to_tsvector(
                                'english'::regconfig, coalesce(caption, '')
                            ),
                            'A'
                        ) ||
                        setweight(
                            to_tsvector(
                                'english'::regconfig, coalesce(breadcrumb, '')
                            ),
                            'B'
                        ) ||
                        setweight(
                            to_tsvector('english'::regconfig, content),
                            'C'
                        )
                    )
                );
                """
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_connection:
            connection.close()


def reset_database(conn=None) -> None:
    """Drop only this application's known tables, then create the unified schema."""
    owns_connection = conn is None
    connection = conn or psycopg2.connect(**database_config())
    try:
        with connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS structured_ingest_state;")
            cur.execute("DROP TABLE IF EXISTS structured_chunks;")
            cur.execute("DROP TABLE IF EXISTS chunks;")
            cur.execute("DROP TABLE IF EXISTS documents;")
        connection.commit()
        setup_database(connection)
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_connection:
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the unified retrieval schema.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing ingested documents/chunks before creating the schema.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation for --reset.",
    )
    args = parser.parse_args()
    if args.reset and not args.yes:
        parser.error("--reset permanently deletes ingested data; pass --yes to confirm")
    if args.reset:
        reset_database()
        print("Unified database schema reset and ready")
    else:
        setup_database()
        print("Unified database schema ready")


if __name__ == "__main__":
    main()
