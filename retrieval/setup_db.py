#TODO: Table/Chart format is not handled yet.

import psycopg2

from retrieval.utils import DB_CONFIG, load_config

# TODO: Switching embedding models (e.g. to an AWS-hosted model later) changes
# config.yaml's local.embedding.dimensions and requires recreating the
# `embedding` column plus its index and re-ingesting every document.
EMBEDDING_DIMENSIONS = load_config()["local"]["embedding"]["dimensions"]


def setup_database() -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # content_hash: sha256, lets ingestion skip unchanged files
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id serial PRIMARY KEY,
                    filename text UNIQUE NOT NULL,
                    page_count int NOT NULL,
                    content_hash text NOT NULL,
                    ingested_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )

            # sub_document: "Appendix D / 22-11...CC Effectiveness Workstream"
            # breadcrumb: 8 Grid Design > 8.17 Inspections > 8.1.7.6 Aging report
            # content_type: 'narrative' (default), 'table', 'figure'
            # embedding: embedding model
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id serial PRIMARY KEY,
                    document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    sub_document text,
                    breadcrumb text,
                    section_number text,
                    page_start int NOT NULL,
                    page_end int NOT NULL,
                    chunk_index int NOT NULL,
                    content_type text NOT NULL DEFAULT 'narrative',
                    content text NOT NULL,
                    token_count int NOT NULL,
                    embedding_model text NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (document_id, page_start, chunk_index)
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
        conn.commit()
    finally:
        conn.close()
        print("Database Ready")


if __name__ == "__main__":
    setup_database()
