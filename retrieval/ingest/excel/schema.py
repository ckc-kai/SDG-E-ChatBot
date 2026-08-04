"""Versioned, idempotent schema migrations for the Excel retrieval tables.

These migrations are additive. They never drop or rewrite the PDF ``documents``
and ``chunks`` rows; the only change to the existing schema is widening the
``chunks.content_type`` constraint so Excel cards can share the tuned hybrid
retrieval path.
"""

from __future__ import annotations

import logging

from psycopg2 import sql

logger = logging.getLogger(__name__)

MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""

EXCEL_CONTENT_TYPE = "excel_card"

MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "0001_excel_sources_and_revisions",
        """
        CREATE TABLE IF NOT EXISTS excel_sources (
            id                  serial PRIMARY KEY,
            source_key          text NOT NULL UNIQUE,
            table_number        int  NOT NULL UNIQUE,
            title               text NOT NULL DEFAULT '',
            active_revision_id  bigint,
            created_at          timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS excel_revisions (
            id                  bigserial PRIMARY KEY,
            source_id           int NOT NULL
                                    REFERENCES excel_sources(id) ON DELETE CASCADE,
            source_file         text NOT NULL,
            source_hash         text NOT NULL,
            ingest_signature    text NOT NULL,
            contract_version    text NOT NULL,
            state               text NOT NULL
                                    CHECK (state IN (
                                        'staging', 'active', 'superseded', 'failed'
                                    )),
            column_inventory    jsonb NOT NULL DEFAULT '{}'::jsonb,
            unknown_columns     text[] NOT NULL DEFAULT '{}',
            source_row_count    int NOT NULL DEFAULT 0,
            fact_count          int NOT NULL DEFAULT 0,
            record_count        int NOT NULL DEFAULT 0,
            card_count          int NOT NULL DEFAULT 0,
            source_uri          text,
            error_summary       text,
            created_at          timestamptz NOT NULL DEFAULT now(),
            activated_at        timestamptz
        );

        CREATE UNIQUE INDEX IF NOT EXISTS excel_one_active_revision_idx
            ON excel_revisions (source_id)
            WHERE state = 'active';
        """,
    ),
    (
        "0002_excel_sources_active_revision_fk",
        """
        ALTER TABLE excel_sources
            DROP CONSTRAINT IF EXISTS excel_sources_active_revision_fk;
        ALTER TABLE excel_sources
            ADD CONSTRAINT excel_sources_active_revision_fk
            FOREIGN KEY (active_revision_id)
            REFERENCES excel_revisions(id)
            ON DELETE SET NULL;
        """,
    ),
    (
        "0003_excel_facts",
        """
        CREATE TABLE IF NOT EXISTS excel_facts (
            id                    bigserial PRIMARY KEY,
            revision_id           bigint NOT NULL
                                      REFERENCES excel_revisions(id) ON DELETE CASCADE,
            table_number          int  NOT NULL,
            record_id             text NOT NULL,
            source_metric_number  text,
            series_id             text,
            semantic_metric_key   text NOT NULL,
            metric_name           text NOT NULL,
            measure_name          text NOT NULL,
            utility_id            text,
            reporting_year        int,
            reporting_quarter     int,
            source_vintage_year   int,
            year_basis            text,
            period_end_date       date,
            hftd_tier             text,
            line_type             text,
            dimensions            jsonb NOT NULL DEFAULT '{}'::jsonb,
            unit                  text,
            value_numeric         numeric,
            value_raw             text,
            value_text            text,
            comments              text,
            blank_meaning         text,
            provenance            jsonb NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS excel_facts_lookup_idx
            ON excel_facts (
                revision_id, table_number, semantic_metric_key,
                reporting_year, reporting_quarter
            );

        CREATE INDEX IF NOT EXISTS excel_facts_series_idx
            ON excel_facts (revision_id, table_number, source_metric_number);
        """,
    ),
    (
        "0004_excel_records",
        """
        CREATE TABLE IF NOT EXISTS excel_records (
            id                    bigserial PRIMARY KEY,
            revision_id           bigint NOT NULL
                                      REFERENCES excel_revisions(id) ON DELETE CASCADE,
            table_number          int  NOT NULL,
            record_id             text NOT NULL,
            entity_key            text,
            entity_type           text NOT NULL,
            title                 text,
            utility_id            text,
            reporting_year        int,
            reporting_quarter     int,
            hftd_tier             text,
            line_type             text,
            date_start            date,
            date_due              date,
            date_end              date,
            status                text,
            attributes            jsonb NOT NULL DEFAULT '{}'::jsonb,
            searchable_text       text,
            provenance            jsonb NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS excel_records_lookup_idx
            ON excel_records (
                revision_id, table_number, entity_type,
                reporting_year, reporting_quarter
            );

        CREATE INDEX IF NOT EXISTS excel_records_entity_key_idx
            ON excel_records (revision_id, table_number, entity_key);
        """,
    ),
    (
        "0005_chunks_allow_excel_card",
        """
        ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_content_type_check;
        ALTER TABLE chunks ADD CONSTRAINT chunks_content_type_check
            CHECK (content_type IN
                   ('narrative', 'table', 'figure', 'excel_card'));
        ALTER TABLE documents ALTER COLUMN page_count SET DEFAULT 0;
        """,
    ),
    (
        "0007_excel_card_link",
        # Lets card activation and revision activation stay consistent.
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS excel_revision_id bigint;
        CREATE INDEX IF NOT EXISTS chunks_excel_revision_idx
            ON chunks (excel_revision_id)
            WHERE excel_revision_id IS NOT NULL;
        """,
    ),
)

# HNSW builds on an existing production corpus must not run inside the regular
# transactional migration path: plain CREATE INDEX blocks writers and a late
# failure rolls back every earlier build. Each statement below runs and commits
# independently with a short lock timeout. A retry skips valid indexes and
# replaces any invalid artifact left by an interrupted concurrent build.
CONCURRENT_INDEX_MIGRATIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "0006_chunks_corpus_lane_indexes",
        (
            (
                "chunks_excel_embedding_idx",
                """CREATE INDEX CONCURRENTLY IF NOT EXISTS
                   chunks_excel_embedding_idx
                   ON chunks USING hnsw (embedding vector_cosine_ops)
                   WHERE content_type = 'excel_card'""",
            ),
            (
                "chunks_excel_contextual_embedding_idx",
                """CREATE INDEX CONCURRENTLY IF NOT EXISTS
                   chunks_excel_contextual_embedding_idx
                   ON chunks USING hnsw (contextual_embedding vector_cosine_ops)
                   WHERE content_type = 'excel_card'""",
            ),
        ),
    ),
    (
        "0008_pdf_evidence_group_indexes",
        tuple(
            (
                f"chunks_{content_type}_{column}_idx",
                f"""CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    chunks_{content_type}_{column}_idx
                    ON chunks USING hnsw ({column_name} vector_cosine_ops)
                    WHERE content_type = '{content_type}'""",
            )
            for content_type in ("narrative", "table", "figure")
            for column, column_name in (
                ("embedding", "embedding"),
                ("contextual_embedding", "contextual_embedding"),
            )
        ),
    ),
)


def applied_versions(cur) -> set[str]:
    cur.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def _apply_concurrent_index_migration(
    conn,
    version: str,
    indexes: tuple[tuple[str, str], ...],
) -> None:
    """Build each HNSW index without a long writer-blocking table lock."""
    conn.commit()
    previous_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '5s'")
            cur.execute("SET statement_timeout = '0'")
            for index_name, create_sql in indexes:
                cur.execute(
                    """
                    SELECT i.indisvalid
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indexrelid
                    WHERE c.relname = %s
                    """,
                    (index_name,),
                )
                state = cur.fetchone()
                if state and not state[0]:
                    cur.execute(
                        sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                            sql.Identifier(index_name)
                        )
                    )
                cur.execute(create_sql)
            cur.execute("RESET lock_timeout")
            cur.execute("RESET statement_timeout")
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("RESET lock_timeout")
                cur.execute("RESET statement_timeout")
        except Exception:
            logger.warning("Could not reset migration session timeouts", exc_info=True)
        finally:
            conn.autocommit = previous_autocommit

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s) "
            "ON CONFLICT (version) DO NOTHING",
            (version,),
        )
    conn.commit()


def migrate(conn) -> list[str]:
    """Apply pending migrations. Returns the versions newly applied."""
    newly_applied: list[str] = []
    with conn.cursor() as cur:
        cur.execute(MIGRATIONS_TABLE)
        conn.commit()
        done = applied_versions(cur)
        for version, sql in MIGRATIONS:
            if version in done:
                continue
            logger.info("Applying migration %s", version)
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
            conn.commit()
            newly_applied.append(version)
        for version, indexes in CONCURRENT_INDEX_MIGRATIONS:
            if version in done:
                continue
            logger.info("Applying concurrent index migration %s", version)
            _apply_concurrent_index_migration(conn, version, indexes)
            newly_applied.append(version)
    return newly_applied
