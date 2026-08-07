"""Idempotent schema migration for PDF figure attachments."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIGRATION_VERSION = "0008_chunk_links"
MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""
CHUNK_LINKS_SQL = """
CREATE TABLE IF NOT EXISTS chunk_links (
    id              bigserial PRIMARY KEY,
    source_chunk_id int NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_chunk_id int REFERENCES chunks(id) ON DELETE CASCADE,
    relation        text NOT NULL,
    link_tier       text NOT NULL,
    confidence      real NOT NULL DEFAULT 0,
    context_text    text,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunk_links_source_idx
    ON chunk_links (source_chunk_id, relation);
CREATE INDEX IF NOT EXISTS chunk_links_target_idx
    ON chunk_links (target_chunk_id, relation);
"""


def migrate(conn) -> bool:
    """Create the figure-link schema if its migration has not been applied."""
    with conn.cursor() as cur:
        cur.execute(MIGRATIONS_TABLE)
        conn.commit()
        cur.execute(
            "SELECT 1 FROM schema_migrations WHERE version = %s",
            (MIGRATION_VERSION,),
        )
        if cur.fetchone():
            return False
        logger.info("Applying migration %s", MIGRATION_VERSION)
        cur.execute(CHUNK_LINKS_SQL)
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)",
            (MIGRATION_VERSION,),
        )
        conn.commit()
    return True
