"""Backfill contextual embeddings on existing chunks without re-ingestion.

This updates only ``contextual_embedding`` and
``contextual_embedding_model``. Chunk ids, boundaries, raw content, and raw
embeddings are preserved.
"""

from __future__ import annotations

import argparse
import logging

from psycopg2.extras import execute_values
from tqdm import tqdm

from retrieval.contextual_embeddings import (
    CONTEXTUAL_EMBEDDING_RECIPE,
    contextual_embedding_text_for_model,
)
from retrieval.setup_db import setup_database
from retrieval.utils import connect_db, get_embedding_model, load_config

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = load_config()["local"]["embedding"]["model"]


def _selection_condition(force: bool) -> str:
    if force:
        return "TRUE"
    return (
        "(c.contextual_embedding IS NULL "
        "OR c.contextual_embedding_model IS DISTINCT FROM %s "
        "OR c.contextual_embedding_recipe IS DISTINCT FROM %s)"
    )


def _condition_params(force: bool) -> tuple:
    return () if force else (EMBEDDING_MODEL_NAME, CONTEXTUAL_EMBEDDING_RECIPE)


def count_pending(conn, force: bool) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*)
            FROM chunks c
            WHERE {_selection_condition(force)}
            """,
            _condition_params(force),
        )
        return cur.fetchone()[0]


def fetch_batch(conn, force: bool, after_id: int, batch_size: int) -> list[tuple]:
    params = (*_condition_params(force), after_id, batch_size)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, d.filename, c.breadcrumb, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {_selection_condition(force)}
              AND c.id > %s
            ORDER BY c.id
            LIMIT %s
            """,
            params,
        )
        return cur.fetchall()


def update_batch(conn, rows: list[tuple], model) -> None:
    contextual_texts = [
        contextual_embedding_text_for_model(
            source_pdf,
            breadcrumb,
            content,
            model.tokenizer,
            model.max_seq_length,
        )
        for _, source_pdf, breadcrumb, content in rows
    ]
    embeddings = model.encode(
        contextual_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    values = [
        (
            chunk_id,
            EMBEDDING_MODEL_NAME,
            CONTEXTUAL_EMBEDDING_RECIPE,
            embedding.tolist(),
        )
        for (chunk_id, *_), embedding in zip(rows, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            UPDATE chunks AS c
            SET contextual_embedding_model = data.embedding_model,
                contextual_embedding_recipe = data.embedding_recipe,
                contextual_embedding = data.embedding
            FROM (VALUES %s) AS data(id, embedding_model, embedding_recipe, embedding)
            WHERE c.id = data.id
            """,
            values,
            template="(%s, %s, %s, %s::vector)",
        )
    conn.commit()


def backfill(batch_size: int, force: bool) -> int:
    setup_database()
    conn = connect_db()
    updated = 0
    try:
        total = count_pending(conn, force)
        if not total:
            logger.info("All contextual embeddings are already current.")
            return 0

        model = get_embedding_model()
        after_id = 0
        with tqdm(total=total, desc="Contextual embeddings", unit="chunk") as progress:
            while rows := fetch_batch(conn, force, after_id, batch_size):
                update_batch(conn, rows, model)
                after_id = rows[-1][0]
                updated += len(rows)
                progress.update(len(rows))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill contextual embeddings without rebuilding chunks."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks encoded and committed per batch (default: 64).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every contextual embedding, including current rows.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    updated = backfill(args.batch_size, args.force)
    print(f"Updated {updated} contextual embedding(s) with {EMBEDDING_MODEL_NAME}.")


if __name__ == "__main__":
    main()
