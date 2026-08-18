"""Attach describing context to figure chunks and re-embed them.

Enriching ``chunks.content`` without regenerating the vector would leave the
stored text and its embedding silently inconsistent, so this command always
re-embeds every figure it rewrites — raw and contextual alike, through the same
functions ingestion uses.

    uv run python -m retrieval.ingest.pdf.refresh_figures
    uv run python -m retrieval.ingest.pdf.refresh_figures --links-only
"""

from __future__ import annotations

import argparse
import hashlib
import logging

from retrieval.contextual_embeddings import (
    CONTEXTUAL_EMBEDDING_RECIPE,
    contextual_embedding_text_for_model,
)
from retrieval.ingest.pdf.figure_context import (
    build_links,
    enrich_figure_content,
    persist_links,
)
from retrieval.ingest.pdf.schema import migrate
from retrieval.source_manifest import title_for_filename
from retrieval.utils import connect_db, get_embedding_model

logger = logging.getLogger(__name__)


def reembed_figures(conn, model, chunk_ids: list[int]) -> int:
    if not chunk_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.filename, c.breadcrumb, c.content
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            ORDER BY c.id
            """,
            (chunk_ids,),
        )
        rows = cur.fetchall()

    contents = [row[3] for row in rows]
    raw_vectors = model.encode(
        contents, normalize_embeddings=True, show_progress_bar=False
    )
    contextual_inputs = [
        contextual_embedding_text_for_model(
            title_for_filename(filename),
            breadcrumb,
            content,
            model.tokenizer,
            model.max_seq_length,
        )
        for _, filename, breadcrumb, content in rows
    ]
    contextual_vectors = model.encode(
        contextual_inputs, normalize_embeddings=True, show_progress_bar=False
    )

    with conn.cursor() as cur:
        for (chunk_id, _, _, content), raw, contextual in zip(
            rows, raw_vectors, contextual_vectors, strict=True
        ):
            cur.execute(
                """
                UPDATE chunks
                SET embedding = %s,
                    contextual_embedding = %s,
                    contextual_embedding_recipe = %s,
                    content_hash = %s,
                    token_count = %s
                WHERE id = %s
                """,
                (
                    raw.tolist(),
                    contextual.tolist(),
                    CONTEXTUAL_EMBEDDING_RECIPE,
                    hashlib.sha256(content.encode()).hexdigest(),
                    len(model.tokenizer.encode(content, add_special_tokens=False)),
                    chunk_id,
                ),
            )
    conn.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="Resolve and store figure links without rewriting figure content.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        migrate(conn)
        links = build_links(conn)
        persist_links(conn, links)
        tiers: dict[str, int] = {}
        for link in links:
            tiers[link.link_tier] = tiers.get(link.link_tier, 0) + 1
        logger.info("Resolved %d figure links: %s", len(links), tiers)

        if args.links_only:
            print(f"Stored {len(links)} figure links (content unchanged).")
            return

        model = get_embedding_model()
        updated = enrich_figure_content(conn, links, model)
        enriched_ids = [
            link.figure_chunk_id for link in links if link.context_text
        ]
        reembedded = reembed_figures(conn, model, enriched_ids)
        print(
            f"Enriched {updated} figure chunk(s) and re-embedded {reembedded}.\n"
            f"Link tiers: {tiers}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
