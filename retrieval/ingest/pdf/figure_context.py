"""Locate the prose a PDF figure illustrates, and link the two.

A figure is essentially never a standalone answer; it visualises a passage of
text. Its stored ``content`` averages ~87 tokens, so a cross-encoder
systematically under-scores it against ~298-token narrative prose — which is
what the additive caption prior was compensating for.

This module finds each figure's describing context with a deterministic,
precision-first cascade (no vision model, no LLM) so the figure can be attached
to the passage it supports instead of competing with it.

Measured coverage on the current corpus of 507 figure chunks:

* 336 figures expose a parseable label in their caption (``Figure 4.1-1``);
* 299 of those (89%) have that label mentioned in same-document prose;
* 234 of those mentions fall within +/-3 pages of the figure.

Tier 1 is the only tier that yields a *specific* narrative chunk, which is what
the attachment role needs; Tiers 2-3 supply context text with a section-level
link only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LABEL_RE = re.compile(
    r"(?i)\b((?:figure|fig\.?|table|map|chart|exhibit)\s*[0-9][0-9.\-]*)"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

TIER_REFERENCE = "label_reference"
TIER_SECTION = "enclosing_section"
TIER_PAGE = "page_context"

# A same-document label mention this far from the figure is likely a different
# figure with a similar label, or a list of figures.
NEAR_PAGE_WINDOW = 3
MAX_CONTEXT_CHARS = 900


@dataclass(frozen=True)
class FigureLink:
    figure_chunk_id: int
    supports_chunk_id: int | None
    link_tier: str
    confidence: float
    context_text: str


def extract_label(caption: str | None) -> str | None:
    if not caption:
        return None
    match = LABEL_RE.search(caption)
    return match.group(1).strip() if match else None


def _sentences_around(text: str, needle: str, window: int = 1) -> str:
    """Return the sentence containing ``needle`` plus neighbours."""
    sentences = SENTENCE_SPLIT_RE.split(" ".join(text.split()))
    lowered = needle.lower()
    for index, sentence in enumerate(sentences):
        if lowered in sentence.lower():
            start = max(0, index - window)
            end = min(len(sentences), index + window + 1)
            return " ".join(sentences[start:end])
    return ""


def find_reference_link(
    cur, figure_chunk_id: int, document_id: int, page_start: int, label: str
) -> FigureLink | None:
    """Tier 1: prose in the same document that names this figure's label.

    Nearest-page mentions are preferred, since a long filing repeats similar
    labels across chapters and also carries a list-of-figures page.
    """
    cur.execute(
        """
        SELECT c.id, c.content, abs(c.page_start - %s) AS page_distance
        FROM chunks c
        WHERE c.document_id = %s
          AND c.content_type = 'narrative'
          AND c.content ILIKE %s
        ORDER BY page_distance
        LIMIT 5
        """,
        (page_start, document_id, f"%{label}%"),
    )
    for chunk_id, content, page_distance in cur.fetchall():
        excerpt = _sentences_around(content, label)
        if not excerpt:
            continue
        # A mention far from the figure is weaker evidence, but still a link.
        confidence = 0.9 if page_distance <= NEAR_PAGE_WINDOW else 0.6
        return FigureLink(
            figure_chunk_id=figure_chunk_id,
            supports_chunk_id=chunk_id,
            link_tier=TIER_REFERENCE,
            confidence=confidence,
            context_text=excerpt[:MAX_CONTEXT_CHARS],
        )
    return None


def find_section_link(
    cur, figure_chunk_id: int, document_id: int, page_start: int, breadcrumb: str
) -> FigureLink | None:
    """Tier 2: the narrative chunk whose section encloses the figure's page."""
    cur.execute(
        """
        SELECT c.id, c.content
        FROM chunks c
        WHERE c.document_id = %s
          AND c.content_type = 'narrative'
          AND c.page_start <= %s
          AND c.page_end >= %s
        ORDER BY (c.page_end - c.page_start), c.chunk_index
        LIMIT 1
        """,
        (document_id, page_start, page_start),
    )
    row = cur.fetchone()
    if not row:
        return None
    chunk_id, content = row
    excerpt = " ".join(content.split())[:MAX_CONTEXT_CHARS]
    return FigureLink(
        figure_chunk_id=figure_chunk_id,
        supports_chunk_id=chunk_id,
        link_tier=TIER_SECTION,
        confidence=0.4,
        context_text=excerpt,
    )


def build_links(conn) -> list[FigureLink]:
    """Resolve a describing context for every figure chunk."""
    links: list[FigureLink] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.document_id, c.page_start, c.caption, c.breadcrumb
            FROM chunks c
            WHERE c.content_type = 'figure'
            ORDER BY c.id
            """
        )
        figures = cur.fetchall()

        for figure_id, document_id, page_start, caption, breadcrumb in figures:
            label = extract_label(caption)
            link = None
            if label:
                link = find_reference_link(
                    cur, figure_id, document_id, page_start, label
                )
            if link is None:
                link = find_section_link(
                    cur, figure_id, document_id, page_start, breadcrumb or ""
                )
            if link is None:
                link = FigureLink(
                    figure_chunk_id=figure_id,
                    supports_chunk_id=None,
                    link_tier=TIER_PAGE,
                    confidence=0.1,
                    context_text="",
                )
            links.append(link)
    return links


def persist_links(conn, links: list[FigureLink]) -> None:
    from psycopg2.extras import execute_values

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunk_links WHERE relation = 'figure_supports'")
        rows = [
            (
                link.figure_chunk_id,
                link.supports_chunk_id,
                "figure_supports",
                link.link_tier,
                link.confidence,
                link.context_text or None,
            )
            for link in links
        ]
        execute_values(
            cur,
            """
            INSERT INTO chunk_links (
                source_chunk_id, target_chunk_id, relation,
                link_tier, confidence, context_text
            ) VALUES %s
            """,
            rows,
        )
    conn.commit()


MARKER = "\nDescribing context: "
# Room reserved for the "Document:"/"Section:"/"Chunk:" contextual prefix that
# contextual_embedding_text_for_model prepends.
CONTEXT_PREFIX_TOKENS = 64


def enrich_figure_content(conn, links: list[FigureLink], model) -> int:
    """Append the resolved describing context to each figure's stored content.

    The appended text is clearly delimited so it stays distinguishable from the
    caption and deterministic page context already present, and so a later
    re-run replaces rather than compounds it.

    The context is trimmed against the live tokenizer.
    ``contextual_embedding_text_for_model`` *raises* rather than truncating when
    the contextual form exceeds the model window, which would abort the refresh
    after content had already been rewritten — leaving stored text inconsistent
    with its embedding. Budgeting here keeps the two in step.
    """
    tokenizer = model.tokenizer
    updated = 0
    with conn.cursor() as cur:
        for link in links:
            if not link.context_text:
                continue
            cur.execute(
                "SELECT content, coalesce(breadcrumb, '') FROM chunks WHERE id = %s",
                (link.figure_chunk_id,),
            )
            row = cur.fetchone()
            if not row:
                continue
            base = row[0].split(MARKER)[0]
            # These filings carry very long breadcrumbs ("Appendix A - ... >
            # 8.1.5.2 ..."), so the prefix must be measured, not assumed.
            breadcrumb_tokens = len(
                tokenizer.encode(row[1], add_special_tokens=False)
            )
            budget = model.max_seq_length - breadcrumb_tokens - CONTEXT_PREFIX_TOKENS
            base_tokens = len(tokenizer.encode(base, add_special_tokens=False))
            remaining = budget - base_tokens - 8
            if remaining < 20:
                continue  # no room to say anything useful

            context = link.context_text
            context_ids = tokenizer(
                context, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            if len(context_ids) > remaining:
                context = context[: context_ids[remaining - 1][1]].rstrip() + "…"

            cur.execute(
                "UPDATE chunks SET content = %s WHERE id = %s",
                (f"{base}{MARKER}{context}", link.figure_chunk_id),
            )
            updated += 1
    conn.commit()
    return updated
