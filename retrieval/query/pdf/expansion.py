"""Parent-child retrieval: rank on small chunks, answer from whole sections.

Retrieval and generation want opposite things from a chunk. Ranking wants it
small -- 86% of this corpus's ground-truth evidence runs are a single page, and a
512-token chunk is about one page, so a small chunk is a precise unit to match
against. Generation wants it large, because a figure quoted without the sentence
that defines it is not usable evidence.

Parent-child resolves that by separating the two: the child stays the retrieval
and ranking unit, and only the *content handed to the model* is widened to the
surrounding section. Nothing about scoring changes, so a bad expansion cannot
promote an irrelevant chunk -- it can only add or fail to add context.

The section is reconstructed from the children themselves, keyed on
``(document_id, section_ordinal)``. No parent text is stored: the children are
already a complete, ordered cover of their section.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

# A section is ~573 tokens at the median and 2,028 at p95, so 2,500 leaves 96% of
# sections uncapped while bounding what a single result can cost the prompt.
PARENT_TOKEN_BUDGET = 2500

# Children overlap by `token_overlap` tokens, so a naive concatenation repeats
# text at every seam. This bounds how far back the de-overlap search looks; it is
# a generous multiple of the configured overlap in characters.
_MAX_SEAM_CHARS = 1200

_EXPANDABLE_CONTENT_TYPES = frozenset({"narrative"})


def _same_section(anchor, candidate) -> bool:
    return (
        candidate.source_pdf == anchor.source_pdf
        and candidate.sub_document == anchor.sub_document
        and candidate.breadcrumb == anchor.breadcrumb
        and candidate.content_type == "narrative"
    )


def select_sibling_context(
    anchors: Iterable,
    candidates: Iterable,
    *,
    token_budget: int = 2000,
    max_per_anchor: int = 2,
) -> tuple:
    """Select nearby section siblings without exceeding the expansion budget.

    This pure selector is deliberately not enabled in the production path. It
    supports the plan's isolated narrative-coverage experiment while keeping
    added context bounded and deterministic.
    """
    if token_budget <= 0 or max_per_anchor <= 0:
        raise ValueError("expansion limits must be positive")
    anchors = tuple(anchors)
    candidates = tuple(candidates)
    anchor_ids = {anchor.chunk_id for anchor in anchors}
    selected = []
    selected_ids: set = set()
    used_tokens = 0
    for anchor in anchors:
        siblings = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.chunk_id not in anchor_ids
                and candidate.chunk_id not in selected_ids
                and _same_section(anchor, candidate)
            ),
            key=lambda candidate: (
                abs(candidate.chunk_index - anchor.chunk_index),
                candidate.chunk_index,
            ),
        )
        taken = 0
        for sibling in siblings:
            tokens = max(0, int(sibling.token_count))
            if used_tokens + tokens > token_budget:
                continue
            selected.append(sibling)
            selected_ids.add(sibling.chunk_id)
            used_tokens += tokens
            taken += 1
            if taken >= max_per_anchor or used_tokens >= token_budget:
                break
    return tuple(selected)


def join_without_overlap(previous: str, piece: str) -> str:
    """Append `piece`, dropping the prefix it repeats from the end of `previous`.

    Token-window chunking makes each child begin with the tail of the one before
    it. `deduplicate_exact_content` cannot help here because overlapping chunks
    are not byte-identical, so the seam is removed by finding the longest suffix
    of the text so far that the next piece starts with.
    """
    if not previous:
        return piece
    if not piece:
        return previous
    window = min(len(previous), len(piece), _MAX_SEAM_CHARS)
    for size in range(window, 0, -1):
        if previous.endswith(piece[:size]):
            return previous + piece[size:]
    return f"{previous}\n{piece}"


def select_parent_window(
    siblings: list, anchor_index: int, token_budget: int = PARENT_TOKEN_BUDGET
) -> list:
    """Contiguous run of siblings around the anchor that fits the token budget.

    Growth is outward from the anchor rather than from the start of the section,
    so an over-long section keeps the part that was actually retrieved instead of
    truncating to its opening paragraphs. `siblings` must be ordered by
    chunk_index; each element needs `.chunk_index` and `.token_count`.
    """
    positions = [
        position
        for position, sibling in enumerate(siblings)
        if sibling.chunk_index == anchor_index
    ]
    if not positions:
        return []
    start = end = positions[0]
    used = siblings[start].token_count

    while True:
        before = siblings[start - 1].token_count if start > 0 else None
        after = siblings[end + 1].token_count if end + 1 < len(siblings) else None
        # Prefer following text over preceding text: a chunk more often needs
        # what comes next to complete a sentence or a table lead-in.
        if after is not None and used + after <= token_budget:
            end += 1
            used += after
        elif before is not None and used + before <= token_budget:
            start -= 1
            used += before
        else:
            break
    return siblings[start : end + 1]


@dataclass(frozen=True)
class SectionSibling:
    """One child chunk of a section, as reconstruction needs it."""

    chunk_id: int
    chunk_index: int
    content: str
    token_count: int


def _fetch_section_siblings(
    conn, chunk_ids: list[int]
) -> dict[int, tuple[tuple[int, int], list[SectionSibling]]]:
    """Each anchor's section key and its narrative siblings, keyed by anchor id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH anchors AS (
                SELECT id, document_id, section_ordinal
                FROM chunks
                WHERE id = ANY(%s) AND content_type = 'narrative'
            )
            SELECT a.id, a.document_id, a.section_ordinal,
                   sibling.id, sibling.chunk_index,
                   sibling.content, sibling.token_count
            FROM anchors a
            JOIN chunks sibling
              ON sibling.document_id = a.document_id
             AND sibling.section_ordinal = a.section_ordinal
             AND sibling.content_type = 'narrative'
            ORDER BY a.id, sibling.chunk_index
            """,
            (chunk_ids,),
        )
        rows = cur.fetchall()

    sections: dict[int, tuple[tuple[int, int], list[SectionSibling]]] = {}
    for anchor_id, document_id, ordinal, chunk_id, index, content, tokens in rows:
        _, siblings = sections.setdefault(anchor_id, ((document_id, ordinal), []))
        siblings.append(
            SectionSibling(
                chunk_id=chunk_id,
                chunk_index=index,
                content=content,
                token_count=tokens,
            )
        )
    return sections


def expand_to_parent_sections(
    conn, results: list, *, token_budget: int = PARENT_TOKEN_BUDGET
) -> list:
    """Widen each narrative result's content to its surrounding section.

    Ranking order and scores are returned untouched -- only `content` and
    `token_count` change. Structured chunks are left alone: a table or a figure
    is already the complete unit it represents.

    A failure here degrades to the un-expanded results rather than failing the
    query, because narrower context is a worse answer, not a broken one.
    """
    anchor_ids = [
        result.query_object.chunk_id
        for result in results
        if result.query_object.content_type in _EXPANDABLE_CONTENT_TYPES
    ]
    if not anchor_ids:
        return results

    try:
        sections_by_anchor = _fetch_section_siblings(conn, anchor_ids)
    except Exception:
        logger.warning(
            "Parent expansion query failed; returning un-expanded results",
            exc_info=True,
        )
        # Degrading gracefully means degrading the *connection* too. A failed
        # statement leaves psycopg2 in an aborted transaction, so without this
        # rollback every subsequent query on the same connection dies with
        # InFailedSqlTransaction -- turning a recoverable "narrower context"
        # into a hard failure several questions later, far from the cause.
        try:
            conn.rollback()
        except Exception:
            logger.warning("Rollback after failed parent expansion failed",
                           exc_info=True)
        return results

    expanded = []
    already_expanded: set[tuple[int, int]] = set()
    for result in results:
        query_object = result.query_object
        section = sections_by_anchor.get(query_object.chunk_id)
        if section is None:
            expanded.append(result)
            continue

        section_key, siblings = section
        # Two hits from one section would otherwise each pull in the same 2,500
        # tokens, spending most of the prompt budget on a repeat. The later hit
        # keeps its own chunk, which still adds its ranking signal.
        if len(siblings) == 1 or section_key in already_expanded:
            expanded.append(result)
            continue
        already_expanded.add(section_key)

        window = select_parent_window(siblings, query_object.chunk_index, token_budget)
        if len(window) <= 1:
            expanded.append(result)
            continue

        content = ""
        for sibling in window:
            content = join_without_overlap(content, sibling.content)
        expanded.append(
            replace(
                result,
                query_object=replace(
                    query_object,
                    content=content,
                    token_count=sum(sibling.token_count for sibling in window),
                ),
            )
        )
    return expanded
