"""Query the WMP knowledge base across narrative AND structured chunks.

Additive companion to `retrieval.query` (which stays narrative-only and keeps
its accepted evaluation behavior). This module runs the same accepted pipeline
shape -- hybrid raw+contextual search with a raw-preserving union pool and a
cross-encoder rerank -- but each dense search UNIONs `chunks` with
`structured_chunks`, so tables and figures compete with narrative text for the
same candidate slots.

Candidates are keyed as "<origin>:<id>" ("narrative:123" / "structured:45")
because the two tables have independent serial ids. Structured candidates carry
their `caption`, `structured_data` (exact table grid), and `image_path`
(cropped figure PNG) so an answering layer can quote precise cells or show the
figure.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from retrieval.query import query_rewrite
from retrieval.utils import (
    connect_db,
    get_embedding_model,
    get_reranker_model,
    load_config,
)

logger = logging.getLogger(__name__)

_config = load_config()["local"]
RETRIEVAL_TOP_K = _config["retrieval"]["retrieval_top_k"]
RERANK_TOP_K = _config["retrieval"]["rerank_top_k"]

_EMBEDDING_COLUMNS = {
    "raw": "embedding",
    "contextual": "contextual_embedding",
}


@dataclass(frozen=True)
class StructuredQueryObject:
    key: str  # "<origin>:<id>", unique across both tables
    chunk_id: int
    origin: str  # 'narrative' | 'structured'
    source_pdf: str
    sub_document: str | None
    breadcrumb: str
    section_number: str | None
    page_start: int
    page_end: int
    chunk_index: int
    content_type: str
    content: str
    caption: str | None
    structured_data: dict | None
    image_path: str | None
    token_count: int
    distance: float


@dataclass(frozen=True)
class RankedStructuredResult:
    query_object: StructuredQueryObject
    rerank_score: float


@dataclass(frozen=True)
class StructuredRetrievalDiagnostics:
    search_queries: list[str]
    candidate_sets: list[list[StructuredQueryObject]]
    pooled_candidates: list[StructuredQueryObject]
    reranked_candidates: list[RankedStructuredResult]


def _search_by_vector(
    query_vector,
    top_k: int,
    conn,
    *,
    embedding_mode: str,
) -> list[StructuredQueryObject]:
    if embedding_mode not in _EMBEDDING_COLUMNS:
        raise ValueError(
            f"Dense search requires one of {tuple(_EMBEDDING_COLUMNS)}; "
            f"got {embedding_mode!r}"
        )
    column = _EMBEDDING_COLUMNS[embedding_mode]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT u.origin, u.id, d.filename, u.sub_document, u.breadcrumb,
                   u.section_number, u.page_start, u.page_end, u.chunk_index,
                   u.content_type, u.content, u.caption, u.structured_data,
                   u.image_path, u.token_count, u.distance
            FROM (
                SELECT 'narrative' AS origin, c.id, c.document_id,
                       c.sub_document, c.breadcrumb, c.section_number,
                       c.page_start, c.page_end, c.chunk_index, c.content_type,
                       c.content, NULL::text AS caption,
                       NULL::jsonb AS structured_data, NULL::text AS image_path,
                       c.token_count, c.{column} <=> %s AS distance
                FROM chunks c
                WHERE c.{column} IS NOT NULL
                UNION ALL
                SELECT 'structured', s.id, s.document_id,
                       s.sub_document, s.breadcrumb, s.section_number,
                       s.page_start, s.page_end, s.chunk_index, s.content_type,
                       s.content, s.caption, s.structured_data, s.image_path,
                       s.token_count, s.{column} <=> %s AS distance
                FROM structured_chunks s
                WHERE s.{column} IS NOT NULL
            ) u
            JOIN documents d ON d.id = u.document_id
            ORDER BY u.distance
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        )
        rows = cur.fetchall()

    return [
        StructuredQueryObject(
            key=f"{row[0]}:{row[1]}",
            chunk_id=row[1],
            origin=row[0],
            source_pdf=row[2],
            sub_document=row[3],
            breadcrumb=row[4],
            section_number=row[5],
            page_start=row[6],
            page_end=row[7],
            chunk_index=row[8],
            content_type=row[9],
            content=row[10],
            caption=row[11],
            structured_data=row[12],
            image_path=row[13],
            token_count=row[14],
            distance=row[15],
        )
        for row in rows
    ]


def raw_preserving_union(
    raw_candidates: list[StructuredQueryObject],
    contextual_candidates: list[StructuredQueryObject],
) -> list[StructuredQueryObject]:
    """Keep the complete raw ranking, append contextual-only candidates.

    Same pooling rule the accepted narrative pipeline uses, keyed on the
    cross-table candidate key.
    """
    combined = list(raw_candidates)
    seen_keys = {candidate.key for candidate in raw_candidates}
    for candidate in contextual_candidates:
        if candidate.key not in seen_keys:
            combined.append(candidate)
            seen_keys.add(candidate.key)
    return combined


def _merge_candidates(
    candidate_sets: list[list[StructuredQueryObject]],
) -> list[StructuredQueryObject]:
    best_by_key: dict[str, StructuredQueryObject] = {}
    for candidates in candidate_sets:
        for candidate in candidates:
            existing = best_by_key.get(candidate.key)
            if existing is None or candidate.distance < existing.distance:
                best_by_key[candidate.key] = candidate
    return list(best_by_key.values())


def _candidate_text(candidate: StructuredQueryObject) -> str:
    """Reranker input: same shape as the narrative pipeline, with an explicit
    element-type prefix so the cross-encoder can weigh 'this is a table'."""
    context: list[str] = []
    if candidate.breadcrumb:
        context.append(f"Section: {candidate.breadcrumb}")
    if candidate.section_number:
        context.append(f"Section number: {candidate.section_number}")
    if candidate.content_type != "narrative":
        context.append(f"Element: {candidate.content_type}")
    context.append(candidate.content)
    return "\n".join(context)


def rerank(
    question: str,
    candidates: list[StructuredQueryObject],
    top_k: int | None,
) -> list[RankedStructuredResult]:
    if not candidates:
        return []
    pairs = [(question, _candidate_text(candidate)) for candidate in candidates]
    scores = get_reranker_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        RankedStructuredResult(query_object=candidate, rerank_score=float(score))
        for candidate, score in ranked
    ]


def retrieve_structured_with_diagnostics(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = "off",
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> tuple[list[RankedStructuredResult], StructuredRetrievalDiagnostics]:
    """rewrite -> hybrid union search over both tables -> merge -> rerank."""
    search_queries = query_rewrite(question, mode=rewrite_mode)
    model = get_embedding_model()

    candidate_sets: list[list[StructuredQueryObject]] = []
    for search_query in search_queries:
        query_vector = model.encode(search_query.strip(), normalize_embeddings=True)
        raw_candidates = _search_by_vector(
            query_vector, retrieval_top_k, conn, embedding_mode="raw"
        )
        contextual_candidates = _search_by_vector(
            query_vector, retrieval_top_k, conn, embedding_mode="contextual"
        )
        candidate_sets.append(raw_preserving_union(raw_candidates, contextual_candidates))

    pooled = _merge_candidates(candidate_sets)
    all_ranked = rerank(question, pooled, top_k=None)
    diagnostics = StructuredRetrievalDiagnostics(
        search_queries=search_queries,
        candidate_sets=candidate_sets,
        pooled_candidates=pooled,
        reranked_candidates=all_ranked,
    )
    return all_ranked[:rerank_top_k], diagnostics


def retrieve_structured(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = "off",
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> list[RankedStructuredResult]:
    ranked, _ = retrieve_structured_with_diagnostics(
        question,
        conn,
        rewrite_mode=rewrite_mode,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query narrative + structured (table/figure) chunks together."
    )
    parser.add_argument("question", type=str, help="Question to ask.")
    parser.add_argument(
        "--rewrite-mode",
        choices=("auto", "off", "always"),
        default="off",
        help="Query decomposition mode (default: off).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        results = retrieve_structured(args.question, conn, rewrite_mode=args.rewrite_mode)
    finally:
        conn.close()

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        qo = result.query_object
        marker = f"[{qo.content_type}]" if qo.origin == "structured" else "[text]"
        print(f"\n[{rank}] {marker} rerank_score={result.rerank_score:.4f} distance={qo.distance:.4f}")
        print(f"    {qo.source_pdf} (p.{qo.page_start}-{qo.page_end})")
        print(f"    {qo.breadcrumb}")
        if qo.image_path:
            print(f"    image: {qo.image_path}")
        print(f"    {qo.content[:300]}")


if __name__ == "__main__":
    main()
