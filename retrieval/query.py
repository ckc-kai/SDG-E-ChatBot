"""Query the WMP knowledge base.

1. query_rewrite: rewrite complex/causal questions
2. search: embeds each (sub-)question with the same model used at ingestion
   time and does a pgvector cosine-distance search against the raw, contextual,
   or hybrid raw+contextual document embeddings in `chunks`
3. Candidate pooling: results from every sub-question are merged and
   deduped by chunk id, so a multi-part question doesn't just get scored fragment-by-fragment
4. rerank: cross-encoder on original questions and pooled candidates
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, replace

from retrieval.contextual_embeddings import CONTEXTUAL_EMBEDDING_RECIPE
from retrieval.failure_log import get_failure_logger
from retrieval.utils import (
    connect_db,
    get_anthropic_client,
    get_embedding_model,
    get_reranker_model,
    load_config,
)

logger = logging.getLogger(__name__)
log_failure = get_failure_logger("query")

_config = load_config()["local"]
QUERY_REWRITE_MODEL = _config["query_rewrite"]["model"]
QUERY_REWRITE_MODE = _config["query_rewrite"].get("mode", "auto")
MAX_REWRITE_SUBQUESTIONS = _config["query_rewrite"].get("max_subquestions", 2)
RETRIEVAL_TOP_K = _config["retrieval"]["retrieval_top_k"]
RERANK_TOP_K = _config["retrieval"]["rerank_top_k"]
RRF_K = _config["retrieval"].get("rrf_k", 60)
DEFAULT_EMBEDDING_MODE = _config["retrieval"].get("embedding_mode", "raw")
DEFAULT_HYBRID_POOL_MODE = _config["retrieval"].get("hybrid_pool_mode", "rrf")
EMBEDDING_MODEL_NAME = _config["embedding"]["model"]
EMBEDDING_MODES = ("raw", "contextual", "hybrid")
HYBRID_POOL_MODES = ("rrf", "union")
_EMBEDDING_COLUMNS = {
    "raw": "embedding",
    "contextual": "contextual_embedding",
}

_REWRITE_MODES = {"auto", "off", "always"}
_INTERROGATIVE = (
    r"(?:what|who|when|where|why|how|which|"
    r"is|are|was|were|do|does|did|has|have|can|could|should|would|will)"
)
_SECOND_INTERROGATIVE_CLAUSE_RE = re.compile(
    rf"(?:\b(?:and|or)\b|[.;?])\s*,?\s*{_INTERROGATIVE}\b",
    re.IGNORECASE,
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a compound question about SDG&E's Wildfire Mitigation Plan into "
    "focused, self-contained sub-questions that can each be answered independently. "
    f"Return no more than {MAX_REWRITE_SUBQUESTIONS} sub-questions. "
    "Respond with ONLY a JSON array of strings, no other text. "
    "If the question is already simple, return a single-element array containing it unchanged."
)

_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")


def _strip_json_fence(text: str) -> str:
    """Strip a leading/trailing ```json ... ``` markdown fence, if present.

    Claude reliably follows "respond with ONLY a JSON array" for the array's
    CONTENT, but still sometimes wraps it in a markdown code fence. That leading
    backtick makes json.loads fail with "Expecting value: line 1 column 1", so we
    strip the fence (if any) before parsing. A no-op on already-bare JSON.
    """
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


_rewrite_call_count = 0
_last_rewrite_trace: dict | None = None


def get_rewrite_call_count() -> int:
    """Total number of Claude API calls made by query_rewrite() since the last reset."""
    return _rewrite_call_count


def reset_rewrite_call_count() -> None:
    global _rewrite_call_count
    _rewrite_call_count = 0


def get_last_rewrite_trace() -> dict | None:
    """Full record of the most recent query_rewrite() call.

    {"question": str, "sub_questions": list[str], "source": str,
     "reason": str, "api_sub_questions": list[str], "error": str | None}
    `source` tells you what actually happened: "simple" = no API call needed,
    "api" = Claude successfully decomposed it, "fallback" = an API call was made
    but failed (see "error"), so the original question was used unchanged.
    """
    return _last_rewrite_trace


@dataclass(frozen=True)
class QueryObject:
    chunk_id: int
    source_pdf: str
    sub_document: str | None
    breadcrumb: str
    section_number: str | None
    page_start: int
    page_end: int
    chunk_index: int
    content_type: str
    content: str
    token_count: int
    distance: float
    retrieval_score: float | None = None


@dataclass(frozen=True)
class RankedResult:
    query_object: QueryObject
    rerank_score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    search_queries: list[str]
    candidate_sets: list[list[QueryObject]]
    channel_candidate_sets: list[dict[str, list[QueryObject]]]
    pooled_candidates: list[QueryObject]
    reranked_candidates: list[RankedResult]


def _decomposition_reason(question: str) -> str | None:
    """Return a high-precision reason to decompose, independent of eval labels.

    Length, auxiliary verbs, identifiers containing periods, and noun lists are
    deliberately not treated as evidence of multiple retrieval intents. The API
    is reserved for a second explicit interrogative clause.
    """
    normalized = " ".join(question.strip().split())
    if normalized.count("?") > 1:
        return "multiple_questions"
    if _SECOND_INTERROGATIVE_CLAUSE_RE.search(normalized):
        return "independent_interrogative_clauses"
    return None


def _needs_decomposition(question: str) -> bool:
    return _decomposition_reason(question) is not None


def _deduplicate_subquestions(question: str, sub_questions: list[str]) -> list[str]:
    seen = {question.casefold()}
    unique: list[str] = []
    for item in sub_questions:
        cleaned = " ".join(item.strip().split())
        signature = cleaned.casefold()
        if not cleaned or signature in seen:
            continue
        seen.add(signature)
        unique.append(cleaned)
        if len(unique) >= MAX_REWRITE_SUBQUESTIONS:
            break
    return unique


def query_rewrite(question: str, mode: str | None = None) -> list[str]:
    global _rewrite_call_count, _last_rewrite_trace
    question = question.strip()
    selected_mode = mode or QUERY_REWRITE_MODE
    if selected_mode not in _REWRITE_MODES:
        raise ValueError(
            f"Unknown rewrite mode {selected_mode!r}; expected one of {sorted(_REWRITE_MODES)}"
        )

    reason = _decomposition_reason(question)
    if selected_mode == "off":
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "disabled",
            "reason": "mode_off",
            "api_sub_questions": [],
            "error": None,
        }
        return [question]
    if selected_mode == "auto" and reason is None:
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "simple",
            "reason": "single_intent",
            "api_sub_questions": [],
            "error": None,
        }
        return [question]
    if selected_mode == "always":
        reason = "mode_always"

    try:
        _rewrite_call_count += 1
        response = get_anthropic_client().messages.create(
            model=QUERY_REWRITE_MODEL,
            max_tokens=512,
            system=_DECOMPOSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        sub_questions = json.loads(_strip_json_fence(response.content[0].text))
        if not isinstance(sub_questions, list) or not sub_questions or not all(
            isinstance(item, str) for item in sub_questions
        ):
            raise ValueError(f"Unexpected rewrite response shape: {sub_questions!r}")
        search_queries = [question, *_deduplicate_subquestions(question, sub_questions)]
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": search_queries,
            "source": "api",
            "reason": reason,
            "api_sub_questions": sub_questions,
            "error": None,
        }
        return search_queries
    except Exception as exc:
        log_failure("query_rewrite", question, exc)
        logger.warning("query_rewrite failed, falling back to original question: %s", exc)
        _last_rewrite_trace = {
            "question": question,
            "sub_questions": [question],
            "source": "fallback",
            "reason": reason,
            "api_sub_questions": [],
            "error": str(exc),
        }
        return [question]


def _validate_embedding_mode(conn, embedding_mode: str) -> str:
    if embedding_mode not in _EMBEDDING_COLUMNS:
        raise ValueError(
            f"Dense search requires one of {tuple(_EMBEDDING_COLUMNS)}; "
            f"got {embedding_mode!r}"
        )
    if embedding_mode == "contextual":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM chunks
                WHERE contextual_embedding IS NULL
                   OR contextual_embedding_model IS DISTINCT FROM %s
                   OR contextual_embedding_recipe IS DISTINCT FROM %s
                """,
                (EMBEDDING_MODEL_NAME, CONTEXTUAL_EMBEDDING_RECIPE),
            )
            incomplete = cur.fetchone()[0]
        if incomplete:
            raise RuntimeError(
                f"Contextual embeddings are missing or stale for {incomplete} chunk(s). "
                "Run `uv run python -m retrieval.backfill_contextual_embeddings` first."
            )
    return _EMBEDDING_COLUMNS[embedding_mode]


def _search_by_vector(
    query_vector,
    top_k: int,
    conn,
    *,
    embedding_mode: str,
) -> list[QueryObject]:
    embedding_column = _validate_embedding_mode(conn, embedding_mode)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, d.filename, c.sub_document, c.breadcrumb, c.section_number,
                   c.page_start, c.page_end, c.chunk_index, c.content_type,
                   c.content, c.token_count, c.{embedding_column} <=> %s AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY distance
            LIMIT %s
            """,
            (query_vector, top_k),
        )
        rows = cur.fetchall()

    return [
        QueryObject(
            chunk_id=row[0],
            source_pdf=row[1],
            sub_document=row[2],
            breadcrumb=row[3],
            section_number=row[4],
            page_start=row[5],
            page_end=row[6],
            chunk_index=row[7],
            content_type=row[8],
            content=row[9],
            token_count=row[10],
            distance=row[11],
        )
        for row in rows
    ]


def search(
    question: str,
    top_k: int,
    conn,
    model,
    *,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
) -> list[QueryObject]:
    if embedding_mode == "hybrid":
        raise ValueError("Use retrieve() for hybrid raw+contextual search.")
    query_vector = model.encode(question.strip(), normalize_embeddings=True)
    return _search_by_vector(
        query_vector,
        top_k,
        conn,
        embedding_mode=embedding_mode,
    )


def reciprocal_rank_fusion(
    candidate_sets: list[list[QueryObject]],
    top_k: int,
    *,
    rrf_k: int = RRF_K,
) -> list[QueryObject]:
    """Fuse ranked channels by chunk id and retain a fixed candidate count."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    representative: dict[int, QueryObject] = {}
    for candidates in candidate_sets:
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
            representative.setdefault(chunk_id, candidate)

    fused_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id),
    )[:top_k]
    return [
        replace(representative[chunk_id], retrieval_score=scores[chunk_id])
        for chunk_id in fused_ids
    ]


def raw_preserving_union(
    raw_candidates: list[QueryObject],
    contextual_candidates: list[QueryObject],
) -> list[QueryObject]:
    """Keep the complete raw ranking, then append contextual-only candidates.

    Raw is the current champion, so A3b must not allow the weaker contextual
    channel to evict a raw candidate. The existing reranker receives the full
    deduplicated union (up to twice the per-channel retrieval count).
    """
    combined = list(raw_candidates)
    seen_ids = {candidate.chunk_id for candidate in raw_candidates}
    for candidate in contextual_candidates:
        if candidate.chunk_id not in seen_ids:
            combined.append(candidate)
            seen_ids.add(candidate.chunk_id)
    return combined


def _merge_candidates(candidate_sets: list[list[QueryObject]]) -> list[QueryObject]:
    best_by_id: dict[int, QueryObject] = {}
    for candidates in candidate_sets:
        for candidate in candidates:
            existing = best_by_id.get(candidate.chunk_id)
            # RRF scores are comparable across hybrid sub-questions. Single-channel
            # candidates retain the existing cosine-distance comparison.
            candidate_quality = (
                candidate.retrieval_score
                if candidate.retrieval_score is not None
                else -candidate.distance
            )
            existing_quality = (
                existing.retrieval_score
                if existing is not None and existing.retrieval_score is not None
                else -existing.distance
                if existing is not None
                else float("-inf")
            )
            if existing is None or candidate_quality > existing_quality:
                best_by_id[candidate.chunk_id] = candidate
    return list(best_by_id.values())


def _candidate_text(candidate: QueryObject) -> str:
    context: list[str] = []
    if candidate.breadcrumb:
        context.append(f"Section: {candidate.breadcrumb}")
    if candidate.section_number:
        context.append(f"Section number: {candidate.section_number}")
    context.append(candidate.content)
    return "\n".join(context)


def rerank(
    question: str, candidates: list[QueryObject], top_k: int | None
) -> list[RankedResult]:
    if not candidates:
        return []
    pairs = [(question, _candidate_text(candidate)) for candidate in candidates]
    scores = get_reranker_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        RankedResult(query_object=candidate, rerank_score=float(score))
        for candidate, score in ranked
    ]


def retrieve_with_diagnostics(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
) -> tuple[list[RankedResult], RetrievalDiagnostics]:
    """
    rewrite -> search -> merge -> rerank
    """
    search_queries = query_rewrite(question, mode=rewrite_mode)
    model = get_embedding_model()
    if embedding_mode not in EMBEDDING_MODES:
        raise ValueError(
            f"Unknown embedding mode {embedding_mode!r}; expected one of {EMBEDDING_MODES}"
        )
    if hybrid_pool_mode not in HYBRID_POOL_MODES:
        raise ValueError(
            f"Unknown hybrid pool mode {hybrid_pool_mode!r}; "
            f"expected one of {HYBRID_POOL_MODES}"
        )

    candidate_sets: list[list[QueryObject]] = []
    channel_candidate_sets: list[dict[str, list[QueryObject]]] = []
    for search_query in search_queries:
        if embedding_mode == "hybrid":
            query_vector = model.encode(search_query.strip(), normalize_embeddings=True)
            raw_candidates = _search_by_vector(
                query_vector,
                retrieval_top_k,
                conn,
                embedding_mode="raw",
            )
            contextual_candidates = _search_by_vector(
                query_vector,
                retrieval_top_k,
                conn,
                embedding_mode="contextual",
            )
            channels = {
                "raw": raw_candidates,
                "contextual": contextual_candidates,
            }
            if hybrid_pool_mode == "union":
                candidates = raw_preserving_union(
                    raw_candidates,
                    contextual_candidates,
                )
            else:
                candidates = reciprocal_rank_fusion(
                    [raw_candidates, contextual_candidates],
                    retrieval_top_k,
                    rrf_k=rrf_k,
                )
        else:
            candidates = search(
                search_query,
                retrieval_top_k,
                conn,
                model,
                embedding_mode=embedding_mode,
            )
            channels = {embedding_mode: candidates}
        candidate_sets.append(candidates)
        channel_candidate_sets.append(channels)

    pooled = _merge_candidates(candidate_sets)
    all_ranked = rerank(question, pooled, top_k=None)
    diagnostics = RetrievalDiagnostics(
        search_queries=search_queries,
        candidate_sets=candidate_sets,
        channel_candidate_sets=channel_candidate_sets,
        pooled_candidates=pooled,
        reranked_candidates=all_ranked,
    )
    return all_ranked[:rerank_top_k], diagnostics


def retrieve(
    question: str,
    conn,
    *,
    rewrite_mode: str | None = None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
) -> list[RankedResult]:
    ranked, _ = retrieve_with_diagnostics(
        question,
        conn,
        rewrite_mode=rewrite_mode,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        embedding_mode=embedding_mode,
        rrf_k=rrf_k,
        hybrid_pool_mode=hybrid_pool_mode,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SDG&E WMP knowledge base.")
    parser.add_argument("question", type=str, help="Question to ask.")
    parser.add_argument(
        "--embedding-mode",
        choices=EMBEDDING_MODES,
        default=DEFAULT_EMBEDDING_MODE,
        help="Dense retrieval strategy: raw, contextual, or hybrid.",
    )
    parser.add_argument(
        "--hybrid-pool-mode",
        choices=HYBRID_POOL_MODES,
        default=DEFAULT_HYBRID_POOL_MODE,
        help=(
            "How hybrid mode builds the reranker pool: fixed-size RRF or a "
            "raw-preserving deduplicated union (default: rrf)."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help="Reciprocal-rank-fusion constant used by hybrid mode (default: 60).",
    )
    args = parser.parse_args()
    if args.rrf_k < 1:
        parser.error("--rrf-k must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        results = retrieve(
            args.question,
            conn,
            embedding_mode=args.embedding_mode,
            rrf_k=args.rrf_k,
            hybrid_pool_mode=args.hybrid_pool_mode,
        )
    finally:
        conn.close()

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        qo = result.query_object
        retrieval_detail = (
            f"rrf_score={qo.retrieval_score:.6f}"
            if qo.retrieval_score is not None
            else f"distance={qo.distance:.4f}"
        )
        print(f"\n[{rank}] rerank_score={result.rerank_score:.4f} {retrieval_detail}")
        print(f"    {qo.source_pdf} (p.{qo.page_start}-{qo.page_end})")
        print(f"    {qo.breadcrumb}")
        print(f"    {qo.content[:300]}")


if __name__ == "__main__":
    main()
