"""Query the WMP knowledge base.

1. query_rewrite: rewrite complex/causal questions
2. search: embeds each (sub-)question with the same model used at ingestion
   time and does a pgvector cosine-distance search against `chunks`
3. Candidate pooling: results from every sub-question are merged and
   deduped by chunk id, so a multi-part question doesn't just get scored fragment-by-fragment
4. rerank: cross-encoder on original questions and pooled candidates
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RankedResult:
    query_object: QueryObject
    rerank_score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    search_queries: list[str]
    candidate_sets: list[list[QueryObject]]
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


def search(question: str, top_k: int, conn, model) -> list[QueryObject]:
    query_vector = model.encode(question.strip(), normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.filename, c.sub_document, c.breadcrumb, c.section_number,
                   c.page_start, c.page_end, c.chunk_index, c.content_type,
                   c.content, c.token_count, c.embedding <=> %s AS distance
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


def _merge_candidates(candidate_sets: list[list[QueryObject]]) -> list[QueryObject]:
    best_by_id: dict[int, QueryObject] = {}
    for candidates in candidate_sets:
        for candidate in candidates:
            existing = best_by_id.get(candidate.chunk_id)
            # if a chunk exist more than once, different sub-question requires the same doc
            if existing is None or candidate.distance < existing.distance:
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
) -> tuple[list[RankedResult], RetrievalDiagnostics]:
    """
    rewrite -> search -> merge -> rerank
    """
    search_queries = query_rewrite(question, mode=rewrite_mode)
    model = get_embedding_model()
    candidate_sets = [
        search(search_query, retrieval_top_k, conn, model)
        for search_query in search_queries
    ]
    pooled = _merge_candidates(candidate_sets)
    all_ranked = rerank(question, pooled, top_k=None)
    diagnostics = RetrievalDiagnostics(
        search_queries=search_queries,
        candidate_sets=candidate_sets,
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
) -> list[RankedResult]:
    ranked, _ = retrieve_with_diagnostics(
        question,
        conn,
        rewrite_mode=rewrite_mode,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SDG&E WMP knowledge base.")
    parser.add_argument("question", type=str, help="Question to ask.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        results = retrieve(args.question, conn)
    finally:
        conn.close()

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        qo = result.query_object
        print(f"\n[{rank}] rerank_score={result.rerank_score:.4f} distance={qo.distance:.4f}")
        print(f"    {qo.source_pdf} (p.{qo.page_start}-{qo.page_end})")
        print(f"    {qo.breadcrumb}")
        print(f"    {qo.content[:300]}")


if __name__ == "__main__":
    main()
