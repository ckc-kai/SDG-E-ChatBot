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
SIMPLE_QUESTION_MAX_WORDS = _config["query_rewrite"]["simple_question_max_words"]
RETRIEVAL_TOP_K = _config["retrieval"]["retrieval_top_k"]
RERANK_TOP_K = _config["retrieval"]["rerank_top_k"]

_COMPOUND_MARKERS = (" and ", " or ", ";", ".")
_WH_WORDS = ("what", "who", "when", "where", "why", "how", "which", "is", "has", "have", "does", "do")

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a compound question about SDG&E's Wildfire Mitigation Plan into "
    "focused, self-contained sub-questions that can each be answered independently. "
    "Respond with ONLY a JSON array of strings, no other text. "
    "If the question is already simple, return a single-element array containing it unchanged."
)


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


def _is_simple_question(question: str) -> bool:
    normalized = question.strip().lower()
    if normalized.count("?") > 1:
        return False
    if any(marker in normalized for marker in _COMPOUND_MARKERS):
        return False
    wh_count = sum(1 for word in _WH_WORDS if re.search(rf"\b{word}\b", normalized))
    if wh_count > 1:
        return False
    return len(normalized.split()) <= SIMPLE_QUESTION_MAX_WORDS


def query_rewrite(question: str) -> list[str]:
    question = question.strip()
    if _is_simple_question(question):
        return [question]

    try:
        response = get_anthropic_client().messages.create(
            model=QUERY_REWRITE_MODEL,
            max_tokens=512,
            system=_DECOMPOSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        sub_questions = json.loads(response.content[0].text)
        if not isinstance(sub_questions, list) or not sub_questions or not all(
            isinstance(item, str) for item in sub_questions
        ):
            raise ValueError(f"Unexpected rewrite response shape: {sub_questions!r}")
        return sub_questions
    except Exception as exc:
        log_failure("query_rewrite", question, exc)
        logger.warning("query_rewrite failed, falling back to original question: %s", exc)
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


def rerank(question: str, candidates: list[QueryObject], top_k: int) -> list[RankedResult]:
    if not candidates:
        return []
    pairs = [(question, candidate.content) for candidate in candidates]
    scores = get_reranker_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [RankedResult(query_object=candidate, rerank_score=float(score)) for candidate, score in ranked[:top_k]]


def retrieve(question: str, conn) -> list[RankedResult]:
    """
    rewrite -> search -> merge -> rerank
    """
    sub_questions = query_rewrite(question)
    model = get_embedding_model()
    candidate_sets = [search(sub_question, RETRIEVAL_TOP_K, conn, model) for sub_question in sub_questions]
    pooled = _merge_candidates(candidate_sets)
    return rerank(question, pooled, RERANK_TOP_K)


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
