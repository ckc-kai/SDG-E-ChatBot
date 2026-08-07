"""Task 3 benchmark helpers.

This module deliberately does not claim that a scripted mock measures prompt
quality. It provides repeatable request construction and response scoring that
can be reused unchanged when a real provider becomes available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation.schemas import AnswerRequest, AnswerResponse, Chunk, ChunkMetadata
from generation.service import AnswerService


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    return rows


def request_from_benchmark_row(row: dict[str, Any]) -> AnswerRequest:
    """Build a Task 3 request from gold evidence for isolated generation tests.

    This bypasses retrieval and therefore must not be reported as end-to-end RAG
    performance. ``content_excerpt`` may also be shorter than the stored chunk.
    """
    chunks = []
    for evidence in row.get("evidence", []):
        source_file = str(evidence["source_pdf"])
        chunks.append(
            Chunk(
                source_id=source_file,
                chunk_id=str(evidence["chunk_id"]),
                content=str(evidence["content_excerpt"]),
                metadata=ChunkMetadata(
                    source_file=source_file,
                    page_start=evidence.get("page_start_db"),
                    page_end=evidence.get("page_end_db_exclusive"),
                    breadcrumb=evidence.get("breadcrumb"),
                    chunk_index=evidence.get("chunk_index"),
                ),
            )
        )
    return AnswerRequest(
        request_id=str(row["id"]),
        question=str(row["question"]),
        chunks=tuple(chunks),
    )


@dataclass(frozen=True)
class EvaluationScore:
    request_id: str
    citation_precision: float
    citation_recall: float
    answer_exact_match: float
    insufficient_context: bool
    warnings: tuple[str, ...]


def score_response(row: dict[str, Any], response: AnswerResponse) -> EvaluationScore:
    gold_ids = {str(item["chunk_id"]) for item in row.get("evidence", [])}
    cited_ids = set(response.cited_chunk_ids)
    correct = len(gold_ids & cited_ids)
    citation_precision = correct / len(cited_ids) if cited_ids else 0.0
    citation_recall = correct / len(gold_ids) if gold_ids else 0.0
    expected = normalize_text(str(row.get("expected_answer", "")))
    actual = normalize_text(response.answer)
    return EvaluationScore(
        request_id=response.request_id,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        # Exact match is intentionally strict and should be supplemented by a
        # reviewed semantic/groundedness rubric once a real model is connected.
        answer_exact_match=1.0 if expected and actual == expected else 0.0,
        insufficient_context=response.insufficient_context,
        warnings=response.warnings,
    )


def aggregate_scores(scores: Iterable[EvaluationScore]) -> dict[str, float | int]:
    rows = list(scores)
    if not rows:
        return {"count": 0, "citation_precision": 0.0, "citation_recall": 0.0, "answer_exact_match": 0.0}
    return {
        "count": len(rows),
        "citation_precision": sum(row.citation_precision for row in rows) / len(rows),
        "citation_recall": sum(row.citation_recall for row in rows) / len(rows),
        "answer_exact_match": sum(row.answer_exact_match for row in rows) / len(rows),
    }


def evaluate_benchmark(
    rows: Iterable[dict[str, Any]], service: AnswerService
) -> tuple[list[EvaluationScore], dict[str, float | int]]:
    """Run isolated Task 3 evaluation with gold evidence supplied as context.

    With a scripted mock this checks orchestration and scoring only. Semantic
    prompt quality requires an approved real provider and human/groundedness
    review; callers must label the provider used in any reported results.
    """
    scores = []
    for row in rows:
        request = request_from_benchmark_row(row)
        response = service.answer(request)
        scores.append(score_response(row, response))
    return scores, aggregate_scores(scores)
