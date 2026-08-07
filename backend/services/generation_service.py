"""Connect Task 2 grouped evidence to the Task 3 answer service."""

from __future__ import annotations

from dataclasses import replace
import time

from generation.adapters import adapt_ranked_results
from generation.providers import create_provider_from_env
from generation.schemas import (
    AnswerRequest,
    AnswerResponse,
    Chunk,
    ChunkMetadata,
    ErrorResponse,
)
from generation.service import AnswerService
from retrieval.query.pdf import EVIDENCE_GROUPS, EvidenceRetrievalResult
from services.retrieval_service import RetrievalBundle


def interleave_grouped_results(evidence: EvidenceRetrievalResult) -> list:
    """Round-robin group ranks so prompt budgeting cannot starve later groups."""
    group_results = [
        evidence.groups[name].results
        for name in EVIDENCE_GROUPS
        if name in evidence.groups
    ]
    longest = max((len(results) for results in group_results), default=0)
    return [
        results[rank]
        for rank in range(longest)
        for results in group_results
        if rank < len(results)
    ]


class GenerationService:
    def __init__(self, answer_service: AnswerService | None = None):
        self._answer_service = answer_service or AnswerService(
            create_provider_from_env()
        )

    def generate(
        self,
        request_id: str,
        question: str,
        bundle: RetrievalBundle,
    ) -> AnswerResponse | ErrorResponse:
        started = time.perf_counter()
        adapter_started = time.perf_counter()
        ranked_results = interleave_grouped_results(bundle.evidence)
        chunks = list(adapt_ranked_results(ranked_results))
        if bundle.verified_excel is not None:
            chunks.insert(0, _verified_excel_chunk(bundle.verified_excel))
        request = AnswerRequest(
            request_id=request_id,
            question=question,
            chunks=tuple(chunks),
        )
        adapter_ms = round((time.perf_counter() - adapter_started) * 1000)
        result = self._answer_service.answer(request)
        timings = result.timings
        if timings is None:
            return result
        return replace(
            result,
            timings=replace(
                timings,
                adapter_ms=adapter_ms,
                generation_total_ms=round((time.perf_counter() - started) * 1000),
            ),
        )

    def warmup(self) -> None:
        """Warm providers that expose a no-answer local preload hook."""
        warmup = getattr(self._answer_service.provider, "warmup", None)
        if callable(warmup):
            warmup()


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _verified_excel_chunk(answer) -> Chunk:
    """Turn Task 2's executed SQL result into explicit grounded evidence."""
    provenance = answer.result.provenance
    first = provenance[0] if provenance else {}
    row_numbers = [
        number
        for item in provenance
        if (number := _int_or_none(item.get("source_row"))) is not None
    ]
    rendered_rows = [
        ", ".join(
            f"{column}={value}"
            for column, value in zip(answer.result.columns, row, strict=False)
        )
        for row in answer.result.rows
    ]
    content = "\n".join(
        [
            "Execution-verified Excel result:",
            f"Question: {answer.question}",
            f"Table: {answer.table_number}",
            *rendered_rows,
            f"Unit: {answer.unit or 'not specified'}",
            f"Contributing facts: {answer.result.contributing_facts}",
        ]
    )
    source_file = first.get("source_file") or f"sdge_table{answer.table_number:02d}.csv"
    return Chunk(
        source_id=str(source_file),
        chunk_id=f"excel-exec-{answer.card_chunk_id}",
        content=content,
        metadata=ChunkMetadata(
            source_file=str(source_file),
            sheet=first.get("source_sheet"),
            row_start=min(row_numbers) if row_numbers else None,
            row_end=max(row_numbers) if row_numbers else None,
            breadcrumb=f"Quarterly Data Report > Table {answer.table_number}",
            content_type="excel_card",
        ),
    )
