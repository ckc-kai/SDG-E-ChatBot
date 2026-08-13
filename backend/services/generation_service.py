"""Connect Task 2 grouped evidence to the Task 3 answer service."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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
            chunks[0:0] = _verified_excel_chunks(bundle.verified_excel)
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


def _verified_excel_chunks(answer) -> list[Chunk]:
    """Turn Task 2's executed SQL result into explicit grounded evidence."""
    selected_keys = tuple(getattr(answer.plan, "select_json_keys", ()))
    required = {
        "annual_quant_target",
        "quant_actual_progress_q1_4",
        "quant_target_units",
    }
    if required.issubset(selected_keys) and "reporting_year" in answer.result.columns:
        return _verified_entity_history_chunks(answer, selected_keys)

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
    return [
        Chunk(
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
    ]


def _decimal(value) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Excel history value is not numeric: {value!r}") from exc


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _percent_complete(actual: Decimal, target: Decimal) -> Decimal | None:
    if target == 0:
        return None
    return (actual / target * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def _format_percent(value: Decimal | None) -> str:
    return "not defined (target is zero)" if value is None else f"{value}%"


def _verified_entity_history_chunks(
    answer, selected_keys: tuple[str, ...]
) -> list[Chunk]:
    columns = answer.result.columns
    year_index = columns.index("reporting_year")
    record_index = columns.index("record_id")
    selected_indexes = {
        key: columns.index(f"selected_{index}")
        for index, key in enumerate(selected_keys)
    }
    row_chunks: list[Chunk] = []
    totals_target = Decimal(0)
    totals_actual = Decimal(0)
    calculation_lines: list[str] = []
    provenance_lines: list[str] = []

    for row_index, row in enumerate(answer.result.rows):
        provenance = (
            answer.result.provenance[row_index]
            if row_index < len(answer.result.provenance)
            else {}
        )
        year = int(row[year_index])
        target = _decimal(row[selected_indexes["annual_quant_target"]])
        actual = _decimal(row[selected_indexes["quant_actual_progress_q1_4"]])
        unit = str(row[selected_indexes["quant_target_units"]])
        percent = _percent_complete(actual, target)
        totals_target += target
        totals_actual += actual
        source_file = (
            provenance.get("source_file") or f"sdge_table{answer.table_number:02d}.csv"
        )
        source_row = _int_or_none(provenance.get("source_row"))
        content = "\n".join(
            [
                "Execution-verified Excel row:",
                f"reporting_year={year}",
                f"record_id={row[record_index]}",
                f"annual_target={_format_decimal(target)}",
                f"q4_year_end_actual={_format_decimal(actual)}",
                f"unit={unit}",
                f"percent_complete={_format_percent(percent)}",
                (
                    "calculation=undefined because target is zero"
                    if percent is None
                    else f"calculation={_format_decimal(actual)} / {_format_decimal(target)} x 100"
                ),
            ]
        )
        row_chunks.append(
            Chunk(
                source_id=str(source_file),
                chunk_id=f"excel-exec-{answer.card_chunk_id}-{year}",
                content=content,
                metadata=ChunkMetadata(
                    source_file=str(source_file),
                    sheet=provenance.get("source_sheet"),
                    row_start=source_row,
                    row_end=source_row,
                    breadcrumb=f"Quarterly Data Report > Table {answer.table_number} > {year}",
                    content_type="excel_card",
                ),
            )
        )
        calculation_lines.append(
            f"{year}: target={_format_decimal(target)}, "
            f"actual={_format_decimal(actual)}, "
            f"percent={_format_percent(percent)}"
        )
        provenance_lines.append(
            f"{year}: {source_file}, "
            f"{provenance.get('source_sheet') or 'Table 1'}, "
            f"row {source_row or 'unknown'}"
        )

    cumulative_percent = _percent_complete(totals_actual, totals_target)
    contributing_sources = tuple(provenance_lines)
    summary = Chunk(
        source_id="Table 1 multi-row calculation",
        chunk_id=f"excel-exec-{answer.card_chunk_id}-summary",
        content="\n".join(
            [
                "Deterministic multi-year calculation from the execution-verified rows:",
                *calculation_lines,
                f"cumulative_target={_format_decimal(totals_target)}",
                f"cumulative_actual={_format_decimal(totals_actual)}",
                f"cumulative_percent_complete={_format_percent(cumulative_percent)}",
                f"variance={_format_decimal(totals_actual - totals_target)}",
                (
                    "cumulative_calculation=undefined because target is zero"
                    if cumulative_percent is None
                    else "cumulative_calculation="
                    f"{_format_decimal(totals_actual)} / {_format_decimal(totals_target)} x 100"
                ),
                "Row provenance:",
                *provenance_lines,
            ]
        ),
        metadata=ChunkMetadata(
            source_file="Table 1 multi-row calculation",
            sheet="Table 1",
            breadcrumb="Quarterly Data Report > Table 1 > cumulative calculation",
            content_type="excel_card",
            contributing_sources=contributing_sources,
        ),
    )
    return [*row_chunks, summary]
