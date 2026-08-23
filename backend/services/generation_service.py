"""Connect grouped Task 2 evidence to the Task 3 answer service."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from generation.adapters import adapt_ranked_results
from generation.computation import CalculationResult
from generation.features import feature_enabled
from generation.followup import resolve_followup_question
from generation.multistep import (
    PlannedSubanswer,
    build_synthesis_prompt,
    protect_synthesis_coverage,
)
from generation.planning import (
    RetrievalPlan,
    build_retrieval_plan,
    supports_multistep_generation,
)
from generation.providers import create_provider_from_env
from generation.providers.base import ProviderError
from generation.providers.ollama import ANSWER_SCHEMA
from generation.routing import RouteDecision, route_question
from generation.schemas import (
    AnswerRequest,
    AnswerResponse,
    AnswerTimings,
    Chunk,
    ChunkMetadata,
    ErrorResponse,
    ModelAnswer,
)
from generation.service import AnswerService, ModelOutputError, parse_model_answer
from generation.citation_validation import validate_and_hydrate_citations
from generation.excel_rollup import display_columns, roll_up
from retrieval.query.excel.manifest import load_manifest
from retrieval.query.pdf import EVIDENCE_GROUPS, EvidenceRetrievalResult
from retrieval.utils import connect_db
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


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Keep the highest-ranked copy of equivalent evidence text."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk.content).strip().casefold()
        key = normalized or f"empty:{chunk.chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


class GenerationService:
    def __init__(
        self,
        answer_service: AnswerService | None = None,
        planner_provider=None,
        answer_service_factory=None,
    ):
        self._answer_service_factory = answer_service_factory
        if answer_service is None:
            self._answer_service_factory = (
                answer_service_factory
                or (lambda: AnswerService(create_provider_from_env()))
            )
            answer_service = self._answer_service_factory()
        self._answer_service = answer_service
        self._planner_provider = planner_provider

    def generate(
        self,
        request_id: str,
        question: str,
        bundle: RetrievalBundle,
    ) -> AnswerResponse | ErrorResponse:
        started = time.perf_counter()
        adapter_started = time.perf_counter()
        chunks = _chunks_from_bundle(bundle)
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
                generation_total_ms=round(
                    (time.perf_counter() - started) * 1000
                ),
            ),
        )

    def generate_mixed(
        self,
        request_id: str,
        question: str,
        bundle: RetrievalBundle,
        plan: RetrievalPlan,
    ) -> AnswerResponse | ErrorResponse:
        """Use parallel subanswers only for a complete multi-step model plan."""
        if (
            not feature_enabled("multistep_generation")
            or not supports_multistep_generation(plan)
            or len(bundle.step_bundles) != len(plan.steps)
        ):
            return self.generate(request_id, question, bundle)

        started = time.perf_counter()

        def answer_step(index_bundle):
            index, step_bundle = index_bundle
            service = (
                self._answer_service_factory()
                if self._answer_service_factory is not None
                else self._answer_service
            )
            worker = GenerationService(answer_service=service)
            return worker.generate(
                f"{request_id}_step_{index + 1}",
                plan.steps[index].question,
                step_bundle,
            )

        with ThreadPoolExecutor(max_workers=len(plan.steps)) as executor:
            responses = tuple(executor.map(
                answer_step, enumerate(bundle.step_bundles)
            ))
        if any(isinstance(response, ErrorResponse) for response in responses):
            return self.generate(request_id, question, bundle)

        subanswers = tuple(
            PlannedSubanswer(
                requirement_ids=plan.steps[index].requirement_ids,
                answer=ModelAnswer(
                    answer=response.answer,
                    cited_chunk_ids=response.cited_chunk_ids,
                    insufficient_context=response.insufficient_context,
                    answered_requirements=response.answered_requirements,
                    missing_requirements=response.missing_requirements,
                ),
            )
            for index, response in enumerate(responses)
        )
        synthesis_service = (
            self._answer_service_factory()
            if self._answer_service_factory is not None
            else self._answer_service
        )
        provider = synthesis_service.provider
        prompt = build_synthesis_prompt(question, plan.requirements, subanswers)
        try:
            structured = getattr(provider, "generate_structured", None)
            raw = (
                structured(prompt, ANSWER_SCHEMA)
                if callable(structured)
                else provider.generate(prompt)
            )
            synthesized = protect_synthesis_coverage(
                parse_model_answer(raw), plan.requirements, subanswers
            )
            chunks = _chunks_from_bundle(bundle)
            registry = {chunk.chunk_id: chunk for chunk in chunks}
            allowed_ids = tuple(dict.fromkeys(
                chunk_id
                for item in subanswers
                for chunk_id in item.answer.cited_chunk_ids
                if chunk_id in registry
            ))
            validation_request = AnswerRequest(
                request_id=request_id,
                question=question,
                chunks=tuple(registry[chunk_id] for chunk_id in allowed_ids),
            )
            valid_ids, citations, warnings = validate_and_hydrate_citations(
                validation_request, synthesized
            )
        except (
            ProviderError,
            ModelOutputError,
            TimeoutError,
            ConnectionError,
        ):
            return self.generate(request_id, question, bundle)

        latency_ms = round((time.perf_counter() - started) * 1000)
        input_tokens = sum(
            response.timings.model_input_tokens or 0
            for response in responses
            if response.timings is not None
        ) + (getattr(getattr(provider, "last_usage", None), "input_tokens", 0) or 0)
        output_tokens = sum(
            response.timings.model_output_tokens or 0
            for response in responses
            if response.timings is not None
        ) + (getattr(getattr(provider, "last_usage", None), "output_tokens", 0) or 0)
        return AnswerResponse(
            request_id=request_id,
            answer=synthesized.answer,
            cited_chunk_ids=valid_ids,
            citations=citations,
            insufficient_context=synthesized.insufficient_context,
            model_id=provider.model_id,
            latency_ms=latency_ms,
            warnings=warnings,
            timings=AnswerTimings(
                generation_total_ms=latency_ms,
                model_input_tokens=input_tokens,
                model_output_tokens=output_tokens,
            ),
            answered_requirements=synthesized.answered_requirements,
            missing_requirements=synthesized.missing_requirements,
        )

    def plan_retrieval(self, question: str) -> RetrievalPlan:
        if self._planner_provider is None:
            self._planner_provider = _create_planner_provider()
        plan = build_retrieval_plan(question, self._planner_provider)
        if plan.source != "fallback":
            return plan
        # Local planner failure would silently degrade a multi-part question
        # to one broad step; escalate once to the hosted structured planner.
        escalation = self._escalation_planner_provider()
        if escalation is None:
            return plan
        escalated = build_retrieval_plan(question, escalation)
        return escalated if escalated.source == "model" else plan

    _escalation_provider = None
    _escalation_failed = False

    def _escalation_planner_provider(self):
        if self._escalation_provider is not None or self._escalation_failed:
            return self._escalation_provider
        if not os.getenv("GROQ_API_KEY", "").strip():
            self._escalation_failed = True
            return None
        values = dict(os.environ)
        values["GROQ_MODEL"] = values.get(
            "TASK3_PLANNER_ESCALATION_MODEL", "openai/gpt-oss-120b"
        )
        values["GROQ_MAX_TOKENS"] = "700"
        values["GROQ_REASONING_EFFORT"] = "low"
        try:
            self._escalation_provider = create_provider_from_env(
                "groq", environ=values
            )
        except Exception:  # noqa: BLE001 - degrade to the local fallback plan
            self._escalation_failed = True
            self._escalation_provider = None
        return self._escalation_provider

    def route_retrieval(self, question: str) -> RouteDecision:
        """Apply deterministic routing and consult one local judge only if needed."""
        route = route_question(question, judge_enabled=False)
        if not route.uncertain or not feature_enabled("support_evidence_judge"):
            return route
        if self._planner_provider is None:
            self._planner_provider = _create_planner_provider()
        return route_question(question, judge=self._planner_provider)

    def resolve_followup(self, question: str, history) -> str:
        """Resolve an underspecified follow-up without treating history as evidence."""
        if self._planner_provider is None:
            self._planner_provider = _create_planner_provider()
        return resolve_followup_question(question, history, self._planner_provider)

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


def _chunks_from_bundle(bundle: RetrievalBundle) -> list[Chunk]:
    ranked_results = interleave_grouped_results(bundle.evidence)
    chunks = list(adapt_ranked_results(ranked_results))
    if feature_enabled("cross_resource_computation"):
        for index, note in reversed(
            tuple(enumerate(getattr(bundle, "computation_notes", ()), start=1))
        ):
            chunks.insert(0, _computation_note_chunk(note, index))
        for index, calculation in reversed(
            tuple(enumerate(bundle.calculations, start=1))
        ):
            chunks.insert(0, _calculation_chunk(calculation, index))
    # Row windows lead the ranked evidence, with verified executions above
    # them. Moving them after the ranked evidence was measured and reverted:
    # 48.24 against 51.37, with refusals back up from 6 to 9 of 27. Evidence
    # the model has to scroll past is evidence it declines to use.
    for row_slice in reversed(getattr(bundle, "excel_rows", ())):
        chunks.insert(0, _excel_rows_chunk(row_slice))
    verified_items = bundle.verified_excels or (
        (bundle.verified_excel,) if bundle.verified_excel is not None else ()
    )
    for answer in reversed(verified_items):
        chunks[0:0] = _verified_excel_chunks(answer)
    # The corpus's shape, above everything derived from it. Several
    # questions are answerable only from coverage and schema -- a year the
    # workbooks do not reach, a field a table does not carry, two tables
    # that share no key -- and with none of it in the prompt the model
    # estimated instead of saying so.
    manifest_chunk = _workbook_manifest_chunk(bundle)
    if manifest_chunk is not None:
        chunks.insert(0, manifest_chunk)
    chunks = deduplicate_chunks(chunks)
    # Different verified executions can share one card id. Preserve each
    # distinct result under a unique id so citation validation is unambiguous.
    seen_ids: dict[str, int] = {}
    unique_chunks: list[Chunk] = []
    for chunk in chunks:
        count = seen_ids.get(chunk.chunk_id)
        if count is None:
            seen_ids[chunk.chunk_id] = 0
        else:
            seen_ids[chunk.chunk_id] = count + 1
            chunk = replace(chunk, chunk_id=f"{chunk.chunk_id}-v{count + 1}")
        unique_chunks.append(chunk)
    return unique_chunks


def _calculation_chunk(result: CalculationResult, index: int) -> Chunk:
    rendered_unit = f" {result.unit}" if result.unit else ""
    return Chunk(
        source_id="Derived calculation",
        chunk_id=f"calculation-{index}",
        content=(
            "Deterministic calculation from validated evidence operands:\n"
            f"expression={result.expression}\n"
            f"result={result.value}{rendered_unit}"
        ),
        metadata=ChunkMetadata(
            source_file="Derived calculation",
            breadcrumb="Cross-resource deterministic calculation",
            content_type="calculation",
            contributing_sources=result.contributing_sources,
        ),
    )


def _workbook_manifest_chunk(bundle) -> Chunk | None:
    """Coverage and schema for the workbooks, when the question touched them."""
    if not feature_enabled("workbook_manifest"):
        return None
    touched_excel = bool(
        bundle.verified_excels
        or bundle.verified_excel
        or getattr(bundle, "excel_rows", ())
        or getattr(bundle.evidence, "groups", {}).get("excel")
    )
    if not touched_excel:
        return None
    connection = connect_db()
    try:
        rendered = load_manifest(connection).render()
    finally:
        connection.close()
    if not rendered:
        return None
    return Chunk(
        source_id="SDG&E quarterly data report",
        chunk_id="workbook-manifest",
        content=rendered,
        metadata=ChunkMetadata(
            source_file="SDG&E quarterly data report",
            breadcrumb="Quarterly Data Report > coverage and schema",
            content_type="excel_card",
        ),
    )


def _computation_note_chunk(note: str, index: int) -> Chunk:
    return Chunk(
        source_id="Computation status",
        chunk_id=f"computation-note-{index}",
        content=note,
        metadata=ChunkMetadata(
            source_file="Computation status",
            breadcrumb="Cross-resource computation status",
            content_type="calculation",
        ),
    )


def _create_planner_provider():
    provider_name = os.getenv("TASK3_PLANNER_PROVIDER", "ollama").strip().casefold()
    values = dict(os.environ)
    if provider_name == "ollama":
        values["OLLAMA_MODEL"] = values.get("TASK3_PLANNER_MODEL", "qwen3:4b")
        values["OLLAMA_MAX_TOKENS"] = values.get("TASK3_PLANNER_MAX_TOKENS", "500")
    elif provider_name == "groq":
        values["GROQ_MODEL"] = values.get(
            "TASK3_PLANNER_MODEL", "openai/gpt-oss-20b"
        )
        values["GROQ_MAX_TOKENS"] = values.get("TASK3_PLANNER_MAX_TOKENS", "500")
        values["GROQ_REASONING_EFFORT"] = "low"
    elif provider_name == "deepseek":
        values["DEEPSEEK_MODEL"] = values.get(
            "TASK3_PLANNER_MODEL", "deepseek-v4-flash"
        )
        values["DEEPSEEK_MAX_TOKENS"] = values.get(
            "TASK3_PLANNER_MAX_TOKENS", "500"
        )
    return create_provider_from_env(provider_name, environ=values)


def _excel_rows_chunk(row_slice) -> Chunk:
    """Render a window of workbook rows as one citable evidence chunk."""
    return Chunk(
        source_id=row_slice.source_file,
        chunk_id=f"excel-rows-{row_slice.card_chunk_id}",
        content=row_slice.render(),
        metadata=ChunkMetadata(
            source_file=row_slice.source_file,
            sheet=f"Table {row_slice.table_number}",
            breadcrumb=(
                f"Quarterly Data Report > Table {row_slice.table_number}"
            ),
            content_type="excel_card",
        ),
    )


# Rendering a result as one table chunk with a header row, instead of one
# chunk per row, was measured over three replicates and reverted: 51.23 against
# 54.79 on the sixteen workbook cases, every replicate worse. The compact form
# is cheaper in tokens and easier to read, and it scored worse anyway -- a row
# that is its own chunk is separately citable and carries its own provenance,
# and the per-row path also computes percent-complete arithmetically per row.
# This is the same result the previous session got when it moved row windows
# after the ranked evidence. Do not re-try this without a measurement.


def _verified_excel_chunks(answer) -> list[Chunk]:
    """Turn an executed Excel result into explicit, cited evidence chunks."""
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
    # Generated jsonb group aliases are named back for the reader; see
    # ``generation.excel_rollup.display_columns``.
    display = display_columns(answer.result, answer.plan)
    rendered_rows = [
        ", ".join(
            f"{column}={value}"
            for column, value in zip(display, row, strict=False)
        )
        for row in answer.result.rows
    ]
    # Components without their total is the single largest recoverable Excel
    # loss: 20% of gold figures were fetched as groups and never combined,
    # because the only deterministic roll-up in the system fired for one table
    # read one way. These figures are computed here, never by the model.
    rolled = (
        roll_up(answer.result, answer.plan)
        if feature_enabled("excel_rollups")
        else []
    )
    content = "\n".join(
        [
            "Execution-verified Excel result:",
            f"Question: {answer.question}",
            f"Table: {answer.table_number}",
            *rendered_rows,
            f"Unit: {answer.unit or 'not specified'}",
            f"Contributing facts: {answer.result.contributing_facts}",
            *(
                [
                    "Deterministic roll-ups over the rows above "
                    "(already calculated; quote them, do not recompute):",
                    *[f"  {item.render()}" for item in rolled],
                ]
                if rolled
                else []
            ),
            *[f"Note: {warning}" for warning in answer.result.warnings],
            *(
                [
                    "Missing requested reporting years: "
                    + ", ".join(
                        str(year)
                        for year in answer.bound.get(
                            "missing_reporting_years", ()
                        )
                    )
                ]
                if answer.bound.get("missing_reporting_years")
                else []
            ),
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


def _decimal(value) -> Decimal | None:
    """The value as a Decimal, or None when the workbook left it blank.

    A blank target or actual is a reportable fact -- an activity carried no
    target that year -- not a malformed row. Raising here aborted the whole
    answer once plans widened beyond single-activity histories, where every
    value happened to be populated.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _format_decimal(value: Decimal | None) -> str:
    return "not reported" if value is None else format(value.normalize(), "f")


def _percent_complete(actual: Decimal | None, target: Decimal | None) -> Decimal | None:
    if target is None or actual is None or target == 0:
        return None
    return (actual / target * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def _format_percent(value: Decimal | None) -> str:
    return "not defined (target is zero or absent)" if value is None else f"{value}%"


def _verified_entity_history_chunks(
    answer, selected_keys: tuple[str, ...]
) -> list[Chunk]:
    columns = answer.result.columns
    year_index = columns.index("reporting_year")
    # The plan chooses its own grouping now, so the row identifier may be
    # record_id, entity_key, or title. Any of them names the row for a reader;
    # only the absence of all three is a problem.
    record_index = next(
        (
            columns.index(name)
            for name in ("record_id", "entity_key", "title")
            if name in columns
        ),
        None,
    )
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
        totals_target += target or Decimal(0)
        totals_actual += actual or Decimal(0)
        source_file = (
            provenance.get("source_file")
            or f"sdge_table{answer.table_number:02d}.csv"
        )
        source_row = _int_or_none(provenance.get("source_row"))
        content = "\n".join(
            [
                "Execution-verified Excel row:",
                f"reporting_year={year}",
                f"record={row[record_index] if record_index is not None else 'unnamed'}",
                f"annual_target={_format_decimal(target)}",
                f"q4_year_end_actual={_format_decimal(actual)}",
                f"unit={unit}",
                f"percent_complete={_format_percent(percent)}",
                (
                    "calculation=undefined because the target is zero or absent"
                    if percent is None
                    else f"calculation={_format_decimal(actual)} / "
                         f"{_format_decimal(target)} x 100"
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
                    breadcrumb=(
                        f"Quarterly Data Report > Table {answer.table_number} > {year}"
                    ),
                    content_type="excel_card",
                ),
            )
        )
        calculation_lines.append(
            f"{year}: target={_format_decimal(target)}, "
            f"actual={_format_decimal(actual)}, percent={_format_percent(percent)}"
        )
        provenance_lines.append(
            f"{year}: {source_file}, "
            f"{provenance.get('source_sheet') or 'Table 1'}, "
            f"row {source_row or 'unknown'}"
        )

    cumulative_percent = _percent_complete(totals_actual, totals_target)
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
                *(
                    [
                        "Missing requested reporting years: "
                        + ", ".join(
                            str(year)
                            for year in answer.bound.get(
                                "missing_reporting_years", ()
                            )
                        )
                    ]
                    if answer.bound.get("missing_reporting_years")
                    else []
                ),
            ]
        ),
        metadata=ChunkMetadata(
            source_file="Table 1 multi-row calculation",
            sheet="Table 1",
            breadcrumb=(
                "Quarterly Data Report > Table 1 > cumulative calculation"
            ),
            content_type="excel_card",
            contributing_sources=tuple(provenance_lines),
        ),
    )
    return [*row_chunks, summary]
