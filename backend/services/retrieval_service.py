"""Task 4 wrapper around grouped retrieval and verified Excel execution."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from generation.coverage import assess_plan_coverage
from generation.computation import CalculationResult
from generation.features import feature_enabled
from generation.planning import RetrievalPlan
from retrieval.query.excel.channel import (
    ExcelAnswer,
    answer_from_excel_all,
    is_entity_history_question,
)
from retrieval.query.excel.query import bind_entity_key
from retrieval.query.excel.rows import ExcelRowSlice, fetch_card_rows
from retrieval.query.excel.trace import outcome_trace
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult, retrieve_configured
from retrieval.query.pdf.query import required_source_roles
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)


_CONTENT_TYPE_TO_GROUP = {
    "narrative": "narrative",
    "table": "table",
    "figure": "figure",
    "excel_card": "excel",
}

_WORKBOOK_SIGNAL_RE = re.compile(
    r"\btables?\s+\d+\b|\bQ[1-4]\b|\b(?:QDR|workbook|quarterly data report)\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalTimings:
    connection_ms: int = 0
    grouped_retrieval_ms: int = 0
    excel_verification_ms: int = 0
    total_ms: int = 0


@dataclass(frozen=True)
class RetrievalBundle:
    evidence: EvidenceRetrievalResult
    verified_excel: ExcelAnswer | None = None
    verified_excels: tuple[ExcelAnswer, ...] = ()
    timings: RetrievalTimings = RetrievalTimings()
    plan_diagnostics: dict | None = None
    calculations: tuple[CalculationResult, ...] = ()
    computation_notes: tuple[str, ...] = ()
    excel_rows: tuple[ExcelRowSlice, ...] = ()
    # Diagnostic only: what the Excel channel attempted and what came back.
    # Never rendered into a prompt; read by eval/excel_lane_report.py.
    excel_trace: tuple[dict, ...] = ()


def _groups_for_types(content_types) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_CONTENT_TYPE_TO_GROUP[value] for value in content_types))


def _scoped_step_question(original_question: str, step_question: str) -> str:
    """Keep explicit document roles and entity ids a model planner may drop.

    A step missing the original question's WMP.### entity id is dangerous
    specifically for Excel: the channel would search cards without an entity
    filter and can confidently return a different activity's value instead of
    declining. Re-append the id whenever a step omits it.
    """
    step_question = _reattach_entity_key(original_question, step_question)
    if not required_source_roles(original_question):
        return step_question
    return (
        f"{step_question} Required source scope from the original question: "
        f"{original_question}"
    )


def _reattach_entity_key(original_question: str, step_question: str) -> str:
    original_key = bind_entity_key(original_question)
    if original_key is None or bind_entity_key(step_question) is not None:
        return step_question
    return f"{step_question} ({original_key})"


class RetrievalService:
    def retrieve(
        self,
        question: str,
        *,
        embedding_mode: str | None = None,
        rewrite_mode: str | None = None,
        content_type: str | None = None,
        content_types: tuple[str, ...] | None = None,
    ) -> RetrievalBundle:
        started = time.perf_counter()
        kwargs = {"output_mode": "grouped"}
        if embedding_mode is not None:
            kwargs["embedding_mode"] = embedding_mode
        if rewrite_mode is not None:
            kwargs["rewrite_mode"] = rewrite_mode
        if content_type is not None and content_types is not None:
            raise ValueError("content_type and content_types are mutually exclusive")
        if content_type is not None:
            kwargs["groups"] = (_CONTENT_TYPE_TO_GROUP[content_type],)
        elif content_types is not None:
            kwargs["groups"] = _groups_for_types(content_types)
        elif required_source_roles(question):
            # Regulatory comparisons need plan/guideline prose and tables, not
            # definitions, figures, or an unrelated Excel-card search.
            kwargs["groups"] = ("narrative", "table")

        connection_started = time.perf_counter()
        connection = connect_db()
        connection_ms = round((time.perf_counter() - connection_started) * 1000)
        grouped_retrieval_ms = 0
        excel_verification_ms = 0
        verified_excel = None
        verified_excels: tuple[ExcelAnswer, ...] = ()
        excel_rows: tuple[ExcelRowSlice, ...] = ()
        excel_trace: tuple[dict, ...] = ()
        try:
            source_roles = required_source_roles(question)
            excel_requested = (
                content_type in (None, "excel_card")
                if content_types is None
                else "excel_card" in content_types
            )
            # Workbook signals make the question Excel-answerable even when a
            # planner labeled the step pdf: an exact entity id, a QDR table
            # reference, or a quarter token (the router's own Excel cue). The
            # channel is generate-and-verify, so wrong attempts decline
            # harmlessly.
            if (
                bind_entity_key(question) is not None
                or _WORKBOOK_SIGNAL_RE.search(question)
            ):
                excel_requested = True
            should_try_excel = excel_requested and not source_roles
            excel_attempted = False

            if should_try_excel and is_entity_history_question(question):
                excel_started = time.perf_counter()
                try:
                    excel_result = answer_from_excel_all(question, connection)
                    excel_attempted = True
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                excel_trace = tuple(outcome_trace(excel_result))
                if isinstance(excel_result, tuple) and excel_result:
                    verified_excels = excel_result
                    verified_excel = excel_result[0]
                    if content_type is None and content_types is None:
                        # Exact execution is stronger than semantic cards. Avoid
                        # spending retrieval/context budget after it succeeds.
                        kwargs["groups"] = ()

            retrieval_started = time.perf_counter()
            try:
                result = retrieve_configured(question, connection, **kwargs)
            finally:
                grouped_retrieval_ms = round(
                    (time.perf_counter() - retrieval_started) * 1000
                )

            if should_try_excel and not excel_attempted:
                excel_started = time.perf_counter()
                try:
                    excel_result = answer_from_excel_all(question, connection)
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                excel_trace = tuple(outcome_trace(excel_result))
                if isinstance(excel_result, tuple) and excel_result:
                    verified_excels = excel_result
                    verified_excel = excel_result[0]

            # A retrieved card describes a table; these are the rows it
            # describes. Verified execution, when it fired, is stronger and is
            # rendered ahead of this, so the two do not compete.
            excel_group = getattr(result, "groups", {}).get("excel")
            if excel_group is not None and feature_enabled("excel_row_evidence"):
                excel_rows = fetch_card_rows(
                    question, excel_group.results, connection
                )
        finally:
            connection.close()

        if not isinstance(result, EvidenceRetrievalResult):
            raise TypeError("Task 2 did not return grouped evidence")
        return RetrievalBundle(
            evidence=result,
            excel_rows=excel_rows,
            verified_excel=verified_excel,
            verified_excels=verified_excels,
            excel_trace=excel_trace,
            timings=RetrievalTimings(
                connection_ms=connection_ms,
                grouped_retrieval_ms=grouped_retrieval_ms,
                excel_verification_ms=excel_verification_ms,
                total_ms=round((time.perf_counter() - started) * 1000),
            ),
        )

    def retrieve_plan(
        self, question: str, plan: RetrievalPlan, **kwargs
    ) -> RetrievalBundle:
        """Retrieve each planned subquestion and preserve per-step coverage."""
        started = time.perf_counter()
        def retrieve_step(step):
            return self.retrieve(
                _scoped_step_question(question, step.question),
                content_types=step.content_types,
                **kwargs,
            )

        resource_groups: dict[str, list[tuple[int, object]]] = {}
        for index, step in enumerate(plan.steps):
            resource_groups.setdefault(step.source, []).append((index, step))
        if feature_enabled("parallel_retrieval") and len(resource_groups) > 1:
            # Parallelize the independent PDF and Excel resources while keeping
            # same-resource tasks sequential and shared-model pressure bounded.
            def retrieve_group(indexed_steps):
                return [
                    (index, retrieve_step(step)) for index, step in indexed_steps
                ]

            indexed_bundles = [None] * len(plan.steps)
            with ThreadPoolExecutor(max_workers=min(2, len(resource_groups))) as executor:
                for completed in executor.map(
                    retrieve_group, resource_groups.values()
                ):
                    for index, bundle in completed:
                        indexed_bundles[index] = bundle
            step_bundles = indexed_bundles
        else:
            step_bundles = [retrieve_step(step) for step in plan.steps]

        initial_coverage = assess_plan_coverage(plan, tuple(step_bundles))
        retry_count = 0
        retry_step_index = None
        if (
            initial_coverage.missing_step_indexes
            and feature_enabled("coverage_retry")
        ):
            # One precise retry for the first uncovered atomic task. There is
            # intentionally no loop and no second planner/model call.
            retry_step_index = initial_coverage.missing_step_indexes[0]
            step_bundles[retry_step_index] = retrieve_step(
                plan.steps[retry_step_index]
            )
            retry_count = 1
        coverage = assess_plan_coverage(plan, tuple(step_bundles))
        merged: dict[str, EvidenceGroup] = {}
        group_names = tuple(
            dict.fromkeys(
                name
                for bundle in step_bundles
                for name in bundle.evidence.groups
            )
        )
        for name in group_names:
            step_groups = [
                bundle.evidence.groups[name]
                for bundle in step_bundles
                if name in bundle.evidence.groups
            ]
            combined = []
            seen: set[str] = set()
            longest = max((len(group.results) for group in step_groups), default=0)
            for rank in range(longest):
                for group in step_groups:
                    if rank >= len(group.results):
                        continue
                    ranked = group.results[rank]
                    chunk_id = str(ranked.query_object.chunk_id)
                    if chunk_id not in seen:
                        combined.append(ranked)
                        seen.add(chunk_id)
            exemplar = step_groups[0]
            merged[name] = EvidenceGroup(
                name=name,
                content_types=exemplar.content_types,
                results=combined,
                diagnostics=exemplar.diagnostics,
            )

        verified_list: list[ExcelAnswer] = []
        verified_keys: set[tuple[int, str]] = set()
        for bundle in step_bundles:
            for answer in bundle.verified_excels or (
                (bundle.verified_excel,) if bundle.verified_excel is not None else ()
            ):
                key = (answer.card_chunk_id, answer.question)
                if key not in verified_keys:
                    verified_keys.add(key)
                    verified_list.append(answer)

        plan_excel_trace = [
            trace for bundle in step_bundles for trace in bundle.excel_trace
        ]
        # A decomposed question whose every step was labelled "pdf" never
        # reached the workbooks, even when the figures it asked for -- unit
        # targets missed across the cycle, inspection counts by year -- are
        # held only there. Rather than teach the step router to predict this,
        # attempt the workbooks once on the whole question. The channel is
        # generate-and-verify, so a question the workbooks cannot answer costs
        # one bounded query and contributes nothing to the prompt.
        if not verified_list and not required_source_roles(question):
            excel_connection = connect_db()
            try:
                whole_question = answer_from_excel_all(question, excel_connection)
            except Exception:  # pragma: no cover - defensive, never fatal
                logger.warning("Whole-question Excel attempt failed", exc_info=True)
                whole_question = None
            finally:
                excel_connection.close()
            plan_excel_trace.extend(outcome_trace(whole_question))
            if isinstance(whole_question, tuple):
                for answer in whole_question:
                    key = (answer.card_chunk_id, answer.question)
                    if key not in verified_keys:
                        verified_keys.add(key)
                        verified_list.append(answer)

        merged_excel_rows: tuple[ExcelRowSlice, ...] = ()
        seen_row_scopes: set[tuple] = set()
        for bundle in step_bundles:
            for row_slice in bundle.excel_rows:
                scope = (row_slice.table_number, row_slice.card_chunk_id)
                if scope not in seen_row_scopes:
                    seen_row_scopes.add(scope)
                    merged_excel_rows += (row_slice,)

        calculations: tuple[CalculationResult, ...] = ()
        computation_notes: tuple[str, ...] = ()
        if feature_enabled("cross_resource_computation"):
            from generation.cross_resource import (
                attempt_cross_resource_calculations,
                question_requests_computation,
            )

            if question_requests_computation(question):
                computation_conn = connect_db()
                try:
                    calculations, computation_answers, computation_notes = (
                        attempt_cross_resource_calculations(
                            question, computation_conn
                        )
                    )
                finally:
                    computation_conn.close()
                for answer in computation_answers:
                    key = (answer.card_chunk_id, answer.question)
                    if key not in verified_keys:
                        verified_keys.add(key)
                        verified_list.append(answer)
        verified_items = tuple(verified_list)
        timings = RetrievalTimings(
            connection_ms=sum(item.timings.connection_ms for item in step_bundles),
            grouped_retrieval_ms=sum(
                item.timings.grouped_retrieval_ms for item in step_bundles
            ),
            excel_verification_ms=sum(
                item.timings.excel_verification_ms for item in step_bundles
            ),
            total_ms=round((time.perf_counter() - started) * 1000),
        )
        return RetrievalBundle(
            evidence=EvidenceRetrievalResult(question=question, groups=merged),
            excel_rows=merged_excel_rows,
            verified_excel=verified_items[0] if verified_items else None,
            verified_excels=verified_items,
            excel_trace=tuple(plan_excel_trace),
            timings=timings,
            calculations=calculations,
            computation_notes=computation_notes,
            plan_diagnostics={
                "source": plan.source,
                "trigger_reason": plan.trigger_reason,
                "subquestion_count": len(plan.steps),
                "atomic_task_count": plan.atomic_task_count,
                "dropped_task_count": plan.dropped_task_count,
                "retry_count": retry_count,
                "retry_step_index": retry_step_index,
                "coverage": coverage.to_dict(),
                "steps": [
                    {
                        "question": step.question,
                        "content_types": list(step.content_types),
                        "chunks": {
                            name: [
                                str(result.query_object.chunk_id)
                                for result in bundle.evidence.groups[name].results
                            ]
                            for name in bundle.evidence.groups
                        },
                        "verified_excel": bool(bundle.verified_excel),
                    }
                    for step, bundle in zip(
                        plan.steps, step_bundles, strict=True
                    )
                ],
            },
        )
