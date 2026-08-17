"""Task 4 wrapper around grouped retrieval and verified Excel execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from generation.coverage import assess_plan_coverage
from generation.computation import CalculationResult
from generation.features import feature_enabled
from generation.planning import RetrievalPlan
from retrieval.query.excel.channel import (
    ExcelAnswer,
    answer_from_excel,
    is_entity_history_question,
)
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult, retrieve_configured
from retrieval.query.pdf.query import required_source_roles
from retrieval.utils import connect_db


_CONTENT_TYPE_TO_GROUP = {
    "narrative": "narrative",
    "table": "table",
    "figure": "figure",
    "excel_card": "excel",
}


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


def _groups_for_types(content_types) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_CONTENT_TYPE_TO_GROUP[value] for value in content_types))


def _scoped_step_question(original_question: str, step_question: str) -> str:
    """Keep explicit document roles if a model planner drops part of the scope."""
    if not required_source_roles(original_question):
        return step_question
    return (
        f"{step_question} Required source scope from the original question: "
        f"{original_question}"
    )


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
        try:
            source_roles = required_source_roles(question)
            excel_requested = (
                content_type in (None, "excel_card")
                if content_types is None
                else "excel_card" in content_types
            )
            should_try_excel = excel_requested and not source_roles
            excel_attempted = False

            if should_try_excel and is_entity_history_question(question):
                excel_started = time.perf_counter()
                try:
                    excel_result = answer_from_excel(question, connection)
                    excel_attempted = True
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                if isinstance(excel_result, ExcelAnswer):
                    verified_excel = excel_result
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
                    excel_result = answer_from_excel(question, connection)
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                if isinstance(excel_result, ExcelAnswer):
                    verified_excel = excel_result
        finally:
            connection.close()

        if not isinstance(result, EvidenceRetrievalResult):
            raise TypeError("Task 2 did not return grouped evidence")
        return RetrievalBundle(
            evidence=result,
            verified_excel=verified_excel,
            verified_excels=((verified_excel,) if verified_excel else ()),
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
            verified_excel=verified_items[0] if verified_items else None,
            verified_excels=verified_items,
            timings=timings,
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
