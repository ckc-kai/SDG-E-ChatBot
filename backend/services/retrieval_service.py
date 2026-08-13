"""Task 4 wrapper around Task 2's grouped retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass
import time

from retrieval.query.excel.channel import ExcelAnswer, answer_from_excel
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult, retrieve_configured
from generation.planning import RetrievalPlan
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


def _groups_for_types(content_types) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_CONTENT_TYPE_TO_GROUP[value] for value in content_types))


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

        connection_started = time.perf_counter()
        conn = connect_db()
        connection_ms = round((time.perf_counter() - connection_started) * 1000)
        grouped_retrieval_ms = 0
        excel_verification_ms = 0
        try:
            retrieval_started = time.perf_counter()
            try:
                result = retrieve_configured(question, conn, **kwargs)
            finally:
                grouped_retrieval_ms = round(
                    (time.perf_counter() - retrieval_started) * 1000
                )
            verified_excel = None
            excel_requested = (
                content_type in (None, "excel_card")
                if content_types is None
                else "excel_card" in content_types
            )
            if excel_requested:
                excel_started = time.perf_counter()
                try:
                    excel_result = answer_from_excel(question, conn)
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                if isinstance(excel_result, ExcelAnswer):
                    verified_excel = excel_result
        finally:
            conn.close()
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

    def retrieve_plan(self, question: str, plan: RetrievalPlan, **kwargs) -> RetrievalBundle:
        """Retrieve each subquestion separately and preserve per-step coverage."""
        started = time.perf_counter()
        step_bundles = [
            self.retrieve(step.question, content_types=step.content_types, **kwargs)
            for step in plan.steps
        ]
        bundles = step_bundles
        merged: dict[str, EvidenceGroup] = {}
        group_names = tuple(dict.fromkeys(
            name for bundle in bundles for name in bundle.evidence.groups
        ))
        for name in group_names:
            step_groups = [
                bundle.evidence.groups[name]
                for bundle in bundles
                if name in bundle.evidence.groups
            ]
            combined = []
            seen: set[str] = set()
            longest = max((len(group.results) for group in step_groups), default=0)
            for rank in range(longest):
                for group in step_groups:
                    if rank >= len(group.results):
                        continue
                    result = group.results[rank]
                    chunk_id = str(result.query_object.chunk_id)
                    if chunk_id not in seen:
                        combined.append(result)
                        seen.add(chunk_id)
            exemplar = step_groups[0]
            merged[name] = EvidenceGroup(
                name=name,
                content_types=exemplar.content_types,
                results=combined,
                diagnostics=exemplar.diagnostics,
            )
        timings = RetrievalTimings(
            connection_ms=sum(item.timings.connection_ms for item in bundles),
            grouped_retrieval_ms=sum(item.timings.grouped_retrieval_ms for item in bundles),
            excel_verification_ms=sum(item.timings.excel_verification_ms for item in bundles),
            total_ms=round((time.perf_counter() - started) * 1000),
        )
        verified_list: list[ExcelAnswer] = []
        verified_keys: set[tuple] = set()
        for item in bundles:
            answer = item.verified_excel
            if answer is None:
                continue
            key = (answer.card_chunk_id, answer.question)
            if key not in verified_keys:
                verified_keys.add(key)
                verified_list.append(answer)
        verified_items = tuple(verified_list)
        verified = verified_items[0] if verified_items else None
        return RetrievalBundle(
            EvidenceRetrievalResult(question=question, groups=merged),
            verified,
            verified_items,
            timings,
            {
                "source": plan.source,
                "trigger_reason": plan.trigger_reason,
                "subquestion_count": len(plan.steps),
                "steps": [
                    {
                        "question": step.question,
                        "content_types": list(step.content_types),
                        "chunks": {
                            name: [str(result.query_object.chunk_id) for result in bundle.evidence.groups[name].results]
                            for name in bundle.evidence.groups
                        },
                        "verified_excel": bool(bundle.verified_excel),
                    }
                    for step, bundle in zip(plan.steps, step_bundles, strict=True)
                ],
            },
        )
