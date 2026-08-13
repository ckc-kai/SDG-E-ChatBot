"""Task 4 wrapper around Task 2's grouped retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass
import time

from retrieval.query.excel.channel import (
    ExcelAnswer,
    answer_from_excel,
    is_entity_history_question,
)
from retrieval.query.pdf import EvidenceRetrievalResult, retrieve_configured
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
    timings: RetrievalTimings = RetrievalTimings()


class RetrievalService:
    def retrieve(
        self,
        question: str,
        *,
        embedding_mode: str | None = None,
        rewrite_mode: str | None = None,
        content_type: str | None = None,
    ) -> RetrievalBundle:
        started = time.perf_counter()
        kwargs = {"output_mode": "grouped"}
        if embedding_mode is not None:
            kwargs["embedding_mode"] = embedding_mode
        if rewrite_mode is not None:
            kwargs["rewrite_mode"] = rewrite_mode
        if content_type is not None:
            kwargs["groups"] = (_CONTENT_TYPE_TO_GROUP[content_type],)
        elif required_source_roles(question):
            # Explicit regulatory comparisons need prose and tables, not
            # figures or an unrelated Excel-card search.
            kwargs["groups"] = ("narrative", "table")

        connection_started = time.perf_counter()
        conn = connect_db()
        connection_ms = round((time.perf_counter() - connection_started) * 1000)
        grouped_retrieval_ms = 0
        excel_verification_ms = 0
        try:
            source_roles = required_source_roles(question)
            should_try_excel = content_type in (None, "excel_card") and not source_roles
            exact_history = is_entity_history_question(question)
            verified_excel = None
            excel_attempted = False
            if should_try_excel and exact_history:
                excel_started = time.perf_counter()
                try:
                    excel_result = answer_from_excel(question, conn)
                    excel_attempted = True
                finally:
                    excel_verification_ms = round(
                        (time.perf_counter() - excel_started) * 1000
                    )
                if isinstance(excel_result, ExcelAnswer):
                    verified_excel = excel_result
                    if content_type is None:
                        # Suppress semantic work only after exact execution has
                        # succeeded. A decline keeps normal fallback evidence.
                        kwargs["groups"] = ()

            retrieval_started = time.perf_counter()
            try:
                result = retrieve_configured(question, conn, **kwargs)
            finally:
                grouped_retrieval_ms = round(
                    (time.perf_counter() - retrieval_started) * 1000
                )
            if should_try_excel and not excel_attempted:
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
            timings=RetrievalTimings(
                connection_ms=connection_ms,
                grouped_retrieval_ms=grouped_retrieval_ms,
                excel_verification_ms=excel_verification_ms,
                total_ms=round((time.perf_counter() - started) * 1000),
            ),
        )
