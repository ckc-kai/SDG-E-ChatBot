"""Task 4 wrapper around Task 2's grouped retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass
import time

from retrieval.query.excel.channel import ExcelAnswer, answer_from_excel
from retrieval.query.pdf import EvidenceRetrievalResult, retrieve_configured
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
            if content_type in (None, "excel_card"):
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
