import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from retrieval.query.excel.channel import ExcelAnswer, ExcelDecline
from retrieval.query.excel.query import ExcelQueryPlan
from retrieval.query.pdf import EvidenceRetrievalResult
from services.retrieval_service import RetrievalService


class RetrievalServiceTests(unittest.TestCase):
    @staticmethod
    def _excel_answer():
        plan = ExcelQueryPlan(table_number=1)
        return ExcelAnswer(
            question="q",
            card_chunk_id=1,
            card_caption="card",
            card_score=1.0,
            table_number=1,
            semantic_metric_key=None,
            plan=plan,
            result=SimpleNamespace(),
            bound={},
        )

    @patch("services.retrieval_service.retrieve_configured")
    @patch("services.retrieval_service.connect_db")
    @patch("services.retrieval_service.answer_from_excel")
    def test_returns_grouped_result_and_closes_connection(
        self, answer_from_excel, connect_db, retrieve
    ):
        connection = MagicMock()
        connect_db.return_value = connection
        expected = EvidenceRetrievalResult(question="q", groups={})
        retrieve.return_value = expected

        result = RetrievalService().retrieve("q", embedding_mode="hybrid")

        self.assertIs(result.evidence, expected)
        retrieve.assert_called_once_with(
            "q", connection, output_mode="grouped", embedding_mode="hybrid"
        )
        connection.close.assert_called_once()

    @patch("services.retrieval_service.retrieve_configured")
    @patch("services.retrieval_service.connect_db")
    @patch("services.retrieval_service.answer_from_excel")
    def test_content_filter_maps_to_evidence_group(
        self, answer_from_excel, connect_db, retrieve
    ):
        retrieve.return_value = EvidenceRetrievalResult(question="q", groups={})
        RetrievalService().retrieve("q", content_type="excel_card")
        self.assertEqual(retrieve.call_args.kwargs["groups"], ("excel",))

    @patch("services.retrieval_service.retrieve_configured")
    @patch("services.retrieval_service.connect_db")
    @patch("services.retrieval_service.answer_from_excel")
    def test_exact_entity_history_skips_semantic_groups(
        self, answer_from_excel, connect_db, retrieve
    ):
        retrieve.return_value = EvidenceRetrievalResult(question="q", groups={})
        answer_from_excel.return_value = self._excel_answer()

        RetrievalService().retrieve(
            "Across 2023-2025, show targets and Q4 actuals for WMP.478."
        )

        self.assertEqual(retrieve.call_args.kwargs["groups"], ())
        answer_from_excel.assert_called_once()

    @patch("services.retrieval_service.retrieve_configured")
    @patch("services.retrieval_service.connect_db")
    @patch("services.retrieval_service.answer_from_excel")
    def test_exact_entity_decline_preserves_semantic_fallback(
        self, answer_from_excel, connect_db, retrieve
    ):
        retrieve.return_value = EvidenceRetrievalResult(question="q", groups={})
        answer_from_excel.return_value = ExcelDecline("missing year")

        RetrievalService().retrieve(
            "Across 2023-2025, show targets and Q4 actuals for WMP.478."
        )

        self.assertNotIn("groups", retrieve.call_args.kwargs)

    @patch("services.retrieval_service.retrieve_configured")
    @patch("services.retrieval_service.connect_db")
    @patch("services.retrieval_service.answer_from_excel")
    def test_oeis_guideline_review_uses_only_relevant_pdf_groups(
        self, answer_from_excel, connect_db, retrieve
    ):
        retrieve.return_value = EvidenceRetrievalResult(question="q", groups={})

        RetrievalService().retrieve(
            "Review the WMP from OEIS's perspective against the WMP guidelines."
        )

        self.assertEqual(retrieve.call_args.kwargs["groups"], ("narrative", "table"))
        answer_from_excel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
