import unittest
from unittest.mock import MagicMock, patch

from retrieval.query.pdf import EvidenceRetrievalResult
from services.retrieval_service import RetrievalService


class RetrievalServiceTests(unittest.TestCase):
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
    def test_multiple_content_filters_map_to_unique_groups(
        self, answer_from_excel, connect_db, retrieve
    ):
        retrieve.return_value = EvidenceRetrievalResult(question="q", groups={})
        RetrievalService().retrieve(
            "q", content_types=("narrative", "table", "figure", "table")
        )
        self.assertEqual(
            retrieve.call_args.kwargs["groups"], ("narrative", "table", "figure")
        )


if __name__ == "__main__":
    unittest.main()
