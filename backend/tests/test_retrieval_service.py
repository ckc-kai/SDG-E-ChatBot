"""
Role: unit tests for RetrievalService.

IMPORTANT: real retrieve() function and the real database
connection are both mocked here
"""
import unittest
from unittest.mock import MagicMock, patch


class TestRetrievalService(unittest.TestCase):
    def setUp(self):
        # Patch connect_db BEFORE constructing RetrievalService
        # addCleanup ensures the patch is removed even if a test fails partway through.
        patcher = patch("services.retrieval_service.connect_db")
        self.mock_connect_db = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_connect_db.return_value = MagicMock()

        from services.retrieval_service import RetrievalService

        self.service = RetrievalService()

    @patch("services.retrieval_service.retrieve")
    def test_retrieve_converts_ranked_results_to_sources(self, mock_retrieve):
        fake_query_object = MagicMock(
            chunk_id=1,
            source_pdf="2024_WMP.pdf",
            breadcrumb="Section 3 > Risk Assessment",
            section_number="3.2",
            page_start=10,
            page_end=11,
            content_type="narrative",
            content="x" * 1000,  # deliberately long, to test truncation
            caption=None,
            object_key=None,
        )
        fake_ranked_result = MagicMock(
            query_object=fake_query_object, rerank_score=0.87
        )
        mock_retrieve.return_value = [fake_ranked_result]

        result = self.service.retrieve("What is the wildfire budget?")

        self.assertEqual(len(result), 1)
        source = result[0]
        self.assertEqual(source.source_pdf, "2024_WMP.pdf")
        self.assertEqual(source.page_start, 10)
        self.assertEqual(source.page_end, 11)
        self.assertEqual(source.content_type, "narrative")
        # snippet should be truncated to SNIPPET_MAX_CHARS, not the full 1000 chars
        self.assertLessEqual(len(source.snippet), 500)

    @patch("services.retrieval_service.retrieve")
    def test_retrieve_handles_empty_results(self, mock_retrieve):
        mock_retrieve.return_value = []
        result = self.service.retrieve("A question with no matches")
        self.assertEqual(result, [])

    @patch("services.retrieval_service.retrieve")
    def test_retrieve_passes_embedding_mode_through_when_specified(self, mock_retrieve):
        mock_retrieve.return_value = []
        self.service.retrieve("question", embedding_mode="raw")
        _, kwargs = mock_retrieve.call_args
        self.assertEqual(kwargs.get("embedding_mode"), "raw")

    @patch("services.retrieval_service.retrieve")
    def test_retrieve_omits_embedding_mode_when_not_specified(self, mock_retrieve):
        # Important: should NOT pass embedding_mode=None explicitly,
        # since that would override retrieve()'s own default from
        # config.yaml. Confirms the kwarg is omitted entirely, not passed as None.
        mock_retrieve.return_value = []
        self.service.retrieve("question")
        _, kwargs = mock_retrieve.call_args
        self.assertNotIn("embedding_mode", kwargs)

    @patch("services.retrieval_service.retrieve")
    def test_retrieve_passes_rewrite_mode_through(self, mock_retrieve):
        mock_retrieve.return_value = []
        self.service.retrieve("question", rewrite_mode="off")
        _, kwargs = mock_retrieve.call_args
        self.assertEqual(kwargs.get("rewrite_mode"), "off")


if __name__ == "__main__":
    unittest.main()