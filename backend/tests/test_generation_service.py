"""
Role: tests the CURRENT stubbed behavior of GenerationService.
these tests need rewriting to check real behavior instead 
"""
import unittest

from models.schemas import Source
from services.generation_service import GenerationService


class TestGenerationService(unittest.TestCase):
    def setUp(self):
        self.service = GenerationService()

    def test_generate_returns_string_referencing_question(self):
        answer = self.service.generate("What is the budget?", sources=[])
        self.assertIsInstance(answer, str)
        self.assertIn("What is the budget?", answer)

    def test_generate_accepts_sources_without_error(self):
        fake_source = Source(
            doc_id=1,
            source_pdf="a.pdf",
            breadcrumb="b",
            section_number="1",
            page_start=1,
            page_end=1,
            content_type="narrative",
            snippet="s",
            caption=None,
            object_key=None,
            rerank_score=0.5,
        )
        # Should not raise, even though the current stub ignores sources.
        answer = self.service.generate("question", sources=[fake_source])
        self.assertIsInstance(answer, str)

    def test_generate_accepts_optional_model_param(self):
        answer = self.service.generate("q", sources=[], model="claude-sonnet-4-6")
        self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()