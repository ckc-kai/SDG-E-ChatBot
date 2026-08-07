"""
Role: 
-validates that models/schemas.py actually enforces what we expect 
    - required fields, 
    - optional fields, 
    - wrong types rejected. 

-Catches contract drift early instead of 422 error later during integration

Run: uv run python -m unittest backend.tests.test_schemas
"""
import unittest

from pydantic import ValidationError

from models.schemas import (
    AskRequest,
    AskResponse,
    DocumentMetadata,
    DocumentsResponse,
    ErrorResponse,
    Filters,
    Source,
)


class TestFilters(unittest.TestCase):
    def test_all_fields_optional(self):
        f = Filters()
        self.assertIsNone(f.content_type)
        self.assertIsNone(f.section_number)
        self.assertIsNone(f.page)

    def test_accepts_partial_fields(self):
        f = Filters(content_type="table")
        self.assertEqual(f.content_type, "table")
        self.assertIsNone(f.page)


class TestAskRequest(unittest.TestCase):
    def test_requires_question(self):
        with self.assertRaises(ValidationError):
            AskRequest()

    def test_minimal_valid_request(self):
        req = AskRequest(question="What is the budget?")
        self.assertEqual(req.question, "What is the budget?")
        self.assertIsNone(req.filters)
        self.assertIsNone(req.embedding_mode)

    def test_rejects_non_string_question(self):
        with self.assertRaises(ValidationError):
            AskRequest(question=12345)

    def test_accepts_embedding_and_rewrite_mode(self):
        req = AskRequest(question="q", embedding_mode="hybrid", rewrite_mode="off")
        self.assertEqual(req.embedding_mode, "hybrid")
        self.assertEqual(req.rewrite_mode, "off")


class TestSource(unittest.TestCase):
    def _valid_kwargs(self, **overrides):
        base = dict(
            doc_id=1,
            source_pdf="2024_WMP.pdf",
            breadcrumb="Section 3",
            section_number="3.1",
            page_start=1,
            page_end=2,
            content_type="narrative",
            snippet="some text",
            caption=None,
            object_key=None,
            rerank_score=0.5,
        )
        base.update(overrides)
        return base

    def test_requires_core_fields(self):
        with self.assertRaises(ValidationError):
            Source(doc_id=1)  # missing everything else

    def test_valid_source(self):
        s = Source(**self._valid_kwargs())
        self.assertEqual(s.page_start, 1)
        self.assertEqual(s.content_type, "narrative")

    def test_caption_and_object_key_can_be_none(self):
        s = Source(**self._valid_kwargs(caption=None, object_key=None))
        self.assertIsNone(s.caption)
        self.assertIsNone(s.object_key)


class TestAskResponse(unittest.TestCase):
    def test_accepts_empty_sources_list(self):
        resp = AskResponse(answer="No data found.", sources=[])
        self.assertEqual(resp.sources, [])

    def test_requires_answer(self):
        with self.assertRaises(ValidationError):
            AskResponse(sources=[])


class TestDocumentsResponse(unittest.TestCase):
    def test_wraps_list_of_documents(self):
        doc = DocumentMetadata(doc_id="1", title="2024 WMP", page_count=100)
        resp = DocumentsResponse(documents=[doc])
        self.assertEqual(len(resp.documents), 1)
        self.assertEqual(resp.documents[0].title, "2024 WMP")

    def test_accepts_empty_document_list(self):
        resp = DocumentsResponse(documents=[])
        self.assertEqual(resp.documents, [])


class TestErrorResponse(unittest.TestCase):
    def test_detail_is_optional(self):
        err = ErrorResponse(error="retrieval_timeout")
        self.assertIsNone(err.detail)

    def test_requires_error_field(self):
        with self.assertRaises(ValidationError):
            ErrorResponse()


if __name__ == "__main__":
    unittest.main()