"""
Role: tests POST /api/ask at the HTTP layer, using FastAPI's TestClient.

Requires httpx installed (FastAPI's TestClient depends on it):
    uv add --dev httpx

Run all backend tests via discovery from the backend/ folder:
    uv run python -m unittest discover -s backend/tests -t backend
"""
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from main import app
from models.schemas import Source
from routers.ask import get_generation_service, get_retrieval_service


class TestAskRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        self.fake_retrieval_service = MagicMock()
        self.fake_generation_service = MagicMock()

        app.dependency_overrides[get_retrieval_service] = (
            lambda: self.fake_retrieval_service
        )
        app.dependency_overrides[get_generation_service] = (
            lambda: self.fake_generation_service
        )

    def tearDown(self):
        app.dependency_overrides.clear()

    def _fake_source(self, **overrides):
        base = dict(
            doc_id=1,
            source_pdf="2024_WMP.pdf",
            breadcrumb="Section 3",
            section_number="3.1",
            page_start=5,
            page_end=5,
            content_type="narrative",
            snippet="Some retrieved text.",
            caption=None,
            object_key=None,
            rerank_score=0.9,
        )
        base.update(overrides)
        return Source(**base)

    def test_ask_returns_answer_and_sources(self):
        self.fake_retrieval_service.retrieve.return_value = [self._fake_source()]
        self.fake_generation_service.generate.return_value = "A generated answer."

        response = self.client.post(
            "/api/ask", json={"question": "What is the budget?"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "A generated answer.")
        self.assertEqual(len(body["sources"]), 1)
        self.assertEqual(body["sources"][0]["source_pdf"], "2024_WMP.pdf")

    def test_ask_rejects_missing_question(self):
        response = self.client.post("/api/ask", json={})
        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_ask_handles_empty_retrieval_results(self):
        self.fake_retrieval_service.retrieve.return_value = []
        self.fake_generation_service.generate.return_value = "No information found."

        response = self.client.post(
            "/api/ask", json={"question": "An obscure question"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [])

    def test_ask_passes_embedding_mode_to_retrieval_service(self):
        self.fake_retrieval_service.retrieve.return_value = []
        self.fake_generation_service.generate.return_value = "answer"

        self.client.post(
            "/api/ask",
            json={"question": "q", "embedding_mode": "raw"},
        )

        _, kwargs = self.fake_retrieval_service.retrieve.call_args
        self.assertEqual(kwargs.get("embedding_mode"), "raw")


if __name__ == "__main__":
    unittest.main()