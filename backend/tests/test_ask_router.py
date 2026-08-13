import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from generation.schemas import AnswerResponse, Citation, ErrorResponse
from main import app
from routers.ask import get_generation_service, get_retrieval_service


class AskRouterTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.retrieval = MagicMock()
        self.generation = MagicMock()
        app.dependency_overrides[get_retrieval_service] = lambda: self.retrieval
        app.dependency_overrides[get_generation_service] = lambda: self.generation

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_returns_task3_public_contract(self):
        bundle = MagicMock()
        self.retrieval.retrieve.return_value = bundle
        self.generation.generate.return_value = AnswerResponse(
            request_id="req_client",
            answer="The target is 25%.",
            cited_chunk_ids=("7",),
            citations=(Citation(chunk_id="7", source_pdf="wmp.pdf", page_start=1),),
            insufficient_context=False,
            model_id="fake",
            latency_ms=1,
        )

        response = self.client.post(
            "/api/ask",
            json={"request_id": "req_client", "question": "What is the target?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cited_chunk_ids"], ["7"])
        self.generation.generate.assert_called_once_with(
            "req_client", "What is the target?", bundle
        )

    def test_retrieval_failure_is_stable_public_error(self):
        self.retrieval.retrieve.side_effect = RuntimeError("database password")
        response = self.client.post("/api/ask", json={"question": "q"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "retrieval_failed")
        self.assertNotIn("password", response.text)

    def test_generation_failure_is_stable_public_error(self):
        self.retrieval.retrieve.return_value = MagicMock()
        self.generation.generate.return_value = ErrorResponse("req_client")
        response = self.client.post(
            "/api/ask", json={"request_id": "req_client", "question": "q"}
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"request_id": "req_client", "error": "answer_generation_failed"},
        )

    def test_rejects_missing_question(self):
        self.assertEqual(self.client.post("/api/ask", json={}).status_code, 422)

    def test_accepts_multiple_content_types(self):
        self.retrieval.retrieve.return_value = MagicMock()
        self.generation.generate.return_value = AnswerResponse(
            request_id="req_types", answer="ok", cited_chunk_ids=(), citations=(),
            insufficient_context=False, model_id="fake", latency_ms=1,
        )
        response = self.client.post("/api/ask", json={
            "request_id": "req_types", "question": "q",
            "filters": {"content_types": ["narrative", "table"]},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.retrieval.retrieve.call_args.kwargs["content_types"],
            ("narrative", "table"),
        )

    def test_rejects_single_and_multiple_content_type_together(self):
        response = self.client.post("/api/ask", json={
            "question": "q", "filters": {
                "content_type": "table", "content_types": ["narrative"]
            },
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_filters")


if __name__ == "__main__":
    unittest.main()
