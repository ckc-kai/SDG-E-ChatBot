import unittest

from pydantic import ValidationError

from models.schemas import AskRequest, AskResponse, Citation, ErrorResponse


class SchemaTests(unittest.TestCase):
    def test_minimal_request(self):
        request = AskRequest(question="What is the target?")
        self.assertIsNone(request.request_id)

    def test_request_rejects_missing_or_numeric_question(self):
        with self.assertRaises(ValidationError):
            AskRequest()
        with self.assertRaises(ValidationError):
            AskRequest(question=123)

    def test_public_answer_contract(self):
        response = AskResponse(
            request_id="req_1",
            answer="The target is 25%.",
            cited_chunk_ids=["7"],
            citations=[Citation(chunk_id="7", source_pdf="wmp.pdf", page_start=1)],
            insufficient_context=False,
        )
        self.assertEqual(response.citations[0].chunk_id, "7")

    def test_public_error_contract(self):
        response = ErrorResponse(request_id="req_1", error="retrieval_failed")
        self.assertEqual(response.error, "retrieval_failed")


if __name__ == "__main__":
    unittest.main()
