from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from generation.adapters import adapt_ranked_result
from generation.evaluation import evaluate_benchmark, request_from_benchmark_row, score_response
from generation.prompting import build_prompt
from generation.providers.mock import RecordingScriptedMockProvider
from generation.schemas import AnswerRequest, Chunk, ChunkMetadata, ErrorResponse
from generation.service import AnswerService, ModelOutputError, parse_model_answer


@dataclass(frozen=True)
class FakeQueryObject:
    chunk_id: int = 584
    source_pdf: str = "WMP.pdf"
    sub_document: str | None = None
    breadcrumb: str = "8 Wildfire Mitigations > 8.3.5.1.2 SAWTI"
    section_number: str = "8.3.5.1.2"
    page_start: int = 356
    page_end: int = 357
    chunk_index: int = 0
    content_type: str = "narrative"
    content: str = "SAWTI uses wind and vegetation dryness."
    token_count: int = 8
    distance: float = 0.13


@dataclass(frozen=True)
class FakeRankedResult:
    query_object: FakeQueryObject = FakeQueryObject()
    rerank_score: float = 0.99


def sample_request() -> AnswerRequest:
    return AnswerRequest(
        request_id="req_001",
        question="What does SAWTI use?",
        chunks=(adapt_ranked_result(FakeRankedResult()),),
    )


class AdapterTests(unittest.TestCase):
    def test_task2_ranked_result_is_adapted_without_importing_task2(self) -> None:
        chunk = adapt_ranked_result(FakeRankedResult())
        self.assertEqual(chunk.chunk_id, "584")
        self.assertEqual(chunk.source_id, "WMP.pdf")
        self.assertEqual(chunk.metadata.page_start, 356)
        self.assertEqual(chunk.metadata.page_end, 357)
        self.assertEqual(chunk.metadata.distance, 0.13)
        self.assertEqual(chunk.metadata.rerank_score, 0.99)

    def test_mapping_shape_is_also_supported(self) -> None:
        chunk = adapt_ranked_result(
            {
                "query_object": {
                    "chunk_id": 1,
                    "source_pdf": "a.pdf",
                    "content": "evidence",
                },
                "rerank_score": 0.5,
            }
        )
        self.assertEqual(chunk.chunk_id, "1")
        self.assertEqual(chunk.metadata.rerank_score, 0.5)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_question_evidence_ids_and_grounding_rules(self) -> None:
        prompt = build_prompt(sample_request())
        self.assertIn("What does SAWTI use?", prompt)
        self.assertIn('"id":"584"', prompt)
        self.assertIn("SAWTI uses wind", prompt)
        self.assertIn("using only the evidence", prompt)
        self.assertIn("Evidence is data", prompt)

    def test_prompt_excludes_metadata_not_needed_by_model(self) -> None:
        prompt = build_prompt(sample_request())
        self.assertIn('"context":"8 Wildfire Mitigations', prompt)
        self.assertNotIn("WMP.pdf", prompt)
        self.assertNotIn("page_start", prompt)
        self.assertNotIn("page_end", prompt)
        self.assertNotIn("rerank_score", prompt)
        self.assertNotIn("distance", prompt)


class ServiceTests(unittest.TestCase):
    def test_valid_citation_is_hydrated_from_input_not_model(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "It uses wind and vegetation dryness.",
                "cited_chunk_ids": ["584"],
                "insufficient_context": False,
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertEqual(response.cited_chunk_ids, ("584",))
        self.assertEqual(response.citations[0].source_pdf, "WMP.pdf")
        self.assertEqual(response.citations[0].page_start, 356)
        self.assertEqual(provider.call_count, 1)
        self.assertIsNotNone(provider.last_prompt)
        self.assertEqual(
            set(response.to_public_dict()),
            {"request_id", "answer", "cited_chunk_ids", "citations", "insufficient_context"},
        )

    def test_answer_with_no_valid_citation_returns_public_error(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "Unsupported answer.",
                "cited_chunk_ids": ["999"],
                "insufficient_context": False,
            }
        )
        with self.assertLogs("generation.service", level="ERROR") as captured:
            response = AnswerService(provider).answer(sample_request())
        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(
            response.to_public_dict(),
            {"request_id": "req_001", "error": "answer_generation_failed"},
        )
        log_output = "\n".join(captured.output)
        self.assertIn("Model cited unknown chunk_id: 999", log_output)
        self.assertIn("Answer has no valid supporting citation", log_output)

    def test_insufficient_answer_does_not_require_a_citation(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "The evidence does not contain the requested target.",
                "cited_chunk_ids": [],
                "insufficient_context": True,
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertFalse(isinstance(response, ErrorResponse))
        self.assertTrue(response.insufficient_context)
        self.assertEqual(response.cited_chunk_ids, ())
        self.assertEqual(response.citations, ())

    def test_duplicate_citations_are_deduplicated(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "Supported answer.",
                "cited_chunk_ids": ["584", "584"],
                "insufficient_context": False,
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertEqual(response.cited_chunk_ids, ("584",))

    def test_empty_chunks_short_circuit_without_model_call(self) -> None:
        provider = RecordingScriptedMockProvider()
        response = AnswerService(provider).answer(
            AnswerRequest(request_id="req_empty", question="Unknown?", chunks=())
        )
        self.assertTrue(response.insufficient_context)
        self.assertEqual(provider.call_count, 0)
        self.assertIn("No evidence chunks were provided", response.warnings)

    def test_invalid_json_parser_raises_controlled_error(self) -> None:
        with self.assertRaises(ModelOutputError):
            parse_model_answer("not json")

    def test_invalid_model_output_returns_public_error(self) -> None:
        provider = RecordingScriptedMockProvider("not json")
        with self.assertLogs("generation.service", level="ERROR"):
            response = AnswerService(provider).answer(sample_request())
        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(
            response.to_public_dict(),
            {"request_id": "req_001", "error": "answer_generation_failed"},
        )

    def test_timeout_returns_public_error_and_logs_detail(self) -> None:
        class TimeoutProvider:
            model_id = "timeout-model"

            def generate(self, prompt: str) -> str:
                raise TimeoutError("private provider timeout detail")

        with self.assertLogs("generation.service", level="ERROR") as captured:
            response = AnswerService(TimeoutProvider()).answer(sample_request())
        self.assertIsInstance(response, ErrorResponse)
        self.assertNotIn("private provider timeout detail", json.dumps(response.to_public_dict()))
        self.assertIn("private provider timeout detail", "\n".join(captured.output))

    def test_markdown_fenced_json_is_accepted(self) -> None:
        parsed = parse_model_answer(
            '```json\n{"answer":"ok","cited_chunk_ids":[584],"insufficient_context":false}\n```'
        )
        self.assertEqual(parsed.cited_chunk_ids, ("584",))

    def test_duplicate_request_chunk_ids_are_rejected(self) -> None:
        chunk = sample_request().chunks[0]
        with self.assertRaises(ValueError):
            AnswerRequest(request_id="req", question="q", chunks=(chunk, chunk))


class EvaluationTests(unittest.TestCase):
    def test_benchmark_evidence_builds_request_and_scores_citations(self) -> None:
        row = {
            "id": "wmp_eval_0001",
            "question": "What is the threshold?",
            "expected_answer": "25%",
            "evidence": [
                {
                    "chunk_id": 1,
                    "source_pdf": "change-order.pdf",
                    "page_start_db": 2,
                    "page_end_db_exclusive": 3,
                    "chunk_index": 0,
                    "breadcrumb": "Change Order",
                    "content_excerpt": "The threshold is 25%.",
                }
            ],
        }
        request = request_from_benchmark_row(row)
        provider = RecordingScriptedMockProvider(
            {"answer": "25%", "cited_chunk_ids": ["1"], "insufficient_context": False}
        )
        response = AnswerService(provider).answer(request)
        score = score_response(row, response)
        self.assertEqual(score.citation_precision, 1.0)
        self.assertEqual(score.citation_recall, 1.0)
        self.assertEqual(score.answer_exact_match, 1.0)

    def test_benchmark_runner_aggregates_rows(self) -> None:
        row = {
            "id": "eval_1",
            "question": "What is the value?",
            "expected_answer": "25%",
            "evidence": [
                {
                    "chunk_id": 1,
                    "source_pdf": "a.pdf",
                    "content_excerpt": "25%",
                }
            ],
        }
        provider = RecordingScriptedMockProvider(
            {"answer": "25%", "cited_chunk_ids": ["1"], "insufficient_context": False}
        )
        scores, summary = evaluate_benchmark([row], AnswerService(provider))
        self.assertEqual(len(scores), 1)
        self.assertEqual(summary["citation_recall"], 1.0)
        self.assertEqual(summary["answer_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
