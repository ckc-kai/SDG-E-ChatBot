from __future__ import annotations

import json
import math
import unittest
from dataclasses import dataclass

from generation.adapters import adapt_ranked_result
from generation.citation_validation import validate_and_hydrate_citations
from generation.evaluation import (
    evaluate_benchmark,
    request_from_benchmark_row,
    score_response,
)
from generation.prompting import (
    NO_EVIDENCE_ANSWER,
    SYSTEM_INSTRUCTIONS,
    _base_prompt_tokens,
    _chunk_prompt_tokens,
    build_prompt,
    select_prompt_chunks,
)
from generation.providers.mock import RecordingScriptedMockProvider
from generation.schemas import (
    AnswerRequest,
    Chunk,
    ChunkMetadata,
    ErrorResponse,
    ModelAnswer,
)
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
        self.assertIn('"allowed_citation_ids":["584"]', prompt)
        self.assertIn("SAWTI uses wind", prompt)
        self.assertTrue(prompt.startswith(SYSTEM_INSTRUCTIONS))

    def test_prompt_excludes_citation_display_and_ranking_metadata(self) -> None:
        prompt = build_prompt(sample_request())
        # Context labels the source document so cross-document questions can
        # attribute every excerpt to its filing.
        self.assertIn('"context":"WMP.pdf | 8 Wildfire Mitigations', prompt)
        self.assertNotIn('"source":"WMP.pdf"', prompt)
        self.assertNotIn("page_start", prompt)
        self.assertNotIn("page_end", prompt)
        self.assertNotIn("rerank_score", prompt)
        self.assertNotIn("distance", prompt)

    def test_prompt_has_no_fixed_top_k_cap_when_all_chunks_fit(self) -> None:
        base = sample_request().chunks[0]
        chunks = tuple(
            Chunk(
                source_id=base.source_id,
                chunk_id=str(index),
                content=f"evidence {index}",
                metadata=base.metadata,
            )
            for index in range(1, 8)
        )
        prompt = build_prompt(
            AnswerRequest(request_id="req_ranked", question="Question?", chunks=chunks)
        )
        self.assertIn('"allowed_citation_ids":["1","2","3","4","5","6","7"]', prompt)
        self.assertIn('"id":"7"', prompt)

    def test_prompt_stops_before_exceeding_evidence_token_budget(self) -> None:
        base = sample_request().chunks[0]
        chunks = tuple(
            Chunk(
                source_id=base.source_id,
                chunk_id=str(index),
                content=f"evidence {index}",
                metadata=ChunkMetadata(token_count=600),
            )
            for index in range(1, 4)
        )
        request = AnswerRequest(request_id="req_budget", question="Question?", chunks=chunks)
        budget = (
            _base_prompt_tokens(request)
            + _chunk_prompt_tokens(chunks[0])
            + _chunk_prompt_tokens(chunks[1])
            - 1
        )
        selected = select_prompt_chunks(
            request,
            prompt_token_budget=budget,
            token_safety_factor=1,
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["1"])

    def test_prompt_does_not_replace_higher_ranked_chunk_with_smaller_one(self) -> None:
        chunks = tuple(
            Chunk(
                source_id="a.pdf",
                chunk_id=str(index),
                content=f"evidence {index}",
                metadata=ChunkMetadata(token_count=token_count),
            )
            for index, token_count in ((1, 600), (2, 600), (3, 200))
        )
        request = AnswerRequest(request_id="req_budget", question="Question?", chunks=chunks)
        budget = (
            _base_prompt_tokens(request)
            + _chunk_prompt_tokens(chunks[0])
            + _chunk_prompt_tokens(chunks[2])
        )
        # Budgets are relative to the instruction size so instruction edits do
        # not silently change what this test exercises.
        selected = select_prompt_chunks(
            request,
            prompt_token_budget=budget,
            token_safety_factor=1,
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["1"])

    def test_token_safety_factor_reserves_space_for_tokenizer_mismatch(self) -> None:
        chunks = tuple(
            Chunk(
                source_id="a.pdf",
                chunk_id=str(index),
                content=f"evidence {index}",
                metadata=ChunkMetadata(token_count=350),
            )
            for index in range(1, 3)
        )
        request = AnswerRequest(request_id="req_safety", question="Question?", chunks=chunks)
        budget = _base_prompt_tokens(request) + sum(
            _chunk_prompt_tokens(chunk) for chunk in chunks
        )
        budget = math.ceil(_base_prompt_tokens(request) * 1.25) + 600
        without_margin = select_prompt_chunks(
            request,
            prompt_token_budget=budget,
            token_safety_factor=1,
        )
        with_margin = select_prompt_chunks(
            request,
            prompt_token_budget=budget,
            token_safety_factor=1.25,
        )
        self.assertEqual([chunk.chunk_id for chunk in without_margin], ["1", "2"])
        self.assertEqual([chunk.chunk_id for chunk in with_margin], ["1"])

    def test_oversized_top_chunk_is_truncated_for_prompt_only(self) -> None:
        original = Chunk(
            source_id="a.pdf",
            chunk_id="1",
            content="x" * 1000,
            metadata=ChunkMetadata(token_count=250),
        )
        request = AnswerRequest(request_id="req_large", question="Question?", chunks=(original,))
        base = math.ceil(_base_prompt_tokens(request) * 1.25)
        selected = select_prompt_chunks(request, prompt_token_budget=base + 150)
        self.assertLess(len(selected[0].content), 1000)
        self.assertGreater(len(selected[0].content), 0)
        self.assertEqual(len(request.chunks[0].content), 1000)


class ServiceTests(unittest.TestCase):
    def test_service_rejects_citation_to_chunk_omitted_from_prompt_budget(self) -> None:
        base = sample_request().chunks[0]
        chunks = tuple(
            Chunk(
                source_id=base.source_id,
                chunk_id=str(index),
                content=f"evidence {index}",
                metadata=ChunkMetadata(token_count=600),
            )
            for index in range(1, 7)
        )
        request = AnswerRequest(
            request_id="req_limit", question="Question?", chunks=chunks
        )
        provider = RecordingScriptedMockProvider(
            {
                "answer": "Unsupported by the selected prompt evidence.",
                "cited_chunk_ids": ["6"],
                "insufficient_context": False,
            }
        )
        with self.assertLogs("generation.service", level="ERROR"):
            response = AnswerService(provider, prompt_token_budget=3400).answer(request)
        self.assertIsInstance(response, ErrorResponse)

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
        self.assertIsNotNone(response.timings)
        self.assertGreaterEqual(response.timings.model_call_ms, 0)
        self.assertIsNotNone(provider.last_prompt)
        self.assertEqual(
            set(response.to_public_dict()),
            {
                "request_id",
                "answer",
                "cited_chunk_ids",
                "citations",
                "insufficient_context",
            },
        )

    def test_derived_citation_exposes_all_contributing_sources(self) -> None:
        request = AnswerRequest(
            request_id="req-derived",
            question="What is the cumulative result?",
            chunks=(
                Chunk(
                    source_id="derived",
                    chunk_id="derived-1",
                    content="Cumulative result",
                    metadata=ChunkMetadata(
                        source_file="Derived calculation",
                        contributing_sources=(
                            "2023.xlsx, Table 1, row 20",
                            "2024.xlsx, Table 1, row 22",
                        ),
                    ),
                ),
            ),
        )
        model_answer = ModelAnswer(
            answer="Result [derived-1]",
            cited_chunk_ids=("derived-1",),
            insufficient_context=False,
        )

        _, citations, _ = validate_and_hydrate_citations(request, model_answer)

        self.assertEqual(
            citations[0].contributing_sources,
            (
                "2023.xlsx, Table 1, row 20",
                "2024.xlsx, Table 1, row 22",
            ),
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

    def test_insufficient_answer_preserves_cited_partial_findings(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "A long partial explanation that must not be shown.",
                "cited_chunk_ids": ["584"],
                "insufficient_context": True,
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertFalse(isinstance(response, ErrorResponse))
        self.assertTrue(response.insufficient_context)
        self.assertEqual(response.answer, "A long partial explanation that must not be shown.")
        self.assertEqual(response.cited_chunk_ids, ("584",))
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(provider.call_count, 1)

    def test_uncited_insufficient_answer_preserves_missing_evidence_explanation(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "The evidence does not contain the requested comparison data.",
                "cited_chunk_ids": [],
                "insufficient_context": True,
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertEqual(
            response.answer,
            "The evidence does not contain the requested comparison data.",
        )
        self.assertEqual(response.cited_chunk_ids, ())

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
        self.assertEqual(response.answer, NO_EVIDENCE_ANSWER)
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
        self.assertNotIn(
            "private provider timeout detail", json.dumps(response.to_public_dict())
        )
        self.assertIn("private provider timeout detail", "\n".join(captured.output))

    def test_markdown_fenced_json_is_accepted(self) -> None:
        parsed = parse_model_answer(
            '```json\n{"answer":"ok","cited_chunk_ids":[584],"insufficient_context":false}\n```'
        )
        self.assertEqual(parsed.cited_chunk_ids, ("584",))

    def test_missing_requirement_forces_insufficient_context(self) -> None:
        parsed = parse_model_answer(
            json.dumps(
                {
                    "answer": "The evidence supports 2026, but not 2023.",
                    "cited_chunk_ids": ["584"],
                    "insufficient_context": False,
                    "answered_requirements": ["2026 framework"],
                    "missing_requirements": ["2023 framework"],
                }
            )
        )
        self.assertTrue(parsed.insufficient_context)
        self.assertEqual(parsed.missing_requirements, ("2023 framework",))

    def test_requirement_coverage_fields_are_internal_only(self) -> None:
        provider = RecordingScriptedMockProvider(
            {
                "answer": "It uses wind and vegetation dryness.",
                "cited_chunk_ids": ["584"],
                "insufficient_context": False,
                "answered_requirements": ["What SAWTI uses"],
                "missing_requirements": [],
            }
        )
        response = AnswerService(provider).answer(sample_request())
        self.assertNotIn("answered_requirements", response.to_public_dict())
        self.assertNotIn("missing_requirements", response.to_public_dict())
        self.assertEqual(response.answered_requirements, ("What SAWTI uses",))
        self.assertEqual(response.missing_requirements, ())

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
