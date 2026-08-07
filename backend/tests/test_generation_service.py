import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from generation.schemas import AnswerResponse
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult
from services.generation_service import GenerationService, interleave_grouped_results
from services.retrieval_service import RetrievalBundle


def result(chunk_id: int):
    query_object = SimpleNamespace(
        chunk_id=chunk_id,
        source_pdf="wmp.pdf",
        content=f"evidence {chunk_id}",
        page_start=1,
        page_end=1,
        sub_document=None,
        breadcrumb="Section",
        section_number=None,
        content_type="narrative",
        chunk_index=chunk_id,
        token_count=10,
        distance=0.1,
        structured_data=None,
    )
    return SimpleNamespace(query_object=query_object, rerank_score=1.0)


def group(name: str, results: list):
    return EvidenceGroup(name=name, content_types=(name,), results=results, diagnostics=MagicMock())


class GenerationServiceTests(unittest.TestCase):
    def test_interleaves_group_ranks(self):
        evidence = EvidenceRetrievalResult(
            question="q",
            groups={
                "narrative": group("narrative", [result(1), result(2)]),
                "table": group("table", [result(3), result(4)]),
            },
        )
        self.assertEqual(
            [item.query_object.chunk_id for item in interleave_grouped_results(evidence)],
            [1, 3, 2, 4],
        )

    def test_adapts_full_chunks_and_calls_task3(self):
        answer_service = MagicMock()
        expected = AnswerResponse(
            request_id="req_1",
            answer="answer",
            cited_chunk_ids=("1",),
            citations=(),
            insufficient_context=False,
            model_id="fake",
            latency_ms=1,
        )
        answer_service.answer.return_value = expected
        evidence = EvidenceRetrievalResult(
            question="q",
            groups={"narrative": group("narrative", [result(1)])},
        )

        actual = GenerationService(answer_service).generate(
            "req_1", "q", RetrievalBundle(evidence)
        )

        self.assertIs(actual, expected)
        request = answer_service.answer.call_args.args[0]
        self.assertEqual(request.chunks[0].content, "evidence 1")


if __name__ == "__main__":
    unittest.main()
