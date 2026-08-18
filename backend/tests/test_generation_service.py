import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from generation.schemas import AnswerResponse
from retrieval.query.excel.query import RECORDS, ExcelQueryPlan
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult
from services.generation_service import (
    GenerationService,
    _verified_excel_chunks,
    deduplicate_chunks,
    interleave_grouped_results,
)
from generation.schemas import Chunk, ChunkMetadata
from generation.computation import CalculationResult
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
    return EvidenceGroup(
        name=name,
        content_types=(name,),
        results=results,
        diagnostics=MagicMock(),
    )


class GenerationServiceTests(unittest.TestCase):
    def test_deduplicates_equivalent_evidence_and_keeps_first_metadata(self):
        chunks = [
            Chunk("source-a", "1", "Same  evidence", ChunkMetadata(source_file="a.pdf")),
            Chunk("source-b", "2", " same evidence ", ChunkMetadata(source_file="b.pdf")),
            Chunk("source-c", "3", "Different", ChunkMetadata(source_file="c.pdf")),
        ]
        unique = deduplicate_chunks(chunks)
        self.assertEqual([chunk.chunk_id for chunk in unique], ["1", "3"])

    def test_interleaves_group_ranks(self):
        evidence = EvidenceRetrievalResult(
            question="q",
            groups={
                "narrative": group("narrative", [result(1), result(2)]),
                "table": group("table", [result(3), result(4)]),
            },
        )
        self.assertEqual(
            [
                item.query_object.chunk_id
                for item in interleave_grouped_results(evidence)
            ],
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

    def test_cross_resource_calculation_enters_prompt_with_operand_provenance(self):
        answer_service = MagicMock()
        answer_service.answer.return_value = AnswerResponse(
            request_id="req_calc",
            answer="75%",
            cited_chunk_ids=("calculation-1",),
            citations=(),
            insufficient_context=False,
            model_id="fake",
            latency_ms=1,
        )
        calculation = CalculationResult(
            value=Decimal("75.00"),
            unit="percent",
            expression="(75 / 100) * 100",
            contributing_sources=("qdr.xlsx#R9", "wmp.pdf#p42"),
        )
        evidence = EvidenceRetrievalResult(question="q", groups={})

        GenerationService(answer_service).generate(
            "req_calc", "q", RetrievalBundle(evidence, calculations=(calculation,))
        )

        request = answer_service.answer.call_args.args[0]
        self.assertEqual(request.chunks[0].chunk_id, "calculation-1")
        self.assertIn("result=75.00 percent", request.chunks[0].content)
        self.assertEqual(
            request.chunks[0].metadata.contributing_sources,
            ("qdr.xlsx#R9", "wmp.pdf#p42"),
        )

    def test_verified_excel_history_has_row_citations_and_calculations(self):
        answer = SimpleNamespace(
            card_chunk_id=2460,
            question="Show WMP.478 across 2023-2025",
            table_number=1,
            plan=ExcelQueryPlan(
                table_number=1,
                source=RECORDS,
                operation="select",
                group_by=("reporting_year", "record_id"),
                select_json_keys=(
                    "annual_quant_target",
                    "quant_actual_progress_q1_4",
                    "quant_target_units",
                ),
            ),
            result=SimpleNamespace(
                columns=[
                    "reporting_year",
                    "record_id",
                    "selected_0",
                    "selected_1",
                    "selected_2",
                ],
                rows=[
                    (2023, "r23", "11100", "11755", "Structures"),
                    (2024, "r24", "15450", "16503", "Structures"),
                    (2025, "r25", "13275", "17950", "Structures"),
                ],
                provenance=[
                    {
                        "source_file": "2023.xlsx",
                        "source_sheet": "Table 1",
                        "source_row": "20",
                    },
                    {
                        "source_file": "2024.xlsx",
                        "source_sheet": "Table 1",
                        "source_row": "22",
                    },
                    {
                        "source_file": "2025.xlsx",
                        "source_sheet": "Table 1",
                        "source_row": "22",
                    },
                ],
            ),
        )

        chunks = _verified_excel_chunks(answer)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(
            [chunk.metadata.source_file for chunk in chunks[:3]],
            ["2023.xlsx", "2024.xlsx", "2025.xlsx"],
        )
        self.assertIn("percent_complete=105.9%", chunks[0].content)
        self.assertIn("cumulative_target=39825", chunks[3].content)
        self.assertIn("cumulative_actual=46208", chunks[3].content)
        self.assertIn("cumulative_percent_complete=116.0%", chunks[3].content)
        self.assertEqual(
            chunks[3].metadata.contributing_sources,
            (
                "2023: 2023.xlsx, Table 1, row 20",
                "2024: 2024.xlsx, Table 1, row 22",
                "2025: 2025.xlsx, Table 1, row 22",
            ),
        )

    def test_verified_excel_history_handles_zero_target_without_crashing(self):
        answer = SimpleNamespace(
            card_chunk_id=1,
            question="Show WMP.1 in 2025",
            table_number=1,
            plan=ExcelQueryPlan(
                table_number=1,
                source=RECORDS,
                operation="select",
                select_json_keys=(
                    "annual_quant_target",
                    "quant_actual_progress_q1_4",
                    "quant_target_units",
                ),
            ),
            result=SimpleNamespace(
                columns=[
                    "reporting_year",
                    "record_id",
                    "selected_0",
                    "selected_1",
                    "selected_2",
                ],
                rows=[(2025, "r25", "0", "0", "Structures")],
                provenance=[],
            ),
        )

        chunks = _verified_excel_chunks(answer)

        self.assertIn("percent_complete=not defined", chunks[0].content)
        self.assertIn("cumulative_percent_complete=not defined", chunks[1].content)


if __name__ == "__main__":
    unittest.main()
