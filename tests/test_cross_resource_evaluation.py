import unittest
from decimal import Decimal
from types import SimpleNamespace

from eval.cross_resource_evaluation import (
    aggregate_cross_resource_scores,
    evaluate_cross_resource_rows,
    score_cross_resource_case,
)
from generation.computation import CalculationResult
from generation.planning import RetrievalPlan, RetrievalStep


class CrossResourceEvaluationTests(unittest.TestCase):
    def _row(self):
        return {
            "id": "case-1",
            "question": "What percentage is operand A of operand B?",
            "required_sources": ["excel", "pdf"],
            "expected_value": "75.00",
            "facts": [
                {
                    "source": "excel",
                    "provenance": [
                        {"source_file": "qdr.xlsx", "source_row": "9"}
                    ],
                },
                {
                    "source": "pdf",
                    "provenance": [{"chunk_id": 42}],
                },
            ],
        }

    def test_scores_each_stage_without_treating_retrieval_as_calculation(self):
        plan = RetrievalPlan(
            (
                RetrievalStep("excel fact", ("excel_card",), "excel"),
                RetrievalStep("pdf fact", ("narrative",), "pdf"),
            ),
            source="model",
        )
        excel_answer = SimpleNamespace(
            result=SimpleNamespace(
                provenance=[{"source_file": "qdr.xlsx", "source_row": "9"}]
            )
        )
        bundle = SimpleNamespace(
            evidence=SimpleNamespace(
                groups={
                    "excel": SimpleNamespace(
                        results=[SimpleNamespace(query_object=SimpleNamespace(chunk_id=7))]
                    ),
                    "narrative": SimpleNamespace(
                        results=[SimpleNamespace(query_object=SimpleNamespace(chunk_id=42))]
                    ),
                }
            ),
            verified_excel=excel_answer,
            verified_excels=(excel_answer,),
            calculations=(),
        )

        score = score_cross_resource_case(self._row(), plan, bundle)

        self.assertEqual(score["planned_source_coverage"], 1.0)
        self.assertEqual(score["retrieved_source_coverage"], 1.0)
        self.assertEqual(score["pdf_gold_recall"], 1.0)
        self.assertTrue(score["excel_operand_verified"])
        self.assertFalse(score["calculation_produced"])
        self.assertFalse(score["calculation_correct"])

    def test_matches_a_provenance_bearing_calculation_and_aggregates(self):
        plan = RetrievalPlan(
            (RetrievalStep("all facts", ("narrative", "excel_card"), "both"),),
            source="fallback",
        )
        bundle = SimpleNamespace(
            evidence=SimpleNamespace(groups={}),
            verified_excel=None,
            verified_excels=(),
            calculations=(
                CalculationResult(
                    Decimal("75.00"), "percent", "(75 / 100) * 100", ("a", "b")
                ),
            ),
        )

        score = score_cross_resource_case(self._row(), plan, bundle)
        summary = aggregate_cross_resource_scores([score])

        self.assertEqual(score["planned_source_coverage"], 0.0)
        self.assertTrue(score["calculation_correct"])
        self.assertEqual(summary["calculation_accuracy"], 1.0)

    def test_evaluates_rows_through_injected_planner_and_retriever(self):
        plan = RetrievalPlan(
            (RetrievalStep("pdf fact", ("narrative",), "pdf"),),
            source="model",
        )
        bundle = SimpleNamespace(
            evidence=SimpleNamespace(groups={}),
            verified_excel=None,
            verified_excels=(),
            calculations=(),
        )
        planned_questions = []

        scores = evaluate_cross_resource_rows(
            [self._row()],
            lambda question: planned_questions.append(question) or plan,
            lambda question, selected: bundle,
        )

        self.assertEqual(planned_questions, [self._row()["question"]])
        self.assertEqual(scores[0]["id"], "case-1")


if __name__ == "__main__":
    unittest.main()
