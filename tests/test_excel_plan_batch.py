"""Regression coverage for multi-output Excel model planning."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval.query.excel.channel import (
    ExcelAnswer,
    _model_plan_answers,
    answer_from_excel,
)
from retrieval.query.excel.nl_planner import PLAN_RESPONSE_SCHEMA, build_model_plans
from retrieval.query.excel.query import (
    FACTS,
    ExcelExecutionResult,
    ExcelQueryPlan,
    Filter,
)


def _result(plan: ExcelQueryPlan, value: int) -> ExcelExecutionResult:
    return ExcelExecutionResult(
        plan=plan,
        revision_id=1,
        columns=["value"],
        rows=[(value,)],
        unit="$ Thousands",
        contributing_facts=1,
    )


class ExcelPlanBatchTests(unittest.TestCase):
    @patch("retrieval.query.excel.nl_planner.execute_plan")
    @patch("retrieval.query.excel.nl_planner._merge_question_dimensions")
    @patch("retrieval.query.excel.nl_planner._sanitize_plan")
    @patch("retrieval.query.excel.nl_planner._table_context")
    @patch("retrieval.query.excel.nl_planner._candidate_tables")
    @patch("retrieval.query.excel.nl_planner._get_planner_provider")
    def test_planner_executes_every_atomic_plan(
        self,
        get_provider,
        candidate_tables,
        table_context,
        sanitize,
        merge_dimensions,
        execute,
    ):
        provider = Mock()
        provider.generate_structured.return_value = json.dumps(
            {
                "action": "plan",
                "plans": [
                    {"operation": "aggregate", "filters": []},
                    {"operation": "aggregate", "filters": []},
                ],
            }
        )
        get_provider.return_value = provider
        candidate_tables.return_value = [(11, "Table 11", None)]
        table_context.return_value = {
            "source": FACTS,
            "title": "Table 11",
            "semantic_keys": [],
            "json_keys": [],
            "vocabulary": {},
            "full_vocabulary": {},
        }
        capex = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("expense_type", value="CAPEX", json_key=True),),
        )
        opex = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("expense_type", value="OPEX", json_key=True),),
        )
        sanitize.side_effect = (capex, opex)
        merge_dimensions.side_effect = lambda plan, _question, _conn: plan
        execute.side_effect = (_result(capex, 474), _result(opex, 194))

        outcomes = build_model_plans(
            "CAPEX and OPEX in 2024", [Mock()], Mock(), Mock()
        )

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(execute.call_count, 2)
        schema = provider.generate_structured.call_args.args[1]
        self.assertIs(schema, PLAN_RESPONSE_SCHEMA)
        self.assertIn("plans", schema["properties"])

    @patch("retrieval.query.excel.channel.build_model_plans")
    def test_channel_preserves_two_answers_from_the_same_card(self, build_plans):
        capex = ExcelQueryPlan(
            table_number=11,
            filters=(Filter("expense_type", value="CAPEX", json_key=True),),
        )
        opex = ExcelQueryPlan(
            table_number=11,
            filters=(Filter("expense_type", value="OPEX", json_key=True),),
        )
        build_plans.return_value = (
            (capex, _result(capex, 474)),
            (opex, _result(opex, 194)),
        )
        card = SimpleNamespace(
            query_object=SimpleNamespace(
                chunk_id=11,
                caption="Table 11",
                structured_data={"table_number": 11},
            ),
            rerank_score=0.9,
        )

        answers = _model_plan_answers("CAPEX and OPEX", (card,), Mock(), Mock())

        self.assertEqual(len(answers), 2)
        self.assertEqual([answer.result.rows[0][0] for answer in answers], [474, 194])
        self.assertEqual([answer.bound["batch_size"] for answer in answers], [2, 2])

    @patch("retrieval.query.excel.channel._model_plan_answers")
    @patch("retrieval.query.excel.channel.feature_enabled", return_value=True)
    @patch("retrieval.query.excel.channel.retrieve", return_value=(Mock(),))
    @patch("retrieval.query.excel.channel._exact_history_answer")
    def test_mixed_history_and_spend_does_not_short_circuit(
        self, exact_history, _retrieve, _feature, model_answers
    ):
        history_plan = ExcelQueryPlan(table_number=1)
        history = ExcelAnswer(
            question="history",
            card_chunk_id=1,
            card_caption="Table 1",
            card_score=1.0,
            table_number=1,
            semantic_metric_key=None,
            plan=history_plan,
            result=_result(history_plan, 84),
            bound={},
        )
        capex_plan = ExcelQueryPlan(table_number=11)
        opex_plan = ExcelQueryPlan(table_number=11)
        capex = SimpleNamespace(plan=capex_plan)
        opex = SimpleNamespace(plan=opex_plan)
        exact_history.return_value = history
        model_answers.return_value = (capex, opex)

        answers = answer_from_excel(
            "For WMP.473 in 2023 compare target and actual, then report Territory CAPEX and OPEX.",
            Mock(),
            contracts=Mock(),
            multiple=True,
        )

        self.assertEqual(answers, (history, capex, opex))
        model_answers.assert_called_once()
        residual_question = model_answers.call_args.args[0]
        self.assertIn("Territory CAPEX", residual_question)
        self.assertIn("Territory OPEX", residual_question)
        self.assertNotIn("target", residual_question)

    @patch("retrieval.query.excel.channel._model_plan_answers")
    @patch("retrieval.query.excel.channel.feature_enabled", return_value=True)
    @patch("retrieval.query.excel.channel.retrieve", return_value=(Mock(),))
    @patch("retrieval.query.excel.channel._exact_history_answer")
    def test_mixed_history_and_spend_without_year_does_not_crash(
        self, exact_history, _retrieve, _feature, model_answers
    ):
        history_plan = ExcelQueryPlan(table_number=1)
        history = ExcelAnswer(
            question="history",
            card_chunk_id=1,
            card_caption="Table 1",
            card_score=1.0,
            table_number=1,
            semantic_metric_key=None,
            plan=history_plan,
            result=_result(history_plan, 84),
            bound={},
        )
        exact_history.return_value = history
        model_answers.return_value = ()

        with (
            patch(
                "retrieval.query.excel.channel.bind_entity_key",
                return_value="WMP.473",
            ),
            patch(
                "retrieval.query.excel.channel.is_entity_history_question",
                return_value=True,
            ),
            patch("retrieval.query.excel.channel.bind_years", return_value=()),
        ):
            answer = answer_from_excel(
                "For WMP.473 compare target and actual, then report Territory CAPEX and OPEX.",
                Mock(),
                contracts=Mock(),
                multiple=True,
            )

        self.assertIs(answer, history)
        residual_question = model_answers.call_args.args[0]
        self.assertIn("requested reporting period", residual_question)


if __name__ == "__main__":
    unittest.main()
