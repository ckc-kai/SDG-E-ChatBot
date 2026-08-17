"""Regression tests for deterministic retrieval scope extraction."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from retrieval.query.excel.channel import (
    ExcelAnswer,
    _keep_requested_years,
    _plan_for_card,
    answer_from_excel,
    is_entity_history_question,
)
from retrieval.query.excel.query import (
    RECORDS,
    ExcelQueryPlan,
    bind_entity_key,
    bind_years,
)
from retrieval.query.pdf.query import (
    RetrievalDiagnostics,
    _interleave_ranked_groups,
    required_source_roles,
)
from retrieval.query.pdf import query as pdf_query
from retrieval.utils import _validate_config, load_config


class QueryScopeTests(unittest.TestCase):
    def test_bind_years_expands_a_wmp_cycle_range(self):
        question = "Across the 2023-2025 WMP cycle, show annual and cumulative targets."

        self.assertEqual(bind_years(question), (2023, 2024, 2025))

    def test_bind_entity_key_extracts_exact_wmp_identifier(self):
        question = "Show Distribution Overhead Detailed Inspections (WMP.478)."

        self.assertEqual(bind_entity_key(question), "WMP.478")

    def test_entity_history_route_requires_an_explicit_year(self):
        self.assertFalse(
            is_entity_history_question("What is the annual target for WMP.478?")
        )

    def test_discrete_years_do_not_include_intervening_rows(self):
        result = SimpleNamespace(
            columns=["reporting_year", "value"],
            rows=[(2023, 1), (2024, 2), (2025, 3)],
            provenance=[{"year": 2023}, {"year": 2024}, {"year": 2025}],
        )

        _keep_requested_years({"reporting_years": (2023, 2025)}, result)

        self.assertEqual(result.rows, [(2023, 1), (2025, 3)])
        self.assertEqual(result.provenance, [{"year": 2023}, {"year": 2025}])

    def test_required_source_roles_separate_oeis_decision_and_guidelines(self):
        question = (
            "Review the WMP from OEIS's perspective and adjust the plan to satisfy "
            "OEIS and the WMP guidelines. Use lessons from the 2023-2025 cycle."
        )

        roles = required_source_roles(question)

        self.assertEqual(
            [role.name for role in roles], ["oeis_decision", "wmp_guidelines"]
        )
        self.assertIn("OEIS decision", roles[0].query)
        self.assertIn("WMP guidelines", roles[1].query)
        self.assertEqual(
            roles[0].filename_patterns,
            ("FINAL_SDGE_20232025_WMP_Decision_and_Cover_Letter.pdf",),
        )
        self.assertEqual(
            roles[1].filename_patterns,
            ("FINAL 2026-2028_Wildfire_Mitigation_Plan_Guidelines.pdf",),
        )

    def test_required_source_roles_leave_unrelated_questions_unrouted(self):
        question = "How many inspections were completed in 2024?"

        self.assertEqual(required_source_roles(question), ())

    def test_required_source_roles_cover_both_wmp_cycles_and_guidelines(self):
        question = (
            "Comparing the 2023-2025 WMP, 2026-2028 WMP, and corresponding "
            "guidelines, what are opportunities to improve risk methodology?"
        )

        roles = required_source_roles(question)

        self.assertEqual(
            [role.name for role in roles],
            [
                "2023_wmp",
                "2023_guidelines",
                "2026_wmp",
                "2026_guidelines",
            ],
        )

    def test_unspecified_wmp_guideline_comparison_covers_both_cycles(self):
        roles = required_source_roles(
            "Compare the WMP to the WMP guidelines and flag missing information."
        )

        self.assertEqual(len(roles), 4)
        self.assertEqual(
            {pattern for role in roles for pattern in role.filename_patterns},
            {
                "SDG&E_2023-2023_Base-WMP_R5-redacted.pdf",
                "2023-2025_WMP_TECHNICAL_GUIDELINES.pdf",
                "SDG&E_2026-2028_Base-WMP_R2.pdf",
                "FINAL 2026-2028_Wildfire_Mitigation_Plan_Guidelines.pdf",
            },
        )

    def test_focused_lexical_query_keeps_years_and_distinctive_terms(self):
        expression = pdf_query._focused_lexical_query(
            "How did grid hardening prioritization evolve from 2023 to 2025?"
        )

        self.assertNotIn(" OR ", expression)
        self.assertIn("2023", expression)
        self.assertIn("2025", expression)
        self.assertIn("prioritization", expression)

    def test_new_retrieval_settings_remain_backward_compatible(self):
        config = deepcopy(load_config())
        config["retrieval"].pop("rerank_batch_size", None)
        config["retrieval"]["lexical_query_mode"] = False

        _validate_config(config)

    @patch("retrieval.query.pdf.query.get_reranker_model")
    def test_reranker_uses_configured_small_batch(self, get_model):
        candidate = Mock(
            breadcrumb="Section",
            section_number=None,
            content_type="narrative",
            caption=None,
            content="Evidence",
            retrieval_hint=None,
            content_hash="hash",
        )
        get_model.return_value.predict.return_value = [0.5]

        with patch.object(pdf_query, "RERANK_BATCH_SIZE", 4):
            pdf_query.rerank("question", [candidate], top_k=1)

        self.assertEqual(
            get_model.return_value.predict.call_args.kwargs["batch_size"], 4
        )

    @patch("retrieval.query.excel.channel.dimension_vocabulary", return_value={})
    def test_entity_history_plan_selects_every_year_and_required_metric(self, _):
        question = (
            "Across the 2023-2025 WMP cycle, what were the annual targets, "
            "year-end Q4 actuals, and percent complete for WMP.478?"
        )
        card = {
            "table_number": 1,
            "entity_key": "WMP.478",
            "card_type": "activity",
        }
        contracts = Mock()

        plan, bound = _plan_for_card(question, card, Mock(), contracts)

        self.assertEqual(plan.operation, "select")
        self.assertEqual(plan.group_by, ("reporting_year", "record_id"))
        self.assertEqual(
            plan.select_json_keys,
            (
                "annual_quant_target",
                "quant_actual_progress_q1_4",
                "quant_target_units",
            ),
        )
        self.assertEqual(bound["entity_key"], "WMP.478")
        self.assertEqual(bound["reporting_years"], (2023, 2024, 2025))
        self.assertEqual(
            [(flt.field, flt.operator, flt.value) for flt in plan.filters],
            [
                ("entity_key", "eq", "WMP.478"),
                ("reporting_year", "gte", 2023),
                ("reporting_year", "lte", 2025),
            ],
        )

    @patch("retrieval.query.pdf.query._validate_embedding_mode")
    @patch("retrieval.query.pdf.query._retrieve_source_role_group")
    def test_explicit_source_roles_bypass_unscoped_group_retrieval(
        self, scoped, _validate
    ):
        diagnostics = RetrievalDiagnostics([], [], [], [], [])
        scoped.return_value = ([], diagnostics)

        with patch("retrieval.query.pdf.query.retrieve_with_diagnostics") as unscoped:
            result = pdf_query.retrieve_evidence(
                "Review the WMP from OEIS's perspective against WMP guidelines.",
                MagicMock(),
            )

        self.assertEqual(scoped.call_count, 2)
        self.assertFalse(unscoped.called)
        self.assertEqual(result.groups["figure"].results, [])
        self.assertEqual(result.groups["excel"].results, [])

    def test_source_role_results_alternate_before_prompt_budgeting(self):
        decision = [Mock(rerank_score=3), Mock(rerank_score=2)]
        guidelines = [Mock(rerank_score=4), Mock(rerank_score=1)]

        self.assertEqual(
            _interleave_ranked_groups([decision, guidelines]),
            [decision[0], guidelines[0], decision[1], guidelines[1]],
        )

    @patch("retrieval.query.excel.channel.execute_plan")
    @patch("retrieval.query.excel.channel._plan_for_card")
    @patch("retrieval.query.excel.channel._exact_entity_card")
    @patch("retrieval.query.excel.channel.retrieve")
    def test_exact_entity_history_bypasses_semantic_card_retrieval(
        self, retrieve, exact_card, make_plan, execute
    ):
        question = "Across 2023-2025, show annual targets and Q4 actuals for WMP.478."
        exact_card.return_value = (
            2460,
            "WMP activity — Distribution Overhead Detailed Inspections",
            {"table_number": 1, "entity_key": "WMP.478"},
        )
        plan = ExcelQueryPlan(table_number=1, source=RECORDS, operation="select")
        make_plan.return_value = (
            plan,
            {"entity_key": "WMP.478", "reporting_years": (2023, 2024, 2025)},
        )
        execute.return_value = SimpleNamespace(
            is_answer=True,
            rows=[
                (2023, "record-2023", "11100", "11755", "Structures"),
                (2024, "record-2024", "15450", "16503", "Structures"),
                (2025, "record-2025", "13275", "17950", "Structures"),
            ],
            columns=[
                "reporting_year",
                "record_id",
                "selected_0",
                "selected_1",
                "selected_2",
            ],
            unit=None,
            contributing_facts=1,
            provenance=[],
        )

        outcome = answer_from_excel(question, Mock(), contracts=Mock())

        self.assertIsInstance(outcome, ExcelAnswer)
        self.assertFalse(retrieve.called)


if __name__ == "__main__":
    unittest.main()
