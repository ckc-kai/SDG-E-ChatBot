"""A question needing two queries must produce two executed plans."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from retrieval.query.excel import nl_planner
from retrieval.query.excel.manifest import WorkbookManifest
from retrieval.query.excel.nl_planner import (
    MAX_FOLLOW_UP_PLANS,
    WIDE_MAX_FOLLOW_UP_PLANS,
    _max_follow_up_plans,
    PLAN_RESPONSE_SCHEMA,
    build_model_plans,
)


class SchemaTests(unittest.TestCase):
    def test_follow_ups_validate_exactly_like_the_primary_plan(self):
        """Two shapes that can drift apart become two ways to reject a plan."""
        primary = PLAN_RESPONSE_SCHEMA["properties"]
        follow_up = PLAN_RESPONSE_SCHEMA["properties"]["follow_up_plans"]["items"][
            "properties"
        ]
        body_only = set(primary) - {"action", "reason", "follow_up_plans"}

        self.assertEqual(body_only, set(follow_up))
        for name in body_only:
            self.assertEqual(primary[name], follow_up[name])

    def test_follow_up_count_is_bounded(self):
        """The schema admits the widest cap; the runtime slice picks the active one."""
        self.assertEqual(
            PLAN_RESPONSE_SCHEMA["properties"]["follow_up_plans"]["maxItems"],
            WIDE_MAX_FOLLOW_UP_PLANS,
        )

    def test_the_cap_follows_the_fanout_flag(self):
        self.assertEqual(
            _max_follow_up_plans({"SDGE_FEATURE_EXCEL_WIDE_FANOUT": "0"}),
            MAX_FOLLOW_UP_PLANS,
        )
        self.assertEqual(
            _max_follow_up_plans({"SDGE_FEATURE_EXCEL_WIDE_FANOUT": "1"}),
            WIDE_MAX_FOLLOW_UP_PLANS,
        )


class _Provider:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, prompt, schema):
        return json.dumps(self.payload)


class BuildPlansTests(unittest.TestCase):
    def _run(self, payload, executed):
        calls = []

        def fake_execute_one(
            body, question, candidates, contexts, conn, contracts, **kwargs
        ):
            calls.append(body)
            return executed.pop(0) if executed else None

        with patch.object(nl_planner, "_get_planner_provider",
                          return_value=_Provider(payload)), \
             patch.object(nl_planner, "_candidate_tables",
                          return_value=[(15, "cap", None)]), \
             patch.object(nl_planner, "_table_context", return_value={}), \
             patch.object(nl_planner, "_render_prompt", return_value="p"), \
             patch.object(nl_planner, "_entity_card_options", return_value={}), \
             patch.object(nl_planner, "load_manifest", return_value=WorkbookManifest()), \
             patch.object(nl_planner, "_execute_one", fake_execute_one):
            plans = build_model_plans("q", ["card"], conn=None, contracts=None)
        return plans, calls

    def test_a_follow_up_is_executed_as_well_as_the_primary_plan(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        plans, calls = self._run(payload, [("planA", "resA"), ("planB", "resB")])

        self.assertEqual(len(plans), 2)
        self.assertEqual(len(calls), 2)

    def test_a_follow_up_inherits_the_plan_action(self):
        """Without this every follow-up sanitises as 'model declined'."""
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        _, calls = self._run(payload, [("planA", "resA"), ("planB", "resB")])

        self.assertEqual(calls[1]["action"], "plan")

    def test_a_failed_follow_up_does_not_lose_the_primary_answer(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        plans, _ = self._run(payload, [("planA", "resA"), None])

        self.assertEqual(plans, [("planA", "resA")])

    def test_a_failed_primary_plan_does_not_discard_valid_follow_ups(self):
        """Each plan body stands or falls alone.

        Abandoning the whole question when the primary plan failed threw away
        follow-ups that had already validated and executed, so one bad first
        guess -- a wrong table, a filter the executor refused -- cost every
        figure the model had correctly asked for.
        """
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        plans, calls = self._run(payload, [None, ("planB", "resB")])

        self.assertEqual(plans, [("planB", "resB")])
        self.assertEqual(len(calls), 2)

    def test_every_plan_failing_still_yields_no_answer(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        plans, _ = self._run(payload, [None, None])

        self.assertEqual(plans, [])

    def test_follow_ups_beyond_the_cap_are_ignored(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [
                {"table_number": 15, "operation": "rank"} for _ in range(6)
            ],
        }
        _, calls = self._run(
            payload, [(f"plan{i}", f"res{i}") for i in range(9)]
        )

        self.assertEqual(len(calls), 1 + _max_follow_up_plans())

    def test_a_duplicated_plan_is_only_kept_once(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "aggregate"}],
        }
        plans, _ = self._run(payload, [("same", "resA"), ("same", "resB")])

        self.assertEqual(len(plans), 1)

    def test_no_follow_ups_still_returns_the_single_plan(self):
        payload = {"action": "plan", "table_number": 15, "operation": "aggregate"}
        plans, calls = self._run(payload, [("planA", "resA")])

        self.assertEqual(len(plans), 1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()


class DeclineRetryTests(unittest.TestCase):
    """A decline must be challenged once before the question loses Excel.

    All four zero-evidence beta cases declined for a reason that was wrong:
    a derived figure has no column (real_009, excel_005) or one requested
    year is absent (real_008, excel_006). The retry names the model's own
    reason back to it; every downstream guard is unchanged, so a recovered
    plan is one that would have validated on the first call.
    """

    def _provider(self, replies):
        calls = []

        class _Provider:
            def generate_structured(_self, prompt, schema):
                calls.append(prompt)
                return json.dumps(replies[min(len(calls) - 1, len(replies) - 1)])

        return _Provider(), calls

    def test_a_decline_is_retried_and_can_recover(self):
        provider, calls = self._provider(
            [
                {"action": "decline", "reason": "no cumulative target column"},
                {"action": "plan", "table_number": 15, "operation": "aggregate"},
            ]
        )
        payload = nl_planner._ask_planner(provider, "PROMPT")
        self.assertEqual(payload["action"], "decline")
        retried = nl_planner._ask_planner(
            provider, nl_planner._retry_prompt("PROMPT", payload["reason"])
        )
        self.assertEqual(retried["action"], "plan")
        self.assertEqual(len(calls), 2)

    def test_the_retry_prompt_quotes_the_models_own_reason(self):
        rendered = nl_planner._retry_prompt("PROMPT", "no cumulative target column")
        self.assertIn("no cumulative target column", rendered)
        self.assertIn("DERIVED figure", rendered)
        self.assertIn("Return the part", rendered)
        # The original prompt survives intact: same tables, same schema.
        self.assertTrue(rendered.startswith("PROMPT"))

    def test_a_missing_reason_still_renders(self):
        rendered = nl_planner._retry_prompt("PROMPT", "")
        self.assertIn("(no reason given)", rendered)

    def test_a_failed_planner_call_returns_none(self):
        class _Broken:
            def generate_structured(self, prompt, schema):
                return "not json"

        self.assertIsNone(nl_planner._ask_planner(_Broken(), "PROMPT"))


class EvidenceBudgetTests(unittest.TestCase):
    """Composition must not reintroduce the flooding failure.

    Raising the plan ceiling from 2 to 4 raises worst-case evidence with it,
    and "more evidence in front of the model" is the one intervention this
    lane has measured as harmful three separate times. The budget keeps the
    first plan unconditionally and then skips only what does not fit.
    """

    class _Res:
        def __init__(self, n):
            self.rows = [()] * n

    class _Plan:
        table_number = 1

    def _pairs(self, *counts):
        return [(self._Plan(), self._Res(n)) for n in counts]

    def test_plans_within_budget_are_all_kept(self):
        pairs = self._pairs(10, 20, 30)
        self.assertEqual(len(nl_planner._within_row_budget(pairs, [])), 3)

    def test_the_first_plan_is_kept_even_when_it_alone_exceeds_the_budget(self):
        pairs = self._pairs(nl_planner.MAX_TOTAL_EVIDENCE_ROWS + 50)
        self.assertEqual(len(nl_planner._within_row_budget(pairs, [])), 1)

    def test_an_oversized_later_plan_is_dropped_with_a_reason(self):
        rejections = []
        pairs = self._pairs(200, 200)
        kept = nl_planner._within_row_budget(pairs, rejections)
        self.assertEqual(len(kept), 1)
        self.assertTrue(any("evidence budget" in r for r in rejections))

    def test_a_small_plan_after_an_oversized_one_still_fits(self):
        """A compact aggregate is worth more than another hundred detail rows."""
        pairs = self._pairs(200, 200, 3)
        kept = nl_planner._within_row_budget(pairs, [])
        self.assertEqual([len(r.rows) for _, r in kept], [200, 3])


class PlanRepairTests(unittest.TestCase):
    """A partly-wrong plan is repaired, not discarded.

    ``real_008`` regressed from ~40 to a reproducible 11.2 across three runs
    because the decline-retry produced plans that the newly-added guards then
    rejected wholesale -- ``'sum' needs a numeric column: 'metric_name'`` and
    ``unknown attributes ['inspection_method', 'inspection_type']``. Both
    plans were right about the table, the period and most of the fields. The
    guards are correct; refusing the whole plan over one field was not.
    """

    def _sanitize(self, payload, contexts=None):
        contexts = contexts or {
            11: {
                "source": "excel_facts",
                "semantic_keys": ["wmp_spend"],
                "json_keys": ["expense_type", "initiative"],
                "vocabulary": {},
                "full_vocabulary": {},
                "title": "Table 11",
            }
        }
        return nl_planner._sanitize_plan(
            payload, "2024 spend", [(11, "card", None)], contexts
        )

    def test_a_text_aggregate_target_is_dropped_not_rejected(self):
        plan = self._sanitize(
            {
                "action": "plan",
                "table_number": 11,
                "operation": "aggregate",
                "aggregate": "sum",
                "semantic_metric_key": "wmp_spend",
                "value_column": "metric_name",
                "filters": [{"field": "reporting_year", "value": 2024}],
            }
        )
        self.assertIsNone(plan.value_column)
        self.assertEqual(plan.aggregate, "sum")

    def test_a_numeric_aggregate_target_survives(self):
        plan = self._sanitize(
            {
                "action": "plan",
                "table_number": 11,
                "operation": "aggregate",
                "aggregate": "sum",
                "semantic_metric_key": "wmp_spend",
                "value_column": "reporting_quarter",
                "filters": [{"field": "reporting_year", "value": 2024}],
            }
        )
        self.assertEqual(plan.value_column, "reporting_quarter")

    def test_a_text_column_survives_for_a_non_arithmetic_aggregate(self):
        plan = self._sanitize(
            {
                "action": "plan",
                "table_number": 11,
                "operation": "aggregate",
                "aggregate": "count_distinct",
                "semantic_metric_key": "wmp_spend",
                "value_column": "metric_name",
                "filters": [{"field": "reporting_year", "value": 2024}],
            }
        )
        self.assertEqual(plan.value_column, "metric_name")


class SelectAttributeRepairTests(unittest.TestCase):
    def _contexts(self):
        return {
            1: {
                "source": "excel_records",
                "semantic_keys": [],
                "json_keys": ["annual_quant_target", "quant_target_units"],
                "vocabulary": {},
                "full_vocabulary": {},
                "title": "Table 1",
            }
        }

    def _payload(self, keys):
        return {
            "action": "plan",
            "table_number": 1,
            "operation": "select",
            "select_json_keys": keys,
            "filters": [{"field": "reporting_year", "value": 2024}],
        }

    def test_invented_attributes_are_dropped_and_the_real_ones_kept(self):
        plan = nl_planner._sanitize_plan(
            self._payload(["annual_quant_target", "inspection_method"]),
            "detailed inspections",
            [(1, "card", None)],
            self._contexts(),
        )
        self.assertEqual(plan.select_json_keys, ("annual_quant_target",))

    def test_a_select_with_nothing_real_left_is_still_refused(self):
        with self.assertRaises(nl_planner._PlanRejected):
            nl_planner._sanitize_plan(
                self._payload(["inspection_method", "inspection_type"]),
                "detailed inspections",
                [(1, "card", None)],
                self._contexts(),
            )


class SelectMustReturnSomethingTests(unittest.TestCase):
    """Group keys alone are not evidence.

    ``excel_001`` asks which activities closed 2025 delayed or cancelled and
    why. The plan found exactly the right nine rows and returned
    ``[reporting_year, record_id]`` -- nine opaque hashes, no title, no
    status, no reason. It scored as "insufficient context" while sitting on
    the answer.
    """

    def _contexts(self):
        return {
            1: {
                "source": "excel_records",
                "semantic_keys": [],
                "json_keys": ["corrective_actions_if_delayed", "annual_quant_target"],
                "vocabulary": {},
                "full_vocabulary": {},
                "title": "Table 1",
            }
        }

    def _sanitize(self, keys):
        return nl_planner._sanitize_plan(
            {
                "action": "plan",
                "table_number": 1,
                "operation": "select",
                "select_json_keys": keys,
                "filters": [{"field": "reporting_year", "value": 2025}],
            },
            "which activities were delayed or cancelled and why",
            [(1, "card", None)],
            self._contexts(),
        )

    def test_promoted_column_names_become_group_columns(self):
        plan = self._sanitize(["status", "title", "corrective_actions_if_delayed"])
        self.assertIn("status", plan.group_by)
        self.assertIn("title", plan.group_by)
        self.assertEqual(
            plan.select_json_keys, ("corrective_actions_if_delayed",)
        )

    def test_a_select_of_only_promoted_columns_still_returns_them(self):
        plan = self._sanitize(["status", "title"])
        self.assertIn("status", plan.group_by)
        self.assertIn("title", plan.group_by)

    def test_a_select_with_nothing_at_all_to_return_is_refused(self):
        with self.assertRaises(nl_planner._PlanRejected):
            self._sanitize([])

    def test_a_real_attribute_select_is_untouched(self):
        plan = self._sanitize(["corrective_actions_if_delayed"])
        self.assertEqual(
            plan.select_json_keys, ("corrective_actions_if_delayed",)
        )
