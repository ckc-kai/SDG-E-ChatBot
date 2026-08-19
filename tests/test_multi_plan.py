"""A question needing two queries must produce two executed plans."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from retrieval.query.excel import nl_planner
from retrieval.query.excel.nl_planner import (
    MAX_FOLLOW_UP_PLANS,
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
        self.assertEqual(
            PLAN_RESPONSE_SCHEMA["properties"]["follow_up_plans"]["maxItems"],
            MAX_FOLLOW_UP_PLANS,
        )


class _Provider:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, prompt, schema):
        return json.dumps(self.payload)


class BuildPlansTests(unittest.TestCase):
    def _run(self, payload, executed):
        calls = []

        def fake_execute_one(body, question, candidates, contexts, conn, contracts):
            calls.append(body)
            return executed.pop(0) if executed else None

        with patch.object(nl_planner, "_get_planner_provider",
                          return_value=_Provider(payload)), \
             patch.object(nl_planner, "_candidate_tables",
                          return_value=[(15, "cap", None)]), \
             patch.object(nl_planner, "_table_context", return_value={}), \
             patch.object(nl_planner, "_render_prompt", return_value="p"), \
             patch.object(nl_planner, "_entity_card_options", return_value={}), \
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

    def test_a_failed_primary_plan_abandons_the_question(self):
        payload = {
            "action": "plan", "table_number": 15, "operation": "aggregate",
            "follow_up_plans": [{"table_number": 15, "operation": "rank"}],
        }
        plans, calls = self._run(payload, [None, ("planB", "resB")])

        self.assertEqual(plans, [])
        self.assertEqual(len(calls), 1)

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

        self.assertEqual(len(calls), 1 + MAX_FOLLOW_UP_PLANS)

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
