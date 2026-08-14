import json
import unittest

from generation.planning import CONTENT_TYPES, build_retrieval_plan, needs_planning
from generation.providers.base import ProviderError


class FakeProvider:
    model_id = "fake"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response

    def generate_structured(self, prompt, schema):
        return self.generate(prompt)


class RetrievalPlanningTests(unittest.TestCase):
    def test_between_years_single_fact_is_not_complex(self):
        self.assertFalse(needs_planning(
            "How many wildfires occurred between 2015 and 2022?"
        ))

    def test_short_comparison_is_complex(self):
        self.assertTrue(needs_planning("Compare the two WMP cycles."))

    def test_simple_question_uses_all_types_without_model_call(self):
        provider = FakeProvider()
        plan = build_retrieval_plan("What is the target?", provider)
        self.assertEqual(plan.steps[0].content_types, CONTENT_TYPES)
        self.assertEqual(provider.calls, 0)

    def test_complex_question_is_decomposed_with_soft_types(self):
        provider = FakeProvider(json.dumps({"subquestions": [
            {"question": "What targets were missed?"},
            {"question": "Why were they missed?"},
        ]}))
        plan = build_retrieval_plan(
            "What targets were missed, why were they missed, and were causes external?",
            provider,
        )
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].content_types, CONTENT_TYPES)

    def test_every_subquestion_searches_all_evidence_types(self):
        provider = FakeProvider(json.dumps({"subquestions": [{
            "question": "What numbers were reported from 2022 through 2025?",
        }]}))
        plan = build_retrieval_plan(
            "Compare the reports and identify what numbers were reported from 2022 through 2025.",
            provider,
        )
        self.assertEqual(
            plan.steps[0].content_types, CONTENT_TYPES
        )

    def test_plan_is_limited_to_two_subquestions(self):
        provider = FakeProvider(json.dumps({"subquestions": [
            {"question": "one"}, {"question": "two"}, {"question": "three"}
        ]}))
        plan = build_retrieval_plan("Compare multiple WMP cycles.", provider)
        self.assertEqual([step.question for step in plan.steps], ["one", "two"])

    def test_invalid_or_failed_plan_never_narrows_retrieval(self):
        provider = FakeProvider(error=ProviderError("offline"))
        plan = build_retrieval_plan("Compare the two WMP cycles and explain why.", provider)
        self.assertEqual(plan.source, "fallback")
        self.assertEqual(plan.steps[0].content_types, CONTENT_TYPES)


if __name__ == "__main__":
    unittest.main()
