import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation.planning import (
    CONTENT_TYPES,
    Requirement,
    RetrievalPlan,
    RetrievalStep,
    build_retrieval_plan,
    needs_planning,
    supports_multistep_generation,
)
from generation.providers.base import ProviderError


class FakeProvider:
    model_id = "fake"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        if self.error:
            raise self.error
        return self.response

    def generate_structured(self, prompt, schema):
        return self.generate(prompt)


class RetrievalPlanningTests(unittest.TestCase):
    def test_multistep_generation_requires_complete_model_plan(self):
        steps = (
            RetrievalStep("Find requirement A", ("narrative",), requirement_ids=("R1",)),
            RetrievalStep("Find requirement B", ("narrative",), requirement_ids=("R2",)),
        )
        requirements = (Requirement("R1", "A"), Requirement("R2", "B"))
        self.assertTrue(supports_multistep_generation(
            RetrievalPlan(steps, source="model", requirements=requirements)
        ))
        self.assertFalse(supports_multistep_generation(
            RetrievalPlan(steps[:1], source="model", requirements=requirements[:1])
        ))
        self.assertFalse(supports_multistep_generation(
            RetrievalPlan(steps, source="fallback", requirements=requirements)
        ))
        self.assertFalse(supports_multistep_generation(
            RetrievalPlan(steps, source="model", dropped_task_count=1, requirements=requirements)
        ))

    def test_between_years_single_fact_is_not_complex(self):
        self.assertFalse(needs_planning(
            "How many wildfires occurred between 2015 and 2022?"
        ))

    def test_reported_metric_over_year_range_uses_planner(self):
        self.assertTrue(needs_planning(
            "How many ignitions were reported from 2022-2025?"
        ))

    def test_short_comparison_is_complex(self):
        self.assertTrue(needs_planning("Compare the two WMP cycles."))

    def test_reported_result_reconciliation_is_complex(self):
        self.assertTrue(needs_planning(
            "Did the approved target make it into what SDG&E reported?"
        ))

    def test_simple_question_uses_all_types_without_model_call(self):
        provider = FakeProvider()
        plan = build_retrieval_plan("What is the target?", provider)
        self.assertEqual(plan.steps[0].content_types, ("narrative",))
        self.assertEqual(plan.steps[0].source, "pdf")
        self.assertEqual(provider.calls, 0)

    def test_complex_question_is_decomposed_with_soft_types(self):
        provider = FakeProvider(json.dumps({"tasks": [
            {"question": "What targets were missed?", "source": "excel"},
            {"question": "Why were they missed?", "source": "pdf"},
        ]}))
        plan = build_retrieval_plan(
            "What targets were missed, why were they missed, and were causes external?",
            provider,
        )
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].content_types, ("excel_card",))
        self.assertEqual(plan.steps[1].content_types, ("narrative",))

    def test_complex_planner_receives_project_domain_vocabulary(self):
        provider = FakeProvider(json.dumps({"tasks": [{
            "question": "Review the Wildfire Mitigation Plan against OEIS expectations.",
            "source": "pdf",
        }]}))
        build_retrieval_plan(
            "Review the WMP from OEIS's perspective for a future filing cycle.",
            provider,
        )
        self.assertIn("WMP means Wildfire Mitigation Plan", provider.last_prompt)
        self.assertIn("OEIS means the California Office", provider.last_prompt)
        self.assertIn("not automatically a request for forecast", provider.last_prompt)
        self.assertIn("For longitudinal performance questions", provider.last_prompt)
        self.assertIn("one PDF task for the filing or change", provider.last_prompt)
        self.assertIn("calculation operands, not derived values", provider.last_prompt)
        self.assertIn("percent complete maps to the task", provider.last_prompt)
        self.assertIn("Do not create one task per metric", provider.last_prompt)
        self.assertIn("MUST produce exactly one task", provider.last_prompt)
        self.assertNotIn("optional role", provider.last_prompt)
        self.assertNotIn("WMP.473", provider.last_prompt)

    def test_every_subquestion_searches_all_evidence_types(self):
        provider = FakeProvider(json.dumps({"tasks": [{
            "question": "What numbers were reported from 2022 through 2025?",
            "source": "pdf",
            "need_table": True,
        }]}))
        plan = build_retrieval_plan(
            "Compare the reports and identify what numbers were reported from 2022 through 2025.",
            provider,
        )
        self.assertEqual(
            plan.steps[0].content_types, ("narrative", "table")
        )

    def test_plan_merges_six_tasks_into_four_without_dropping_requirements(self):
        provider = FakeProvider(json.dumps({
            "requirements": [
                {"id": f"R{index}", "text": f"requirement {index}"}
                for index in range(1, 7)
            ],
            "tasks": [
            {"question": str(index), "source": "pdf", "metric": str(index)}
            for index in range(1, 7)
            ],
        }))
        payload = json.loads(provider.response)
        for index, task in enumerate(payload["tasks"], start=1):
            task["requirement_ids"] = [f"R{index}"]
        provider.response = json.dumps(payload)
        plan = build_retrieval_plan("Compare multiple WMP cycles.", provider)
        self.assertEqual(len(plan.steps), 4)
        self.assertEqual(plan.atomic_task_count, 6)
        self.assertEqual(plan.dropped_task_count, 0)
        self.assertEqual(
            {item for step in plan.steps for item in step.requirement_ids},
            {f"R{index}" for index in range(1, 7)},
        )

    def test_equivalent_atomic_tasks_are_deduplicated_before_branch_cap(self):
        provider = FakeProvider(json.dumps({"tasks": [
            {
                "question": "Find the 2025 target.",
                "source": "excel",
                "table_role": "qdr_table_01",
                "metric": "annual target",
                "period": "2025",
            },
            {
                "question": "Find that target another way.",
                "source": "excel",
                "table_role": "qdr_table_01",
                "metric": "annual target",
                "period": "2025",
            },
        ]}))
        plan = build_retrieval_plan("Compare reports and explain the target.", provider)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.atomic_task_count, 2)

    def test_underspecified_tasks_with_different_questions_are_not_collapsed(self):
        provider = FakeProvider(json.dumps({"tasks": [
            {"question": "Find metric A.", "source": "pdf"},
            {"question": "Find metric B.", "source": "pdf"},
        ]}))
        plan = build_retrieval_plan("Compare reports and explain both metrics.", provider)
        self.assertEqual([step.question for step in plan.steps], [
            "Find metric A.", "Find metric B."
        ])

    def test_invalid_or_failed_plan_never_narrows_retrieval(self):
        provider = FakeProvider(error=ProviderError("offline"))
        plan = build_retrieval_plan("Compare the two WMP cycles and explain why.", provider)
        self.assertEqual(plan.source, "fallback")
        self.assertEqual(plan.steps[0].content_types, CONTENT_TYPES)

    def test_planner_trace_records_raw_output_and_rejection_reason(self):
        with tempfile.TemporaryDirectory() as trace_dir, patch.dict(
            os.environ,
            {"TASK3_PLANNER_TRACE_DIR": trace_dir},
        ):
            provider = FakeProvider(response="not-json")
            plan = build_retrieval_plan("Compare the two WMP cycles.", provider)

            self.assertEqual(plan.source, "fallback")
            traces = list(Path(trace_dir).glob("planner-*.json"))
            self.assertEqual(len(traces), 1)
            record = json.loads(traces[0].read_text(encoding="utf-8"))
            self.assertEqual(record["provider_model_id"], "fake")
            self.assertEqual(record["raw_output"], "not-json")
            self.assertEqual(record["outcome"], "rejected")
            self.assertEqual(record["rejection"]["type"], "JSONDecodeError")

    def test_planner_can_replay_an_accepted_raw_output_without_model_call(self):
        payload = json.dumps({"tasks": [{
            "question": "Find the first WMP cycle.",
            "source": "pdf",
        }]})
        with tempfile.TemporaryDirectory() as trace_dir, patch.dict(
            os.environ,
            {"TASK3_PLANNER_TRACE_DIR": trace_dir},
        ):
            original = build_retrieval_plan(
                "Compare the two WMP cycles.", FakeProvider(response=payload)
            )
            replay_file = next(Path(trace_dir).glob("planner-*.json"))

            replay_provider = FakeProvider(error=ProviderError("must not be called"))
            with patch.dict(
                os.environ,
                {
                    "TASK3_PLANNER_REPLAY_FILE": str(replay_file),
                    "TASK3_PLANNER_TRACE_DIR": "",
                },
            ):
                replayed = build_retrieval_plan(
                    "Compare the two WMP cycles.", replay_provider
                )

        self.assertEqual(original, replayed)
        self.assertEqual(replay_provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
