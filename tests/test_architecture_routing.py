import json
import unittest

from generation.providers.base import ProviderError
from generation.routing import route_question


class FakeJudge:
    model_id = "fake"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def generate_structured(self, prompt, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class ResourceRoutingTests(unittest.TestCase):
    def test_simple_pdf_question_uses_narrative_only(self):
        route = route_question("How does SDG&E define wildfire risk?")
        self.assertTrue(route.need_pdf)
        self.assertFalse(route.need_excel)
        self.assertEqual(route.pdf_content_types, ("narrative",))
        self.assertFalse(route.uncertain)

    def test_explicit_figure_and_workbook_cues_are_deterministic(self):
        figure = route_question("According to Figure 7-2, what does the chart show?")
        workbook = route_question(
            "In the quarterly activity workbook, what was WMP.455's 2024 target?"
        )
        self.assertEqual(figure.pdf_content_types, ("narrative", "figure"))
        self.assertFalse(figure.need_excel)
        self.assertTrue(workbook.need_excel)
        self.assertFalse(workbook.need_pdf)

    def test_uncertain_route_uses_at_most_one_structured_judge_call(self):
        judge = FakeJudge(json.dumps({
            "need_narrative": True,
            "need_table": True,
            "need_figure": False,
            "need_excel": False,
            "uncertain": False,
        }))
        route = route_question(
            "What exact values are shown for the mitigation categories?", judge=judge
        )
        self.assertEqual(judge.calls, 1)
        self.assertEqual(route.pdf_content_types, ("narrative", "table"))
        self.assertEqual(route.source, "judge")

    def test_failed_judge_fails_open_to_bounded_pdf_support(self):
        judge = FakeJudge(error=ProviderError("offline"))
        route = route_question(
            "What exact values are shown for the mitigation categories?", judge=judge
        )
        self.assertEqual(judge.calls, 1)
        self.assertEqual(
            route.pdf_content_types, ("narrative", "table", "figure")
        )
        self.assertTrue(route.uncertain)
        self.assertEqual(route.source, "fail_open")


if __name__ == "__main__":
    unittest.main()
