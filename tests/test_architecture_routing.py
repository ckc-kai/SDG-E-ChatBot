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
    def test_every_question_requests_every_evidence_group(self):
        """Grouped retrieval ranks groups independently, so withholding one can
        only lose evidence. Routing therefore no longer predicts resources."""
        route = route_question("How does SDG&E define wildfire risk?")
        self.assertTrue(route.need_pdf)
        self.assertTrue(route.need_excel)
        self.assertEqual(
            route.pdf_content_types, ("narrative", "table", "figure")
        )
        self.assertEqual(
            route.content_types,
            ("narrative", "table", "figure", "excel_card"),
        )

    def test_stakeholder_phrasing_still_reaches_the_workbook(self):
        """The regression this replaced a classifier for: a question answerable
        only from the workbook, phrased without the words "workbook" or "QDR"."""
        plain = route_question(
            "How many poles were replaced in 2024, and what was the target?"
        )
        self.assertTrue(plain.need_excel)
        self.assertIn("excel_card", plain.content_types)

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
        # A judge may add evidence but never remove a deterministic group.
        self.assertEqual(
            route.pdf_content_types, ("narrative", "table", "figure")
        )
        self.assertTrue(route.need_excel)
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
