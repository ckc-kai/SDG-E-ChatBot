import unittest
from types import SimpleNamespace

from generation.coverage import assess_plan_coverage
from generation.planning import RetrievalPlan, RetrievalStep
from retrieval.query.pdf import EvidenceGroup, EvidenceRetrievalResult


class CoverageLedgerTests(unittest.TestCase):
    @staticmethod
    def _bundle(groups, *, verified=False):
        return SimpleNamespace(
            evidence=EvidenceRetrievalResult(question="q", groups=groups),
            verified_excel=object() if verified else None,
            verified_excels=(),
        )

    def test_support_type_is_covered_only_when_its_group_has_evidence(self):
        plan = RetrievalPlan((
            RetrievalStep("q", ("narrative", "table"), "pdf"),
        ))
        narrative = EvidenceGroup("narrative", ("narrative",), [object()], SimpleNamespace())
        table = EvidenceGroup("table", ("table",), [], SimpleNamespace())
        ledger = assess_plan_coverage(
            plan, (self._bundle({"narrative": narrative, "table": table}),)
        )
        self.assertFalse(ledger.covered)
        self.assertEqual(ledger.missing_step_indexes, (0,))
        self.assertEqual(ledger.items[0].missing_content_types, ("table",))

    def test_verified_excel_covers_excel_task_without_semantic_card(self):
        plan = RetrievalPlan((RetrievalStep("q", ("excel_card",), "excel"),))
        ledger = assess_plan_coverage(plan, (self._bundle({}, verified=True),))
        self.assertTrue(ledger.covered)


if __name__ == "__main__":
    unittest.main()
