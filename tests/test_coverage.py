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

    def test_wrong_wmp_cycle_does_not_cover_scoped_step(self):
        plan = RetrievalPlan((RetrievalStep(
            "Find the model.",
            ("narrative",),
            "pdf",
            document_role="2026-2028 WMP",
            period="2026-2028",
        ),))
        wrong = SimpleNamespace(
            query_object=SimpleNamespace(
                source_pdf="SDG&E_2023-2023_Base-WMP_R5-redacted.pdf"
            )
        )
        narrative = EvidenceGroup(
            "narrative", ("narrative",), [wrong], SimpleNamespace()
        )

        ledger = assess_plan_coverage(
            plan, (self._bundle({"narrative": narrative}),)
        )

        self.assertFalse(ledger.covered)
        self.assertEqual(ledger.missing_step_indexes, (0,))

    def test_partial_verified_excel_does_not_claim_full_coverage(self):
        plan = RetrievalPlan((RetrievalStep(
            "reported metric from 2022-2025", ("excel_card",), "excel"
        ),))
        verified = SimpleNamespace(bound={"missing_reporting_years": (2022,)})
        bundle = SimpleNamespace(
            evidence=EvidenceRetrievalResult(question="q", groups={}),
            verified_excel=verified,
            verified_excels=(verified,),
        )

        self.assertFalse(assess_plan_coverage(plan, (bundle,)).covered)


if __name__ == "__main__":
    unittest.main()
