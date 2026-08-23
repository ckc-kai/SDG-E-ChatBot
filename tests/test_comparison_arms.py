"""Comparison arms are read from the question, never from a fixed list.

The eval set is a reference, not the target: a question naming cycles or years
this corpus has never held must be detected exactly like one it has.
"""

from __future__ import annotations

from generation.comparison import asks_for_comparison, detect_axis, missing_arms


class TestAxisDetection:
    def test_two_cycles_are_one_axis_of_two_arms_not_four_years(self):
        axis = detect_axis(
            "Between the 2023-2025 WMP and the 2026-2028 WMP, how did risk ranking evolve?"
        )
        assert axis.kind == "cycle"
        assert axis.arms == ("2023-2025", "2026-2028")

    def test_cycles_absent_from_this_corpus_detect_identically(self):
        axis = detect_axis("Compare the 2030-2032 plan with the 2033-2035 plan")
        assert axis.kind == "cycle"
        assert axis.arms == ("2030-2032", "2033-2035")

    def test_several_years_are_arms(self):
        axis = detect_axis("show target versus actual for 2023, 2024 and 2025")
        assert axis.kind == "year"
        assert axis.arms == ("2023", "2024", "2025")

    def test_quarters_are_arms(self):
        assert detect_axis("Compare Q1 and Q3 outage counts").arms == ("Q1", "Q3")

    def test_a_single_period_is_not_a_comparison(self):
        assert not asks_for_comparison("Pull our 2022 actuals for inspections")
        assert not asks_for_comparison("What were our biggest 2024 spend programs?")


class TestMissingArms:
    EVIDENCE = "sdge__wmp__2023-2025__r5-redacted__2025-07-25.pdf"

    def test_an_arm_with_no_evidence_is_reported(self):
        absent = missing_arms(
            "Between the 2023-2025 WMP and the 2026-2028 WMP, what changed?",
            self.EVIDENCE,
        )
        assert absent == ("2026-2028",)

    def test_both_arms_present_reports_nothing(self):
        absent = missing_arms(
            "Between the 2023-2025 WMP and the 2026-2028 WMP, what changed?",
            self.EVIDENCE + " sdge__wmp__2026-2028__r2__2025-05-23.pdf",
        )
        assert absent == ()

    def test_a_non_comparison_never_reports_a_missing_arm(self):
        assert missing_arms("What did we spend in 2024?", "") == ()
