import unittest
from decimal import Decimal

from generation.computation import (
    CalculationError,
    CalculationSpec,
    EvidenceOperand,
    FactRequirement,
    TypedComputationPlan,
    execute_calculation,
)


class CrossResourceComputationTests(unittest.TestCase):
    def _plan(self, operation="ratio_percent"):
        return TypedComputationPlan(
            facts=(
                FactRequirement("actual", "excel", "completed_units", "2025"),
                FactRequirement("target", "pdf", "planned_units", "2025"),
            ),
            calculation=CalculationSpec(
                operation=operation,
                left_ref="actual",
                right_ref="target",
            ),
        )

    def test_ratio_is_decimal_and_keeps_both_operand_provenances(self):
        result = execute_calculation(
            self._plan(),
            {
                "actual": EvidenceOperand(
                    Decimal("75"), "units", "2025", ("qdr.xlsx#Table 1!R9",)
                ),
                "target": EvidenceOperand(
                    Decimal("100"), "units", "2025", ("wmp.pdf#p42",)
                ),
            },
        )
        self.assertEqual(result.value, Decimal("75.00"))
        self.assertEqual(result.unit, "percent")
        self.assertEqual(
            result.contributing_sources,
            ("qdr.xlsx#Table 1!R9", "wmp.pdf#p42"),
        )

    def test_missing_operand_and_zero_denominator_are_rejected(self):
        with self.assertRaisesRegex(CalculationError, "missing operand"):
            execute_calculation(self._plan(), {})
        with self.assertRaisesRegex(CalculationError, "zero denominator"):
            execute_calculation(
                self._plan(),
                {
                    "actual": EvidenceOperand(Decimal("1"), "units", "2025", ("a",)),
                    "target": EvidenceOperand(Decimal("0"), "units", "2025", ("b",)),
                },
            )

    def test_period_or_unit_mismatch_is_rejected(self):
        operands = {
            "actual": EvidenceOperand(Decimal("75"), "miles", "2025", ("a",)),
            "target": EvidenceOperand(Decimal("100"), "poles", "2024", ("b",)),
        }
        with self.assertRaisesRegex(CalculationError, "period"):
            execute_calculation(self._plan(), operands)

    def test_change_percent_accepts_planned_different_periods(self):
        plan = TypedComputationPlan(
            facts=(
                FactRequirement("new", "excel", "miles", "2025", "miles"),
                FactRequirement("old", "pdf", "miles", "2024", "miles"),
            ),
            calculation=CalculationSpec("change_percent", "new", "old"),
        )
        result = execute_calculation(
            plan,
            {
                "new": EvidenceOperand(Decimal("120"), "miles", "2025", ("new",)),
                "old": EvidenceOperand(Decimal("100"), "miles", "2024", ("old",)),
            },
        )
        self.assertEqual(result.value, Decimal("20.00"))


if __name__ == "__main__":
    unittest.main()
