"""Deterministic roll-ups must combine components, and must refuse to lie.

The 20% "fetched as components and never combined" bucket in
``logs/progress/2026-08-19-extended.md`` is what this module exists to close.
The refusal cases matter at least as much as the arithmetic: ``real_009``
reported 2,053,923.98 as a real figure by adding inspections, poles, trees and
miles together, and a roll-up that repeated that mistake would be worse than
no roll-up at all.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from generation.excel_rollup import RollUp, roll_up
from retrieval.query.excel.query import (
    FACTS,
    RECORDS,
    ExcelExecutionResult,
    ExcelQueryPlan,
)


def _result(
    *,
    columns,
    rows,
    aggregate="sum",
    operation="aggregate",
    unit="dollars",
    warnings=None,
    source=FACTS,
):
    plan = ExcelQueryPlan(
        table_number=11,
        source=source,
        operation=operation,
        aggregate=aggregate,
        group_by=tuple(columns[:-1]),
    )
    return ExcelExecutionResult(
        plan=plan,
        revision_id=1,
        columns=list(columns),
        rows=[tuple(row) for row in rows],
        unit=unit,
        contributing_facts=len(rows),
        warnings=list(warnings or []),
    )


def _labels(rolled):
    return [item.label for item in rolled]


def _by_label(rolled, label):
    return next(item for item in rolled if item.label == label)


class TestGroupTotal:
    def test_grouped_sum_gets_a_total(self):
        result = _result(
            columns=["expense_type", "value"],
            rows=[["CAPEX", Decimal("473986")], ["OPEX", Decimal("193968")]],
        )
        rolled = roll_up(result)
        assert _by_label(rolled, "group_total").value == Decimal("667954")

    def test_total_carries_the_result_unit(self):
        result = _result(
            columns=["expense_type", "value"],
            rows=[["CAPEX", Decimal("10")], ["OPEX", Decimal("5")]],
        )
        assert _by_label(roll_up(result), "group_total").unit == "dollars"

    def test_blank_groups_are_excluded_and_declared(self):
        result = _result(
            columns=["tier", "value"],
            rows=[["Tier 2", Decimal("10")], ["Tier 3", None], ["Non-HFTD", Decimal("5")]],
        )
        total = _by_label(roll_up(result), "group_total")
        assert total.value == Decimal("15")
        assert "1 groups blank and excluded" in total.calculation


class TestRefusals:
    def test_a_mixed_unit_result_is_never_totalled(self):
        """real_009: inspections + poles + trees + miles is not a number."""
        result = _result(
            columns=["entity_key", "value"],
            rows=[["WMP.549", Decimal("35")], ["WMP.473", Decimal("84.43")]],
            warnings=["result spans several units: Poles, Miles, Trees"],
        )
        assert roll_up(result) == []

    @pytest.mark.parametrize("aggregate", ["avg", "min", "max"])
    def test_non_additive_aggregates_are_never_totalled(self, aggregate):
        """Averaging averages and maxing maxima do not total."""
        result = _result(
            columns=["tier", "value"],
            rows=[["Tier 2", Decimal("10")], ["Tier 3", Decimal("20")]],
            aggregate=aggregate,
        )
        assert roll_up(result) == []

    def test_a_select_scan_is_not_rolled_up(self):
        result = _result(
            columns=["record_id", "selected_0"],
            rows=[["a", "1.0"], ["b", "2.0"]],
            operation="select",
            source=RECORDS,
        )
        assert roll_up(result) == []

    def test_an_ungrouped_scalar_is_not_rolled_up(self):
        result = _result(columns=["value"], rows=[[Decimal("932371")]])
        assert roll_up(result) == []

    def test_a_single_group_is_not_rolled_up(self):
        result = _result(columns=["tier", "value"], rows=[["Tier 2", Decimal("10")]])
        assert roll_up(result) == []

    def test_a_text_measure_column_is_not_rolled_up(self):
        result = _result(
            columns=["tier", "status"],
            rows=[["Tier 2", "Delayed"], ["Tier 3", "Complete"]],
        )
        assert roll_up(result) == []


class TestShares:
    def test_shares_sum_to_one_hundred(self):
        result = _result(
            columns=["expense_type", "value"],
            rows=[["CAPEX", Decimal("750")], ["OPEX", Decimal("250")]],
        )
        shares = [item for item in roll_up(result) if item.label.startswith("share[")]
        assert [item.value for item in shares] == [Decimal("75.0"), Decimal("25.0")]
        assert all(item.unit == "%" for item in shares)

    def test_a_share_shows_its_arithmetic(self):
        result = _result(
            columns=["expense_type", "value"],
            rows=[["CAPEX", Decimal("750")], ["OPEX", Decimal("250")]],
        )
        share = next(
            item for item in roll_up(result) if item.label == "share[expense_type=CAPEX]"
        )
        assert share.calculation == "750 / 1000 x 100"

    def test_a_negative_component_suppresses_shares(self):
        """A share of a mixed-sign set is not a proportion of anything."""
        result = _result(
            columns=["tier", "value"],
            rows=[["Tier 2", Decimal("-10")], ["Tier 3", Decimal("30")]],
        )
        rolled = roll_up(result)
        assert _by_label(rolled, "group_total").value == Decimal("20")
        assert not [item for item in rolled if item.label.startswith("share[")]

    def test_a_zero_total_suppresses_shares(self):
        result = _result(
            columns=["tier", "value"],
            rows=[["Tier 2", Decimal("0")], ["Tier 3", Decimal("0")]],
        )
        assert not [
            item for item in roll_up(result) if item.label.startswith("share[")
        ]


class TestYearSubtotals:
    def test_year_and_dimension_grouping_gets_per_year_subtotals(self):
        result = _result(
            columns=["reporting_year", "expense_type", "value"],
            rows=[
                [2023, "CAPEX", Decimal("100")],
                [2023, "OPEX", Decimal("50")],
                [2024, "CAPEX", Decimal("200")],
                [2024, "OPEX", Decimal("75")],
            ],
        )
        rolled = roll_up(result)
        assert _by_label(rolled, "subtotal[reporting_year=2023]").value == Decimal("150")
        assert _by_label(rolled, "subtotal[reporting_year=2024]").value == Decimal("275")
        assert _by_label(rolled, "group_total").value == Decimal("425")

    def test_year_as_the_only_grouping_gets_no_subtotals(self):
        """Each row already is its year's total; restating it adds nothing."""
        result = _result(
            columns=["reporting_year", "value"],
            rows=[[2023, Decimal("100")], [2024, Decimal("200")]],
        )
        assert not [
            item for item in roll_up(result) if item.label.startswith("subtotal[")
        ]


class TestRendering:
    def test_a_rollup_renders_its_value_unit_and_arithmetic(self):
        item = RollUp(
            label="group_total",
            value=Decimal("667954"),
            calculation="sum of 2 group values",
            unit="dollars",
        )
        assert item.render() == "group_total=667954 dollars (sum of 2 group values)"


class TestDisplayColumns:
    """Generated jsonb aliases must read as the dimension they group.

    ``compile_plan`` emits ``group_0`` for a jsonb group key so that
    model-supplied text never lands in a SQL identifier position. The evidence
    block then said ``group_0=CAPEX``, which tells the answering model
    nothing. The name is restored at render time, where it is data.
    """

    def test_a_generated_alias_is_named_back(self):
        from generation.excel_rollup import display_columns
        from retrieval.query.excel.query import GroupKey

        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            group_by=(GroupKey("expense_type", json_key=True),),
        )
        result = _result(columns=["group_0", "value"], rows=[["CAPEX", Decimal("1")]])
        assert display_columns(result, plan) == ["expense_type", "value"]

    def test_typed_columns_are_left_alone(self):
        from generation.excel_rollup import display_columns

        plan = ExcelQueryPlan(
            table_number=11, source=FACTS, group_by=("reporting_year",)
        )
        result = _result(
            columns=["reporting_year", "value"], rows=[[2024, Decimal("1")]]
        )
        assert display_columns(result, plan) == ["reporting_year", "value"]

    def test_an_alias_without_a_matching_group_key_survives_unchanged(self):
        from generation.excel_rollup import display_columns

        plan = ExcelQueryPlan(table_number=11, source=FACTS, group_by=())
        result = _result(columns=["group_0", "value"], rows=[["x", Decimal("1")]])
        assert display_columns(result, plan) == ["group_0", "value"]

    def test_share_labels_use_the_dimension_name(self):
        from retrieval.query.excel.query import GroupKey

        plan = ExcelQueryPlan(
            table_number=2,
            source=FACTS,
            aggregate="sum",
            group_by=(GroupKey("expense_type", json_key=True),),
        )
        result = _result(
            columns=["group_0", "value"],
            rows=[["CAPEX", Decimal("750")], ["OPEX", Decimal("250")]],
        )
        labels = [item.label for item in roll_up(result, plan)]
        assert "share[expense_type=CAPEX]" in labels
        assert not any("group_0" in label for label in labels)


class TestTruncationRefusal:
    """A total over a truncated head is a wrong number stated confidently.

    9 of 22 executed results on the beta set came back sitting exactly on
    their row limit with nothing in the evidence to say so. Totalling the
    eight largest initiatives and labelling it "group_total" would be the
    same class of defect as summing across incompatible units.
    """

    def _truncated(self):
        result = _result(
            columns=["initiative", "value"],
            rows=[["A", Decimal("10")], ["B", Decimal("5")]],
        )
        result.warnings = [
            "Only the first 2 rows are shown and more exist; this is a "
            "partial view, not the complete set."
        ]
        return result

    def test_a_truncated_result_is_never_totalled(self):
        assert roll_up(self._truncated()) == []

    def test_the_same_rows_untruncated_are_totalled(self):
        result = self._truncated()
        result.warnings = []
        assert any(item.label == "group_total" for item in roll_up(result))
