"""The overlapping-scope contract, and its re-derivation from the live corpus.

A semantic rule that encodes a wrong relation is worse than no rule, so the
containment claim is not merely asserted here -- it is re-queried against the
active revisions. If a future ingest makes HFTD exceed Territory in any cell,
this fails rather than quietly scoping every spend answer to the wrong subset.

The integration tests need a live database and are marked accordingly.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from generation.excel_rollup import roll_up
from retrieval.query.excel.query import (
    FACTS,
    ExcelExecutionResult,
    ExcelQueryPlan,
    Filter,
    _apply_canonical_scope,
)
from retrieval.query.excel.scopes import (
    OVERLAPPING_SCOPES,
    overlapping_scope,
    scope_is_pinned,
)


class TestContractShape:
    def test_table_eleven_is_the_only_overlapping_table(self):
        """Every other table partitions into Tier 2 / Tier 3 / Non-HFTD."""
        assert set(OVERLAPPING_SCOPES) == {11}

    def test_territory_is_the_canonical_whole(self):
        scope = overlapping_scope(11)
        assert scope.canonical == "Territory"
        assert scope.contained == ("HFTD",)

    def test_a_partitioning_table_has_no_contract(self):
        assert overlapping_scope(2) is None
        assert overlapping_scope(15) is None


class TestCanonicalScopeRewrite:
    def _plan(self, **kwargs):
        base = dict(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate="sum",
            filters=(Filter("reporting_year", "eq", 2024),),
        )
        base.update(kwargs)
        return ExcelQueryPlan(**base)

    def test_an_unscoped_sum_is_pinned_to_territory(self):
        scoped, warnings = _apply_canonical_scope(self._plan())
        assert Filter("hftd_tier", "eq", "Territory") in scoped.filters
        assert warnings and "double-count" in warnings[0]

    def test_an_explicitly_scoped_plan_is_untouched(self):
        plan = self._plan(
            filters=(
                Filter("reporting_year", "eq", 2024),
                Filter("hftd_tier", "eq", "HFTD"),
            )
        )
        scoped, warnings = _apply_canonical_scope(plan)
        assert scoped == plan and warnings == []

    def test_grouping_by_the_scope_column_is_untouched(self):
        """Each row is labelled with its scope, so nothing is conflated."""
        plan = self._plan(group_by=("hftd_tier",))
        scoped, warnings = _apply_canonical_scope(plan)
        assert scoped == plan and warnings == []

    def test_a_partitioning_table_is_untouched(self):
        plan = self._plan(table_number=2)
        scoped, warnings = _apply_canonical_scope(plan)
        assert scoped == plan and warnings == []

    @pytest.mark.parametrize("aggregate", ["count", "min", "max"])
    def test_non_magnitude_aggregates_are_untouched(self, aggregate):
        plan = self._plan(aggregate=aggregate)
        scoped, warnings = _apply_canonical_scope(plan)
        assert scoped == plan and warnings == []

    def test_scope_is_pinned_detects_both_filter_and_group(self):
        scope = overlapping_scope(11)
        assert scope_is_pinned(scope, self._plan(group_by=("hftd_tier",)))
        assert scope_is_pinned(
            scope,
            self._plan(
                filters=(Filter("hftd_tier", "eq", "Territory"),)
            ),
        )
        assert not scope_is_pinned(scope, self._plan())


class TestRollUpRefusesOverlap:
    def test_territory_and_hftd_groups_are_never_totalled(self):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate="sum",
            group_by=("hftd_tier",),
        )
        result = ExcelExecutionResult(
            plan=plan,
            revision_id=1,
            columns=["hftd_tier", "value"],
            rows=[("Territory", Decimal("667954")), ("HFTD", Decimal("612666"))],
            unit="dollars",
            contributing_facts=336,
        )
        assert roll_up(result, plan) == []

    def test_a_partitioning_tier_breakdown_is_still_totalled(self):
        plan = ExcelQueryPlan(
            table_number=2,
            source=FACTS,
            operation="aggregate",
            aggregate="sum",
            group_by=("hftd_tier",),
        )
        result = ExcelExecutionResult(
            plan=plan,
            revision_id=1,
            columns=["hftd_tier", "value"],
            rows=[
                ("HFTD Tier 2", Decimal("10")),
                ("HFTD Tier 3", Decimal("20")),
                ("Non-HFTD", Decimal("70")),
            ],
            unit="miles",
            contributing_facts=3,
        )
        rolled = roll_up(result, plan)
        total = next(item for item in rolled if item.label == "group_total")
        assert total.value == Decimal("100")


@pytest.mark.integration
class TestContractHoldsAgainstTheCorpus:
    """Re-derive the containment rule rather than trusting the constant."""

    @pytest.fixture(scope="class")
    def conn(self):
        from retrieval.utils import connect_db

        connection = connect_db()
        yield connection
        connection.close()

    def test_hftd_never_exceeds_territory_in_any_cell(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH cells AS (
                    SELECT f.reporting_year,
                           f.dimensions->>'initiative'   AS initiative,
                           f.dimensions->>'expense_type' AS expense_type,
                           f.hftd_tier,
                           sum(f.value_numeric) AS v
                    FROM excel_facts f
                    JOIN excel_revisions r
                      ON r.id = f.revision_id AND r.state = 'active'
                    WHERE f.table_number = 11 AND f.hftd_tier IS NOT NULL
                    GROUP BY 1, 2, 3, 4
                )
                SELECT count(*) FILTER (WHERE hftd IS NOT NULL AND terr IS NOT NULL),
                       count(*) FILTER (WHERE hftd > terr + 0.01)
                FROM (
                    SELECT reporting_year, initiative, expense_type,
                           max(v) FILTER (WHERE hftd_tier = 'Territory') AS terr,
                           max(v) FILTER (WHERE hftd_tier = 'HFTD')      AS hftd
                    FROM cells GROUP BY 1, 2, 3
                ) paired
                """
            )
            comparable, violations = cur.fetchone()
        assert comparable > 0, "no comparable cells: the rule cannot be checked"
        assert violations == 0, (
            f"{violations} cells have HFTD > Territory; 'HFTD is contained in "
            "Territory' no longer holds and the contract must be revisited"
        )

    def test_only_table_eleven_uses_the_overlapping_vocabulary(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT f.table_number
                FROM excel_facts f
                JOIN excel_revisions r
                  ON r.id = f.revision_id AND r.state = 'active'
                WHERE f.hftd_tier IN ('Territory', 'HFTD')
                """
            )
            tables = {row[0] for row in cur.fetchall()}
        assert tables == set(OVERLAPPING_SCOPES)
