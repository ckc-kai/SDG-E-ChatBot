"""The typed plan DSL must express analytic questions, and must never lie.

Every assertion here is a shape the beta workbook questions actually need, or
a compiled-SQL defect that produced confident wrong numbers. The two numeric
cases are measured against the live v2 corpus and are recorded in
``logs/progress/2026-08-19-extended.md``:

  annual_quant_target = 0            text comparison -> 4 rows, truth 10
  quant_actual_progress_q1_4 > 1000  text comparison -> 102 rows, truth 41

The second is the dangerous one: text ordering puts '2.0' above '1000', so the
executor returned rows that satisfied nothing the caller asked for, and the
channel reported the result as execution-verified.
"""

from __future__ import annotations

import pytest

from retrieval.ingest.excel.contracts import load_contracts
from retrieval.query.excel.query import (
    FACTS,
    RECORDS,
    ExcelQueryPlan,
    Filter,
    GroupKey,
    Having,
    PlanError,
    compile_plan,
)


@pytest.fixture(scope="module")
def contracts():
    return load_contracts()


def _sql(plan: ExcelQueryPlan, contracts) -> tuple[str, list]:
    return compile_plan(plan, contracts.for_table(plan.table_number))


class TestNumericJsonFilters:
    def test_numeric_cast_compiles_to_a_numeric_comparison(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(
                Filter("annual_quant_target", "eq", 0, json_key=True, cast="numeric"),
            ),
            group_by=("record_id",),
        )
        sql, params = _sql(plan, contracts)

        assert "::numeric" in sql
        # The value travels as a number, never str()-ed into a text comparison.
        assert 0 in params and "0" not in params

    def test_numeric_cast_tolerates_blank_and_non_numeric_attributes(self, contracts):
        """A row whose attribute is '' or 'n/a' must be skipped, not crash."""
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(
                Filter("annual_quant_target", "gt", 100, json_key=True, cast="numeric"),
            ),
            group_by=("record_id",),
        )
        sql, _ = _sql(plan, contracts)

        assert "~" in sql or "NULLIF" in sql

    def test_ordering_operators_on_json_require_an_explicit_cast(self, contracts):
        """Lexicographic ordering on text is never what the caller meant."""
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("quant_actual_progress_q1_4", "gt", 1000, json_key=True),),
            group_by=("record_id",),
        )
        with pytest.raises(PlanError, match="cast"):
            _sql(plan, contracts)

    def test_equality_on_json_without_a_cast_still_compiles_as_text(self, contracts):
        """Dimension equality is genuinely textual and must keep working."""
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("expense_type", "eq", "CAPEX", json_key=True),),
        )
        sql, params = _sql(plan, contracts)

        assert "->> %s = %s" in sql
        assert "CAPEX" in params


class TestGroupByJsonKeys:
    def test_group_by_a_json_dimension(self, contracts):
        """'biggest capital programs' groups by initiative, not by record_id."""
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="rank",
            aggregate="sum",
            group_by=(GroupKey("wmp_initiative", json_key=True),),
        )
        sql, params = _sql(plan, contracts)

        assert "dimensions ->> %s" in sql
        assert "GROUP BY" in sql
        assert "wmp_initiative" in params

    def test_json_group_key_alias_is_generated_not_caller_supplied(self, contracts):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="rank",
            aggregate="sum",
            group_by=(GroupKey("wmp_initiative", json_key=True),),
        )
        sql, _ = _sql(plan, contracts)

        assert "group_0" in sql
        assert "wmp_initiative" not in sql  # travels as a parameter only

    def test_typed_group_by_still_accepts_bare_strings(self, contracts):
        plan = ExcelQueryPlan(
            table_number=13,
            source=RECORDS,
            aggregate="count",
            group_by=("reporting_year", "status"),
        )
        sql, _ = _sql(plan, contracts)

        assert "t.reporting_year" in sql and "t.status" in sql


class TestRecordAggregates:
    def test_sum_over_a_cast_json_attribute_on_records(self, contracts):
        """excel_records had no aggregable value column at all before this."""
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="aggregate",
            aggregate="sum",
            value_json_key="quant_actual_progress_q1_4",
            group_by=("reporting_year",),
        )
        sql, params = _sql(plan, contracts)

        assert "sum(" in sql and "::numeric" in sql
        assert "quant_actual_progress_q1_4" in params

    def test_count_star_on_records_is_unchanged(self, contracts):
        plan = ExcelQueryPlan(
            table_number=13, source=RECORDS, aggregate="count", group_by=("status",)
        )
        sql, _ = _sql(plan, contracts)

        assert "count(*)" in sql

    def test_count_null_reports_the_blank_cohort(self, contracts):
        """excel_002's gold answer leads with 50,446 unpopulated priorities."""
        plan = ExcelQueryPlan(
            table_number=13,
            source=RECORDS,
            aggregate="count_null",
            value_column="status",
            group_by=("reporting_year",),
        )
        sql, _ = _sql(plan, contracts)

        assert "count(*) FILTER" in sql and "IS NULL" in sql


class TestHaving:
    def test_having_filters_the_aggregate(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="aggregate",
            aggregate="sum",
            value_json_key="quant_actual_progress_q1_4",
            group_by=("entity_key",),
            having=Having("gt", 0),
        )
        sql, params = _sql(plan, contracts)

        assert "HAVING" in sql
        assert 0 in params

    def test_having_without_group_by_is_refused(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="aggregate",
            aggregate="sum",
            value_json_key="quant_actual_progress_q1_4",
            having=Having("gt", 0),
        )
        with pytest.raises(PlanError, match="group"):
            _sql(plan, contracts)


class TestInjectionSurface:
    @pytest.mark.parametrize(
        "hostile",
        ["x'; DROP TABLE excel_facts; --", "value) OR 1=1 --", "a\"b", "a\\b"],
    )
    def test_hostile_json_field_names_never_reach_the_statement(
        self, contracts, hostile
    ):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter(hostile, "eq", "x", json_key=True),),
        )
        sql, params = _sql(plan, contracts)

        assert hostile not in sql
        assert hostile in params

    def test_hostile_typed_column_is_rejected(self, contracts):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("x'; DROP TABLE excel_facts; --", "eq", 1),),
        )
        with pytest.raises(PlanError):
            _sql(plan, contracts)

    def test_hostile_group_key_is_parameterized(self, contracts):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            aggregate="sum",
            group_by=(GroupKey("a'; DROP TABLE x; --", json_key=True),),
        )
        sql, params = _sql(plan, contracts)

        assert "DROP TABLE" not in sql
        assert "a'; DROP TABLE x; --" in params

    def test_unknown_cast_is_rejected(self, contracts):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("x", "eq", 1, json_key=True, cast="numeric); DROP --"),),
        )
        with pytest.raises(PlanError, match="cast"):
            _sql(plan, contracts)


class TestSetOperators:
    def test_in_compiles_to_a_single_any_predicate(self, contracts):
        """'delayed or cancelled' is one filter, not a plan per value."""
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("status", "in", ["Delayed", "Cancelled"]),),
            group_by=("entity_key",),
        )
        sql, params = _sql(plan, contracts)

        assert "= ANY(%s)" in sql
        assert ["Delayed", "Cancelled"] in params

    def test_not_in_compiles_to_all(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("status", "not_in", ["Complete"]),),
            group_by=("entity_key",),
        )
        sql, _ = _sql(plan, contracts)

        assert "<> ALL(%s)" in sql

    def test_an_empty_set_is_refused(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("status", "in", []),),
            group_by=("entity_key",),
        )
        with pytest.raises(PlanError, match="non-empty"):
            _sql(plan, contracts)

    def test_a_scalar_value_for_a_set_operator_is_refused(self, contracts):
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("status", "in", "Delayed"),),
            group_by=("entity_key",),
        )
        with pytest.raises(PlanError, match="non-empty list"):
            _sql(plan, contracts)

    def test_set_values_are_bound_not_interpolated(self, contracts):
        hostile = "x'); DROP TABLE excel_records; --"
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="select",
            filters=(Filter("status", "in", [hostile]),),
            group_by=("entity_key",),
        )
        sql, params = _sql(plan, contracts)

        assert "DROP TABLE" not in sql
        assert [hostile] in params


class TestMetadataScopeParity:
    """The scope queries must accept every filter the main statement does.

    They compiled filters independently and drifted: an operator that worked in
    the result raised KeyError while gathering that result's own unit and
    provenance metadata.
    """

    @pytest.mark.parametrize(
        "flt",
        [
            Filter("status", "in", ["Delayed", "Cancelled"]),
            Filter("annual_quant_target", "eq", 0, json_key=True, cast="numeric"),
            Filter("reporting_year", "gte", 2023),
        ],
    )
    def test_record_scope_compiles_every_supported_filter(self, contracts, flt):
        from retrieval.query.excel.query import _record_scope

        plan = ExcelQueryPlan(table_number=1, source=RECORDS, filters=(flt,))
        where, params = _record_scope(plan, contracts.for_table(1))

        assert where and params

    def test_fact_scope_compiles_a_set_filter(self, contracts):
        from retrieval.query.excel.query import _fact_scope

        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            filters=(Filter("expense_type", "in", ["CAPEX", "OPEX"], json_key=True),),
        )
        where, params = _fact_scope(plan, contracts.for_table(11))

        assert "ANY" in where
        assert ["CAPEX", "OPEX"] in params


class TestFactAggregateSemantics:
    """Counting fact rows counts reporting slots, not events."""

    def test_count_on_a_fact_table_becomes_a_sum(self):
        from retrieval.query.excel.nl_planner import _sanitize_plan

        contexts = {
            6: {
                "source": FACTS,
                "semantic_keys": ["vegetation_contact"],
                "json_keys": [],
                "vocabulary": {},
                "full_vocabulary": {},
                "title": "Ignitions by driver",
            }
        }
        plan = _sanitize_plan(
            {
                "action": "plan",
                "table_number": 6,
                "operation": "aggregate",
                "aggregate": "count",
                "semantic_metric_key": "vegetation_contact",
                "filters": [{"field": "reporting_year", "operator": "eq", "value": 2024}],
            },
            "how many ignitions in 2024",
            [(6, "card", None)],
            contexts,
        )

        assert plan.aggregate == "sum"

    def test_count_on_a_record_table_is_left_alone(self):
        from retrieval.query.excel.nl_planner import _sanitize_plan

        contexts = {
            13: {
                "source": RECORDS,
                "semantic_keys": [],
                "json_keys": ["equipment_type"],
                "vocabulary": {},
                "full_vocabulary": {},
                "title": "Open work orders",
            }
        }
        plan = _sanitize_plan(
            {
                "action": "plan",
                "table_number": 13,
                "operation": "aggregate",
                "aggregate": "count",
                "filters": [{"field": "reporting_year", "operator": "eq", "value": 2025}],
            },
            "how many work orders in 2025",
            [(13, "card", None)],
            contexts,
        )

        assert plan.aggregate == "count"


class TestCompiledStatementIsWellFormed:
    """Structural invariants that must hold for every plan the DSL accepts."""

    _HOSTILE = [
        "x'; DROP TABLE excel_facts; --",
        "a) OR 1=1 --",
        'a"b',
        "a\\b",
        "'; COMMIT; --",
        "%s",
    ]

    @pytest.mark.parametrize("hostile", _HOSTILE)
    @pytest.mark.parametrize("source,table", [(RECORDS, 1), (FACTS, 11)])
    def test_every_placeholder_has_exactly_one_parameter(
        self, contracts, hostile, source, table
    ):
        """Arity is the real injection invariant.

        A field name that reached the statement text instead of the parameter
        list would change the placeholder count. Checking `hostile not in sql`
        cannot catch a value that is itself "%s"; counting can.
        """
        plans = [
            ExcelQueryPlan(
                table_number=table,
                source=source,
                filters=(Filter(hostile, "eq", "v", json_key=True),),
            ),
            ExcelQueryPlan(
                table_number=table,
                source=source,
                filters=(Filter(hostile, "in", [hostile], json_key=True),),
            ),
            ExcelQueryPlan(
                table_number=table,
                source=source,
                filters=(Filter(hostile, "gt", 1, json_key=True, cast="numeric"),),
            ),
            ExcelQueryPlan(
                table_number=table,
                source=source,
                aggregate="count",
                group_by=(GroupKey(hostile, json_key=True),),
            ),
        ]
        for plan in plans:
            try:
                sql, params = _sql(plan, contracts)
            except PlanError:
                continue
            assert sql.count("%s") == len(params)
            if hostile != "%s":
                assert hostile not in sql

    def test_an_arithmetic_aggregate_with_no_target_is_refused(self, contracts):
        """sum(*) is not SQL; refuse rather than send it to the database."""
        plan = ExcelQueryPlan(table_number=1, source=RECORDS, aggregate="sum")

        with pytest.raises(PlanError, match="value_json_key"):
            _sql(plan, contracts)

    def test_count_star_on_records_remains_valid(self, contracts):
        plan = ExcelQueryPlan(table_number=1, source=RECORDS, aggregate="count")
        sql, _ = _sql(plan, contracts)

        assert "count(*)" in sql


@pytest.mark.integration
class TestRecordUnitSafety:
    """A record aggregate must not add quantities of different kinds.

    Table 1 keeps the unit in a sibling attribute rather than a typed column,
    so the fact tables' unit guard never saw it. Summing annual_quant_target
    across the table added Inspections, Structures, Poles, Trees, Miles and
    Capacitors -- 23 distinct units -- into 2,053,923.98, executed cleanly, and
    was reported as an answer.
    """

    def _conn(self):
        try:
            from retrieval.utils import connect_db

            return connect_db()
        except Exception:  # pragma: no cover - environment dependent
            pytest.skip("no database available")

    def test_summing_across_units_is_refused(self, contracts):
        from retrieval.query.excel.query import execute_plan

        connection = self._conn()
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="aggregate",
            aggregate="sum",
            value_json_key="annual_quant_target",
        )
        try:
            with pytest.raises(PlanError, match="incompatible units"):
                execute_plan(plan, connection, contracts=contracts)
        finally:
            connection.close()

    def test_grouping_by_the_entity_keeps_each_row_single_unit(self, contracts):
        from retrieval.query.excel.query import execute_plan

        connection = self._conn()
        plan = ExcelQueryPlan(
            table_number=1,
            source=RECORDS,
            operation="aggregate",
            aggregate="sum",
            value_json_key="annual_quant_target",
            group_by=("entity_key",),
            limit=5,
        )
        try:
            result = execute_plan(plan, connection, contracts=contracts)
        finally:
            connection.close()

        assert result.rows

    def test_a_count_is_never_unit_sensitive(self, contracts):
        from retrieval.query.excel.query import execute_plan

        connection = self._conn()
        plan = ExcelQueryPlan(
            table_number=1, source=RECORDS, operation="aggregate", aggregate="count"
        )
        try:
            result = execute_plan(plan, connection, contracts=contracts)
        finally:
            connection.close()

        assert result.rows


class TestAggregateTargetIsTyped:
    """``sum``/``avg`` over a text column must never become SQL.

    Found live on the 768d bge corpus: the planner answered "how much did we
    spend on capital and O&M in 2024" with a follow-up plan whose
    ``value_column`` was ``unit``. That compiled to ``sum(t.unit)``, which
    Postgres rejects with a bare ProgrammingError rather than a PlanError --
    so it escaped the per-plan handler, aborted the transaction, and took
    every later plan for the same question down with it.
    """

    @pytest.mark.parametrize("aggregate", ["sum", "avg"])
    @pytest.mark.parametrize("column", ["unit", "measure_name", "metric_name"])
    def test_arithmetic_over_a_text_column_is_refused(
        self, contracts, aggregate, column
    ):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate=aggregate,
            value_column=column,
            filters=(Filter("reporting_year", "eq", 2024),),
        )
        with pytest.raises(PlanError, match="numeric"):
            _sql(plan, contracts)

    @pytest.mark.parametrize("aggregate", ["sum", "avg"])
    def test_the_default_fact_measure_still_compiles(self, contracts, aggregate):
        """Naming no column falls through to the fact table's numeric measure."""
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate=aggregate,
            filters=(Filter("reporting_year", "eq", 2024),),
        )
        sql, _ = _sql(plan, contracts)
        assert f"{aggregate}(t.value_numeric)" in sql

    @pytest.mark.parametrize("aggregate", ["sum", "avg"])
    def test_an_allowlisted_numeric_column_still_compiles(self, contracts, aggregate):
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate=aggregate,
            value_column="reporting_quarter",
            filters=(Filter("reporting_year", "eq", 2024),),
        )
        sql, _ = _sql(plan, contracts)
        assert f"{aggregate}(t.reporting_quarter)" in sql

    @pytest.mark.parametrize("aggregate", ["min", "max", "count", "count_distinct"])
    def test_non_arithmetic_aggregates_over_text_are_still_allowed(
        self, contracts, aggregate
    ):
        """``max(unit)`` returns a value that exists in a row; it is not arithmetic."""
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate=aggregate,
            value_column="unit",
            filters=(Filter("reporting_year", "eq", 2024),),
        )
        sql, _ = _sql(plan, contracts)
        assert "t.unit" in sql


class TestTruncationIsDisclosed:
    """A result cut off at its limit must say so, in the result itself.

    The executor asks for ``limit + 1`` rows and returns ``limit``: the extra
    row is a probe, never evidence. Without this the answering model read the
    eight largest initiatives as the complete set and either presented a
    partial list as exhaustive or reported "insufficient context".
    """

    def test_the_compiled_statement_still_carries_the_plan_limit(self, contracts):
        """The probe happens at execution; compilation is unchanged."""
        plan = ExcelQueryPlan(
            table_number=11,
            source=FACTS,
            operation="aggregate",
            aggregate="sum",
            filters=(Filter("reporting_year", "eq", 2024),),
            group_by=(GroupKey("initiative", json_key=True),),
            limit=8,
        )
        sql, params = _sql(plan, contracts)
        assert sql.rstrip().endswith("LIMIT %s")
        assert params[-1] == 8
