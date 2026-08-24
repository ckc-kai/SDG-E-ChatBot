"""Validated, parameterized execution of Excel fact/record queries.

An ``ExcelQueryPlan`` is data, never SQL. Every identifier that reaches the
compiled statement comes from a code allowlist keyed on the reviewed contract;
every value travels as a psycopg2 parameter. There is no string interpolation of
caller-supplied text anywhere in this module.

The executor is also the Excel routing signal used by the retrieval gate: a plan
that validates, compiles, and returns rows is positive evidence that the question
was answerable from the spreadsheets. See ``docs/retrieval_ranking_fix_plan.md``
Phase 4.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal

from retrieval.ingest.excel.contracts import (
    ContractSet,
    TableContract,
    load_contracts,
)
from generation.features import feature_enabled
from retrieval.query.excel.scopes import overlapping_scope, scope_is_pinned

logger = logging.getLogger(__name__)

FACTS = "excel_facts"
RECORDS = "excel_records"
SOURCES = (FACTS, RECORDS)

AGGREGATES = {
    "sum": "sum({col})",
    "avg": "avg({col})",
    "count": "count({col})",
    "min": "min({col})",
    "max": "max({col})",
    # A blank is a reportable fact, not an absence to be dropped. Table 13
    # leaves 86.9% of its GO 95 priority field unpopulated, and an answer that
    # reports the populated counts without the blank cohort misrepresents the
    # data more than one that reports nothing.
    "count_null": "count(*) FILTER (WHERE {col} IS NULL)",
    "count_distinct": "count(DISTINCT {col})",
}
ORDERING_OPERATORS = {"gt", "gte", "lt", "lte"}
OPERATORS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
# "delayed or cancelled", "Level 2 or Level 3". Expressing these as a set of
# ``ne`` complements needed the full vocabulary up front and silently included
# any value the vocabulary had missed.
SET_OPERATORS = {"in": "= ANY(%s)", "not_in": "<> ALL(%s)"}
MAX_SET_VALUES = 25

# jsonb ``->>`` yields text. Comparing it without a cast orders '2.0' above
# '1000' and reads '0' as unequal to a stored '0.0', so a plan that looked
# validated returned rows matching nothing the caller asked for -- and the
# channel reported that as execution-verified. Casts are a fixed code
# allowlist; nothing caller-supplied ever reaches the statement.
CASTS = {"numeric": "::numeric", "date": "::date", "text": ""}
# Guards the cast against rows whose attribute is blank or non-numeric, which
# would otherwise abort the whole statement with a conversion error.
_NUMERIC_GUARD = r"^-?[0-9]+(\.[0-9]+)?$"
# Record tables keep the unit of a numeric attribute in a sibling attribute
# rather than in a typed column, so the fact tables' unit guard never saw them.
# Summing ``annual_quant_target`` across table 1 added Inspections, Structures,
# Poles, Trees, Miles and Capacitors -- 23 distinct units -- into one number
# and reported it as an answer.
_UNIT_ATTRIBUTE_RE = re.compile(r"(^|_)units?$", re.I)

# Typed columns that may be filtered, grouped, or ordered on. Anything else must
# be addressed through the jsonb payload, by parameterized key.
FACT_COLUMNS = {
    "table_number",
    "record_id",
    "source_metric_number",
    "series_id",
    "semantic_metric_key",
    "metric_name",
    "measure_name",
    "utility_id",
    "reporting_year",
    "reporting_quarter",
    "source_vintage_year",
    "year_basis",
    "period_end_date",
    "hftd_tier",
    "line_type",
    "unit",
}
RECORD_COLUMNS = {
    "table_number",
    "record_id",
    "entity_key",
    "entity_type",
    "title",
    "utility_id",
    "reporting_year",
    "reporting_quarter",
    "hftd_tier",
    "line_type",
    "date_start",
    "date_due",
    "date_end",
    "status",
}
JSON_COLUMN = {FACTS: "dimensions", RECORDS: "attributes"}
VALUE_COLUMN = {FACTS: "value_numeric", RECORDS: None}

# ``sum`` and ``avg`` are arithmetic: naming a text column as their target
# compiles to ``sum(t.unit)``, which Postgres rejects at execution time with a
# bare ProgrammingError rather than a PlanError -- aborting the transaction and
# taking every later plan in the same question down with it. The DSL knows the
# column types, so refuse it while it is still a plan.
ARITHMETIC_AGGREGATES = {"sum", "avg"}
NUMERIC_COLUMNS = {
    "value_numeric",
    "reporting_year",
    "reporting_quarter",
    "source_vintage_year",
}

MAX_LIMIT = 200
DEFAULT_LIMIT = 50
# Tables 14/15 carry two year axes; a bare year filter is ambiguous there.
DUAL_YEAR_AXIS_TABLES = {14, 15}


class PlanError(ValueError):
    """The plan is malformed, unsafe, or ambiguous. Never becomes SQL."""


class ClarificationNeeded(PlanError):
    """The plan is well-formed but the question is under-specified."""


@dataclass(frozen=True)
class Filter:
    field: str
    operator: str = "eq"
    value: Any = None
    json_key: bool = False
    # Only meaningful with json_key: how to read the text jsonb returns.
    cast: str | None = None


@dataclass(frozen=True)
class GroupKey:
    """A grouping column, which may live inside the jsonb payload.

    Spend is grouped by initiative and expense type; work orders by priority.
    Those are jsonb keys, so before this existed the planner could only group
    by ``record_id`` -- opaque hashes -- and "our biggest capital programs"
    came back as a ranked list of meaningless identifiers.
    """

    field: str
    json_key: bool = False
    cast: str | None = None


@dataclass(frozen=True)
class Having:
    """A predicate on the aggregate, for 'beat the total but missed a year'."""

    operator: str = "gt"
    value: Any = 0


@dataclass(frozen=True)
class ExcelQueryPlan:
    table_number: int
    source: Literal["excel_facts", "excel_records"] = FACTS
    semantic_metric_key: str | None = None
    measure_name: str | None = None
    operation: Literal["aggregate", "select", "rank"] = "aggregate"
    aggregate: str = "sum"
    filters: tuple[Filter, ...] = ()
    group_by: tuple[Any, ...] = ()
    # Aggregate a numeric attribute held in the record jsonb payload.
    # ``excel_records`` has no typed value column, so before this the only
    # aggregate available over tables 1 and 13 was count(*).
    value_json_key: str | None = None
    # Aggregate (or count nulls of) an allowlisted typed column.
    value_column: str | None = None
    having: Having | None = None
    # Reviewed JSON attributes to return for entity-record lookups. Keys remain
    # SQL parameters; generated aliases are fixed and never user controlled.
    select_json_keys: tuple[str, ...] = ()
    order_by: str | None = None
    descending: bool = True
    limit: int = DEFAULT_LIMIT
    require_year_basis: bool = True


@dataclass
class ExcelExecutionResult:
    plan: ExcelQueryPlan
    revision_id: int | None
    columns: list[str]
    rows: list[tuple]
    unit: str | None
    contributing_facts: int
    provenance: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blank_meanings: list[str] = field(default_factory=list)
    # The statement that actually ran. Carried so an evaluation run can tell a
    # figure that came from an executed query apart from one the answering
    # model read off a card, which was previously indistinguishable.
    sql: str | None = None
    sql_params: list[Any] = field(default_factory=list)

    @property
    def is_answer(self) -> bool:
        """Non-empty, non-null result: the Excel routing signal."""
        if not self.rows:
            return False
        return any(cell is not None for row in self.rows for cell in row[-1:])

    def scalar(self) -> Decimal | None:
        if not self.rows:
            return None
        value = self.rows[0][-1]
        return value if isinstance(value, Decimal) else value


def _columns_for(source: str) -> set[str]:
    return FACT_COLUMNS if source == FACTS else RECORD_COLUMNS


def _as_group_key(entry: Any) -> GroupKey:
    """Accept a bare column name or a GroupKey, so old plans still compile."""
    return entry if isinstance(entry, GroupKey) else GroupKey(str(entry))


def _cast_suffix(cast: str | None) -> str:
    """Resolve a cast name against the code allowlist. Never interpolated raw."""
    if cast is None:
        return ""
    if cast not in CASTS:
        raise PlanError(f"unknown cast {cast!r}")
    return CASTS[cast]


def _compile_filter(
    flt: Filter, alias: str, json_column: str, params: list[Any]
) -> str:
    """One filter as a parameterized predicate.

    Shared by the main statement and by the unit/provenance scope queries.
    They used to compile filters separately, and the copies drifted: a new
    operator worked in the result but raised KeyError while gathering the
    metadata for that same result.
    """
    if flt.json_key:
        expression = _json_expression(
            f"{alias}.{json_column}", flt.cast, params, flt.field
        )
    else:
        expression = f"{alias}.{flt.field}"
    if flt.operator in SET_OPERATORS:
        values = list(flt.value)
        if flt.json_key and not flt.cast:
            values = [str(item) for item in values]
        params.append(values)
        return f"{expression} {SET_OPERATORS[flt.operator]}"
    if flt.json_key:
        params.append(flt.value if flt.cast else str(flt.value))
    else:
        params.append(flt.value)
    return f"{expression} {OPERATORS[flt.operator]} %s"


def _json_expression(
    json_column: str, cast: str | None, params: list[Any], field_name: str
) -> str:
    """``t.payload ->> %s`` read at the requested type, key bound as a parameter.

    A numeric cast is guarded by a regex test so a blank or non-numeric
    attribute yields NULL -- and so fails the predicate -- instead of aborting
    the whole statement with a conversion error.
    """
    suffix = _cast_suffix(cast)
    if cast == "numeric":
        params.extend([field_name, _NUMERIC_GUARD, field_name])
        return (
            f"(CASE WHEN {json_column} ->> %s ~ %s "
            f"THEN {json_column} ->> %s END)::numeric"
        )
    params.append(field_name)
    return f"({json_column} ->> %s){suffix}" if suffix else f"{json_column} ->> %s"


def _validate(plan: ExcelQueryPlan, contract: TableContract) -> None:
    if plan.source not in SOURCES:
        raise PlanError(f"unknown source {plan.source!r}")
    if plan.aggregate not in AGGREGATES:
        raise PlanError(f"unknown aggregate {plan.aggregate!r}")
    if not 1 <= plan.limit <= MAX_LIMIT:
        raise PlanError(f"limit must be within 1..{MAX_LIMIT}")

    allowed = _columns_for(plan.source)
    for flt in plan.filters:
        if flt.operator in SET_OPERATORS:
            if not isinstance(flt.value, (list, tuple)) or not flt.value:
                raise PlanError(f"{flt.operator!r} needs a non-empty list")
            if len(flt.value) > MAX_SET_VALUES:
                raise PlanError(f"{flt.operator!r} list exceeds {MAX_SET_VALUES}")
        elif flt.operator not in OPERATORS:
            raise PlanError(f"unknown operator {flt.operator!r}")
        if not flt.json_key and flt.field not in allowed:
            raise PlanError(f"{flt.field!r} is not a filterable column")
        if flt.json_key:
            _cast_suffix(flt.cast)
            # Text ordering puts '2.0' above '1000'. A caller asking for a
            # magnitude comparison on a jsonb attribute always means the
            # number, so refuse rather than silently answer a different
            # question -- this was returning 102 rows where 41 were correct.
            if flt.operator in ORDERING_OPERATORS and flt.cast in (None, "text"):
                raise PlanError(
                    f"{flt.field!r} uses {flt.operator!r} on a jsonb attribute "
                    "without a cast; text ordering is not numeric ordering"
                )
        if flt.cast is not None and not flt.json_key:
            raise PlanError("cast applies only to jsonb attributes")

    for entry in plan.group_by:
        key = _as_group_key(entry)
        _cast_suffix(key.cast)
        if not key.json_key and key.field not in allowed:
            raise PlanError(f"{key.field!r} is not groupable")
    if plan.order_by and plan.order_by not in allowed | {"value", "aggregate"}:
        raise PlanError(f"{plan.order_by!r} is not orderable")

    if plan.value_column is not None and plan.value_column not in allowed:
        raise PlanError(f"{plan.value_column!r} is not an aggregable column")
    if (
        plan.value_column is not None
        and plan.aggregate in ARITHMETIC_AGGREGATES
        and plan.value_column not in NUMERIC_COLUMNS
    ):
        raise PlanError(
            f"{plan.aggregate!r} needs a numeric column; "
            f"{plan.value_column!r} is not one"
        )
    if plan.value_json_key is not None and plan.value_column is not None:
        raise PlanError("value_json_key and value_column are mutually exclusive")
    if plan.value_json_key is not None and plan.source != RECORDS:
        raise PlanError("value_json_key is supported only for entity records")
    if plan.having is not None:
        if plan.having.operator not in OPERATORS:
            raise PlanError(f"unknown having operator {plan.having.operator!r}")
        if not plan.group_by:
            raise PlanError("having requires a group_by")
        if plan.operation == "select":
            raise PlanError("having requires an aggregate operation")
    if plan.select_json_keys and plan.operation != "select":
        raise PlanError("select_json_keys requires operation='select'")
    if plan.select_json_keys and plan.source != RECORDS:
        raise PlanError("select_json_keys is supported only for entity records")

    # Tables 14/15: refuse a bare reporting-year filter, because the same year
    # means two different things depending on year_basis.
    if plan.table_number in DUAL_YEAR_AXIS_TABLES and plan.require_year_basis:
        fields = {f.field for f in plan.filters}
        if "reporting_year" in fields and not (
            {"year_basis", "source_vintage_year"} & fields
        ):
            raise ClarificationNeeded(
                f"Table {plan.table_number} reports a submission vintage year and "
                "a separate reporting year. Specify year_basis or "
                "source_vintage_year, or ask about a specific submission."
            )


def compile_plan(
    plan: ExcelQueryPlan, contract: TableContract
) -> tuple[str, list[Any]]:
    """Return parameterized SQL and its parameters. Identifiers are allowlisted."""
    _validate(plan, contract)
    source = plan.source
    json_column = JSON_COLUMN[source]
    params: list[Any] = []

    where = ["r.state = 'active'", "t.table_number = %s"]
    params.append(plan.table_number)
    if plan.semantic_metric_key and source == FACTS:
        where.append("t.semantic_metric_key = %s")
        params.append(plan.semantic_metric_key)
    if plan.measure_name and source == FACTS:
        where.append("t.measure_name = %s")
        params.append(plan.measure_name)
    for flt in plan.filters:
        where.append(_compile_filter(flt, "t", json_column, params))

    # Contract-declared duplicate policy (Table 13 snapshots repeat work orders).
    dedupe = contract.dedupe.get("default_filter") if contract.dedupe else None
    if dedupe and source == RECORDS:
        where.append(f"t.{json_column} ->> %s = %s")
        params.extend([dedupe["field"], str(dedupe["value"])])

    where_sql = " AND ".join(where)

    # Group keys are compiled before the SELECT list so their bound parameters
    # keep source order; jsonb keys become generated aliases, never caller text.
    group_params: list[Any] = []
    group_expressions: list[str] = []
    group_aliases: list[str] = []
    for index, entry in enumerate(plan.group_by):
        key = _as_group_key(entry)
        if key.json_key:
            expression = _json_expression(
                f"t.{json_column}", key.cast, group_params, key.field
            )
            alias = f"group_{index}"
        else:
            expression = f"t.{key.field}"
            alias = key.field
        group_expressions.append(expression)
        group_aliases.append(alias)
    group_select = ", ".join(
        expression if expression == f"t.{alias}" else f"{expression} AS {alias}"
        for expression, alias in zip(group_expressions, group_aliases)
    )
    # Group and order by select-list ordinal rather than repeating the
    # expression. A jsonb group key binds its key name as a parameter, and
    # repeating the expression would bind it twice, in an order that is easy
    # to get wrong and impossible to see in the compiled string.
    group_ordinals = ", ".join(str(index + 1) for index in range(len(group_expressions)))
    where_params = params
    params = [*group_params, *where_params]

    value_column = VALUE_COLUMN[source]

    if plan.operation == "select":
        cols = group_select or "t.record_id"
        order_cols = group_ordinals or "1"
        tail = f", t.{value_column}" if value_column else ""
        json_select = "".join(
            f", t.{json_column} ->> %s AS selected_{index}"
            for index, _ in enumerate(plan.select_json_keys)
        )
        sql = (
            f"SELECT {cols}{tail}{json_select} FROM {source} t "
            f"JOIN excel_revisions r ON r.id = t.revision_id "
            f"WHERE {where_sql} ORDER BY {order_cols} LIMIT %s"
        )
        return sql, [*plan.select_json_keys, *params, plan.limit]

    # Aggregate target, in precedence order: an explicit jsonb attribute, an
    # explicit typed column, then the source's natural value column.
    agg_params: list[Any] = []
    if plan.value_json_key is not None:
        cast = None if plan.aggregate.startswith("count") else "numeric"
        target_sql = _json_expression(
            f"t.{json_column}", cast, agg_params, plan.value_json_key
        )
    elif plan.value_column is not None:
        target_sql = f"t.{plan.value_column}"
    elif value_column:
        target_sql = f"t.{value_column}"
    else:
        target_sql = "*"
    agg_sql = AGGREGATES[plan.aggregate].format(col=target_sql)
    if plan.aggregate == "count" and target_sql == "*":
        agg_sql = "count(*)"
    elif target_sql == "*":
        # sum(*)/avg(*) are not SQL. A record table has no typed value column,
        # so an arithmetic aggregate there needs value_json_key or
        # value_column; without one the plan is malformed, and refusing here
        # keeps a broken statement from reaching the database.
        raise PlanError(
            f"aggregate {plan.aggregate!r} needs value_json_key or value_column "
            f"on {plan.source}"
        )

    select_parts = ([group_select] if group_select else []) + [f"{agg_sql} AS value"]
    sql = (
        f"SELECT {', '.join(select_parts)} FROM {source} t "
        f"JOIN excel_revisions r ON r.id = t.revision_id "
        f"WHERE {where_sql}"
    )
    # Placeholder order follows the statement text: group keys and the
    # aggregate target appear in the SELECT list, then the WHERE predicates.
    params = [*group_params, *agg_params, *where_params]
    if group_ordinals:
        sql += f" GROUP BY {group_ordinals}"
    if plan.having is not None:
        # HAVING cannot reference a select alias in PostgreSQL, so the
        # aggregate expression -- and its parameters -- are emitted again.
        sql += f" HAVING {agg_sql} {OPERATORS[plan.having.operator]} %s"
        having_params: list[Any] = []
        if plan.value_json_key is not None:
            cast = None if plan.aggregate.startswith("count") else "numeric"
            _json_expression(
                f"t.{json_column}", cast, having_params, plan.value_json_key
            )
        params.extend([*having_params, plan.having.value])
    if plan.operation == "rank" or plan.order_by:
        direction = "DESC" if plan.descending else "ASC"
        sql += f" ORDER BY value {direction} NULLS LAST"
    elif group_ordinals:
        sql += f" ORDER BY {group_ordinals}"
    sql += " LIMIT %s"
    params.append(plan.limit)
    return sql, params


def _apply_canonical_scope(plan: ExcelQueryPlan) -> tuple[ExcelQueryPlan, list[str]]:
    """Pin an overlapping scope column the plan left open.

    Table 11 reports every WMP dollar twice -- once scoped ``Territory`` and
    once scoped to the ``HFTD`` subset inside it. A plan that neither filters
    nor groups that column sums both and double-counts the overlap: this is
    ``excel_003``, where 2024 CAPEX came back as 932,371 instead of 473,986.

    An unscoped aggregate is rewritten to the canonical whole rather than
    refused, because "how much did we spend" does have a right answer --
    the territory figure -- and refusing returns nothing at all. The rewrite
    is reported as a warning so the trace and the evidence both show which
    scope was actually read.
    """
    if not feature_enabled("excel_canonical_scope"):
        return plan, []
    scope = overlapping_scope(plan.table_number)
    if scope is None or plan.source != FACTS:
        return plan, []
    if plan.operation == "select" or scope_is_pinned(scope, plan):
        return plan, []
    if plan.aggregate not in {"sum", "avg"}:
        # count/min/max over the union do not double-count a magnitude.
        return plan, []
    scoped = replace(
        plan,
        filters=(*plan.filters, Filter(scope.column, "eq", scope.canonical)),
    )
    return scoped, [
        f"{scope.column} values overlap ({', '.join(scope.values)}); "
        f"scoped to the canonical {scope.canonical!r} rather than summing "
        f"across them, which would double-count "
        f"{', '.join(scope.contained)}."
    ]


def _resolve_unambiguous_year_basis(plan: ExcelQueryPlan, conn) -> ExcelQueryPlan:
    """Skip the dual-axis clarification when the requested year is unambiguous.

    Tables 14/15 store 2023/2024 vintages whose reporting year equals the
    vintage year, and the 2025 vintage reports risk year 2026. A bare
    reporting-year filter that maps to exactly one active vintage cannot be
    misread, so it executes; a year with zero or several matching vintages
    still raises the clarification.
    """
    if plan.table_number not in DUAL_YEAR_AXIS_TABLES or not plan.require_year_basis:
        return plan
    fields = {flt.field for flt in plan.filters}
    if "reporting_year" not in fields or {"year_basis", "source_vintage_year"} & fields:
        return plan
    years = [
        flt.value
        for flt in plan.filters
        if flt.field == "reporting_year" and flt.operator == "eq"
    ]
    if len(years) != 1:
        return plan
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT f.source_vintage_year
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state = 'active'
            WHERE f.table_number = %s AND f.reporting_year = %s
            """,
            (plan.table_number, years[0]),
        )
        vintages = [row[0] for row in cur.fetchall()]
    if len(vintages) == 1:
        return replace(plan, require_year_basis=False)
    return plan


def execute_plan(
    plan: ExcelQueryPlan,
    conn,
    *,
    contracts: ContractSet | None = None,
) -> ExcelExecutionResult:
    contracts = contracts or load_contracts()
    contract = contracts.for_table(plan.table_number)
    plan = _resolve_unambiguous_year_basis(plan, conn)
    plan, scope_warnings = _apply_canonical_scope(plan)
    sql, params = compile_plan(plan, contract)

    with conn.cursor() as cur:
        # Ask for one row beyond the limit. If it comes back, the result is a
        # truncated head of a larger set, and saying so is the difference
        # between "the eight largest initiatives" and a list the answering
        # model presents as complete. 9 of 22 executed results on the beta set
        # sat exactly at their limit with nothing in the evidence to say so.
        # ``compile_plan`` always ends with an unconditional " LIMIT %s", so
        # the row cap is the final parameter. Checking rather than assuming
        # means a future change to the compiler degrades to "no truncation
        # probe" instead of silently rewriting some other parameter.
        probe_params = list(params)
        can_probe = bool(probe_params) and probe_params[-1] == plan.limit
        if can_probe:
            probe_params[-1] = plan.limit + 1
        cur.execute(sql, probe_params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        truncated = can_probe and len(rows) > plan.limit
        rows = rows[: plan.limit]

        unit = None
        mixed_units: list[str] = []
        contributing = 0
        blanks: list[str] = []
        revision_id = None
        provenance: list[dict[str, Any]] = []
        if plan.source == FACTS:
            meta_where, meta_params = _fact_scope(plan, contract)
            cur.execute(
                f"""
                SELECT count(f.value_numeric),
                       array_agg(DISTINCT f.unit) FILTER (WHERE f.unit IS NOT NULL),
                       array_agg(DISTINCT f.blank_meaning)
                         FILTER (WHERE f.value_numeric IS NULL
                                   AND f.blank_meaning IS NOT NULL),
                       min(f.revision_id)
                FROM excel_facts f
                JOIN excel_revisions r ON r.id = f.revision_id
                WHERE {meta_where}
                """,
                meta_params,
            )
            contributing, units, blank_list, revision_id = cur.fetchone()
            units = units or []
            blanks = list(blank_list or [])
            if len(units) > 1:
                # Mixed units are only a hazard when one number is the sum of
                # all of them. A plan that groups by the metric -- or by any
                # dimension that separates them -- returns one row per unit,
                # each labeled, which is a legitimate breakdown rather than an
                # apples-and-oranges total. Refusing those declined questions
                # the corpus answers exactly.
                separating_columns = {
                    "semantic_metric_key", "metric_name", "measure_name", "unit"
                }
                separates_units = any(
                    (entry.field if isinstance(entry, GroupKey) else str(entry))
                    in separating_columns
                    for entry in plan.group_by
                ) or plan.operation == "select"
                if not separates_units:
                    raise PlanError(
                        "refusing to aggregate incompatible units: "
                        + ", ".join(units)
                    )
                mixed_units = sorted(units)
            unit = units[0] if len(units) == 1 else None
            cur.execute(
                f"""
                SELECT f.provenance, f.value_raw
                FROM excel_facts f
                JOIN excel_revisions r ON r.id = f.revision_id
                WHERE {meta_where} AND f.value_numeric IS NOT NULL
                LIMIT 5
                """,
                meta_params,
            )
            provenance = [
                {**(row[0] or {}), "value_raw": row[1]} for row in cur.fetchall()
            ]
        else:
            mixed_units = []
            meta_where, meta_params = _record_scope(plan, contract)
            if plan.value_json_key and plan.aggregate in _UNIT_SENSITIVE_AGGREGATES:
                mixed_units = _record_units_in_scope(
                    cur, plan, contract, meta_where, meta_params
                )
                if len(mixed_units) > 1 and not _groups_separate_units(plan):
                    raise PlanError(
                        "refusing to aggregate "
                        f"{plan.value_json_key!r} across incompatible units: "
                        + ", ".join(mixed_units[:8])
                        + (" ..." if len(mixed_units) > 8 else "")
                    )
            cur.execute(
                f"""
                SELECT count(*), min(rec.revision_id)
                FROM excel_records rec
                JOIN excel_revisions r ON r.id = rec.revision_id
                WHERE {meta_where}
                """,
                meta_params,
            )
            contributing, revision_id = cur.fetchone()
            provenance_limit = plan.limit if plan.operation == "select" else 5
            provenance_order = ""
            if plan.operation == "select":
                order_columns = plan.group_by or ("record_id",)
                provenance_order = " ORDER BY " + ", ".join(
                    f"rec.{column}" for column in order_columns
                )
            cur.execute(
                f"""
                SELECT rec.provenance
                FROM excel_records rec
                JOIN excel_revisions r ON r.id = rec.revision_id
                WHERE {meta_where}
                {provenance_order}
                LIMIT %s
                """,
                [*meta_params, provenance_limit],
            )
            provenance = [row[0] or {} for row in cur.fetchall()]

    warnings: list[str] = list(scope_warnings)
    if truncated:
        warnings.append(
            f"Only the first {plan.limit} rows are shown and more exist; this "
            "is a partial view, not the complete set. Do not present it as "
            "exhaustive or total it as if it were."
        )
    if len(mixed_units) > 1:
        warnings.append(
            "This result spans several units ("
            + ", ".join(mixed_units)
            + "); read each row against its own metric and never total them."
        )
    if plan.table_number in DUAL_YEAR_AXIS_TABLES:
        warnings.append(
            "Table "
            f"{plan.table_number} separates submission vintage from reporting year; "
            "state both in any answer."
        )
    return ExcelExecutionResult(
        plan=plan,
        revision_id=revision_id,
        columns=columns,
        rows=rows,
        unit=unit,
        contributing_facts=contributing or 0,
        provenance=provenance,
        warnings=warnings,
        blank_meanings=blanks,
        sql=sql,
        sql_params=list(params),
    )


# min/max pick an existing row's value, so they stay meaningful across units
# as long as the unit travels with the answer; sum and avg do not.
_UNIT_SENSITIVE_AGGREGATES = {"sum", "avg"}


def _groups_separate_units(plan: ExcelQueryPlan) -> bool:
    """True when each returned row belongs to a single unit."""
    separating = {
        "semantic_metric_key", "metric_name", "measure_name", "unit",
        "entity_key", "record_id", "title",
    }
    for entry in plan.group_by:
        field = entry.field if isinstance(entry, GroupKey) else str(entry)
        if field in separating:
            return True
        if isinstance(entry, GroupKey) and entry.json_key and _UNIT_ATTRIBUTE_RE.search(field):
            return True
    return False


def _record_units_in_scope(
    cur,
    plan: ExcelQueryPlan,
    contract: TableContract,
    meta_where: str,
    meta_params: list[Any],
) -> list[str]:
    """Distinct unit labels covered by a record aggregate's scope."""
    unit_keys = [
        key for key in contract.json_dimensions if _UNIT_ATTRIBUTE_RE.search(key)
    ]
    if not unit_keys:
        return []
    cur.execute(
        f"""
        SELECT DISTINCT rec.attributes ->> %s
        FROM excel_records rec
        JOIN excel_revisions r ON r.id = rec.revision_id
        WHERE {meta_where}
          AND rec.attributes ->> %s IS NOT NULL
          AND rec.attributes ->> %s <> ''
        """,
        [unit_keys[0], *meta_params, unit_keys[0], unit_keys[0]],
    )
    return sorted(row[0] for row in cur.fetchall())


def _fact_scope(plan: ExcelQueryPlan, contract: TableContract) -> tuple[str, list[Any]]:
    """The same predicate as the main query, for unit/provenance metadata."""
    where = ["r.state = 'active'", "f.table_number = %s"]
    params: list[Any] = [plan.table_number]
    if plan.semantic_metric_key:
        where.append("f.semantic_metric_key = %s")
        params.append(plan.semantic_metric_key)
    if plan.measure_name:
        where.append("f.measure_name = %s")
        params.append(plan.measure_name)
    for flt in plan.filters:
        where.append(_compile_filter(flt, "f", "dimensions", params))
    return " AND ".join(where), params


def _record_scope(
    plan: ExcelQueryPlan,
    contract: TableContract,
) -> tuple[str, list[Any]]:
    """The record equivalent of ``_fact_scope`` for counts and provenance."""
    where = ["r.state = 'active'", "rec.table_number = %s"]
    params: list[Any] = [plan.table_number]
    for flt in plan.filters:
        where.append(_compile_filter(flt, "rec", "attributes", params))
    dedupe = contract.dedupe.get("default_filter") if contract.dedupe else None
    if dedupe:
        where.append("rec.attributes ->> %s = %s")
        params.extend([dedupe["field"], str(dedupe["value"])])
    return " AND ".join(where), params


# --------------------------------------------------------------------------
# Deterministic parameter binding from a question (no LLM, no classifier)
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20[2-3][0-9])\b")
_YEAR_RANGE_RE = re.compile(r"\b(20[2-3][0-9])\s*[-\u2013\u2014]\s*(20[2-3][0-9])\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_WMP_ENTITY_RE = re.compile(r"\bWMP\.\d+[A-Za-z]?\b", re.IGNORECASE)


def bind_years(question: str) -> tuple[int, ...]:
    """Return every explicitly requested year, expanding inclusive ranges."""
    years = {int(value) for value in _YEAR_RE.findall(question)}
    for start_text, end_text in _YEAR_RANGE_RE.findall(question):
        start, end = int(start_text), int(end_text)
        if start <= end and end - start <= 10:
            years.update(range(start, end + 1))
    return tuple(sorted(years))


def bind_entity_key(question: str) -> str | None:
    """Extract an exact WMP activity identifier without an LLM call."""
    match = _WMP_ENTITY_RE.search(question)
    return match.group(0).upper() if match else None


def bind_period(question: str) -> dict[str, int]:
    """Extract only a 4-digit year and Q1-Q4 token. Nothing else is bound."""
    bound: dict[str, int] = {}
    years = bind_years(question)
    if len(years) == 1:
        bound["reporting_year"] = years[0]
    quarters = _QUARTER_RE.findall(question)
    if len(set(quarters)) == 1:
        bound["reporting_quarter"] = int(quarters[0])
    return bound


def bind_dimensions(question: str, vocabulary: dict[str, list[str]]) -> dict[str, str]:
    """Match question text against the active revision's dimension vocabulary.

    Values come from the database, not the question, so an unmatched phrase can
    never reach SQL. Longer values are tested first so "HFTD Tier 3" wins over a
    substring of itself.

    Two guards matter here. Purely numeric vocabulary values are skipped: the
    corpus stores bookkeeping attributes such as ``exact_duplicate_index = '2'``,
    and a question mentioning "2024 Q2" would otherwise bind every one of them
    and silently filter the answer down to zero rows. Matching is also
    word-bounded, so "Pole" does not match "poles" inside another word.
    """
    lowered = question.lower()
    bound: dict[str, str] = {}
    for field_name, values in vocabulary.items():
        for value in sorted((v for v in values if v), key=len, reverse=True):
            candidate = value.strip()
            if len(candidate) < 3:
                continue
            # Bookkeeping counters and codes are not things a user names.
            if candidate.replace(".", "").replace("-", "").isdigit():
                continue
            if re.search(rf"(?<!\w){re.escape(candidate.lower())}(?!\w)", lowered):
                bound[field_name] = value
                break
    return bound


def dimension_vocabulary(
    conn, table_number: int, *, source: str = FACTS
) -> dict[str, list[str]]:
    """Distinct promoted and jsonb dimension values for one active table.

    Entity tables (1, 13) store their dimensions on ``excel_records.attributes``
    rather than ``excel_facts.dimensions``, so the source must be selectable or
    those tables silently return an empty vocabulary and nothing binds.
    """
    table = RECORDS if source == RECORDS else FACTS
    json_column = JSON_COLUMN[table]
    vocabulary: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        for column in ("hftd_tier", "line_type"):
            cur.execute(
                f"""
                SELECT DISTINCT f.{column} FROM {table} f
                JOIN excel_revisions r ON r.id = f.revision_id AND r.state='active'
                WHERE f.table_number = %s AND f.{column} IS NOT NULL
                """,
                (table_number,),
            )
            values = [row[0] for row in cur.fetchall()]
            if values:
                vocabulary[column] = values
        cur.execute(
            f"""
            SELECT key, array_agg(DISTINCT value)
            FROM (
                SELECT d.key, d.value
                FROM {table} f
                JOIN excel_revisions r
                  ON r.id = f.revision_id AND r.state = 'active'
                CROSS JOIN LATERAL jsonb_each_text(f.{json_column}) AS d(key, value)
                WHERE f.table_number = %s
            ) pairs
            GROUP BY key
            HAVING count(DISTINCT value) BETWEEN 2 AND 200
            """,
            (table_number,),
        )
        for key, values in cur.fetchall():
            vocabulary[key] = list(values)
    return vocabulary
