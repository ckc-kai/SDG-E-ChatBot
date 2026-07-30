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
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from retrieval.ingest.excel.contracts import (
    ContractSet,
    TableContract,
    load_contracts,
)

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
}
OPERATORS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}

# Typed columns that may be filtered, grouped, or ordered on. Anything else must
# be addressed through the jsonb payload, by parameterized key.
FACT_COLUMNS = {
    "table_number", "record_id", "source_metric_number", "series_id",
    "semantic_metric_key", "metric_name", "measure_name", "utility_id",
    "reporting_year", "reporting_quarter", "source_vintage_year", "year_basis",
    "period_end_date", "hftd_tier", "line_type", "unit",
}
RECORD_COLUMNS = {
    "table_number", "record_id", "entity_key", "entity_type", "title",
    "utility_id", "reporting_year", "reporting_quarter", "hftd_tier",
    "line_type", "date_start", "date_due", "date_end", "status",
}
JSON_COLUMN = {FACTS: "dimensions", RECORDS: "attributes"}
VALUE_COLUMN = {FACTS: "value_numeric", RECORDS: None}

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


@dataclass(frozen=True)
class ExcelQueryPlan:
    table_number: int
    source: Literal["excel_facts", "excel_records"] = FACTS
    semantic_metric_key: str | None = None
    measure_name: str | None = None
    operation: Literal["aggregate", "select", "rank"] = "aggregate"
    aggregate: str = "sum"
    filters: tuple[Filter, ...] = ()
    group_by: tuple[str, ...] = ()
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


def _validate(plan: ExcelQueryPlan, contract: TableContract) -> None:
    if plan.source not in SOURCES:
        raise PlanError(f"unknown source {plan.source!r}")
    if plan.aggregate not in AGGREGATES:
        raise PlanError(f"unknown aggregate {plan.aggregate!r}")
    if not 1 <= plan.limit <= MAX_LIMIT:
        raise PlanError(f"limit must be within 1..{MAX_LIMIT}")

    allowed = _columns_for(plan.source)
    for flt in plan.filters:
        if flt.operator not in OPERATORS:
            raise PlanError(f"unknown operator {flt.operator!r}")
        if not flt.json_key and flt.field not in allowed:
            raise PlanError(f"{flt.field!r} is not a filterable column")
    for column in (*plan.group_by, *( (plan.order_by,) if plan.order_by else () )):
        if column not in allowed and column not in {"value", "aggregate"}:
            raise PlanError(f"{column!r} is not groupable/orderable")
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


def compile_plan(plan: ExcelQueryPlan, contract: TableContract) -> tuple[str, list[Any]]:
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
        if flt.json_key:
            where.append(f"t.{json_column} ->> %s {OPERATORS[flt.operator]} %s")
            params.extend([flt.field, str(flt.value)])
        else:
            where.append(f"t.{flt.field} {OPERATORS[flt.operator]} %s")
            params.append(flt.value)

    # Contract-declared duplicate policy (Table 13 snapshots repeat work orders).
    dedupe = contract.dedupe.get("default_filter") if contract.dedupe else None
    if dedupe and source == RECORDS:
        where.append(f"t.{json_column} ->> %s = %s")
        params.extend([dedupe["field"], str(dedupe["value"])])

    where_sql = " AND ".join(where)
    group_sql = ", ".join(f"t.{c}" for c in plan.group_by)
    value_column = VALUE_COLUMN[source]

    if plan.operation == "select":
        select_columns = list(plan.group_by) or ["record_id"]
        cols = ", ".join(f"t.{c}" for c in select_columns)
        tail = f", t.{value_column}" if value_column else ""
        json_select = "".join(
            f", t.{json_column} ->> %s AS selected_{index}"
            for index, _ in enumerate(plan.select_json_keys)
        )
        sql = (
            f"SELECT {cols}{tail}{json_select} FROM {source} t "
            f"JOIN excel_revisions r ON r.id = t.revision_id "
            f"WHERE {where_sql} ORDER BY {cols} LIMIT %s"
        )
        return sql, [*plan.select_json_keys, *params, plan.limit]

    target = value_column if value_column else "*"
    agg_sql = AGGREGATES[plan.aggregate].format(col=f"t.{target}")
    if plan.aggregate == "count" and not value_column:
        agg_sql = "count(*)"

    select_parts = ([group_sql] if group_sql else []) + [f"{agg_sql} AS value"]
    sql = (
        f"SELECT {', '.join(select_parts)} FROM {source} t "
        f"JOIN excel_revisions r ON r.id = t.revision_id "
        f"WHERE {where_sql}"
    )
    if group_sql:
        sql += f" GROUP BY {group_sql}"
    if plan.operation == "rank" or plan.order_by:
        direction = "DESC" if plan.descending else "ASC"
        sql += f" ORDER BY value {direction} NULLS LAST"
    elif group_sql:
        sql += f" ORDER BY {group_sql}"
    sql += " LIMIT %s"
    params.append(plan.limit)
    return sql, params


def execute_plan(
    plan: ExcelQueryPlan,
    conn,
    *,
    contracts: ContractSet | None = None,
) -> ExcelExecutionResult:
    contracts = contracts or load_contracts()
    contract = contracts.for_table(plan.table_number)
    sql, params = compile_plan(plan, contract)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        unit = None
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
                raise PlanError(
                    "refusing to aggregate incompatible units: " + ", ".join(units)
                )
            unit = units[0] if units else None
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
            meta_where, meta_params = _record_scope(plan, contract)
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
            cur.execute(
                f"""
                SELECT rec.provenance
                FROM excel_records rec
                JOIN excel_revisions r ON r.id = rec.revision_id
                WHERE {meta_where}
                LIMIT 5
                """,
                meta_params,
            )
            provenance = [row[0] or {} for row in cur.fetchall()]

    warnings: list[str] = []
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
    )


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
        if flt.json_key:
            where.append(f"f.dimensions ->> %s {OPERATORS[flt.operator]} %s")
            params.extend([flt.field, str(flt.value)])
        else:
            where.append(f"f.{flt.field} {OPERATORS[flt.operator]} %s")
            params.append(flt.value)
    return " AND ".join(where), params


def _record_scope(
    plan: ExcelQueryPlan,
    contract: TableContract,
) -> tuple[str, list[Any]]:
    """The record equivalent of ``_fact_scope`` for counts and provenance."""
    where = ["r.state = 'active'", "rec.table_number = %s"]
    params: list[Any] = [plan.table_number]
    for flt in plan.filters:
        if flt.json_key:
            where.append(f"rec.attributes ->> %s {OPERATORS[flt.operator]} %s")
            params.extend([flt.field, str(flt.value)])
        else:
            where.append(f"rec.{flt.field} {OPERATORS[flt.operator]} %s")
            params.append(flt.value)
    dedupe = contract.dedupe.get("default_filter") if contract.dedupe else None
    if dedupe:
        where.append("rec.attributes ->> %s = %s")
        params.extend([dedupe["field"], str(dedupe["value"])])
    return " AND ".join(where), params


# --------------------------------------------------------------------------
# Deterministic parameter binding from a question (no LLM, no classifier)
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20[2-3][0-9])\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)


def bind_period(question: str) -> dict[str, int]:
    """Extract only a 4-digit year and Q1-Q4 token. Nothing else is bound."""
    bound: dict[str, int] = {}
    years = _YEAR_RE.findall(question)
    if len(set(years)) == 1:
        bound["reporting_year"] = int(years[0])
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
