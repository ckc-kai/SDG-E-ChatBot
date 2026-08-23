"""Model-formulated Excel query plans, validated by the typed executor.

The deterministic channel in ``channel.py`` covers entity history and simple
count/sum aggregates. Comparisons, trends, rankings, and single-attribute
lookups need a plan shape the heuristics never emit. This module asks a
structured-output model to propose an ``ExcelQueryPlan`` and then refuses to
run anything that is not grounded in the reviewed contracts:

- the table must come from the retrieved candidate cards;
- semantic metric keys must exist in the active revision for that table;
- filter fields must be allowlisted typed columns or real jsonb keys;
- equality values for vocabulary-backed dimensions must exist in the corpus;
- vintage/year-basis filters are stripped unless the question names a
  submission, so the tables 14/15 clarification contract stays deterministic.

A sanitized plan still goes through ``compile_plan``'s allowlists and the
executor's unit checks, so the model can never widen the query surface.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import psycopg2

from dataclasses import replace

from generation.features import feature_enabled
from generation.providers.base import ProviderError
from retrieval.ingest.excel.contracts import ContractSet
from retrieval.query.excel.manifest import WorkbookManifest, load_manifest
from retrieval.query.excel.query import (
    AGGREGATES,
    ARITHMETIC_AGGREGATES,
    NUMERIC_COLUMNS,
    FACT_COLUMNS,
    FACTS,
    GroupKey,
    Having,
    MAX_LIMIT,
    OPERATORS,
    ORDERING_OPERATORS as _ORDERING_OPERATORS,
    SET_OPERATORS,
    RECORD_COLUMNS,
    RECORDS,
    ExcelExecutionResult,
    ExcelQueryPlan,
    Filter,
    PlanError,
    bind_dimensions,
    bind_entity_key,
    dimension_vocabulary,
    execute_plan,
)

logger = logging.getLogger(__name__)

ENTITY_TABLES = {1, 13}
DUAL_YEAR_AXIS_TABLES = {14, 15}
DEFAULT_PLANNER_MODEL = "openai/gpt-oss-120b"
MAX_CANDIDATE_TABLES = 4
MAX_VOCAB_VALUES = 25
# Two extra queries cover the aggregate-plus-detail and two-table shapes
# without letting one question fan out into a survey of the workbook.
MAX_FOLLOW_UP_PLANS = 2
# The gold answers state a mean of 6.1 figures and the baseline executes 1.19
# plans per question, so two follow-ups is a hard ceiling well below what the
# questions need. Four is still bounded -- a question cannot turn into a survey
# of the workbook -- and each plan is sanitised and guarded exactly as before.
WIDE_MAX_FOLLOW_UP_PLANS = 4
# Raising the plan ceiling raises the worst-case evidence volume with it, and
# "more evidence in front of the model" is the one intervention this lane has
# measured as harmful three separate times (one-table rendering, open table
# selection, wider grouped-aggregate limits -- about -3.6 points each). Five
# plans against table 1 could put ~500 rows in the prompt. The budget below is
# set just above the largest single result observed on the beta set (128 rows)
# plus room for several aggregates, so composition is bought without
# reintroducing the flooding failure.
MAX_TOTAL_EVIDENCE_ROWS = 260
# Enough to carry every WMP activity in one grouped result; the total evidence
# budget above still bounds how many such results one question can collect.
WIDE_GROUPED_LIMIT = 60


def _max_follow_up_plans(environ=None) -> int:
    return (
        WIDE_MAX_FOLLOW_UP_PLANS
        if feature_enabled("excel_wide_fanout", environ=environ)
        else MAX_FOLLOW_UP_PLANS
    )

_VINTAGE_CUE_RE = re.compile(r"\b(submission|submitted|filing|filed|vintage|year[ _-]basis)\b", re.I)
_SHAPE_CUE_RE = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|difference between"
    r"|change (?:across|over)|across the quarters|trend|quarter[- ]by[- ]quarter"
    r"|highest|lowest|largest|smallest|top\s+(?:\d+|three|five|ten)"
    r"|rank(?:ed|ing)?|breakdown by|by tier|per quarter"
    r"|not (?:met|meet)|missed target|shortfall|repeatedly|under[- ]target"
    r"|behind target)\b",
    re.I,
)
_EITHER_OR_RE = re.compile(r"\bmore\b.+\bor\b", re.I)

# The shape of one plan, shared by the primary plan and every follow-up so
# the two can never validate differently.
_PLAN_BODY_PROPERTIES: dict[str, Any] = {
    "table_number": {"type": "integer"},
    "semantic_metric_key": {"type": ["string", "null"]},
    "operation": {"type": "string", "enum": ["aggregate", "select", "rank"]},
    "aggregate": {
        "type": "string",
        "enum": ["sum", "avg", "count", "min", "max", "count_null", "count_distinct"],
    },
    "filters": {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "operator": {
                    "type": "string",
                    "enum": sorted(set(OPERATORS) | set(SET_OPERATORS)),
                },
                "value": {
                    "type": ["string", "number", "array"],
                    "items": {"type": ["string", "number"]},
                },
                # jsonb returns text. A magnitude comparison on a stored
                # number needs this, or the executor refuses the plan rather
                # than ordering '2.0' above '1000'.
                "cast": {"type": ["string", "null"], "enum": ["numeric", "date", "text", None]},
            },
            "required": ["field", "value"],
            "additionalProperties": False,
        },
    },
    "group_by": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    # Grouping by an attribute inside the jsonb payload -- spend by initiative,
    # work orders by priority -- rather than by an opaque record_id.
    "group_by_json_keys": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    # excel_records has no typed numeric column; naming an attribute here is
    # the only way to sum or average one.
    "value_json_key": {"type": ["string", "null"]},
    "value_column": {"type": ["string", "null"]},
    "having": {
        "type": ["object", "null"],
        "properties": {
            "operator": {"type": "string", "enum": sorted(OPERATORS)},
            "value": {"type": "number"},
        },
        "required": ["operator", "value"],
        "additionalProperties": False,
    },
    "select_json_keys": {
        "type": "array",
        "maxItems": 6,
        "items": {"type": "string"},
    },
    "descending": {"type": "boolean"},
    "limit": {"type": "integer"},
}

PLAN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["plan", "decline"]},
        "reason": {"type": "string"},
        **_PLAN_BODY_PROPERTIES,
        "follow_up_plans": {
            # Some questions are two queries, not one: an aggregate total and
            # the ranked detail behind it, or the same metric read from two
            # tables. Asked for a single plan the model used to decline these
            # outright -- "cannot be expressed with a single query plan" -- and
            # the question fell back to prose with no numbers. Each follow-up is
            # sanitised and executed exactly like the primary plan, so this
            # widens what can be asked without widening what can be run.
            "type": "array",
            "maxItems": WIDE_MAX_FOLLOW_UP_PLANS,
            "items": {
                "type": "object",
                "properties": _PLAN_BODY_PROPERTIES,
                "required": ["table_number", "operation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


def _as_number(value: Any) -> float | int | None:
    """The value as a number, or None when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        rendered = str(value).strip().replace(",", "")
        return float(rendered) if "." in rendered else int(rendered)
    except (TypeError, ValueError):
        return None


def question_needs_model_shapes(question: str) -> bool:
    """True for comparison/trend/ranking/grouped questions the heuristics miss."""
    return bool(_SHAPE_CUE_RE.search(question) or _EITHER_OR_RE.search(question))


def _question_matched_values(
    question: str, vocabulary: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Every corpus dimension value the question names, word-bounded."""
    lowered = question.casefold()
    matched: dict[str, list[str]] = {}
    for field, values in vocabulary.items():
        hits = [
            value
            for value in values
            if value
            and len(value.strip()) >= 3
            and not value.replace(".", "").replace("-", "").isdigit()
            and re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", lowered)
        ]
        if hits:
            matched[field] = hits
    return matched


_planner_provider: Any = None
_planner_provider_failed = False

# Why the last planning attempt produced no plan. Diagnostic only: read by the
# channel so an evaluation trace can say which layer refused, instead of
# recording an unexplained decline.
last_rejections: list[str] = []


def _get_planner_provider():
    """Lazily create the structured-output provider for plan formulation."""
    global _planner_provider, _planner_provider_failed
    if _planner_provider is not None or _planner_provider_failed:
        return _planner_provider
    from generation.providers import create_provider_from_env

    provider_name = os.getenv("EXCEL_PLANNER_PROVIDER", "").strip().casefold()
    if not provider_name:
        provider_name = "groq" if os.getenv("GROQ_API_KEY", "").strip() else "ollama"
    values = dict(os.environ)
    if provider_name == "groq":
        values["GROQ_MODEL"] = values.get("EXCEL_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        values["GROQ_MAX_TOKENS"] = values.get("EXCEL_PLANNER_MAX_TOKENS", "700")
        values["GROQ_REASONING_EFFORT"] = "low"
    elif provider_name == "ollama":
        values["OLLAMA_MODEL"] = values.get("EXCEL_PLANNER_MODEL", "qwen3.5:9b")
        values["OLLAMA_MAX_TOKENS"] = values.get("EXCEL_PLANNER_MAX_TOKENS", "700")
    try:
        _planner_provider = create_provider_from_env(provider_name, environ=values)
    except (ProviderError, ValueError):
        logger.warning("Excel model planner provider unavailable; heuristics only")
        _planner_provider_failed = True
        _planner_provider = None
    return _planner_provider


_table_context_cache: dict[int, dict[str, Any]] = {}


def _table_context(conn, table_number: int, contracts: ContractSet) -> dict[str, Any]:
    """Semantic keys, jsonb keys, and bounded dimension vocabulary for one table."""
    cached = _table_context_cache.get(table_number)
    if cached is not None:
        return cached
    source = RECORDS if table_number in ENTITY_TABLES else FACTS
    json_column = "attributes" if source == RECORDS else "dimensions"
    context: dict[str, Any] = {"source": source}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT f.semantic_metric_key
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state = 'active'
            WHERE f.table_number = %s AND f.semantic_metric_key IS NOT NULL
            ORDER BY 1
            """,
            (table_number,),
        )
        context["semantic_keys"] = [row[0] for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT DISTINCT jsonb_object_keys(t.{json_column})
            FROM {source} t
            JOIN excel_revisions r ON r.id = t.revision_id AND r.state = 'active'
            WHERE t.table_number = %s
            """,
            (table_number,),
        )
        context["json_keys"] = sorted(row[0] for row in cur.fetchall())
        vocabulary: dict[str, list[str]] = {}
        for column in ("hftd_tier", "line_type"):
            cur.execute(
                f"""
                SELECT DISTINCT t.{column} FROM {source} t
                JOIN excel_revisions r ON r.id = t.revision_id AND r.state='active'
                WHERE t.table_number = %s AND t.{column} IS NOT NULL
                """,
                (table_number,),
            )
            values = [row[0] for row in cur.fetchall()]
            if values:
                vocabulary[column] = sorted(values)
        cur.execute(
            f"""
            SELECT key, array_agg(DISTINCT value)
            FROM (
                SELECT d.key, d.value
                FROM {source} t
                JOIN excel_revisions r
                  ON r.id = t.revision_id AND r.state = 'active'
                CROSS JOIN LATERAL jsonb_each_text(t.{json_column}) AS d(key, value)
                WHERE t.table_number = %s
            ) pairs
            GROUP BY key
            HAVING count(DISTINCT value) BETWEEN 2 AND %s
            """,
            (table_number, MAX_VOCAB_VALUES),
        )
        for key, values in cur.fetchall():
            vocabulary[key] = sorted(value for value in values if value)
    context["vocabulary"] = vocabulary
    context["full_vocabulary"] = dimension_vocabulary(
        conn, table_number, source=source
    )
    try:
        contract = contracts.for_table(table_number)
        context["title"] = getattr(contract, "title", None) or f"Table {table_number}"
    except (KeyError, ValueError):
        context["title"] = f"Table {table_number}"
    _table_context_cache[table_number] = context
    return context


def _candidate_tables(cards) -> list[tuple[int, str, str | None]]:
    """Unique (table_number, caption, semantic_metric_key) from ranked cards."""
    seen: set[int] = set()
    candidates: list[tuple[int, str, str | None]] = []
    for card in cards:
        data = card.query_object.structured_data or {}
        table_number = data.get("table_number")
        if not isinstance(table_number, int) or table_number in seen:
            continue
        seen.add(table_number)
        candidates.append(
            (
                table_number,
                card.query_object.caption or "",
                data.get("semantic_metric_key"),
            )
        )
        if len(candidates) >= MAX_CANDIDATE_TABLES:
            break
    return candidates


def _entity_card_options(cards) -> dict[int, list[tuple[str, str]]]:
    """Per entity table, the retrieved (entity_key, caption) candidates."""
    options: dict[int, list[tuple[str, str]]] = {}
    for card in cards:
        data = card.query_object.structured_data or {}
        table_number = data.get("table_number")
        entity_key = data.get("entity_key")
        if not isinstance(table_number, int) or not entity_key:
            continue
        rows = options.setdefault(table_number, [])
        if len(rows) < 6 and entity_key not in {key for key, _ in rows}:
            rows.append((str(entity_key), card.query_object.caption or ""))
    return options


def _render_prompt(
    question: str,
    candidates: list[tuple[int, str, str | None]],
    contexts: dict[int, dict[str, Any]],
    entity_options: dict[int, list[tuple[str, str]]] | None = None,
    manifest: WorkbookManifest | None = None,
) -> str:
    lines = [
        "Convert this question about SDG&E quarterly wildfire data into typed",
        "query plans, or decline if no table can answer it.",
        "",
        f"Question: {question}",
        "",
    ]
    # Retrieval ranks cards by text similarity, which is a poor proxy for which
    # table holds a figure: "what caused our ignitions" ranked the
    # utility-defined metric glossary above the ignitions-by-driver table
    # because the glossary's metric names contain the word. Listing every table
    # lets the plan be built against the schema rather than against the
    # ranking, with the retrieved ones flagged as a hint rather than a fence.
    retrieved = {table for table, _, _ in candidates}
    if manifest and manifest.tables:
        lines.append(
            "Every table in the corpus, for judging what the data can and "
            "cannot answer. Build the plan on a table marked as retrieved:"
        )
        for facts in manifest.tables:
            mark = " <- retrieved for this question" if facts.table_number in retrieved else ""
            years = (
                f"{min(facts.years)}-{max(facts.years)}" if facts.years else "none"
            )
            lines.append(
                f"- Table {facts.table_number}: {facts.title}"
                f" ({facts.row_count:,} rows, years {years}){mark}"
            )
            if facts.description:
                lines.append(f"    {facts.description}")
        lines.append("")
    lines.append("Detail for the retrieved tables (most relevant first):")
    for table_number, caption, semantic_key in candidates:
        context = contexts[table_number]
        family = "entity records" if context["source"] == RECORDS else "metric facts"
        lines.append(
            f"- Table {table_number} ({family}): {context['title']}; "
            f"top card: {caption[:140]!r}"
            + (f"; card metric key: {semantic_key}" if semantic_key else "")
        )
        if context["semantic_keys"]:
            lines.append(
                f"  semantic_metric_key values: {', '.join(context['semantic_keys'])}"
            )
        typed = sorted(
            (RECORD_COLUMNS if context["source"] == RECORDS else FACT_COLUMNS)
            - {"table_number", "record_id", "series_id", "source_metric_number"}
        )
        lines.append(f"  typed filter columns: {', '.join(typed)}")
        if context["json_keys"]:
            lines.append(f"  json attribute keys: {', '.join(context['json_keys'])}")
        for field, values in context["vocabulary"].items():
            lines.append(f"  values of {field}: {', '.join(values)}")
        matched = _question_matched_values(question, context["full_vocabulary"])
        if matched:
            lines.append(
                "  dimension values from this table matching the question: "
                + "; ".join(
                    f"{field} in [{', '.join(repr(v) for v in values)}]"
                    for field, values in matched.items()
                )
            )
        for entity_key, caption in (entity_options or {}).get(table_number, ()):
            lines.append(f"  entity card: entity_key={entity_key}: {caption[:110]}")
        # Typed-column domains. Without these the planner had to guess what a
        # column holds, and read "Level 2" (a GO 95 priority in ``status``) as
        # HFTD Tier 2, answering a different question with confident counts.
        for facts in (manifest.tables if manifest else ()):
            if facts.table_number != table_number:
                continue
            for column, values, null_count in facts.value_domains:
                blank = f" ({null_count:,} rows blank)" if null_count else ""
                lines.append(
                    f"  column {column} holds: "
                    + ", ".join(repr(value) for value in values)
                    + blank
                )
            if facts.years:
                lines.append(
                    f"  years present: {min(facts.years)}-{max(facts.years)}"
                )
    lines += [
        "",
        "Rules:",
        "- A question that asks for two separate figures needs two plans, not a",
        "  decline. Put the first in the top-level plan and each additional one",
        f"  in follow_up_plans (up to {_max_follow_up_plans()}), each with its own table_number and",
        "  operation. Typical shapes: a total AND the ranked detail behind it",
        "  ('what was the Tier 3 total, and which segments were highest' ->",
        "  plan: operation=aggregate filtered to the tier; follow_up:",
        "  operation=rank group_by record_id); the same metric on two tables;",
        "  or two different metrics. NEVER decline because one query is not",
        "  enough. Use one plan when one plan genuinely answers the question.",
        "- A plan MUST set top-level table_number, and for metric-fact tables a",
        "  top-level semantic_metric_key naming the metric concept. Neither is",
        "  a filter.",
        "- Use only listed tables, columns, keys, and dimension values. Never invent values.",
        "- Several semantic_metric_key values can share a topic (e.g. duration_of_X,",
        "  frequency_of_X, scope_of_X for the same event type). Match the",
        "  question's own word: 'how many'/'number of'/'total X events' means",
        "  frequency_of_X; 'how long' means duration_of_X; 'how many circuits/",
        "  customers affected' means scope_of_X. Never substitute a different",
        "  key from the same family.",
        "- Every plan needs a time filter: reporting_year (plus reporting_quarter",
        "  when the question names a quarter), or source_vintage_year for a",
        "  submission/filing year. Use source_vintage_year ONLY when the question",
        "  says submission or filing; a bare year means reporting_year.",
        "- 'Compare A or B of one dimension' -> operation=rank, aggregate=sum,",
        "  group_by that dimension, optionally filter out other values.",
        "- When several values of one dimension match the question, it is a",
        "  comparison: group_by that dimension; never filter to just one value.",
        "- 'How did X change across quarters of YEAR' -> operation=aggregate,",
        "  aggregate=sum, group_by ['reporting_quarter'], filter reporting_year.",
        "- 'Which N <things> had highest/lowest X' -> operation=rank,",
        "  aggregate=max, group_by ['record_id'], limit N, descending for highest.",
        "- Several metrics of one scope (e.g. overall utility, wildfire, and",
        "  PSPS risk for one tier or one segment): operation=select on the",
        "  facts table with group_by ['semantic_metric_key'] and filters for",
        "  the scope; every metric's value returns as its own labeled row.",
        "- One attribute of one named activity or work order (entity records):",
        "  operation=select, filter entity_key (pick the entity card whose",
        "  caption names it) and reporting_year, select_json_keys=[the",
        "  attribute]. Example attributes: annual_quant_target,",
        "  quant_actual_progress_q1_4, quant_target_units, status.",
        "- Counting entities: operation=aggregate, aggregate=count on records.",
        "  On metric-fact tables never use count: one fact row is one metric",
        "  observation for one period, so counting rows counts reporting slots",
        "  rather than events. 'How many ignitions/outages/events' is",
        "  aggregate=sum over the metric's own value.",
        "- 'Which activities did not meet / repeatedly missed targets':",
        "  operation=select on the activity records table with select_json_keys",
        "  ['annual_quant_target','quant_actual_progress_q1_4',",
        "  'quant_target_units'], group_by ['reporting_year','record_id',",
        "  'title'], filters covering the requested years, limit 200; the",
        "  target-versus-actual comparison happens downstream.",
        "- Filters default to equality; use operator gte/lte for ranges,",
        "  ne to exclude a value (e.g. exclude 'Non-HFTD' when comparing tiers).",
        "- For 'A or B' of one field use operator 'in' with a list value:",
        "  {field:'status', operator:'in', value:['Delayed','Cancelled']}.",
        "  Use 'not_in' for 'everything except'. Do not emit one plan per",
        "  value, and do not filter to just one of them.",
        "- Do not set limit on a select that asks 'which ...' or 'what ...':",
        "  the answer is every matching row. Set limit only for an explicit",
        "  top-N request.",
        "- A json attribute holding a NUMBER (targets, actuals, spend, miles,",
        "  counts) must set cast='numeric' whenever the filter compares",
        "  magnitude (gt/gte/lt/lte) or tests zero. Without it the comparison",
        "  is textual and '2.0' sorts above '1000'. Example: find work reported",
        "  against no target -> filters [{field:'annual_quant_target',",
        "  operator:'eq', value:0, cast:'numeric'},",
        "  {field:'quant_actual_progress_q1_4', operator:'gt', value:0,",
        "  cast:'numeric'}] with operation=select.",
        "- To break a figure down by an attribute inside the json payload",
        "  (spend by initiative, work orders by priority, miles by activity)",
        "  put that key in group_by_json_keys, NOT in group_by. group_by is",
        "  only for the typed columns listed above. Grouping by record_id",
        "  returns opaque hashes and is almost never what a person asked for.",
        "- Entity-record tables (WMP activities, work orders) DO hold numbers:",
        "  they live in the json payload, so set value_json_key to the",
        "  attribute and the executor sums or averages it. Never decline",
        "  saying a table has no numeric field before checking the attribute",
        "  list above. Example: delivered miles per year ->",
        "  operation=aggregate, aggregate=sum,",
        "  value_json_key='quant_actual_progress_q1_4',",
        "  group_by ['reporting_year'].",
        "- A plan does NOT need a year filter when it groups by",
        "  reporting_year, or when it is a select narrowed by another",
        "  predicate. Questions like 'did we ever...' cover the whole cycle,",
        "  and adding a year to them answers a narrower question than asked.",
        "- When one table holds part of the answer and another holds the rest",
        "  (delivered quantity in the activities table, spend in the spend",
        "  table), use follow_up_plans with the other table_number rather than",
        "  declining because one table is incomplete.",
        "- Blanks are part of the answer. When breaking down by a field that",
        "  is often unpopulated, group by it anyway: a NULL group reports the",
        "  unpopulated cohort, which must be stated rather than dropped. Use",
        "  aggregate=count_null with value_column to count blanks directly.",
        "- 'Did any group exceed / fall short of a threshold' -> aggregate the",
        "  group and add having {operator, value}; having needs a group_by.",
        "- A question about a PROPERTY of the data rather than one entity",
        "  ('did we ever...', 'is it possible that...', 'which activities",
        "  had...') is a scan over the whole table with predicates, NOT a",
        "  lookup of the entity named on the top card. Never filter to a",
        "  single entity_key unless the question itself names that entity.",
        # The four zero-evidence beta cases all declined for one of these two
        # reasons: a derived figure has no column of its own, or one requested
        # period is missing. Both are answerable from components, and both were
        # being thrown away wholesale. See the decline analysis in
        # logs/progress/2026-08-21-excel.md.
        "- NEVER decline because the figure the question names is not stored as",
        "  a column. Cumulative totals, percent complete, variance, share of",
        "  total, per-year subtotals and running totals are all COMPUTED",
        "  DETERMINISTICALLY by the system from the rows you return. Your job",
        "  is to return the components. 'Cumulative three-year target per",
        "  activity' -> operation=aggregate, aggregate=sum,",
        "  value_json_key='annual_quant_target', group_by ['entity_key'] over",
        "  the cycle years; the system adds the years and computes the",
        "  percentage. Returning the components is always better than",
        "  declining.",
        "- NEVER decline because ONE requested period or entity is missing.",
        "  Return the periods that DO exist and let the answer state the gap.",
        "  A question asking 2022-2025 of a corpus holding 2023-2025 is a plan",
        "  filtered to 2023-2025, not a decline: the reader needs the three",
        "  years that exist plus a sentence saying 2022 is not in the",
        "  workbooks. Declining returns neither.",
        "- Use action=decline ONLY when no listed table holds the subject",
        "  matter at all -- not when the arithmetic, the derivation, or one",
        "  period is missing.",
        "- If the question cannot be answered by these tables, action=decline.",
        "Return JSON only.",
    ]
    return "\n".join(lines)


class _PlanRejected(ValueError):
    """The model plan failed grounding validation."""


def _merge_question_dimensions(
    plan: ExcelQueryPlan, question: str, conn
) -> ExcelQueryPlan:
    """Deterministically scope a model plan with dimensions the question names.

    High-cardinality vocabularies (equipment types, event drivers) are too
    large for the planning prompt, so the model cannot know their exact
    spelling. Word-bounded matching against the active revision's values adds
    the missing filter, or fixes the casing of a model-guessed value. Grouped
    dimensions are never collapsed.
    """
    if plan.operation == "select":
        return plan
    vocabulary = dimension_vocabulary(conn, plan.table_number, source=plan.source)
    bound = bind_dimensions(question, vocabulary)
    if not bound:
        return plan
    typed_columns = RECORD_COLUMNS if plan.source == RECORDS else FACT_COLUMNS
    # group_by holds GroupKey objects as well as bare column names. Comparing a
    # field name against the objects never matched, so this added a filter on
    # the very dimension the plan was grouping by and collapsed the breakdown
    # to a single row -- "capital and O&M" came back as CAPEX alone.
    grouped = {
        entry.field if isinstance(entry, GroupKey) else str(entry)
        for entry in plan.group_by
    }
    filters = list(plan.filters)
    changed = False
    for field, value in bound.items():
        if field in grouped:
            continue
        existing = [index for index, flt in enumerate(filters) if flt.field == field]
        if existing:
            for index in existing:
                flt = filters[index]
                if (
                    flt.operator == "eq"
                    and str(flt.value) != value
                    and str(flt.value).casefold() == value.casefold()
                ):
                    filters[index] = replace(flt, value=value)
                    changed = True
            continue
        filters.append(
            Filter(field, value=value, json_key=field not in typed_columns)
        )
        changed = True
    return replace(plan, filters=tuple(filters)) if changed else plan


def _sanitize_plan(
    payload: dict[str, Any],
    question: str,
    candidates: list[tuple[int, str, str | None]],
    contexts: dict[int, dict[str, Any]],
) -> ExcelQueryPlan:
    if payload.get("action") != "plan":
        raise _PlanRejected(str(payload.get("reason") or "model declined"))

    # Models often express the metric concept as a filter; hoist it, and drop
    # utility filters because this corpus holds exactly one utility.
    raw_filters: list[dict[str, Any]] = []
    hoisted_key: str | None = None
    for item in payload.get("filters") or ():
        field = str(item.get("field") or "").strip()
        if field == "semantic_metric_key":
            value = item.get("value")
            hoisted_key = str(value) if value is not None else None
            continue
        if field == "utility_id":
            continue
        raw_filters.append(item)
    semantic_key = payload.get("semantic_metric_key") or hoisted_key or None

    table_number = payload.get("table_number")
    candidate_tables = [table for table, _, _ in candidates]
    if table_number not in contexts:
        # Infer the table from the semantic key when the model omitted it.
        matches = [
            table
            for table in candidate_tables
            if semantic_key
            and semantic_key.casefold()
            in {key.casefold() for key in contexts[table]["semantic_keys"]}
        ]
        if matches:
            table_number = matches[0]
        elif len(candidate_tables) == 1:
            table_number = candidate_tables[0]
        else:
            raise _PlanRejected(f"table {table_number!r} is not a known table")
    context = contexts[table_number]
    source = context["source"]
    typed_columns = RECORD_COLUMNS if source == RECORDS else FACT_COLUMNS
    json_keys = set(context["json_keys"])
    vocabulary = context["vocabulary"]

    if source == FACTS:
        known = {key.casefold(): key for key in context["semantic_keys"]}
        if semantic_key:
            if semantic_key.casefold() not in known:
                raise _PlanRejected(f"unknown semantic key {semantic_key!r}")
            semantic_key = known[semantic_key.casefold()]
        # A missing key is validated after filters: a jsonb dimension filter
        # can scope the facts as precisely as a metric key does.
    else:
        semantic_key = None

    has_vintage_cue = bool(_VINTAGE_CUE_RE.search(question))
    filters: list[Filter] = []
    for item in raw_filters:
        field = str(item.get("field") or "").strip()
        operator = str(item.get("operator") or "eq")
        value = item.get("value")
        if operator in SET_OPERATORS:
            if not isinstance(value, (list, tuple)) or not value:
                raise _PlanRejected(f"{operator!r} needs a non-empty list")
            is_json = field not in typed_columns
            if is_json and field not in json_keys:
                raise _PlanRejected(f"unknown filter field {field!r}")
            if not is_json and field not in typed_columns:
                raise _PlanRejected(f"unknown filter field {field!r}")
            if field in {"reporting_year", "reporting_quarter", "source_vintage_year"}:
                rendered_set: list[Any] = []
                for item_value in value:
                    number = _as_number(re.sub(r"(?i)^q", "", str(item_value).strip()))
                    if number is None:
                        raise _PlanRejected(f"non-integer period value {item_value!r}")
                    rendered_set.append(int(number))
            else:
                rendered_set = [str(item_value) for item_value in value]
            filters.append(
                Filter(field, operator=operator, value=rendered_set, json_key=is_json)
            )
            continue
        if operator not in OPERATORS or value is None:
            raise _PlanRejected(f"invalid filter {item!r}")
        if field in {"source_vintage_year", "year_basis"} and not has_vintage_cue:
            # The question did not name a submission; keep the executor's
            # dual-axis clarification behavior deterministic.
            continue
        if (
            field == "reporting_year"
            and table_number in DUAL_YEAR_AXIS_TABLES
            and has_vintage_cue
        ):
            # "2025 submission" means the vintage axis on dual-axis tables.
            field = "source_vintage_year"
        if field in typed_columns:
            if field in {"reporting_year", "reporting_quarter", "source_vintage_year"}:
                rendered = re.sub(r"(?i)^q", "", str(value).strip())
                try:
                    value = int(rendered)
                except (TypeError, ValueError) as exc:
                    raise _PlanRejected(f"non-integer period value {value!r}") from exc
            filters.append(Filter(field, operator=operator, value=value))
        elif field in json_keys:
            cast = item.get("cast") or None
            if cast not in (None, "numeric", "date", "text"):
                raise _PlanRejected(f"unknown cast {cast!r}")
            numeric_value = _as_number(value)
            # A magnitude comparison is always about the number. Supplying the
            # cast the model forgot is safer than compiling a text comparison
            # that silently answers a different question -- and the executor
            # refuses an uncast ordering filter anyway.
            if cast is None and operator in _ORDERING_OPERATORS:
                if numeric_value is None:
                    raise _PlanRejected(
                        f"{field!r} compared with {operator!r} against "
                        f"non-numeric {value!r}"
                    )
                cast = "numeric"
            if cast == "numeric":
                if numeric_value is None:
                    raise _PlanRejected(f"non-numeric value {value!r} for {field!r}")
                filters.append(
                    Filter(field, operator=operator, value=numeric_value,
                           json_key=True, cast="numeric")
                )
                continue
            rendered = str(value)
            known_values = vocabulary.get(field)
            if operator == "eq" and known_values and rendered not in known_values:
                raise _PlanRejected(
                    f"value {rendered!r} not in corpus vocabulary for {field!r}"
                )
            filters.append(
                Filter(field, operator=operator, value=rendered, json_key=True,
                       cast=cast)
            )
        else:
            raise _PlanRejected(f"unknown filter field {field!r}")

    operation = payload.get("operation") or "aggregate"
    group_by = tuple(
        str(column) for column in payload.get("group_by") or () if str(column).strip()
    )
    if operation == "rank" and source == FACTS and any(
        column not in typed_columns for column in group_by
    ):
        # Ranked "which segments/circuits" intents group by the fact's entity
        # row; models name the concept (segment_id) rather than the column.
        group_by = ("record_id",)
    for column in group_by:
        if column not in typed_columns:
            raise _PlanRejected(f"{column!r} is not groupable")
    json_group_keys = tuple(
        str(key)
        for key in payload.get("group_by_json_keys") or ()
        if str(key).strip()
    )
    unknown_group_keys = [key for key in json_group_keys if key not in json_keys]
    if unknown_group_keys:
        raise _PlanRejected(f"unknown group attributes {unknown_group_keys!r}")

    value_json_key = payload.get("value_json_key") or None
    if value_json_key is not None:
        value_json_key = str(value_json_key)
        if source != RECORDS or value_json_key not in json_keys:
            # Fact tables aggregate value_numeric; naming an attribute there is
            # a category error, not a reason to lose an otherwise valid plan.
            value_json_key = None
    value_column = payload.get("value_column") or None
    if value_column is not None:
        value_column = str(value_column)
        if value_column not in typed_columns:
            value_column = None
        elif (
            payload.get("aggregate") in ARITHMETIC_AGGREGATES
            and value_column not in NUMERIC_COLUMNS
        ):
            # The model named a label column -- metric_name, unit -- as the
            # thing to sum. The executor refuses that outright and correctly
            # so, but refusing the whole plan cost real_008 every figure it
            # had: what was meant is the table's own measure, which is what
            # dropping the column falls through to.
            value_column = None

    raw_having = payload.get("having") or None
    having = None
    if isinstance(raw_having, dict):
        having_value = _as_number(raw_having.get("value"))
        having_operator = str(raw_having.get("operator") or "gt")
        if having_value is not None and having_operator in OPERATORS:
            having = Having(operator=having_operator, value=having_value)

    select_json_keys = tuple(
        str(key) for key in payload.get("select_json_keys") or () if str(key).strip()
    )
    if select_json_keys and source != RECORDS:
        # Fact tables carry the value in value_numeric; the extra attributes
        # are decoration the executor cannot select. Keep the plan itself.
        select_json_keys = ()
    if select_json_keys:
        # A model naming ``title`` or ``status`` is naming a real record column,
        # not a hallucinated field: those are promoted out of the attributes
        # jsonb and the executor already returns them. Rejecting the whole plan
        # over a decorative select threw away otherwise valid queries, so drop
        # the promoted names and keep the plan. A name that is neither a jsonb
        # attribute nor a column is still invented, and still rejected.
        # A promoted name is not decoration -- ``status`` and ``title`` are
        # what make a returned row identifiable. Dropping them left
        # ``excel_001`` returning nine bare record_ids for "which activities
        # were delayed and why". Move them into the grouping, where the
        # executor returns them as real columns.
        promoted = tuple(
            key for key in select_json_keys
            if key in typed_columns and key not in group_by
        )
        if promoted:
            group_by = (*group_by, *promoted)
        select_json_keys = tuple(
            key for key in select_json_keys if key not in typed_columns
        )
        unknown = [key for key in select_json_keys if key not in json_keys]
        if unknown:
            # Dropping an invented attribute is already how a promoted column
            # name is handled two lines above; rejecting the plan instead threw
            # away the attributes the model got RIGHT alongside the one it
            # invented. Only a select with nothing left to select is refused.
            select_json_keys = tuple(
                key for key in select_json_keys if key in json_keys
            )
            if not select_json_keys:
                raise _PlanRejected(f"unknown attributes {unknown!r}")
            logger.info("dropped invented attributes %r from a select", unknown)
        operation = "select"
        if not group_by:
            group_by = ("reporting_year", "record_id")

    # When the question names several values of one typed dimension (for
    # example "HFTD Tier 2 or HFTD Tier 3") it is comparing them: group by the
    # dimension instead of filtering to one value, and restrict the group to
    # exactly the mentioned values so unrequested rows cannot displace them.
    full_vocabulary = context.get("full_vocabulary") or vocabulary
    question_matches = _question_matched_values(question, full_vocabulary)
    mentioned_counts: dict[str, int] = {}
    for column, mentioned in question_matches.items():
        if len(mentioned) < 2 or column not in typed_columns:
            continue
        if any(flt.field == column and flt.operator == "eq" for flt in filters):
            filters = [
                flt
                for flt in filters
                if not (flt.field == column and flt.operator == "eq")
            ]
        if column not in group_by and operation in {"aggregate", "rank"}:
            group_by = (*group_by, column)
    for column in group_by:
        values = full_vocabulary.get(column)
        if not values:
            continue
        mentioned = question_matches.get(column, ())
        if not mentioned or len(mentioned) == len(values):
            continue
        mentioned_counts[column] = len(mentioned)
        excluded = {
            str(flt.value)
            for flt in filters
            if flt.field == column and flt.operator == "ne"
        }
        for value in values:
            if value not in mentioned and value not in excluded:
                filters.append(Filter(column, operator="ne", value=value))

    limit = payload.get("limit")
    wants_top_n = bool(
        re.search(r"\b(top|first|highest|lowest|largest|smallest|biggest)\b",
                  question, re.I)
    )
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        # Raising this to 60 for grouped aggregates was measured and reverted
        # with the table-selection change above. A wider default fills the
        # prompt with rows the question did not ask for, and the same effect
        # showed up when large results were rendered as one block: more
        # evidence in front of the model made the answers worse, not better.
        # A ``select`` scan is different and still returns everything it
        # matches -- that is what the question asked for.
        # Eight groups is far too few for the per-entity questions this lane
        # now reaches: "cumulative targets per activity" spans ~60 activities,
        # and eight of them read as either a fabricated complete list or as
        # "insufficient context". A wider default was measured and reverted
        # once before, but it was reverted *together with* open table
        # selection, so the two were never separated. It is retried here with
        # truncation now disclosed, which is what made the narrow cap look
        # safe.
        wide = feature_enabled("excel_wide_fanout")
        if group_by or json_group_keys:
            limit = WIDE_GROUPED_LIMIT if wide else 8
        else:
            limit = 1
    if operation == "select" and not wants_top_n:
        # A scan answers "which activities...", and the answer is however many
        # rows match. Capping it at eight silently truncated the answer to the
        # first few and the model reported those as the complete set -- nine
        # delayed activities became two. Only an explicit top-N narrows this.
        limit = MAX_LIMIT
    if mentioned_counts:
        limit = max(limit, max(mentioned_counts.values()))
    # An entity filter must come from the question, not from whichever card
    # retrieval happened to rank first. "Is it possible we reported completed
    # work against a zero target?" is a scan over every activity; anchoring it
    # to the top card's WMP.1191 returned three rows about an unrelated
    # activity and let the model conclude, with execution "verification", that
    # no such case existed.
    named_entity = bind_entity_key(question)
    entity_filters = [flt for flt in filters if flt.field == "entity_key"]
    if entity_filters and named_entity is None:
        raise _PlanRejected(
            "plan filters to an entity the question does not name; "
            "a property question is a scan, not a lookup"
        )
    if entity_filters and named_entity is not None:
        if any(str(flt.value) != named_entity for flt in entity_filters):
            raise _PlanRejected(
                f"plan filters to an entity other than {named_entity!r}"
            )

    # A period keeps an unscoped ``sum`` from silently spanning every year in
    # the corpus. It is not required when the plan is scoped some other way:
    #
    # - grouping by reporting_year labels each year in the result, so nothing
    #   is silently conflated;
    # - a ``select`` narrowed by a real predicate is a scan, and "did we ever
    #   report work against a zero target?" deliberately asks about the whole
    #   cycle. Demanding a year here rejected the correct plan for exactly the
    #   questions that must not name one.
    period_fields = {flt.field for flt in filters}
    group_fields = {
        entry.field if isinstance(entry, GroupKey) else str(entry)
        for entry in (*group_by, *(GroupKey(k, True) for k in json_group_keys))
    }
    non_period_filters = [
        flt
        for flt in filters
        if flt.field not in {"reporting_year", "reporting_quarter",
                             "source_vintage_year", "year_basis"}
    ]
    scoped_otherwise = (
        "entity_key" in period_fields
        or "reporting_year" in group_fields
        or (operation == "select" and non_period_filters)
    )
    if not period_fields & {"reporting_year", "source_vintage_year"} and not scoped_otherwise:
        # The guard exists so a bare ``sum`` cannot silently add three years
        # together and present the total as one year's figure. Grouping by
        # year serves that purpose and answers the question; refusing serves
        # it and answers nothing. Questions about a whole cycle -- "cumulative
        # targets over the three year cycle" -- name no single year on
        # purpose, and were being declined for saying exactly what they meant.
        if operation in {"aggregate", "rank"} and "reporting_year" in typed_columns:
            group_by = (*group_by, "reporting_year")
        else:
            raise _PlanRejected("plan names no reporting period")
    # A fact table holds many unrelated metrics, so an unscoped sum spans all
    # of them. Any of these keeps that from happening: naming the metric,
    # filtering a dimension, or grouping by something that labels each row
    # with the metric it belongs to.
    metric_labeled = "semantic_metric_key" in group_by
    grouped_by_dimension = bool(json_group_keys) or bool(
        set(group_by) - {"reporting_year", "reporting_quarter"}
    )
    if (
        source == FACTS
        and semantic_key is None
        and not metric_labeled
        and not grouped_by_dimension
        and not any(flt.json_key for flt in filters)
    ):
        # Label every row with its metric rather than refusing outright; the
        # answer then carries the breakdown the question implied.
        if "semantic_metric_key" in typed_columns and operation != "select":
            group_by = ("semantic_metric_key", *group_by)
            metric_labeled = True
        else:
            raise _PlanRejected(
                "fact plans need a semantic metric key or a scoping dimension filter"
            )
    metric_labeled_select = operation == "select" and metric_labeled
    if metric_labeled_select:
        # One labeled row per metric of the scope; never truncate the set.
        limit = max(limit, 50)

    descending = bool(payload.get("descending", True))
    if re.search(r"\b(more|most|higher|highest|larger|largest|greater)\b", question, re.I):
        descending = True
    elif re.search(r"\b(fewer|fewest|less|least|lower|lowest|smaller|smallest)\b", question, re.I):
        descending = False

    aggregate = str(
        payload.get("aggregate") or ("count" if source == RECORDS else "sum")
    )
    if aggregate not in AGGREGATES:
        raise _PlanRejected(f"unknown aggregate {aggregate!r}")
    # count_null needs something to test for nullity.
    if aggregate == "count_null" and value_column is None and value_json_key is None:
        aggregate = "count"
    # A fact row is one metric observation for one period and scope, so
    # counting fact rows counts reporting slots, never events. "How many
    # ignitions" is sum(value_numeric): counting returned 96 for a driver whose
    # actual 2024 total was 14, and the model reported the row count as the
    # ignition count. Cardinality questions use count_distinct.
    if source == FACTS and aggregate == "count":
        aggregate = "sum"
    # A record-table sum with nothing to sum degrades to a count rather than
    # failing: the group breakdown is still the useful half of the answer.
    if (
        source == RECORDS
        and value_json_key is None
        and value_column is None
        and aggregate in {"sum", "avg", "min", "max"}
    ):
        aggregate = "count"
    if having is not None and (operation == "select" or not (group_by or json_group_keys)):
        having = None

    if operation == "select" and source == RECORDS and not select_json_keys:
        # A select that returns nothing but a period and an opaque id is not
        # evidence: nine ``record_id`` hashes answer "which activities were
        # delayed" only for a reader who already knows what they are, and
        # answer "and why" not at all. A named column -- title, status,
        # entity_key -- is enough; nothing at all is not. Refusing lets the
        # channel fall through to a plan that does return something.
        informative = {
            entry.field if isinstance(entry, GroupKey) else str(entry)
            for entry in group_by
        } - {"reporting_year", "reporting_quarter", "record_id"}
        if not informative and not json_group_keys:
            raise _PlanRejected(
                "select returns only a period and an opaque row id; "
                "group keys alone are not evidence"
            )

    combined_group_by: tuple[Any, ...] = (
        *group_by,
        *(GroupKey(key, json_key=True) for key in json_group_keys),
    )
    return ExcelQueryPlan(
        table_number=table_number,
        source=source,
        semantic_metric_key=semantic_key,
        operation=operation,
        aggregate=aggregate,
        filters=tuple(filters),
        group_by=combined_group_by,
        value_json_key=value_json_key,
        value_column=value_column,
        having=having,
        select_json_keys=select_json_keys,
        descending=descending,
        limit=limit,
    )


_RETRIEVAL_BLAME_RE = re.compile(
    r"\b(not retrieved|no(?:t)? (?:in|among) the retrieved|retrieved tables?"
    r"|available tables?|listed tables?|candidate tables?|no table)\b",
    re.I,
)


def _blames_retrieval(reason: str) -> bool:
    """True when the decline is about which tables were offered, not the data."""
    return bool(_RETRIEVAL_BLAME_RE.search(reason or ""))


def _all_table_candidates(
    manifest: WorkbookManifest,
    contexts: dict[int, dict[str, Any]],
    conn,
    contracts: ContractSet,
):
    """Every manifest table as a selectable candidate, with its context loaded."""
    widened_contexts = dict(contexts)
    candidates: list[tuple[int, str, str | None]] = []
    for facts in manifest.tables:
        number = facts.table_number
        if number not in widened_contexts:
            try:
                widened_contexts[number] = _table_context(conn, number, contracts)
            except (KeyError, ValueError):
                continue
        candidates.append((number, facts.title or "", None))
    return candidates, widened_contexts


def _ask_planner(provider, prompt: str) -> dict[str, Any] | None:
    """One structured planning call. None means the call itself failed."""
    try:
        structured = getattr(provider, "generate_structured", None)
        raw = (
            structured(prompt, PLAN_RESPONSE_SCHEMA)
            if callable(structured)
            else provider.generate(prompt)
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise _PlanRejected("plan response must be a JSON object")
        return payload
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Excel model planning failed: %s", exc)
        last_rejections.clear()
        last_rejections.append(f"planner call failed: {exc}")
        return None


def _retry_prompt(prompt: str, reason: str) -> str:
    """The same prompt, with the model's own decline reason challenged.

    Deliberately not a free re-ask: it names the reason, restates the two
    rules that decline most often violates, and asks for the components. The
    schema and every downstream guard are unchanged, so a retry can only
    produce a plan that would have been accepted the first time.
    """
    return "\n".join(
        [
            prompt,
            "",
            "You already answered this question with action=decline, saying:",
            f"  {reason or '(no reason given)'}",
            "",
            "That is very likely wrong. Before declining again, check:",
            "- Is the missing thing a DERIVED figure (cumulative, total,",
            "  percent complete, variance, share, running total)? The system",
            "  computes those from the rows you return. Return the components.",
            "- Is only SOME of the requested range missing? Return the part",
            "  that exists; the answer states the gap.",
            "- Does any listed table hold the underlying subject at all, even",
            "  under a different name? If so, plan against it.",
            "Decline again only if no listed table holds this subject matter.",
            "Return JSON only.",
        ]
    )


def build_model_plan(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
) -> tuple[ExcelQueryPlan, ExcelExecutionResult] | None:
    """Back-compatible single-plan entry point; None means fall back."""
    plans = build_model_plans(question, cards, conn, contracts)
    return plans[0] if plans else None


def build_model_plans(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
) -> list[tuple[ExcelQueryPlan, ExcelExecutionResult]]:
    """Formulate, validate and execute one or more plans; [] means fall back."""
    provider = _get_planner_provider()
    if provider is None or not cards:
        return []
    candidates = _candidate_tables(cards)
    if not candidates:
        return []
    # The manifest is an aid; without it the planner falls back to the
    # retrieved candidates rather than failing to plan at all.
    manifest = load_manifest(conn)
    # Offering all fourteen tables as selectable was measured over three
    # replicates and reverted: 51.20 against 54.79 on the beta cases and a
    # lower held-out fact coverage too. It did fix individual table choices --
    # "what caused our ignitions" moved from the metric glossary to the
    # ignitions table -- but net it gave the planner more ways to be wrong than
    # to be right. The manifest still describes every table in the prompt, as
    # context for saying what the corpus cannot answer; the plan is still built
    # against the tables retrieval actually surfaced.
    contexts: dict[int, dict[str, Any]] = {}
    for table_number, _, _ in candidates:
        try:
            contexts[table_number] = _table_context(conn, table_number, contracts)
        except (KeyError, ValueError):
            continue
    if not contexts:
        return []
    prompt = _render_prompt(
        question,
        candidates,
        contexts,
        _entity_card_options(cards),
        manifest,
    )
    payload = _ask_planner(provider, prompt)
    if payload is None:
        return []

    # A decline costs the question every Excel figure it might have had, and
    # the measured declines were not "this corpus lacks the subject" -- they
    # were "the derived figure has no column" and "one of the requested years
    # is missing". Both are answerable from components. Re-ask once, naming
    # the model's own reason, before accepting that there is nothing to run.
    if payload.get("action") != "plan" and feature_enabled("excel_wide_fanout"):
        reason = str(payload.get("reason") or "").strip()
        # Some declines are honest reports of a *retrieval* failure -- real_009
        # declined saying "the relevant WMP activities table is not retrieved",
        # which was true. Offering every table as selectable on the normal path
        # was measured and reverted (51.20 against 54.79), because it gave the
        # planner more ways to be wrong than to be right. Offering them only
        # here is a different trade: the narrow path has already returned
        # nothing, so the downside is bounded by "still nothing".
        retry_candidates, retry_contexts = candidates, contexts
        if _blames_retrieval(reason) and manifest:
            widened, widened_contexts = _all_table_candidates(
                manifest, contexts, conn, contracts
            )
            if widened:
                retry_candidates, retry_contexts = widened, widened_contexts
        retry_prompt = _retry_prompt(
            _render_prompt(
                question,
                retry_candidates,
                retry_contexts,
                _entity_card_options(cards),
                manifest,
            ),
            reason,
        )
        retried = _ask_planner(provider, retry_prompt)
        if retried is not None and retried.get("action") == "plan":
            logger.info("Excel planner recovered from a decline: %s", reason[:120])
            payload = retried
            candidates, contexts = retry_candidates, retry_contexts
        else:
            last_rejections.clear()
            last_rejections.append(f"declined, and declined again on retry: {reason}")
            return []

    executed: list[tuple[ExcelQueryPlan, ExcelExecutionResult]] = []
    rejections: list[str] = []
    follow_ups = payload.get("follow_up_plans")
    # A follow-up carries only a plan body; ``action`` lives once on the
    # envelope. Without it every follow-up sanitises as "model declined".
    bodies = [payload] + [
        {**body, "action": "plan"}
        for body in (follow_ups if isinstance(follow_ups, list) else [])
        if isinstance(body, dict)
    ][: _max_follow_up_plans()]
    for body in bodies:
        # A rejected primary plan used to discard every follow-up with it, so
        # one bad first guess threw away queries that had already validated.
        # Each body stands or falls alone.
        outcome = _execute_one(
            body, question, candidates, contexts, conn, contracts,
            rejections=rejections,
        )
        if outcome is None:
            continue
        if any(plan == outcome[0] for plan, _ in executed):
            continue
        executed.append(outcome)
    executed = _within_row_budget(executed, rejections)
    last_rejections.clear()
    last_rejections.extend(rejections)
    return executed


def _within_row_budget(executed, rejections: list[str]):
    """Every plan that fits the evidence budget, in plan order.

    The first plan is always kept -- a question that produced exactly one
    large result must still get it. After that a plan is skipped only if it
    would push the total over budget, and later, smaller plans are still
    considered: a compact aggregate is usually worth more to the answer than
    another hundred detail rows.
    """
    kept, rows = [], 0
    for plan, result in executed:
        count = len(getattr(result, "rows", None) or ())
        if kept and rows + count > MAX_TOTAL_EVIDENCE_ROWS:
            rejections.append(
                f"dropped a {count}-row plan on table "
                f"{getattr(plan, 'table_number', '?')}: "
                f"evidence budget of {MAX_TOTAL_EVIDENCE_ROWS} rows already at {rows}"
            )
            continue
        kept.append((plan, result))
        rows += count
    return kept


_UNIT_REFUSAL_RE = re.compile(r"across incompatible units", re.I)
# Ordered by how much a reader gets from the label.
_UNIT_SEPARATING_GROUPS = ("entity_key", "title", "record_id")


def _regroup_for_units(plan: ExcelQueryPlan, exc: PlanError) -> ExcelQueryPlan | None:
    """The same plan grouped so each row carries one unit, or None.

    Returns None for anything that is not a unit refusal, and for a plan the
    executor would refuse again, so the caller's rejection path is unchanged
    in every other case.
    """
    if not _UNIT_REFUSAL_RE.search(str(exc)):
        return None
    if plan.source != RECORDS or plan.operation == "select":
        return None
    columns = RECORD_COLUMNS
    existing = {
        entry.field if isinstance(entry, GroupKey) else str(entry)
        for entry in plan.group_by
    }
    key = next(
        (name for name in _UNIT_SEPARATING_GROUPS
         if name in columns and name not in existing),
        None,
    )
    if key is None:
        return None
    # Each activity is one unit, so the per-activity rows are comparable and
    # the cycle total the question asked for is a sum over that activity's
    # years -- which the deterministic roll-up computes.
    return replace(plan, group_by=(*plan.group_by, key), limit=max(plan.limit, 50))


def _execute_one(
    payload: dict,
    question: str,
    candidates,
    contexts,
    conn,
    contracts: ContractSet,
    *,
    rejections: list[str] | None = None,
) -> tuple[ExcelQueryPlan, ExcelExecutionResult] | None:
    """Sanitise, execute and guard one plan body. None means discard it."""
    def reject(reason: str) -> None:
        if rejections is not None:
            rejections.append(reason)

    try:
        plan = _sanitize_plan(payload, question, candidates, contexts)
        plan = _merge_question_dimensions(plan, question, conn)
        result = execute_plan(plan, conn, contracts=contracts)
    except _PlanRejected as exc:
        logger.info("Excel model plan rejected: %s", exc)
        reject(f"rejected: {exc}")
        return None
    except PlanError as exc:
        # A unit refusal is the executor saying "this sum spans Poles, Trees
        # and Miles" -- which is right, and which is also almost never what
        # the question wanted. real_009 asks for cumulative targets *per
        # activity*, and per-activity groups are exactly what the guard
        # already permits. Repair the plan once by grouping on the entity
        # rather than losing the query: this is a deterministic rewrite to
        # a shape the same guard accepts, not a retry of the same statement.
        repaired = _regroup_for_units(plan, exc) if plan is not None else None
        if repaired is not None:
            try:
                result = execute_plan(repaired, conn, contracts=contracts)
            except (PlanError, ValueError, TypeError) as repair_exc:
                reject(f"refused by executor: {exc}; repair also refused: {repair_exc}")
                return None
            except psycopg2.Error as repair_exc:
                conn.rollback()
                reject(f"repair failed in the database: {repair_exc}")
                return None
            logger.info("Excel plan repaired by grouping on the entity: %s", exc)
            plan = repaired
        else:
            # Includes ClarificationNeeded; surface through the heuristic path.
            logger.info("Excel model plan refused by executor: %s", exc)
            reject(f"refused by executor: {exc}")
            return None
    except (ValueError, TypeError) as exc:
        logger.warning("Excel model plan could not be built: %s", exc)
        reject(f"could not build: {exc}")
        return None
    except psycopg2.Error as exc:
        # A statement the DSL let through but Postgres refused. Without this
        # the exception escapes the whole question, and -- worse -- leaves the
        # connection in a failed transaction, so every *later* plan for the
        # same question dies on "current transaction is aborted" and the
        # question is reported as having no Excel evidence. Roll back and
        # reject only this plan.
        conn.rollback()
        logger.warning("Excel model plan failed in the database: %s", exc)
        reject(f"database error: {str(exc).strip().splitlines()[0]}")
        return None
    if plan.operation == "select":
        selected_indexes = [
            index
            for index, column in enumerate(result.columns)
            if column.startswith("selected_")
        ]
        has_selected_value = any(
            row[index] is not None
            for row in result.rows
            for index in selected_indexes
        )
        if selected_indexes and not has_selected_value:
            reject("select returned no values for the requested attributes")
            return None
        if not result.rows:
            reject("select returned no rows")
            return None
    elif not result.is_answer:
        reject("aggregate returned no usable value")
        return None
    # Mirror the heuristic channel's "nothing matched" guards: zero matching
    # evidence means the plan does not fit the question, not that the answer
    # is zero.
    if plan.source == FACTS and result.contributing_facts == 0:
        reject("no facts matched the plan's scope")
        return None
    if (
        plan.source == RECORDS
        and plan.operation == "aggregate"
        and plan.aggregate == "count"
        and result.rows
        and not result.rows[0][-1]
    ):
        reject("count matched zero records")
        return None
    return plan, result
