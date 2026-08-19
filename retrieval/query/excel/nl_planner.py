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

from dataclasses import replace

from generation.providers.base import ProviderError
from retrieval.ingest.excel.contracts import ContractSet
from retrieval.query.excel.query import (
    FACT_COLUMNS,
    FACTS,
    MAX_LIMIT,
    OPERATORS,
    RECORD_COLUMNS,
    RECORDS,
    ExcelExecutionResult,
    ExcelQueryPlan,
    Filter,
    PlanError,
    bind_dimensions,
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
    "aggregate": {"type": "string", "enum": ["sum", "avg", "count", "min", "max"]},
    "filters": {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "operator": {
                    "type": "string",
                    "enum": sorted(OPERATORS),
                },
                "value": {"type": ["string", "number"]},
            },
            "required": ["field", "value"],
            "additionalProperties": False,
        },
    },
    "group_by": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
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
            "maxItems": MAX_FOLLOW_UP_PLANS,
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
) -> str:
    lines = [
        "Convert this question about SDG&E quarterly wildfire data into typed",
        "query plans, or decline if no table can answer it.",
        "",
        f"Question: {question}",
        "",
        "Candidate tables (from retrieval, most relevant first):",
    ]
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
    lines += [
        "",
        "Rules:",
        "- A question that asks for two separate figures needs two plans, not a",
        "  decline. Put the first in the top-level plan and each additional one",
        f"  in follow_up_plans (up to {MAX_FOLLOW_UP_PLANS}), each with its own table_number and",
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
        "- 'Which activities did not meet / repeatedly missed targets':",
        "  operation=select on the activity records table with select_json_keys",
        "  ['annual_quant_target','quant_actual_progress_q1_4',",
        "  'quant_target_units'], group_by ['reporting_year','record_id',",
        "  'title'], filters covering the requested years, limit 200; the",
        "  target-versus-actual comparison happens downstream.",
        "- Filters default to equality; use operator gte/lte for ranges,",
        "  ne to exclude a value (e.g. exclude 'Non-HFTD' when comparing tiers).",
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
    grouped = set(plan.group_by)
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
    if table_number not in candidate_tables:
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
            raise _PlanRejected(
                f"table {table_number!r} is not a retrieved candidate"
            )
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
        if operator == "in" and isinstance(value, list):
            # The executor has no IN; keep the restriction as the complement.
            known_values = vocabulary.get(field)
            if known_values and all(str(v) in known_values for v in value):
                rendered_values = {str(v) for v in value}
                filters.extend(
                    Filter(
                        field,
                        operator="ne",
                        value=other,
                        json_key=field not in typed_columns,
                    )
                    for other in known_values
                    if other not in rendered_values
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
            rendered = str(value)
            known_values = vocabulary.get(field)
            if operator == "eq" and known_values and rendered not in known_values:
                raise _PlanRejected(
                    f"value {rendered!r} not in corpus vocabulary for {field!r}"
                )
            filters.append(Filter(field, operator=operator, value=rendered, json_key=True))
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
        select_json_keys = tuple(
            key for key in select_json_keys if key not in typed_columns
        )
        unknown = [key for key in select_json_keys if key not in json_keys]
        if unknown:
            raise _PlanRejected(f"unknown attributes {unknown!r}")
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
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        limit = 8 if group_by else 1
    if mentioned_counts:
        limit = max(limit, max(mentioned_counts.values()))
    period_fields = {flt.field for flt in filters}
    has_entity_filter = "entity_key" in period_fields
    if not period_fields & {"reporting_year", "source_vintage_year"} and not has_entity_filter:
        raise _PlanRejected("plan names no reporting period")
    metric_labeled_select = (
        operation == "select" and "semantic_metric_key" in group_by
    )
    if (
        source == FACTS
        and semantic_key is None
        and not metric_labeled_select
        and not any(flt.json_key for flt in filters)
    ):
        raise _PlanRejected(
            "fact plans need a semantic metric key or a scoping dimension filter"
        )
    if metric_labeled_select:
        # One labeled row per metric of the scope; never truncate the set.
        limit = max(limit, 50)

    descending = bool(payload.get("descending", True))
    if re.search(r"\b(more|most|higher|highest|larger|largest|greater)\b", question, re.I):
        descending = True
    elif re.search(r"\b(fewer|fewest|less|least|lower|lowest|smaller|smallest)\b", question, re.I):
        descending = False

    return ExcelQueryPlan(
        table_number=table_number,
        source=source,
        semantic_metric_key=semantic_key,
        operation=operation,
        aggregate=str(payload.get("aggregate") or ("count" if source == RECORDS else "sum")),
        filters=tuple(filters),
        group_by=group_by,
        select_json_keys=select_json_keys,
        descending=descending,
        limit=limit,
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
    contexts = {
        table_number: _table_context(conn, table_number, contracts)
        for table_number, _, _ in candidates
    }
    prompt = _render_prompt(
        question, candidates, contexts, _entity_card_options(cards)
    )
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
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Excel model planning failed: %s", exc)
        return []

    executed: list[tuple[ExcelQueryPlan, ExcelExecutionResult]] = []
    follow_ups = payload.get("follow_up_plans")
    # A follow-up carries only a plan body; ``action`` lives once on the
    # envelope. Without it every follow-up sanitises as "model declined".
    bodies = [payload] + [
        {**body, "action": "plan"}
        for body in (follow_ups if isinstance(follow_ups, list) else [])
        if isinstance(body, dict)
    ][:MAX_FOLLOW_UP_PLANS]
    for index, body in enumerate(bodies):
        outcome = _execute_one(body, question, candidates, contexts, conn, contracts)
        if outcome is None:
            # A failed follow-up is not a failed answer: the primary plan may
            # still stand on its own.
            if index == 0:
                return []
            continue
        if any(plan == outcome[0] for plan, _ in executed):
            continue
        executed.append(outcome)
    return executed


def _execute_one(
    payload: dict,
    question: str,
    candidates,
    contexts,
    conn,
    contracts: ContractSet,
) -> tuple[ExcelQueryPlan, ExcelExecutionResult] | None:
    """Sanitise, execute and guard one plan body. None means discard it."""
    try:
        plan = _sanitize_plan(payload, question, candidates, contexts)
        plan = _merge_question_dimensions(plan, question, conn)
        result = execute_plan(plan, conn, contracts=contracts)
    except _PlanRejected as exc:
        logger.info("Excel model plan rejected: %s", exc)
        return None
    except PlanError as exc:
        # Includes ClarificationNeeded; surface through the heuristic path.
        logger.info("Excel model plan refused by executor: %s", exc)
        return None
    except (ValueError, TypeError) as exc:
        logger.warning("Excel model plan could not be built: %s", exc)
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
            return None
        if not result.rows:
            return None
    elif not result.is_answer:
        return None
    # Mirror the heuristic channel's "nothing matched" guards: zero matching
    # evidence means the plan does not fit the question, not that the answer
    # is zero.
    if plan.source == FACTS and result.contributing_facts == 0:
        return None
    if (
        plan.source == RECORDS
        and plan.operation == "aggregate"
        and plan.aggregate == "count"
        and result.rows
        and not result.rows[0][-1]
    ):
        return None
    return plan, result
