"""Execution-verified Excel answering — a separate channel, not a peer lane.

Putting Excel cards into the same ranked list as PDF chunks does not work, and
score calibration cannot rescue it: a card for "overhead circuit miles" is
genuinely topically similar to prose about overhead circuit miles, so no
monotonic score transform can recover the user's intent. Measured cost of trying:
narrative hit@1 collapses from 0.847 to 0.207.

So Excel is gated differently. Rather than *predicting* whether a question is an
Excel question, this module *attempts* the Excel answer and lets the executor
decide:

1. retrieve the best card from the Excel lane alone;
2. build a query plan from that card's reviewed ``filter_spec`` metadata, binding
   only a year, a quarter, and dimension values drawn from the database;
3. execute it under full contract validation.

A plan that validates and returns a non-null row is positive evidence the
question was answerable from the spreadsheets. Anything else — no card, no
bindable period, refused plan, empty result — declines, and the PDF lanes answer
instead. Generate-and-verify, not classify-and-hope.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from generation.features import feature_enabled
from retrieval.ingest.excel.contracts import ContractSet, load_contracts
from retrieval.query.excel.nl_planner import (
    build_model_plans,
    last_rejections,
    question_needs_model_shapes,
)
from retrieval.query.excel.query import (
    FACTS,
    RECORDS,
    ExcelExecutionResult,
    ExcelQueryPlan,
    Filter,
    PlanError,
    bind_entity_key,
    bind_dimensions,
    bind_period,
    bind_years,
    dimension_vocabulary,
    execute_plan,
)
from retrieval.query.lanes import EXCEL
from retrieval.query.pdf.query import retrieve

logger = logging.getLogger(__name__)

# Formerly 0.30, as a pre-filter against questions that are not about a table.
# It was rejecting the *correct* card on nine of the sixteen Excel-scope beta
# questions -- "Table 1 - WMP activities" at 0.05, "Table 11 - WMP spend by
# initiative" at 0.10 -- because a cross-encoder trained on passage relevance
# scores a terse catalogue descriptor ("Stored records: 128") near zero however
# well it matches. The score is not a credibility signal on this text shape.
#
# The gate was also redundant. This module is generate-and-verify by design: a
# card that does not describe the question yields a plan that fails contract
# validation or returns no row, and declines there instead. Letting execution
# decide costs one bounded query and removes a threshold nothing could calibrate.
MIN_CARD_SCORE = 0.0
# The corpus reports 2023 Q4 through 2025 Q4, so an activity's entire recorded
# history fits well inside this many rows.
_FULL_HISTORY_ROW_LIMIT = 12
# Entity tables answer record lookups; metric tables answer aggregates.
ENTITY_TABLES = {1: "wmp_activity", 13: "work_order"}

# A count over an entity table returns a number for almost any filter, so
# execution success is weak evidence there. Require the question to actually ask
# for a count, otherwise a narrative question can be handed a meaningless tally.
_COUNT_CUES = ("how many", "number of", "count of", "how much")
_ENTITY_HISTORY_CUES = (
    "target",
    "actual",
    "progress",
    "percent complete",
    "completion",
    "cumulative",
)
_ENTITY_SERIES_CUES = (
    *_ENTITY_HISTORY_CUES,
    "reported",
    "numbers",
    "values",
    "trend",
    "chart",
    "graph",
)
_TABLE1_HISTORY_FIELDS = (
    "annual_quant_target",
    "quant_actual_progress_q1_4",
    "quant_target_units",
)
_TARGET_ONLY_CUES = ("target",)
_ACTUAL_ONLY_CUES = ("actual", "progress")
_FULL_HISTORY_CUES = (
    "percent",
    "complete",
    "completion",
    "cumulative",
    "status",
    "compare",
    "versus",
    " vs",
)


def _history_select_keys(question: str) -> tuple[str, ...]:
    """Narrow the history select to the attribute the question actually asks."""
    lowered = question.casefold()
    wants_target = any(cue in lowered for cue in _TARGET_ONLY_CUES)
    wants_actual = any(cue in lowered for cue in _ACTUAL_ONLY_CUES)
    wants_full = any(cue in lowered for cue in _FULL_HISTORY_CUES)
    if wants_full or (wants_target and wants_actual) or not (wants_target or wants_actual):
        return _TABLE1_HISTORY_FIELDS
    if wants_target:
        return ("annual_quant_target",)
    return ("quant_actual_progress_q1_4",)


def is_entity_history_question(question: str) -> bool:
    """True when an exact WMP id and time-series value request are explicit."""
    lowered = question.casefold()
    return (
        bind_entity_key(question) is not None
        and bool(bind_years(question))
        and any(cue in lowered for cue in _ENTITY_HISTORY_CUES)
    )


@dataclass
class ExcelAnswer:
    question: str
    card_chunk_id: int
    card_caption: str
    card_score: float
    table_number: int
    semantic_metric_key: str | None
    plan: ExcelQueryPlan
    result: ExcelExecutionResult
    bound: dict[str, Any]

    @property
    def unit(self) -> str | None:
        return self.result.unit


@dataclass
class ExcelDecline:
    reason: str
    card_caption: str | None = None
    card_score: float | None = None
    # Why each model plan was thrown out, when one was attempted. A decline
    # that only says "entity-table card" hides the fact that the planner had
    # already produced something and the executor refused it.
    planner_rejections: tuple[str, ...] = ()


def _exact_entity_card(conn, entity_key: str) -> tuple[int, str, dict] | None:
    """Resolve an exact reviewed entity card without semantic retrieval."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, caption, structured_data
            FROM chunks
            WHERE content_type = 'excel_card'
              AND structured_data ->> 'entity_key' = %s
            ORDER BY id
            LIMIT 1
            """,
            (entity_key,),
        )
        row = cur.fetchone()
    return (row[0], row[1] or "", row[2] or {}) if row else None


def _missing_requested_years(
    bound: dict[str, Any], result: ExcelExecutionResult
) -> tuple[int, ...]:
    requested = tuple(bound.get("reporting_years", ()))
    if not requested or "reporting_year" not in result.columns:
        return ()
    year_index = result.columns.index("reporting_year")
    returned = {int(row[year_index]) for row in result.rows}
    return tuple(year for year in requested if year not in returned)


def _has_complete_history_values(result: ExcelExecutionResult) -> bool:
    """Every selected history attribute must be present for every row."""
    required = tuple(
        column for column in result.columns if column.startswith("selected_")
    )
    if not required:
        return False
    indexes = tuple(result.columns.index(column) for column in required)
    return bool(result.rows) and all(
        all(row[index] is not None for index in indexes) for row in result.rows
    )


def _keep_requested_years(bound: dict[str, Any], result: ExcelExecutionResult) -> None:
    """Remove interval rows that were not explicitly requested by the user."""
    requested = set(bound.get("reporting_years", ()))
    if not requested or "reporting_year" not in result.columns:
        return
    year_index = result.columns.index("reporting_year")
    kept_indexes = [
        index
        for index, row in enumerate(result.rows)
        if int(row[year_index]) in requested
    ]
    result.rows = [result.rows[index] for index in kept_indexes]
    result.provenance = [
        result.provenance[index]
        for index in kept_indexes
        if index < len(result.provenance)
    ]


def _plan_for_card(
    question: str,
    card_data: dict,
    conn,
    contracts: ContractSet,
) -> tuple[ExcelQueryPlan, dict[str, Any]]:
    table_number = int(card_data["table_number"])
    contracts.for_table(table_number)  # validates the table is reviewed
    semantic_key = card_data.get("semantic_metric_key")

    is_entity_table = table_number in ENTITY_TABLES
    bound: dict[str, Any] = dict(bind_period(question))
    entity_key = bind_entity_key(question)
    years = bind_years(question)
    lowered = question.casefold()

    if (
        table_number == 1
        and entity_key
        and years
        and any(cue in lowered for cue in _ENTITY_HISTORY_CUES)
    ):
        bound.update(entity_key=entity_key, reporting_years=years)
        filters = [Filter("entity_key", value=entity_key)]
        if len(years) == 1:
            filters.append(Filter("reporting_year", value=years[0]))
        else:
            filters.extend(
                [
                    Filter("reporting_year", operator="gte", value=years[0]),
                    Filter("reporting_year", operator="lte", value=years[-1]),
                ]
            )
        return (
            ExcelQueryPlan(
                table_number=table_number,
                source=RECORDS,
                operation="select",
                filters=tuple(filters),
                group_by=("reporting_year", "record_id"),
                select_json_keys=_history_select_keys(question),
                descending=False,
                limit=max(3, years[-1] - years[0] + 1),
            ),
            bound,
        )

    vocabulary = dimension_vocabulary(
        conn, table_number, source=RECORDS if is_entity_table else FACTS
    )
    dimensions = bind_dimensions(question, vocabulary)

    filters: list[Filter] = []
    promoted = {"hftd_tier", "line_type"}
    for field, value in bound.items():
        filters.append(Filter(field, value=value))
    for field, value in dimensions.items():
        filters.append(Filter(field, value=value, json_key=field not in promoted))
    bound.update(dimensions)

    plan = ExcelQueryPlan(
        table_number=table_number,
        source=RECORDS if is_entity_table else FACTS,
        semantic_metric_key=None if is_entity_table else semantic_key,
        operation="aggregate",
        aggregate="count" if is_entity_table else "sum",
        filters=tuple(filters),
    )
    return plan, bound


def answer_from_excel(
    question: str,
    conn,
    *,
    contracts: ContractSet | None = None,
    min_card_score: float = MIN_CARD_SCORE,
) -> ExcelAnswer | ExcelDecline:
    """Attempt an exact Excel answer; decline rather than guess."""
    outcome = answer_from_excel_all(
        question, conn, contracts=contracts, min_card_score=min_card_score
    )
    return outcome[0] if isinstance(outcome, tuple) else outcome


def answer_from_excel_all(
    question: str,
    conn,
    *,
    contracts: ContractSet | None = None,
    min_card_score: float = MIN_CARD_SCORE,
) -> tuple[ExcelAnswer, ...] | ExcelDecline:
    """Every exact Excel answer the question supports, or one decline.

    A question can legitimately need more than one query -- a tier total and
    the segments behind it, or the same metric on two tables -- and returning
    only the first threw away figures the model then had to invent or refuse.
    """
    contracts = contracts or load_contracts()
    entity_key = bind_entity_key(question)
    if entity_key and is_entity_history_question(question):
        exact = _exact_history_answer(question, entity_key, conn, contracts)
        return (exact,) if isinstance(exact, ExcelAnswer) else exact

    years = bind_years(question)
    if years:
        if any(cue in question.casefold() for cue in _ENTITY_SERIES_CUES):
            entity_card = _resolve_entity_card(question, conn)
            if entity_card is not None and len(years) >= 2:
                entity_key = str(
                    entity_card.query_object.structured_data.get("entity_key")
                    or ""
                )
                history = _card_entity_history_answer(
                    question, entity_card, entity_key, years, conn, contracts
                )
                if history is not None:
                    return history
        resolved = _resolve_metric_card(question, conn)
        if resolved is not None:
            if len(years) >= 2:
                history = _card_fact_history_answer(
                    question, resolved, years, conn, contracts
                )
                if history is not None:
                    return history
            else:
                exact = _card_answer(
                    question, (resolved,), conn, contracts, min_card_score
                )
                if isinstance(exact, ExcelAnswer):
                    return exact

    cards = retrieve(question, conn, rewrite_mode="off", lanes=(EXCEL,))
    if not cards:
        return ExcelDecline("no Excel card retrieved")

    model_allowed = feature_enabled("model_excel_planner")
    if model_allowed:
        # Formerly gated on a keyword regex for comparison/trend/ranking cues.
        # That regex excluded eleven of the sixteen workbook-dependent beta
        # questions -- every "break down by", "did we ever", "is it possible
        # that" -- and sent them to card heuristics that cannot express a
        # GROUP BY or a predicate scan. It was the same failure as the
        # ``_EXCEL_RE`` routing gate and the ``MIN_CARD_SCORE`` floor: a hand
        # written classifier starving the one component able to answer.
        #
        # The channel is generate-and-verify, so attempting costs one bounded
        # query when the planner has nothing useful to say. ``question_needs_
        # model_shapes`` survives only as a hint in the prompt.
        answers = _model_plan_answers(question, cards, conn, contracts)
        if answers:
            return tuple(answers)

    outcome = _card_answer(question, cards, conn, contracts, min_card_score)
    if isinstance(outcome, ExcelAnswer):
        return (outcome,)
    if model_allowed and last_rejections:
        outcome = replace(outcome, planner_rejections=tuple(last_rejections))
    return outcome


def _model_plan_answer(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
) -> ExcelAnswer | None:
    """The first validated model plan, for callers that want a single answer."""
    answers = _model_plan_answers(question, cards, conn, contracts)
    return answers[0] if answers else None


def _model_plan_answers(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
) -> list[ExcelAnswer]:
    """Wrap every validated, executed model plan in the channel's answer type."""
    return [
        _answer_for_plan(question, cards, plan, result)
        for plan, result in build_model_plans(question, cards, conn, contracts)
    ]


def _answer_for_plan(question: str, cards, plan, result) -> ExcelAnswer:
    card = next(
        (
            item
            for item in cards
            if (item.query_object.structured_data or {}).get("table_number")
            == plan.table_number
        ),
        cards[0],
    )
    return ExcelAnswer(
        question=question,
        card_chunk_id=card.query_object.chunk_id,
        card_caption=card.query_object.caption or "",
        card_score=card.rerank_score,
        table_number=plan.table_number,
        semantic_metric_key=plan.semantic_metric_key,
        plan=plan,
        result=result,
        bound={"model_plan": True},
    )


def _exact_history_answer(
    question: str,
    entity_key: str,
    conn,
    contracts: ContractSet,
) -> ExcelAnswer | ExcelDecline:
    exact_card = _exact_entity_card(conn, entity_key)
    if exact_card is None:
        return ExcelDecline(f"no reviewed Excel entity card for {entity_key}")
    card_chunk_id, card_caption, card_data = exact_card
    try:
        plan, bound = _plan_for_card(question, card_data, conn, contracts)
        result = execute_plan(plan, conn, contracts=contracts)
    except (PlanError, KeyError, ValueError) as exc:
        return ExcelDecline(f"exact entity plan refused: {exc}", card_caption, 1.0)
    _keep_requested_years(bound, result)
    if not result.is_answer:
        return ExcelDecline(
            "exact entity plan returned no usable value", card_caption, 1.0
        )
    missing_years = _missing_requested_years(bound, result)
    if missing_years:
        return ExcelDecline(
            "missing requested years: "
            + ", ".join(str(year) for year in missing_years),
            card_caption,
            1.0,
        )
    if not _has_complete_history_values(result):
        return ExcelDecline(
            "one or more requested years lacks target, actual, or unit data",
            card_caption,
            1.0,
        )
    return ExcelAnswer(
        question=question,
        card_chunk_id=card_chunk_id,
        card_caption=card_caption,
        card_score=1.0,
        table_number=plan.table_number,
        semantic_metric_key=plan.semantic_metric_key,
        plan=plan,
        result=result,
        bound=bound,
    )


def _card_entity_history_answer(
    question: str,
    card,
    entity_key: str,
    years: tuple[int, ...],
    conn,
    contracts: ContractSet,
) -> ExcelAnswer | None:
    """History select for an activity resolved by its retrieved entity card.

    An empty ``years`` means the question named no year, which for a row select
    is a request for the activity's whole recorded history rather than a reason
    to decline. The reporting-period gate further down exists to stop an
    aggregate from silently summing every year in the corpus; a select
    aggregates nothing, so it does not need that protection.
    """
    filters = [Filter("entity_key", value=entity_key)]
    if len(years) == 1:
        filters.append(Filter("reporting_year", value=years[0]))
    elif years:
        filters.extend(
            [
                Filter("reporting_year", operator="gte", value=years[0]),
                Filter("reporting_year", operator="lte", value=years[-1]),
            ]
        )
    plan = ExcelQueryPlan(
        table_number=1,
        source=RECORDS,
        operation="select",
        filters=tuple(filters),
        group_by=("reporting_year", "record_id"),
        select_json_keys=_history_select_keys(question),
        descending=False,
        limit=(
            max(3, years[-1] - years[0] + 1) if years else _FULL_HISTORY_ROW_LIMIT
        ),
    )
    try:
        result = execute_plan(plan, conn, contracts=contracts)
    except PlanError:
        return None
    bound = {"entity_key": entity_key, "reporting_years": years}
    _keep_requested_years(bound, result)
    if not result.rows or not _has_complete_history_values(result):
        return None
    missing_years = _missing_requested_years(bound, result)
    if missing_years:
        bound["missing_reporting_years"] = missing_years
    return ExcelAnswer(
        question=question,
        card_chunk_id=card.query_object.chunk_id,
        card_caption=card.query_object.caption or "",
        card_score=card.rerank_score,
        table_number=1,
        semantic_metric_key=None,
        plan=plan,
        result=result,
        bound=bound,
    )


# Words too generic to identify one activity out of 128.
_ENTITY_TITLE_STOPWORDS = frozenset(
    {
        "and", "the", "of", "for", "in", "to", "a", "an", "or", "with", "on",
        "wmp", "program", "programme", "programs", "activity", "activities",
        "work", "system", "total", "annual", "quarterly", "sdge", "sdg",
        "inspection", "inspections", "assessment", "assessments", "other",
    }
)


def _question_names_card_entity(question: str, card) -> bool:
    """True when the question itself identifies the card's activity.

    Retrieval ranking is not a claim that the user asked about this entity --
    only that the text was nearby. Requiring the question to carry the
    activity's own distinguishing words keeps an entity lookup from standing in
    for a question about the whole table.
    """
    data = card.query_object.structured_data or {}
    entity_key = str(data.get("entity_key") or "")
    lowered = question.casefold()
    if entity_key and entity_key.casefold() in lowered:
        return True
    title = str(data.get("title") or card.query_object.caption or "")
    # Strip the card's boilerplate prefix so it cannot supply the match.
    title = re.sub(r"(?i)^\s*wmp activity\s*[-\u2014:]*\s*", "", title)
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", title.casefold())
        if len(term) > 3 and term not in _ENTITY_TITLE_STOPWORDS
    }
    if not terms:
        return False
    matched = {term for term in terms if re.search(rf"(?<!\w){term}", lowered)}
    # Most of the activity's distinguishing words, not merely one shared noun.
    return len(matched) >= max(2, (len(terms) + 1) // 2)


_FACT_HISTORY_CUES = (
    "reported",
    "reporting",
    "from",
    "through",
    "between",
    "across",
    "trend",
    "total",
    "how many",
    "how much",
)
_METRIC_GENERIC_TOKENS = {
    "all",
    "amount",
    "count",
    "metric",
    "number",
    "of",
    "reported",
    "total",
    "value",
    "wmp",
    "activity",
}


@dataclass(frozen=True)
class _ResolvedMetricQueryObject:
    chunk_id: int
    caption: str
    structured_data: dict[str, Any]


@dataclass(frozen=True)
class _ResolvedMetricCard:
    query_object: _ResolvedMetricQueryObject
    rerank_score: float = 1.0


def _resolve_entity_card(question: str, conn) -> _ResolvedMetricCard | None:
    """Resolve one reviewed Table 1 activity by its official display name.

    Matching is catalog-driven rather than tied to individual WMP IDs. Exact
    short names beat broader activities, while equally specific matches decline
    so ambiguous phrases never select an arbitrary inspection program.
    """
    question_tokens = _metric_tokens(question)
    if not question_tokens:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, caption, structured_data
            FROM chunks
            WHERE content_type = 'excel_card'
              AND structured_data ->> 'table_number' = '1'
              AND structured_data ->> 'entity_key' IS NOT NULL
            ORDER BY id
            """
        )
        rows = cur.fetchall()

    by_entity: dict[str, tuple[float, int, int, str, dict[str, Any]]] = {}
    for chunk_id, caption, structured_data in rows:
        data = structured_data or {}
        entity_key = str(data.get("entity_key") or "")
        if not entity_key or entity_key in by_entity:
            continue
        display = re.split(
            r"\s+[\u2013\u2014-]\s+", str(caption or ""), maxsplit=1
        )[-1]
        label_tokens = _metric_tokens(display)
        if not label_tokens:
            continue
        overlap = label_tokens & question_tokens
        coverage = len(overlap) / len(label_tokens)
        by_entity[entity_key] = (
            coverage,
            len(overlap),
            int(chunk_id),
            str(caption or ""),
            data,
        )

    ranked = sorted(
        by_entity.values(), key=lambda item: (-item[0], -item[1], item[2])
    )
    if not ranked or ranked[0][0] < 0.75:
        return None
    best_signature = ranked[0][:2]
    if len(ranked) > 1 and ranked[1][:2] == best_signature:
        return None
    coverage, _overlap, chunk_id, caption, data = ranked[0]
    return _ResolvedMetricCard(
        _ResolvedMetricQueryObject(chunk_id, caption, data), coverage
    )


def _metric_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))
    normalized = {
        token[:-1] if len(token) > 4 and token.endswith("s") else token
        for token in tokens
        if token not in _METRIC_GENERIC_TOKENS
    }
    return {token for token in normalized if len(token) >= 3}


def _resolve_metric_card(question: str, conn) -> _ResolvedMetricCard | None:
    """Resolve an unambiguous reviewed metric card without vector ranking."""
    question_tokens = _metric_tokens(question)
    if not question_tokens:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, caption, structured_data
            FROM chunks
            WHERE content_type = 'excel_card'
              AND structured_data ->> 'semantic_metric_key' IS NOT NULL
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    for chunk_id, caption, structured_data in rows:
        data = structured_data or {}
        tokens = _metric_tokens(str(data.get("semantic_metric_key") or ""))
        if not tokens:
            continue
        overlap = tokens & question_tokens
        if not overlap:
            continue
        ranked.append(
            (len(overlap) / len(tokens), int(chunk_id), str(caption or ""), data)
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked or ranked[0][0] < 0.75:
        return None
    if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
        return None
    score, chunk_id, caption, data = ranked[0]
    return _ResolvedMetricCard(
        _ResolvedMetricQueryObject(chunk_id, caption, data), score
    )


def _card_fact_history_answer(
    question: str,
    card,
    years: tuple[int, ...],
    conn,
    contracts: ContractSet,
) -> ExcelAnswer | None:
    """Execute a comparable fact metric across an explicit year interval.

    Unlike Table 1 entity history, reviewed QDR fact tables carry the value in
    ``value_numeric``.  Grouping one semantic metric by reporting year provides
    exact partial evidence while recording any requested year absent from the
    active corpus; the generator can then answer what is known without silently
    claiming full-period coverage.
    """
    if len(years) < 2 or not any(
        cue in question.casefold() for cue in _FACT_HISTORY_CUES
    ):
        return None
    card_data = card.query_object.structured_data or {}
    table_number = card_data.get("table_number")
    semantic_key = card_data.get("semantic_metric_key")
    if table_number in ENTITY_TABLES or not isinstance(table_number, int):
        return None
    if not semantic_key:
        return None
    try:
        vocabulary = dimension_vocabulary(conn, table_number, source=FACTS)
        dimensions = bind_dimensions(question, vocabulary)
        filters = [
            Filter("reporting_year", operator="gte", value=years[0]),
            Filter("reporting_year", operator="lte", value=years[-1]),
            *(
                Filter(
                    field,
                    value=value,
                    json_key=field not in {"hftd_tier", "line_type"},
                )
                for field, value in dimensions.items()
            ),
        ]
        plan = ExcelQueryPlan(
            table_number=table_number,
            source=FACTS,
            semantic_metric_key=str(semantic_key),
            operation="aggregate",
            aggregate="sum",
            filters=tuple(filters),
            group_by=("reporting_year",),
            descending=False,
            limit=max(3, len(years)),
        )
        result = execute_plan(plan, conn, contracts=contracts)
    except (PlanError, KeyError, TypeError, ValueError):
        return None
    if not result.is_answer:
        return None
    bound: dict[str, Any] = {
        "reporting_years": years,
        "dimensions": dimensions,
    }
    missing_years = _missing_requested_years(bound, result)
    if missing_years:
        bound["missing_reporting_years"] = missing_years
    return ExcelAnswer(
        question=question,
        card_chunk_id=card.query_object.chunk_id,
        card_caption=card.query_object.caption or "",
        card_score=card.rerank_score,
        table_number=table_number,
        semantic_metric_key=str(semantic_key),
        plan=plan,
        result=result,
        bound=bound,
    )


def _card_answer(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
    min_card_score: float,
) -> ExcelAnswer | ExcelDecline:
    lowered = question.lower()
    best = cards[0]
    card_data = best.query_object.structured_data or {}
    if best.rerank_score < min_card_score:
        return ExcelDecline(
            "best card scored below the credibility floor",
            card_caption=best.query_object.caption,
            card_score=best.rerank_score,
        )
    if "table_number" not in card_data:
        return ExcelDecline(
            "card carries no table reference",
            card_caption=best.query_object.caption,
            card_score=best.rerank_score,
        )

    table_number = card_data.get("table_number")
    years = bind_years(question)
    if table_number not in ENTITY_TABLES and years:
        history = _card_fact_history_answer(
            question, best, years, conn, contracts
        )
        if history is not None:
            return history
    is_entity_history = (
        table_number == 1
        and bind_entity_key(question) is not None
        and any(cue in lowered for cue in _ENTITY_HISTORY_CUES)
    )

    # The question names an activity without its WMP id; the retrieved entity
    # card resolves the id, so the same deterministic history select applies.
    #
    # Only when the question actually names that activity. The cue test alone
    # fires on any question containing "target" or "actual", so "is it possible
    # we reported completed work against a zero target?" -- a scan over all 128
    # activities -- was answered with the history of whichever activity ranked
    # first, and reported as execution-verified. A property question is a scan,
    # not a lookup.
    card_entity_key = card_data.get("entity_key")
    card_years = years
    if (
        table_number == 1
        and not is_entity_history
        and card_entity_key
        and _question_names_card_entity(question, best)
        and any(cue in lowered for cue in _ENTITY_HISTORY_CUES)
    ):
        answer = _card_entity_history_answer(
            question, best, str(card_entity_key), card_years, conn, contracts
        )
        if answer is not None:
            return answer
    if (
        table_number in ENTITY_TABLES
        and not is_entity_history
        and not any(cue in lowered for cue in _COUNT_CUES)
    ):
        return ExcelDecline(
            "entity-table card but the question does not ask for a count",
            best.query_object.caption,
            best.rerank_score,
        )
    if table_number not in ENTITY_TABLES and not card_data.get("semantic_metric_key"):
        return ExcelDecline(
            "card names no metric concept, so a sum would span the whole table",
            best.query_object.caption,
            best.rerank_score,
        )

    try:
        plan, bound = _plan_for_card(question, card_data, conn, contracts)
    except (PlanError, KeyError, ValueError) as exc:
        return ExcelDecline(
            f"could not build a valid plan: {exc}",
            best.query_object.caption,
            best.rerank_score,
        )

    # A period is the minimum specificity for a defensible exact answer;
    # without one, a "sum" silently spans every year in the corpus.
    if not {
        "reporting_year",
        "reporting_years",
        "source_vintage_year",
    }.intersection(bound):
        return ExcelDecline(
            "question names no reporting period",
            best.query_object.caption,
            best.rerank_score,
        )

    try:
        result = execute_plan(plan, conn, contracts=contracts)
    except PlanError as exc:
        # Includes the deliberate tables 14/15 year-basis clarification.
        return ExcelDecline(
            f"plan refused: {exc}", best.query_object.caption, best.rerank_score
        )

    if not result.is_answer:
        return ExcelDecline(
            "plan returned no usable value",
            best.query_object.caption,
            best.rerank_score,
        )

    # "Nothing matched" is not evidence that the question was an Excel question.
    # An entity count of zero, or a sum with no contributing facts, means the
    # retrieved card was simply wrong — so decline instead of reporting a 0.
    scalar = result.rows[0][-1]
    if plan.source == RECORDS and not scalar:
        return ExcelDecline(
            "no matching records, so the card does not fit the question",
            best.query_object.caption,
            best.rerank_score,
        )
    if plan.source == FACTS and result.contributing_facts == 0:
        return ExcelDecline(
            "no contributing facts, so the card does not fit the question",
            best.query_object.caption,
            best.rerank_score,
        )

    return ExcelAnswer(
        question=question,
        card_chunk_id=best.query_object.chunk_id,
        card_caption=best.query_object.caption or "",
        card_score=best.rerank_score,
        table_number=plan.table_number,
        semantic_metric_key=plan.semantic_metric_key,
        plan=plan,
        result=result,
        bound=bound,
    )
