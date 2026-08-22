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
from dataclasses import dataclass
from typing import Any

from generation.features import feature_enabled
from retrieval.ingest.excel.contracts import ContractSet, load_contracts
from retrieval.query.excel.nl_planner import (
    build_model_plans,
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

# A card this weak is not credible evidence that the question is about a table.
MIN_CARD_SCORE = 0.30
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


def _history_group_by(question: str) -> tuple[str, ...]:
    """Select the typed activity status column when the question requests it."""
    columns = ("reporting_year", "record_id")
    if "status" in question.casefold():
        return (*columns, "status")
    return columns


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


def _has_complete_history_values(
    result: ExcelExecutionResult, *, require_status: bool = False
) -> bool:
    """Every selected history attribute must be present for every row."""
    required = tuple(
        column for column in result.columns if column.startswith("selected_")
    )
    if not required:
        return False
    indexes = tuple(result.columns.index(column) for column in required)
    complete = bool(result.rows) and all(
        all(row[index] is not None for index in indexes) for row in result.rows
    )
    if not complete or not require_status:
        return complete
    if "status" not in result.columns:
        return False
    status_index = result.columns.index("status")
    return all(
        row[status_index] is not None and str(row[status_index]).strip()
        for row in result.rows
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
                group_by=_history_group_by(question),
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
    multiple: bool = False,
) -> ExcelAnswer | tuple[ExcelAnswer, ...] | ExcelDecline:
    """Attempt exact Excel answers; optionally preserve an atomic plan batch."""
    contracts = contracts or load_contracts()
    entity_key = bind_entity_key(question)
    history_answer: ExcelAnswer | None = None
    if entity_key and is_entity_history_question(question):
        exact_history = _exact_history_answer(question, entity_key, conn, contracts)
        requests_spend_pair = all(
            term in question.casefold() for term in ("capex", "opex")
        )
        if not requests_spend_pair or not isinstance(exact_history, ExcelAnswer):
            return exact_history
        # A mixed Table 1 + Table 11 question must not stop after history.
        history_answer = exact_history

    years = bind_years(question)
    if years and history_answer is None:
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

    model_question = question
    if history_answer is not None:
        lowered = question.casefold()
        scope = (
            "Territory"
            if "territory" in lowered
            else "HFTD" if "hftd" in lowered else "requested"
        )
        period = (
            f" in reporting year {years[0]}"
            if years
            else " for the requested reporting period"
        )
        model_question = (
            f"What are the {scope} CAPEX and {scope} OPEX amounts for "
            f"{entity_key}{period}?"
        )

    cards = retrieve(model_question, conn, rewrite_mode="off", lanes=(EXCEL,))
    if not cards:
        return history_answer or ExcelDecline("no Excel card retrieved")

    model_allowed = feature_enabled("model_excel_planner")
    if model_allowed and (
        question_needs_model_shapes(model_question)
        # A question naming an exact activity id without a history intent
        # needs an entity-scoped fact query the heuristics cannot build.
        or entity_key is not None
    ):
        # Comparisons, trends, and rankings need plan shapes the heuristics
        # never emit; a validated model plan answers them exactly.
        answers = _model_plan_answers(model_question, cards, conn, contracts)
        if answers:
            combined = ((history_answer,) if history_answer else ()) + answers
            return combined if multiple else combined[0]
        if history_answer is not None:
            return history_answer

    if history_answer is not None:
        return history_answer

    outcome = _card_answer(question, cards, conn, contracts, min_card_score)
    if isinstance(outcome, ExcelAnswer) or not model_allowed:
        return outcome
    if outcome.reason.startswith("plan refused"):
        # Keep the deliberate dual-year-axis clarification deterministic.
        return outcome
    answers = _model_plan_answers(question, cards, conn, contracts)
    if answers:
        return answers if multiple else answers[0]
    return outcome


def _model_plan_answers(
    question: str,
    cards,
    conn,
    contracts: ContractSet,
) -> tuple[ExcelAnswer, ...]:
    """Wrap every validated model plan without collapsing sibling outputs."""
    outcomes = build_model_plans(question, cards, conn, contracts)
    if not outcomes:
        return ()
    answers: list[ExcelAnswer] = []
    for plan, result in outcomes:
        card = next(
            (
                item
                for item in cards
                if (item.query_object.structured_data or {}).get("table_number")
                == plan.table_number
            ),
            cards[0],
        )
        answers.append(
            ExcelAnswer(
                question=question,
                card_chunk_id=card.query_object.chunk_id,
                card_caption=card.query_object.caption or "",
                card_score=card.rerank_score,
                table_number=plan.table_number,
                semantic_metric_key=plan.semantic_metric_key,
                plan=plan,
                result=result,
                bound={"model_plan": True, "batch_size": len(outcomes)},
            )
        )
    return tuple(answers)


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
    if not _has_complete_history_values(
        result, require_status="status" in question.casefold()
    ):
        return ExcelDecline(
            "one or more requested years lacks a requested history value",
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
    """History select for an activity resolved by its retrieved entity card."""
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
    plan = ExcelQueryPlan(
        table_number=1,
        source=RECORDS,
        operation="select",
        filters=tuple(filters),
        group_by=_history_group_by(question),
        select_json_keys=_history_select_keys(question),
        descending=False,
        limit=max(3, years[-1] - years[0] + 1),
    )
    try:
        result = execute_plan(plan, conn, contracts=contracts)
    except PlanError:
        return None
    bound = {"entity_key": entity_key, "reporting_years": years}
    _keep_requested_years(bound, result)
    if not result.rows or not _has_complete_history_values(
        result,
        require_status="status" in question.casefold(),
    ):
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
    card_entity_key = card_data.get("entity_key")
    card_years = years
    if (
        table_number == 1
        and not is_entity_history
        and card_entity_key
        and card_years
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
