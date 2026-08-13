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
from dataclasses import dataclass
from typing import Any

from retrieval.ingest.excel.contracts import ContractSet, load_contracts
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
_TABLE1_HISTORY_FIELDS = (
    "annual_quant_target",
    "quant_actual_progress_q1_4",
    "quant_target_units",
)


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


def _has_complete_history_values(result: ExcelExecutionResult) -> bool:
    """The deterministic history renderer requires target, actual, and unit."""
    required = ("selected_0", "selected_1", "selected_2")
    if not all(column in result.columns for column in required):
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
                select_json_keys=_TABLE1_HISTORY_FIELDS,
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
    contracts = contracts or load_contracts()
    lowered = question.lower()
    entity_key = bind_entity_key(question)
    if entity_key and is_entity_history_question(question):
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

    cards = retrieve(question, conn, rewrite_mode="off", lanes=(EXCEL,))
    if not cards:
        return ExcelDecline("no Excel card retrieved")

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
    is_entity_history = (
        table_number == 1
        and bind_entity_key(question) is not None
        and any(cue in lowered for cue in _ENTITY_HISTORY_CUES)
    )
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
