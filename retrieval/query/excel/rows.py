"""Attach the workbook rows behind a retrieved Excel card to the evidence.

An ``excel_card`` chunk is a catalogue entry: it says a table exists, what it
covers and how many records it holds. It carries no values. So a question whose
answer is a number in the workbook puts the answering model in an impossible
position -- it is shown a description of the data and asked for the data -- and
the model does the correct thing and reports that it cannot answer.

``retrieval.query.excel.channel`` was the intended way out: build a query plan,
validate it against the table contract, execute it. When it fires, it is
authoritative and should be preferred. But it declines whenever the question is
not shaped like a plan it can build, which on the frozen beta set is most of
them, and the fallback was a description with no numbers.

This module is the fallback that should have existed. It does not plan, compute
or aggregate: it selects a bounded window of the rows the card points at,
narrowed by whatever the question makes explicit, and renders them for the
prompt. The model then reads real values instead of a promise of values. Nothing
here decides what the answer is, so an imperfect narrowing costs prompt tokens
rather than correctness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from retrieval.query.excel.query import bind_years


logger = logging.getLogger(__name__)

# Tables 1 and 13 are entity tables held row-per-record; the rest are fact
# tables held as tidy metric rows.
_ENTITY_TABLES = {1, 13}
# Enough rows for a year-by-year or category-by-category reading without
# crowding out narrative evidence. The corpus's widest relevant slice -- one
# metric across every quarter of the cycle -- is well inside this.
MAX_ROWS_PER_CARD = 40
# More than this and the prompt is being filled with catalogue, not evidence.
MAX_CARDS = 3
# Deterministic per-year/per-tier aggregates over each card's scope were built
# and measured here, and removed: they scored 47.42 against 51.37 for the row
# window alone. The totals were correct, but a large block of numbers at the top
# of the prompt pulled attention off narrative evidence, and multi-PDF questions
# lost more than the Excel questions gained.


@dataclass(frozen=True)
class ExcelRowSlice:
    table_number: int
    card_chunk_id: str
    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    truncated: bool

    @property
    def source_file(self) -> str:
        return f"sdge_table{self.table_number:02d}_rag_ready.csv"

    def render(self) -> str:
        """Render as a pipe table, the densest form the model reads reliably."""
        header = " | ".join(self.columns)
        divider = " | ".join("---" for _ in self.columns)
        body = "\n".join(
            " | ".join(
                _render_cell(column, value)
                for column, value in zip(self.columns, row)
            )
            for row in self.rows
        )
        note = (
            f"\n(showing the first {len(self.rows)} matching rows; more exist)"
            if self.truncated
            else ""
        )
        return (
            f"Workbook rows from Table {self.table_number} — {self.caption}\n"
            f"{header}\n{divider}\n{body}{note}"
        )


_FACT_COLUMNS = (
    "metric_name",
    "reporting_year",
    "reporting_quarter",
    "hftd_tier",
    "line_type",
    "value_raw",
    "unit",
    "dimensions",
)
_RECORD_COLUMNS = (
    "entity_key",
    "title",
    "reporting_year",
    "reporting_quarter",
    "status",
    "attributes",
)
_JSON_COLUMNS = {"dimensions", "attributes"}
# A Table 1 record carries dozens of administrative fields; these are the ones
# that answer "what was planned and what happened".
_ATTRIBUTE_ALLOWLIST = (
    "annual_quant_target",
    "quant_actual_progress_q1_4",
    "quant_target_units",
    "percent_complete",
    "units",
    "wmp_category",
    "delay_or_cancellation_reason",
)


def _render_cell(column: str, value) -> str:
    """Flatten a jsonb cell to ``key=value`` pairs; pass anything else through."""
    if value is None:
        return ""
    if column not in _JSON_COLUMNS or not isinstance(value, dict):
        return str(value)
    items = (
        (key, item)
        for key, item in value.items()
        if item not in (None, "", [], {})
        and (column != "attributes" or key in _ATTRIBUTE_ALLOWLIST)
    )
    return "; ".join(f"{key}={item}" for key, item in items) or ""


def _fetch(conn, sql: str, params: list) -> list[tuple]:
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def _slice_for_card(question: str, card, conn) -> ExcelRowSlice | None:
    data = card.query_object.structured_data or {}
    table_number = data.get("table_number")
    if not isinstance(table_number, int):
        return None

    years = bind_years(question)
    params: list = [table_number]
    where = ["r.state = 'active'", "t.table_number = %s"]

    if table_number in _ENTITY_TABLES:
        columns, source = _RECORD_COLUMNS, "excel_records"
        if entity_key := data.get("entity_key"):
            where.append("t.entity_key = %s")
            params.append(str(entity_key))
    else:
        columns, source = _FACT_COLUMNS, "excel_facts"
        # A concept card names the one metric it describes; a table-overview
        # card names none, and then every metric in the table is in scope.
        if metric_key := data.get("semantic_metric_key"):
            where.append("t.semantic_metric_key = %s")
            params.append(str(metric_key))

    if years:
        where.append("t.reporting_year = ANY(%s)")
        params.append(list(years))

    selected = ", ".join(f"t.{column}" for column in columns)
    sql = (
        f"SELECT {selected} FROM {source} t "
        f"JOIN excel_revisions r ON r.id = t.revision_id "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY t.reporting_year NULLS LAST, t.reporting_quarter NULLS LAST "
        f"LIMIT %s"
    )
    # One extra row is fetched purely to detect truncation honestly.
    rows = _fetch(conn, sql, [*params, MAX_ROWS_PER_CARD + 1])
    if not rows:
        return None
    return ExcelRowSlice(
        table_number=table_number,
        card_chunk_id=str(card.query_object.chunk_id),
        caption=card.query_object.caption or f"Table {table_number}",
        columns=columns,
        rows=tuple(rows[:MAX_ROWS_PER_CARD]),
        truncated=len(rows) > MAX_ROWS_PER_CARD,
    )


def fetch_card_rows(question: str, cards, conn) -> tuple[ExcelRowSlice, ...]:
    """Return row windows for the best few retrieved Excel cards."""
    slices: list[ExcelRowSlice] = []
    seen_scopes: set[tuple] = set()
    for card in cards:
        if len(slices) >= MAX_CARDS:
            break
        data = card.query_object.structured_data or {}
        scope = (data.get("table_number"), data.get("semantic_metric_key"),
                 data.get("entity_key"))
        if scope in seen_scopes:
            continue
        seen_scopes.add(scope)
        try:
            row_slice = _slice_for_card(question, card, conn)
        except Exception:
            # Evidence enrichment must never break retrieval.
            logger.exception("Could not read workbook rows for a card")
            continue
        if row_slice is not None:
            slices.append(row_slice)
    return tuple(slices)
