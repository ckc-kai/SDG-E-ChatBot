"""Rendering an executed result must not assume one particular plan shape."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.generation_service import _verified_excel_chunks  # noqa: E402

_HISTORY_KEYS = (
    "annual_quant_target",
    "quant_actual_progress_q1_4",
    "quant_target_units",
)


@dataclass
class _Plan:
    select_json_keys: tuple[str, ...]


@dataclass
class _Result:
    columns: list[str]
    rows: list[tuple]
    provenance: list[dict]
    unit: str | None = None
    contributing_facts: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Answer:
    question: str
    table_number: int
    plan: _Plan
    result: _Result
    unit: str | None = None
    card_chunk_id: int = 1
    card_caption: str = "Table 1"
    card_score: float = 0.5
    semantic_metric_key: str | None = None


def _answer(columns: list[str], rows: list[tuple]) -> _Answer:
    return _Answer(
        question="which activities?",
        table_number=1,
        plan=_Plan(_HISTORY_KEYS),
        result=_Result(columns=columns, rows=rows, provenance=[{}] * len(rows)),
    )


@pytest.mark.parametrize(
    "identifier", ["record_id", "entity_key", "title"]
)
def test_history_rendering_accepts_any_row_identifier(identifier):
    """The planner chooses its own grouping; record_id is no longer implied.

    Grouping by entity_key -- which reads far better than an opaque record id
    -- used to raise ValueError inside answer generation and abort the run.
    """
    columns = ["reporting_year", identifier, "selected_0", "selected_1", "selected_2"]
    rows = [(2024, "WMP.483", "0.0", "2225.0", "Structures")]

    chunks = _verified_excel_chunks(_answer(columns, rows))

    assert chunks
    assert any("2225" in chunk.content for chunk in chunks)


def test_history_rendering_survives_a_result_with_no_identifier_column():
    columns = ["reporting_year", "selected_0", "selected_1", "selected_2"]
    rows = [(2024, "0.0", "2225.0", "Structures")]

    chunks = _verified_excel_chunks(_answer(columns, rows))

    assert chunks


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_target_is_reported_not_fatal(blank):
    """An activity with no target that year is a fact worth stating.

    Raising on it aborted answer generation entirely once plans widened past
    single-activity histories, where every value happened to be populated.
    """
    columns = ["reporting_year", "entity_key", "selected_0", "selected_1", "selected_2"]
    rows = [(2024, "WMP.475", blank, "0.54", "Miles")]

    chunks = _verified_excel_chunks(_answer(columns, rows))

    assert chunks
    joined = "\n".join(chunk.content for chunk in chunks)
    assert "not reported" in joined
    assert "0.54" in joined


def test_a_large_result_still_renders_one_chunk_per_row():
    """The compact single-table form was measured and reverted.

    One chunk per row scored 54.79 against 51.23 for a single table chunk,
    every replicate better, because a row that is its own chunk is separately
    citable and carries its own provenance. See
    logs/progress/2026-08-19-extended.md.
    """
    columns = ["reporting_year", "entity_key", "selected_0", "selected_1", "selected_2"]
    rows = [(2024, f"WMP.{index}", "1.0", "2.0", "Miles") for index in range(60)]

    chunks = _verified_excel_chunks(_answer(columns, rows))

    assert len(chunks) > 10


def test_a_small_history_renders_per_row():
    columns = ["reporting_year", "entity_key", "selected_0", "selected_1", "selected_2"]
    rows = [(year, "WMP.483", "10.0", "12.0", "Structures") for year in (2023, 2024)]

    chunks = _verified_excel_chunks(_answer(columns, rows))

    assert len(chunks) > 1


def test_a_mixed_unit_warning_reaches_the_evidence():
    """A grouped result spanning units must say so rather than be refused."""
    columns = ["semantic_metric_key", "value"]
    rows = [("overhead_circuit_miles", 100), ("number_of_ignitions", 5)]
    answer = _answer(columns, rows)
    answer.plan = _Plan(())
    answer.result.warnings = ["This result spans several units (Circuit miles, # ignitions)"]

    chunks = _verified_excel_chunks(answer)

    assert any("spans several units" in chunk.content for chunk in chunks)
