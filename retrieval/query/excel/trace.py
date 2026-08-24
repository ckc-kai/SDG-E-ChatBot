"""JSON-safe traces of what the Excel channel actually did.

Before this module an evaluation run recorded ``verified_excel: true`` and
nothing else, so a wrong figure could not be attributed: it was impossible to
tell whether the answering model had read a number out of an executed query,
off a card's prose, or invented it. Worse, ``verified_excel`` was satisfied by
any plan that returned a non-empty row, including one whose filters were
compiled as text comparisons and returned the wrong rows.

A trace records the plan, the statement that ran, and a bounded sample of what
came back, so ``eval/excel_lane_report.py`` can answer the only question that
matters when a number is wrong: which layer produced it.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from datetime import date, datetime
from typing import Any

# A trace is diagnostic, not evidence. Keep it small enough that writing one
# per case does not turn answers.json into a database dump.
MAX_TRACE_ROWS = 25


def _jsonable(value: Any) -> Any:
    """Coerce psycopg2 and dataclass values into something json.dump accepts."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def plan_trace(plan: Any) -> dict[str, Any]:
    """The plan as data, with its filters flattened for readability."""
    if plan is None:
        return {}
    return {
        "table_number": getattr(plan, "table_number", None),
        "source": getattr(plan, "source", None),
        "operation": getattr(plan, "operation", None),
        "aggregate": getattr(plan, "aggregate", None),
        "semantic_metric_key": getattr(plan, "semantic_metric_key", None),
        "measure_name": getattr(plan, "measure_name", None),
        "filters": [
            {
                "field": flt.field,
                "operator": flt.operator,
                "value": _jsonable(flt.value),
                "json_key": flt.json_key,
                "cast": getattr(flt, "cast", None),
            }
            for flt in getattr(plan, "filters", ())
        ],
        "group_by": _jsonable(getattr(plan, "group_by", ())),
        "select_json_keys": _jsonable(getattr(plan, "select_json_keys", ())),
        "having": _jsonable(getattr(plan, "having", None)),
        "order_by": getattr(plan, "order_by", None),
        "descending": getattr(plan, "descending", None),
        "limit": getattr(plan, "limit", None),
    }


def answer_trace(answer: Any) -> dict[str, Any]:
    """What one executed Excel answer ran and what came back."""
    result = getattr(answer, "result", None)
    rows = list(getattr(result, "rows", []) or [])
    return {
        "outcome": "answer",
        "table_number": getattr(answer, "table_number", None),
        "card_chunk_id": getattr(answer, "card_chunk_id", None),
        "card_caption": getattr(answer, "card_caption", None),
        "card_score": getattr(answer, "card_score", None),
        "bound": _jsonable(getattr(answer, "bound", {}) or {}),
        "plan": plan_trace(getattr(answer, "plan", None)),
        "sql": getattr(result, "sql", None),
        "sql_params": _jsonable(getattr(result, "sql_params", []) or []),
        "columns": list(getattr(result, "columns", []) or []),
        "row_count": len(rows),
        "rows": [_jsonable(row) for row in rows[:MAX_TRACE_ROWS]],
        "rows_truncated": len(rows) > MAX_TRACE_ROWS,
        "unit": getattr(result, "unit", None),
        "contributing_facts": getattr(result, "contributing_facts", None),
        "blank_meanings": _jsonable(getattr(result, "blank_meanings", []) or []),
        "warnings": _jsonable(getattr(result, "warnings", []) or []),
    }


def decline_trace(decline: Any) -> dict[str, Any]:
    """Why the channel refused, which is the more common outcome."""
    return {
        "outcome": "decline",
        "reason": getattr(decline, "reason", str(decline)),
        "card_caption": getattr(decline, "card_caption", None),
        "card_score": getattr(decline, "card_score", None),
        "planner_rejections": list(getattr(decline, "planner_rejections", ()) or ()),
    }


def outcome_trace(outcome: Any) -> list[dict[str, Any]]:
    """Trace whatever ``answer_from_excel_all`` returned, answers or decline."""
    if outcome is None:
        return []
    if isinstance(outcome, tuple):
        return [answer_trace(answer) for answer in outcome]
    if hasattr(outcome, "result"):
        return [answer_trace(outcome)]
    return [decline_trace(outcome)]
