"""Deterministic stage scoring for cross-resource computation runs.

The scorer deliberately separates planning, retrieval, operand verification,
and calculation. Finding both evidence lanes is not counted as a correct
calculation unless the runtime actually emits the expected deterministic
``CalculationResult``.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from collections.abc import Callable
from typing import Any, Iterable


def _coverage(required: list[str], observed: list[str]) -> float:
    if not required:
        return 1.0
    remaining = Counter(observed)
    matched = 0
    for source in required:
        if remaining[source] > 0:
            matched += 1
            remaining[source] -= 1
    return matched / len(required)


def _retrieved_chunk_ids(bundle: Any) -> set[str]:
    return {
        str(result.query_object.chunk_id)
        for group in getattr(bundle.evidence, "groups", {}).values()
        for result in group.results
    }


def _retrieved_sources(bundle: Any) -> list[str]:
    groups = getattr(bundle.evidence, "groups", {})
    sources: list[str] = []
    if any(groups.get(name) and groups[name].results for name in ("narrative", "table", "figure")):
        sources.append("pdf")
    verified = getattr(bundle, "verified_excels", ()) or (
        (bundle.verified_excel,) if getattr(bundle, "verified_excel", None) else ()
    )
    if (groups.get("excel") and groups["excel"].results) or verified:
        sources.append("excel")
    return sources


def _gold_pdf_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(item["chunk_id"])
        for fact in row.get("facts", [])
        if fact.get("source") == "pdf"
        for item in fact.get("provenance", [])
        if item.get("chunk_id") is not None
    }


def _excel_operand_verified(row: dict[str, Any], bundle: Any) -> bool:
    expected = {
        (str(item.get("source_file")), str(item.get("source_row")))
        for fact in row.get("facts", [])
        if fact.get("source") == "excel"
        for item in fact.get("provenance", [])
        if item.get("source_file") is not None and item.get("source_row") is not None
    }
    if not expected:
        return False
    answers = getattr(bundle, "verified_excels", ()) or (
        (bundle.verified_excel,) if getattr(bundle, "verified_excel", None) else ()
    )
    actual = {
        (str(item.get("source_file")), str(item.get("source_row")))
        for answer in answers
        for item in getattr(answer.result, "provenance", ())
        if item.get("source_file") is not None and item.get("source_row") is not None
    }
    return expected.issubset(actual)


def _calculation_correct(row: dict[str, Any], bundle: Any) -> bool:
    try:
        expected = Decimal(str(row["expected_value"]))
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return False
    return any(result.value == expected for result in getattr(bundle, "calculations", ()))


def score_cross_resource_case(
    row: dict[str, Any], plan: Any, bundle: Any
) -> dict[str, Any]:
    """Score one case without upgrading partial evidence to answer success."""
    required = [str(source) for source in row.get("required_sources", [])]
    planned = [str(step.source) for step in plan.steps]
    retrieved_ids = _retrieved_chunk_ids(bundle)
    gold_pdf_ids = _gold_pdf_ids(row)
    calculations = tuple(getattr(bundle, "calculations", ()))
    return {
        "id": row["id"],
        "question": row["question"],
        "plan_source": plan.source,
        "planned_sources": planned,
        "planned_source_coverage": _coverage(required, planned),
        "retrieved_source_coverage": _coverage(required, _retrieved_sources(bundle)),
        "pdf_gold_recall": (
            len(gold_pdf_ids & retrieved_ids) / len(gold_pdf_ids)
            if gold_pdf_ids
            else None
        ),
        "excel_operand_verified": _excel_operand_verified(row, bundle),
        "calculation_produced": bool(calculations),
        "calculation_correct": _calculation_correct(row, bundle),
        "expected_value": row.get("expected_value"),
        "actual_calculations": [str(result.value) for result in calculations],
        "plan_diagnostics": getattr(bundle, "plan_diagnostics", None),
    }


def evaluate_cross_resource_rows(
    rows: Iterable[dict[str, Any]],
    plan_question: Callable[[str], Any],
    retrieve_plan: Callable[[str, Any], Any],
) -> list[dict[str, Any]]:
    """Execute stage evaluation through injected planner/retrieval callables."""
    scores = []
    for row in rows:
        question = str(row["question"])
        plan = plan_question(question)
        bundle = retrieve_plan(question, plan)
        scores.append(score_cross_resource_case(row, plan, bundle))
    return scores


def aggregate_cross_resource_scores(
    scores: Iterable[dict[str, Any]],
) -> dict[str, float | int]:
    rows = list(scores)
    count = len(rows)
    if not rows:
        return {"count": 0}

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / count

    pdf_values = [row["pdf_gold_recall"] for row in rows if row["pdf_gold_recall"] is not None]
    return {
        "count": count,
        "planned_source_coverage": mean("planned_source_coverage"),
        "retrieved_source_coverage": mean("retrieved_source_coverage"),
        "pdf_gold_recall": (
            sum(float(value) for value in pdf_values) / len(pdf_values)
            if pdf_values
            else 0.0
        ),
        "excel_operand_verification_rate": sum(
            bool(row["excel_operand_verified"]) for row in rows
        )
        / count,
        "calculation_production_rate": sum(
            bool(row["calculation_produced"]) for row in rows
        )
        / count,
        "calculation_accuracy": sum(
            bool(row["calculation_correct"]) for row in rows
        )
        / count,
    }
