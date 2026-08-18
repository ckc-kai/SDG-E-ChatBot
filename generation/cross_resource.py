"""End-to-end cross-resource computation: bind operands, then calculate.

The typed calculator in ``generation.computation`` refuses anything without
provenance, matching periods, and compatible units. This module produces those
operands:

- a computation planner (structured-output model) decomposes the question into
  two operand subquestions with declared sources;
- Excel operands come from the execution-verified Excel channel, so their
  values carry row-level workbook provenance;
- PDF operands come from a model extraction that is only accepted when the
  extracted number literally appears in the cited chunk's text.

Anything unverifiable simply produces no calculation; the answer model then
reports the limitation instead of an invented number.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from generation.computation import (
    CalculationError,
    CalculationResult,
    CalculationSpec,
    EvidenceOperand,
    FactRequirement,
    TypedComputationPlan,
    execute_calculation,
)
from generation.providers.base import ProviderError

logger = logging.getLogger(__name__)

_COMPUTATION_CUE_RE = re.compile(
    r"\b(what percent(?:age)?|percent(?:age)? (?:is|of)|ratio of|"
    r"difference between|divided by|operand)\b",
    re.I,
)

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

COMPUTATION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "computation_requested": {"type": "boolean"},
        "operation": {
            "type": "string",
            "enum": ["ratio_percent", "difference", "sum", "change_percent"],
        },
        "operands": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string", "enum": ["pdf", "excel"]},
                    "question": {"type": "string"},
                    "metric": {"type": "string"},
                    "period": {"type": "string"},
                },
                "required": ["id", "source", "question", "metric", "period"],
                "additionalProperties": False,
            },
        },
        "left_ref": {"type": "string"},
        "right_ref": {"type": "string"},
    },
    "required": ["computation_requested"],
    "additionalProperties": False,
}

PDF_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "value": {"type": "string"},
        "unit": {"type": ["string", "null"]},
        "chunk_id": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["found"],
    "additionalProperties": False,
}


def question_requests_computation(question: str) -> bool:
    return bool(_COMPUTATION_CUE_RE.search(question))


_OPERAND_WRAPPER_RE = re.compile(
    r"Operand\s+([A-Z])\s+must use the\s+(cleaned QDR workbook|WMP filing)\s+fact:"
    r"\s*(.*?)(?=Operand\s+[A-Z]\s+must use|$)",
    re.S | re.I,
)


def contradictory_operand_sources(question: str) -> str | None:
    """Detect operand instructions whose declared source contradicts the text.

    A fact labeled as coming from the WMP filing whose own wording asks about
    the cleaned quarterly workbook (or vice versa) cannot be verified against
    the declared source; the audited golden behavior is to abstain and name
    the inconsistent operand rather than compute from the wrong source.
    """
    for match in _OPERAND_WRAPPER_RE.finditer(question):
        operand, declared, text = match.groups()
        lowered = text.casefold()
        mentions_workbook = any(
            cue in lowered for cue in ("workbook", "qdr", "quarterly activity")
        )
        mentions_filing = "filing" in lowered
        declared = declared.casefold()
        if "workbook" in declared and mentions_filing and not mentions_workbook:
            return (
                f"operand {operand} is declared as a cleaned QDR workbook fact "
                "but its own wording asks about the WMP filing"
            )
        if "filing" in declared and mentions_workbook and not mentions_filing:
            return (
                f"operand {operand} is declared as a WMP filing fact but its "
                "own wording asks about the cleaned quarterly workbook"
            )
    return None


_provider: Any = None
_provider_failed = False


def _get_provider():
    global _provider, _provider_failed
    if _provider is not None or _provider_failed:
        return _provider
    from generation.providers import create_provider_from_env

    provider_name = os.getenv("CROSS_RESOURCE_PLANNER_PROVIDER", "").strip().casefold()
    if not provider_name:
        provider_name = "groq" if os.getenv("GROQ_API_KEY", "").strip() else "ollama"
    values = dict(os.environ)
    if provider_name == "groq":
        values["GROQ_MODEL"] = values.get(
            "CROSS_RESOURCE_PLANNER_MODEL", "openai/gpt-oss-120b"
        )
        values["GROQ_MAX_TOKENS"] = "700"
        values["GROQ_REASONING_EFFORT"] = "low"
    try:
        _provider = create_provider_from_env(provider_name, environ=values)
    except (ProviderError, ValueError):
        logger.warning("Cross-resource planner provider unavailable")
        _provider_failed = True
        _provider = None
    return _provider


def _structured(provider, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    try:
        generate = getattr(provider, "generate_structured", None)
        raw = generate(prompt, schema) if callable(generate) else provider.generate(prompt)
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.info("Cross-resource structured call failed: %s", exc)
        return None


def _plan_computation(
    question: str, provider
) -> tuple[TypedComputationPlan, dict[str, str]] | None:
    prompt = f"""Decide whether this question asks for one arithmetic calculation
combining exactly two facts, and if so decompose it.

Question: {question}

If it does, return computation_requested=true with:
- operation: ratio_percent ("what percentage is A of B" -> A/B*100),
  difference (A-B), sum (A+B), or change_percent ((A-B)/B*100);
- operands: exactly two, each with a unique id, source "excel" for the cleaned
  quarterly workbook / QDR data or "pdf" for filings and guidelines, a
  standalone factual question that retrieves that single number, the metric
  name, and the period (for example "2024");
- left_ref and right_ref naming the operand ids in the operation's order.
Otherwise return computation_requested=false. Return JSON only."""
    payload = _structured(provider, prompt, COMPUTATION_PLAN_SCHEMA)
    if not payload or payload.get("computation_requested") is not True:
        return None
    try:
        operands = payload["operands"]
        facts = tuple(
            FactRequirement(
                id=str(item["id"]),
                source=str(item["source"]),  # type: ignore[arg-type]
                metric=str(item["metric"]),
                period=str(item["period"]),
            )
            for item in operands
        )
        spec = CalculationSpec(
            operation=str(payload["operation"]),  # type: ignore[arg-type]
            left_ref=str(payload["left_ref"]),
            right_ref=str(payload["right_ref"]),
        )
        plan = TypedComputationPlan(facts=facts, calculation=spec)
    except (KeyError, TypeError, ValueError, CalculationError) as exc:
        logger.info("Cross-resource computation plan invalid: %s", exc)
        return None
    operand_questions = {
        str(item["id"]): str(item["question"]) for item in operands
    }
    return plan, operand_questions


def _normalize_number(text: str) -> str:
    return text.replace(",", "").strip()


def _resolve_excel_operand(
    operand_question: str, requirement: FactRequirement, conn
) -> tuple[EvidenceOperand, Any] | None:
    from retrieval.query.excel.channel import ExcelAnswer, answer_from_excel

    outcome = answer_from_excel(operand_question, conn)
    if not isinstance(outcome, ExcelAnswer):
        logger.info(
            "Excel operand declined for %r: %s",
            requirement.id,
            getattr(outcome, "reason", "unknown"),
        )
        return None
    result = outcome.result
    if not result.rows:
        return None
    selected_indexes = [
        index
        for index, column in enumerate(result.columns)
        if column.startswith("selected_")
    ]
    row = result.rows[0]
    candidate_cells = (
        [row[index] for index in selected_indexes] if selected_indexes else [row[-1]]
    )
    value = next((cell for cell in candidate_cells if cell is not None), None)
    if value is None:
        return None
    provenance = tuple(
        f"{item.get('source_file')} {item.get('source_sheet') or ''} "
        f"row {item.get('source_row')}".strip()
        for item in result.provenance
        if item.get("source_file")
    ) or (f"excel table {outcome.table_number} chunk {outcome.card_chunk_id}",)
    try:
        operand = EvidenceOperand(
            value=_decimal(value),
            unit=result.unit,
            period=requirement.period,
            provenance=provenance,
        )
    except ValueError:
        return None
    # Keep the executed answer so callers can surface row-level provenance.
    return operand, outcome


def _decimal(value):
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(_normalize_number(str(value)))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"operand value {value!r} is not numeric") from exc


def _resolve_pdf_operand(
    operand_question: str,
    requirement: FactRequirement,
    conn,
    provider,
    *,
    query_hint: str = "",
) -> EvidenceOperand | None:
    from retrieval.query.pdf import retrieve_configured

    # A verbose operand sentence dilutes retrieval; a compact entity-led query
    # often ranks the filing's target table first. Search both and merge.
    queries = [f"{operand_question} {query_hint}".strip()]
    if query_hint:
        queries.append(f"{query_hint} {requirement.period} {requirement.metric}")
    candidates: list[Any] = []
    seen_ids: set[str] = set()
    for query in queries:
        evidence = retrieve_configured(
            query,
            conn,
            output_mode="grouped",
            groups=("narrative", "table"),
            rewrite_mode="off",
        )
        for result in [
            item for group in evidence.groups.values() for item in group.results
        ][:12]:
            chunk_id = str(result.query_object.chunk_id)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                candidates.append(result)
    candidates = candidates[:20]
    if not candidates:
        return None
    excerpts = []
    by_id: dict[str, Any] = {}
    for result in candidates:
        chunk_id = str(result.query_object.chunk_id)
        by_id[chunk_id] = result.query_object
        excerpts.append(
            {
                "chunk_id": chunk_id,
                "source": result.query_object.source_pdf,
                "text": result.query_object.content[:1500],
            }
        )
    prompt = f"""Extract one number from the excerpts to answer this question.

Question: {operand_question}
Metric: {requirement.metric}; period: {requirement.period}

Excerpts:
{json.dumps(excerpts, ensure_ascii=False)}

Return found=true only when one excerpt states the exact requested value for
the requested metric and period. Copy value verbatim (digits only, keep the
decimal point), set unit, chunk_id, and quote as the shortest sentence
fragment from that excerpt containing the value. Return JSON only."""
    payload = _structured(provider, prompt, PDF_EXTRACTION_SCHEMA)
    if not payload or payload.get("found") is not True:
        return None
    chunk_id = str(payload.get("chunk_id") or "")
    query_object = by_id.get(chunk_id)
    value_text = _normalize_number(str(payload.get("value") or ""))
    quote = str(payload.get("quote") or "")
    if query_object is None or not value_text:
        return None
    content = " ".join(str(query_object.content).split())
    normalized_quote = " ".join(quote.split())
    # The quote must come from the cited chunk and must contain the value; a
    # model cannot introduce a number the corpus does not state.
    if normalized_quote and normalized_quote not in content:
        return None
    haystack = normalized_quote or content
    numbers = {_normalize_number(match) for match in _NUMBER_RE.findall(haystack)}
    if value_text not in numbers:
        return None
    try:
        return EvidenceOperand(
            value=_decimal(value_text),
            unit=payload.get("unit") or None,
            period=requirement.period,
            provenance=(
                f"{query_object.source_pdf} chunk {chunk_id} "
                f"page {query_object.page_start}",
            ),
        )
    except ValueError:
        return None


def attempt_cross_resource_calculations(
    question: str, conn
) -> tuple[tuple[CalculationResult, ...], tuple[Any, ...], tuple[str, ...]]:
    """Return (calculations, verified excel answers, advisory notes).

    Every failure path returns no calculation plus a note explaining why, so
    the answer model states the limitation rather than approximating.
    """
    if not question_requests_computation(question):
        return (), (), ()
    contradiction = contradictory_operand_sources(question)
    if contradiction:
        return (
            (),
            (),
            (
                "A calculation was requested but was not performed because the "
                f"question's operand sourcing is internally inconsistent: "
                f"{contradiction}. Do not compute the requested result from "
                "any other evidence, even if candidate numbers appear "
                "elsewhere; state this limitation, identify the unsupported "
                "operand, and answer with insufficient_context=true.",
            ),
        )
    provider = _get_provider()
    if provider is None:
        return (), (), ()
    planned = _plan_computation(question, provider)
    if planned is None:
        return (), (), ()
    plan, operand_questions = planned

    operands: dict[str, EvidenceOperand] = {}
    excel_answers: list[Any] = []
    # Excel operands resolve first: their entity keys and captions sharpen the
    # PDF operand query, whose lexical lane can then hit the filing's table.
    ordered = sorted(plan.facts, key=lambda fact: fact.source != "excel")
    for requirement in ordered:
        operand_question = operand_questions.get(requirement.id)
        if not operand_question:
            return (), (), ()
        if requirement.source == "excel":
            resolved = _resolve_excel_operand(operand_question, requirement, conn)
            if resolved is None:
                return (
                    (),
                    tuple(excel_answers),
                    (_unverified_note(requirement, operand_question),),
                )
            operand, answer = resolved
            operands[requirement.id] = operand
            excel_answers.append(answer)
        else:
            hint = " ".join(
                dict.fromkeys(
                    part
                    for answer in excel_answers
                    for part in (
                        str((answer.bound or {}).get("entity_key") or ""),
                        *(
                            str(flt.value)
                            for flt in answer.plan.filters
                            if flt.field == "entity_key"
                        ),
                        str(answer.card_caption or "").replace(
                            "WMP activity — ", ""
                        ),
                    )
                    if part
                )
            )
            operand = _resolve_pdf_operand(
                operand_question, requirement, conn, provider, query_hint=hint
            )
            if operand is None:
                return (
                    (),
                    tuple(excel_answers),
                    (_unverified_note(requirement, operand_question),),
                )
            operands[requirement.id] = operand

    try:
        result = execute_calculation(plan, operands)
    except CalculationError as exc:
        logger.info("Cross-resource calculation refused: %s", exc)
        return (
            (),
            tuple(excel_answers),
            (
                "A calculation was requested but the verified operands were "
                f"not compatible ({exc}); no calculation was performed. State "
                "this limitation instead of guessing.",
            ),
        )
    return (result,), tuple(excel_answers), ()


def _unverified_note(requirement: FactRequirement, operand_question: str) -> str:
    source_label = (
        "the cleaned quarterly workbook"
        if requirement.source == "excel"
        else "the PDF filings"
    )
    return (
        "A calculation was requested but was not performed because operand "
        f"{requirement.id!r} ({operand_question}) could not be verified "
        f"against {source_label}. State exactly which operand lacks verified "
        "evidence and do not guess a value."
    )
