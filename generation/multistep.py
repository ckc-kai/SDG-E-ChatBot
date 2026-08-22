"""Deterministic safety rules for optional multi-step answer synthesis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json

from generation.schemas import ModelAnswer


@dataclass(frozen=True)
class PlannedSubanswer:
    requirement_ids: tuple[str, ...]
    answer: ModelAnswer


SYNTHESIS_INSTRUCTIONS = """Synthesize a final answer to the original question using only the validated subanswers.

WMP means Wildfire Mitigation Plan; OEIS means the California Office of Energy
Infrastructure Safety; QDR means Quarterly Data Report; and SDG&E means San
Diego Gas & Electric. Keep these terms unchanged.

Address every factual requirement in the original question. A subanswer marked
insufficient may support its stated partial findings, but cannot establish its
missing requirements. Do not add facts absent from the subanswers. Cite only
allowed chunk IDs actually needed by the final answer. If any material
requirement remains unsupported, set insufficient_context=true, identify it,
and present only supported partial findings.

Preserve every displayed decimal place from validated subanswers. Do not round
or abbreviate execution-verified values unless the original question asks for
rounding.

answered_requirements and missing_requirements must contain only the supplied
requirement IDs (for example R1), never paraphrases. Put every supplied ID in
exactly one of those arrays.

Return only one JSON object with answer, cited_chunk_ids,
insufficient_context, answered_requirements, and missing_requirements."""


def build_synthesis_prompt(
    question: str,
    requirements,
    subanswers: Iterable[PlannedSubanswer],
) -> str:
    subanswers = tuple(subanswers)
    allowed_ids = tuple(dict.fromkeys(
        chunk_id
        for item in subanswers
        for chunk_id in item.answer.cited_chunk_ids
    ))
    payload = {
        "original_question": question,
        "requirements": [
            {"id": requirement.id, "text": requirement.text}
            for requirement in requirements
        ],
        "allowed_citation_ids": allowed_ids,
        "subanswers": [
            {
                "requirement_ids": item.requirement_ids,
                "answer": item.answer.answer,
                "cited_chunk_ids": item.answer.cited_chunk_ids,
                "insufficient_context": item.answer.insufficient_context,
            }
            for item in subanswers
        ],
    }
    return (
        f"{SYNTHESIS_INSTRUCTIONS}\nINPUT:"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def protect_synthesis_coverage(
    synthesis: ModelAnswer,
    requirements,
    subanswers: Iterable[PlannedSubanswer],
) -> ModelAnswer:
    """Prevent synthesis from weakening evidence limitations found by a step.

    Until requirement IDs are part of the planner contract, this deliberately
    uses a conservative union: synthesis may add missing requirements, but it
    may not remove a subanswer's missing requirement or change an insufficient
    step into a fully supported final response.
    """
    subanswers = tuple(subanswers)
    requirement_ids = tuple(requirement.id for requirement in requirements)
    supported = {
        requirement_id
        for item in subanswers
        if not item.answer.insufficient_context and item.answer.cited_chunk_ids
        for requirement_id in item.requirement_ids
    }
    model_answered = set(synthesis.answered_requirements)
    final_answered = tuple(
        requirement_id for requirement_id in requirement_ids
        if requirement_id in supported and requirement_id in model_answered
    )
    final_missing = tuple(
        requirement_id for requirement_id in requirement_ids
        if requirement_id not in final_answered
    )
    return ModelAnswer(
        answer=synthesis.answer,
        cited_chunk_ids=synthesis.cited_chunk_ids,
        insufficient_context=(
            synthesis.insufficient_context
            or bool(final_missing)
        ),
        answered_requirements=final_answered,
        missing_requirements=final_missing,
    )
