"""Deterministic safety rules for optional multi-step answer synthesis."""

from __future__ import annotations

from collections.abc import Iterable
import json

from generation.schemas import ModelAnswer


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

Return only one JSON object with answer, cited_chunk_ids,
insufficient_context, answered_requirements, and missing_requirements."""


def build_synthesis_prompt(
    question: str,
    subanswers: Iterable[ModelAnswer],
) -> str:
    subanswers = tuple(subanswers)
    allowed_ids = tuple(dict.fromkeys(
        chunk_id for answer in subanswers for chunk_id in answer.cited_chunk_ids
    ))
    payload = {
        "original_question": question,
        "allowed_citation_ids": allowed_ids,
        "subanswers": [
            {
                "answer": answer.answer,
                "cited_chunk_ids": answer.cited_chunk_ids,
                "insufficient_context": answer.insufficient_context,
                "answered_requirements": answer.answered_requirements,
                "missing_requirements": answer.missing_requirements,
            }
            for answer in subanswers
        ],
    }
    return (
        f"{SYNTHESIS_INSTRUCTIONS}\nINPUT:"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def protect_synthesis_coverage(
    synthesis: ModelAnswer,
    subanswers: Iterable[ModelAnswer],
) -> ModelAnswer:
    """Prevent synthesis from weakening evidence limitations found by a step.

    Until requirement IDs are part of the planner contract, this deliberately
    uses a conservative union: synthesis may add missing requirements, but it
    may not remove a subanswer's missing requirement or change an insufficient
    step into a fully supported final response.
    """
    subanswers = tuple(subanswers)
    protected_missing = tuple(dict.fromkeys(
        requirement
        for answer in subanswers
        for requirement in answer.missing_requirements
    ))
    final_missing = tuple(dict.fromkeys(
        (*synthesis.missing_requirements, *protected_missing)
    ))
    return ModelAnswer(
        answer=synthesis.answer,
        cited_chunk_ids=synthesis.cited_chunk_ids,
        insufficient_context=(
            synthesis.insufficient_context
            or any(answer.insufficient_context for answer in subanswers)
            or bool(final_missing)
        ),
        answered_requirements=synthesis.answered_requirements,
        missing_requirements=final_missing,
    )
