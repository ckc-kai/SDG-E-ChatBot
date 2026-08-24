"""Objective account of what evidence a retrieval bundle actually carries.

The answer model decides on its own whether the evidence is sufficient, and it
is measurably wrong about that: on the 2026-08-23 beta run it refused 10 of 27
answerable questions, one of them (``real_004``) holding 128 execution-verified
rows of exactly the target-versus-actual data the question asked for.

This module reports what is in hand without asking a model, so a refusal can be
checked against the record rather than trusted.
"""

from __future__ import annotations

from typing import Any


def _group_counts(bundle) -> dict[str, int]:
    groups = getattr(getattr(bundle, "evidence", None), "groups", None) or {}
    return {
        name: len(tuple(getattr(group, "results", ())))
        for name, group in groups.items()
    }


def _verified_answers(bundle) -> tuple:
    answers = tuple(getattr(bundle, "verified_excels", ()) or ())
    if answers:
        return answers
    single = getattr(bundle, "verified_excel", None)
    return (single,) if single is not None else ()


def verified_row_count(bundle) -> int:
    """Rows returned by execution-verified Excel plans, summed over plans."""
    total = 0
    for answer in _verified_answers(bundle):
        rows = getattr(getattr(answer, "result", None), "rows", ())
        total += len(tuple(rows))
    return total


def evidence_snapshot(bundle) -> dict[str, Any]:
    """Summarise the evidence in hand, for diagnostics and refusal review."""
    groups = _group_counts(bundle)
    return {
        "ranked_chunks": sum(groups.values()),
        "groups": groups,
        "verified_excel_plans": len(_verified_answers(bundle)),
        "verified_rows": verified_row_count(bundle),
        "excel_row_slices": len(tuple(getattr(bundle, "excel_rows", ()) or ())),
    }
