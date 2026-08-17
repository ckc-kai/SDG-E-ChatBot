"""Bounded same-section expansion for isolated parent/child experiments."""

from __future__ import annotations

from collections.abc import Iterable


def _same_section(anchor, candidate) -> bool:
    return (
        candidate.source_pdf == anchor.source_pdf
        and candidate.sub_document == anchor.sub_document
        and candidate.breadcrumb == anchor.breadcrumb
        and candidate.content_type == "narrative"
    )


def select_sibling_context(
    anchors: Iterable,
    candidates: Iterable,
    *,
    token_budget: int = 2000,
    max_per_anchor: int = 2,
) -> tuple:
    """Select nearby section siblings without exceeding the expansion budget.

    This pure selector is deliberately not enabled in the production path. It
    supports the plan's isolated narrative-coverage experiment while keeping
    added context bounded and deterministic.
    """
    if token_budget <= 0 or max_per_anchor <= 0:
        raise ValueError("expansion limits must be positive")
    anchors = tuple(anchors)
    candidates = tuple(candidates)
    anchor_ids = {anchor.chunk_id for anchor in anchors}
    selected = []
    selected_ids: set = set()
    used_tokens = 0
    for anchor in anchors:
        siblings = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.chunk_id not in anchor_ids
                and candidate.chunk_id not in selected_ids
                and _same_section(anchor, candidate)
            ),
            key=lambda candidate: (
                abs(candidate.chunk_index - anchor.chunk_index),
                candidate.chunk_index,
            ),
        )
        taken = 0
        for sibling in siblings:
            tokens = max(0, int(sibling.token_count))
            if used_tokens + tokens > token_budget:
                continue
            selected.append(sibling)
            selected_ids.add(sibling.chunk_id)
            used_tokens += tokens
            taken += 1
            if taken >= max_per_anchor or used_tokens >= token_budget:
                break
    return tuple(selected)
