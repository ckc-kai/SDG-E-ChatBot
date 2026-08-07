"""Retrieval lanes and the soft gate that orders them.

A lane is a set of content types that are scored against each other only after
being ranked *within* the lane. This removes the zero-sum competition that a
single global ranked list forces between prose, tables, figures, and Excel cards.

The gate deliberately contains no question classifier: every lane is always
retrieved, and lane confidence is derived from each lane's own score
distribution (post-retrieval query-performance prediction). A low-confidence
lane is demoted out of the top slot but never removed, so the failure mode is a
worse rank rather than a missing result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev
from typing import Any, Sequence

NARRATIVE = "narrative"
STRUCTURED = "structured"
EXCEL = "excel"

LANE_CONTENT_TYPES: dict[str, tuple[str, ...]] = {
    NARRATIVE: ("narrative",),
    STRUCTURED: ("table", "figure"),
    EXCEL: ("excel_card",),
}
PDF_LANES = (NARRATIVE, STRUCTURED)
ALL_LANES = (NARRATIVE, STRUCTURED, EXCEL)

_CONTENT_TYPE_TO_LANE = {
    content_type: lane
    for lane, content_types in LANE_CONTENT_TYPES.items()
    for content_type in content_types
}

# A lane whose confidence falls below this may not hold rank 1. Its candidates
# stay in the list, so this is a demotion, not a quota.
DEFAULT_CONFIDENCE_FLOOR = 0.0


def lane_of(content_type: str) -> str:
    return _CONTENT_TYPE_TO_LANE.get(content_type, NARRATIVE)


def content_types_for(lanes: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for lane in lanes:
        if lane not in LANE_CONTENT_TYPES:
            raise ValueError(
                f"Unknown lane {lane!r}; expected one of {sorted(LANE_CONTENT_TYPES)}"
            )
        out.extend(LANE_CONTENT_TYPES[lane])
    return tuple(out)


@dataclass
class LaneConfidence:
    """Post-retrieval query-performance prediction for one lane.

    Every signal is computed from the lane's own ranked scores. No labels and no
    inspection of the question are involved, so this needs no training data.
    """

    lane: str
    top1: float = 0.0
    gap: float = 0.0
    mean_top3: float = 0.0
    n_above: int = 0
    spread: float = 0.0
    candidates: int = 0

    @property
    def confidence(self) -> float:
        """Explainable rule: lead with top-1, break ties on discrimination.

        Kept deliberately simple so a ranking decision can always be explained.
        A fitted model over these same signals is the documented next step, and
        only justified if this rule underperforms.
        """
        return self.top1 + 0.10 * self.gap

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "top1": self.top1,
            "gap": self.gap,
            "mean_top3": self.mean_top3,
            "n_above": self.n_above,
            "spread": self.spread,
            "candidates": self.candidates,
            "confidence": self.confidence,
        }


def lane_confidence(
    lane: str, scores: Sequence[float], *, threshold: float = 0.5
) -> LaneConfidence:
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return LaneConfidence(lane=lane)
    top1 = ordered[0]
    second = ordered[1] if len(ordered) > 1 else ordered[0]
    head = ordered[:3]
    return LaneConfidence(
        lane=lane,
        top1=top1,
        gap=top1 - second,
        mean_top3=sum(head) / len(head),
        n_above=sum(1 for s in ordered if s >= threshold),
        spread=pstdev(ordered[:10]) if len(ordered[:10]) > 1 else 0.0,
        candidates=len(ordered),
    )


@dataclass
class LaneOutcome:
    """One lane's ranked results plus its confidence diagnostics."""

    lane: str
    results: list = field(default_factory=list)
    confidence: LaneConfidence | None = None
