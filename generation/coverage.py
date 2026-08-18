"""Deterministic evidence-coverage ledger for bounded retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from generation.planning import RetrievalPlan


_TYPE_TO_GROUP = {
    "narrative": "narrative",
    "table": "table",
    "figure": "figure",
    "excel_card": "excel",
}


@dataclass(frozen=True)
class CoverageItem:
    step_index: int
    question: str
    covered_content_types: tuple[str, ...]
    missing_content_types: tuple[str, ...]

    @property
    def covered(self) -> bool:
        return not self.missing_content_types


@dataclass(frozen=True)
class CoverageLedger:
    items: tuple[CoverageItem, ...]

    @property
    def covered(self) -> bool:
        return all(item.covered for item in self.items)

    @property
    def missing_step_indexes(self) -> tuple[int, ...]:
        return tuple(item.step_index for item in self.items if not item.covered)

    def to_dict(self) -> dict:
        return {
            "covered": self.covered,
            "missing_steps": list(self.missing_step_indexes),
            "items": [
                {
                    "step_index": item.step_index,
                    "question": item.question,
                    "covered_content_types": list(item.covered_content_types),
                    "missing_content_types": list(item.missing_content_types),
                }
                for item in self.items
            ],
        }


def assess_plan_coverage(plan: RetrievalPlan, bundles: tuple) -> CoverageLedger:
    if len(plan.steps) != len(bundles):
        raise ValueError("plan steps and retrieval bundles must align")
    items: list[CoverageItem] = []
    for index, (step, bundle) in enumerate(zip(plan.steps, bundles, strict=True)):
        covered: list[str] = []
        missing: list[str] = []
        for content_type in step.content_types:
            if content_type == "excel_card" and (
                getattr(bundle, "verified_excel", None) is not None
                or getattr(bundle, "verified_excels", ())
            ):
                covered.append(content_type)
                continue
            group_name = _TYPE_TO_GROUP.get(content_type)
            group = getattr(bundle.evidence, "groups", {}).get(group_name)
            if group is not None and bool(group.results):
                covered.append(content_type)
            else:
                missing.append(content_type)
        items.append(
            CoverageItem(index, step.question, tuple(covered), tuple(missing))
        )
    return CoverageLedger(tuple(items))
