"""Provider-independent, bounded planning for complex retrieval questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from generation.providers.base import ModelProvider, ProviderError
from generation.routing import route_question


CONTENT_TYPES = ("narrative", "table", "figure", "excel_card")
PDF_CONTENT_TYPES = ("narrative", "table", "figure")
DEFAULT_MAX_ATOMIC_TASKS = 6
DEFAULT_MAX_INITIAL_BRANCHES = 4

_TASK_PROPERTIES: dict[str, Any] = {
    "question": {"type": "string"},
    "source": {"type": "string", "enum": ["pdf", "excel"]},
    "document_role": {"type": "string"},
    "table_role": {"type": "string"},
    "entity": {"type": "string"},
    "metric": {"type": "string"},
    "period": {"type": "string"},
    "need_table": {"type": "boolean"},
    "need_figure": {"type": "boolean"},
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "maxItems": DEFAULT_MAX_ATOMIC_TASKS,
            "items": {
                "type": "object",
                "properties": _TASK_PROPERTIES,
                "required": ["question", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"\b(compare|comparing|versus|vs\.?|differences?|evolv(?:e|ed))\b", re.I
)
_AUDIT_RE = re.compile(r"\b(review|evaluate|assess|flag|gap analysis|compliance|opportunit(?:y|ies))\b", re.I)
_COMPOUND_RE = re.compile(
    r"\b(and (?:why|how|what|whether)|as well as)\b|[,;:]\s*(?:why|how|what|whether)\b"
    r"|\b(?:show|explain|identify)\b.*\b(?:and|,)\b.*\b(?:show|explain|identify)\b",
    re.I,
)
_MULTI_SCOPE_RE = re.compile(
    r"\b(across|between)\b.*\b(cycles?|years?|utilities|documents?|guidelines?|WMPs?|QDRs?)\b",
    re.I,
)
_SIDE_BY_SIDE_RE = re.compile(
    r"\b(side by side|alongside)\b|;\s*then\b|\bthen report\b", re.I
)
_TABLE_REF_RE = re.compile(r"\btables?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.I)


def _references_multiple_tables(question: str) -> bool:
    """Distinct workbook tables are independent evidence tasks."""
    numbers: set[str] = set()
    for reference in _TABLE_REF_RE.findall(question):
        numbers.update(re.findall(r"\d+", reference))
    return len(numbers) >= 2


@dataclass(frozen=True)
class RetrievalStep:
    question: str
    content_types: tuple[str, ...]
    source: str = "pdf"
    document_role: str | None = None
    table_role: str | None = None
    entity: str | None = None
    metric: str | None = None
    period: str | None = None

    @property
    def branch_signature(self) -> tuple[str, ...]:
        # Equivalent atomic facts share one retrieval branch even when their
        # surface questions differ.
        signature = tuple(
            (value or "").strip().casefold()
            for value in (
                self.source,
                self.document_role,
                self.table_role,
                self.entity,
                self.metric,
                self.period,
                ",".join(self.content_types),
            )
        )
        has_fact_identity = any(signature[index] for index in range(1, 6))
        return signature if has_fact_identity else (*signature, self.question.casefold())


@dataclass(frozen=True)
class RetrievalPlan:
    steps: tuple[RetrievalStep, ...]
    source: str = "simple"
    trigger_reason: str = "single_task"
    atomic_task_count: int = 1
    dropped_task_count: int = 0


def planning_reason(question: str) -> str | None:
    """Return a conservative reason based on independent evidence tasks."""
    normalized = " ".join(question.split())
    if normalized.count("?") >= 2:
        return "multiple_questions"
    if _COMPARE_RE.search(normalized):
        return "comparison"
    if _AUDIT_RE.search(normalized):
        return "review_or_compliance"
    if _COMPOUND_RE.search(normalized):
        return "compound_tasks"
    if _MULTI_SCOPE_RE.search(normalized):
        return "multi_document_scope"
    if _SIDE_BY_SIDE_RE.search(normalized):
        return "side_by_side_report"
    if _references_multiple_tables(normalized):
        return "multiple_workbook_tables"
    return None


def needs_planning(question: str) -> bool:
    return planning_reason(question) is not None


def _simple_step(question: str) -> RetrievalStep:
    route = route_question(question, judge_enabled=False)
    return RetrievalStep(
        question=question,
        content_types=route.content_types,
        source="excel" if route.need_excel and not route.need_pdf else "pdf",
    )


def fallback_plan(question: str) -> RetrievalPlan:
    # Invalid model output must never narrow retrieval.
    return RetrievalPlan(
        (RetrievalStep(question.strip(), CONTENT_TYPES, "both"),),
        "fallback",
        "planner_failed",
    )


def _clean_optional(value: Any) -> str | None:
    cleaned = " ".join(str(value or "").split())
    return cleaned or None


def _step_from_item(item: dict[str, Any]) -> RetrievalStep | None:
    question = _clean_optional(item.get("question"))
    source = _clean_optional(item.get("source")) or "pdf"
    source = source.casefold()
    if not question or source not in {"pdf", "excel"}:
        return None
    if source == "excel":
        content_types = ("excel_card",)
    else:
        selected = ["narrative"]
        if item.get("need_table") is True:
            selected.append("table")
        if item.get("need_figure") is True:
            selected.append("figure")
        content_types = tuple(selected)
    return RetrievalStep(
        question=question,
        content_types=content_types,
        source=source,
        document_role=_clean_optional(item.get("document_role")),
        table_role=_clean_optional(item.get("table_role")),
        entity=_clean_optional(item.get("entity")),
        metric=_clean_optional(item.get("metric")),
        period=_clean_optional(item.get("period")),
    )


def _legacy_tasks(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Read the previous planner shape during rollout and cached-run replay."""
    items = payload.get("subquestions")
    if not isinstance(items, list):
        return None
    converted = []
    for item in items:
        if isinstance(item, dict):
            converted.append({"question": item.get("question"), "source": "pdf"})
    return converted


def build_retrieval_plan(
    question: str,
    provider: ModelProvider,
    *,
    max_tasks: int = DEFAULT_MAX_ATOMIC_TASKS,
    max_branches: int = DEFAULT_MAX_INITIAL_BRANCHES,
    max_subquestions: int | None = None,
) -> RetrievalPlan:
    """Build at most four initial branches from up to six atomic tasks.

    ``max_subquestions`` remains as a rollout compatibility alias.  Planner
    failure broadens to every evidence type and never makes a second model call.
    """
    question = " ".join(question.strip().split())
    if not question:
        raise ValueError("question must not be empty")
    if max_subquestions is not None:
        max_tasks = max_subquestions
    if max_tasks <= 0 or max_branches <= 0:
        raise ValueError("planning limits must be positive")
    reason = planning_reason(question)
    if reason is None:
        return RetrievalPlan((_simple_step(question),), "simple")

    exact_shape = {
        "tasks": [
            {
                "question": "standalone factual search question",
                "source": "pdf",
                "document_role": "optional role",
                "table_role": "optional role",
                "entity": "optional entity",
                "metric": "optional metric",
                "period": "optional period",
                "need_table": False,
                "need_figure": False,
            }
        ]
    }
    prompt = f"""Create a typed evidence plan for this regulatory-data question.
Produce at most {max_tasks} atomic factual tasks. Return exactly this JSON shape:
{json.dumps(exact_shape, separators=(",", ":"))}

Every task MUST contain the keys "question" and "source". "source" MUST be
exactly "pdf" or "excel". Use "excel" for values reported in SDG&E's cleaned
quarterly workbook (QDR tables): activity targets, actuals, status (Table 1);
spend/CAPEX/OPEX (Table 11); circuit-mile inventories and upgrades (Tables
7-9); ignitions, events, findings, weather days (Tables 2-6, 10); risk by
tier or segment (Tables 14-15); work orders (Table 13). Use "pdf" for filings,
guidelines, decisions, and narrative content. Use only the keys shown above;
never use source_type, narrative_needed, calculations, or explanatory text.
For PDF, narrative is automatic; set need_table or need_figure only when
needed. Excel tasks do not need PDF support flags. Preserve entities (keep
exact ids like WMP.473 in task questions), periods, metrics, document roles,
and table roles from the question. Do not assume an answer or a cause.

Original question:
{question}"""
    try:
        structured = getattr(provider, "generate_structured", None)
        raw = (
            structured(prompt, PLAN_SCHEMA)
            if callable(structured)
            else provider.generate(prompt)
        )
        payload = json.loads(_FENCE_RE.sub("", raw.strip()).strip())
        if not isinstance(payload, dict):
            raise ValueError("plan must be an object")
        items = payload.get("tasks")
        if not isinstance(items, list):
            items = _legacy_tasks(payload)
        if not isinstance(items, list) or not items:
            raise ValueError("tasks must be a non-empty list")

        candidates = [
            step
            for item in items[:max_tasks]
            if isinstance(item, dict) and (step := _step_from_item(item)) is not None
        ]
        if not candidates:
            raise ValueError("plan contained no valid retrieval tasks")
        deduplicated: list[RetrievalStep] = []
        seen: set[tuple[str, ...]] = set()
        for step in candidates:
            signature = step.branch_signature
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append(step)
        selected = deduplicated[:max_branches]
        return RetrievalPlan(
            tuple(selected),
            "model",
            reason,
            atomic_task_count=len(candidates),
            dropped_task_count=max(0, len(deduplicated) - len(selected)),
        )
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
        return fallback_plan(question)
