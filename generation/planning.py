"""Provider-independent, bounded planning for complex retrieval questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
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
    "requirement_ids": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
    },
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
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
    "required": ["requirements", "tasks"],
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
_RECONCILE_RE = re.compile(
    r"\b(?:make it into|made it into|reported (?:target|actual|result)|"
    r"target(?:s)? (?:versus|vs\.?|against) actual(?:s)?|"
    r"requested (?:versus|vs\.?|against) reported|"
    r"approved (?:versus|vs\.?|against) reported)\b",
    re.I,
)
_TABLE_REF_RE = re.compile(r"\btables?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.I)
_YEAR_RANGE_RE = re.compile(
    r"\b20[2-3]\d\s*(?:-|\u2013|\u2014|through|to)\s*20[2-3]\d\b", re.I
)
_REPORTED_METRIC_RE = re.compile(
    r"\b(?:reported|reporting|QDR|quarterly data)\b.*"
    r"\b(?:how many|how much|total|sum|count|trend|change|values?|numbers?)\b"
    r"|\b(?:how many|how much|total|sum|count|trend|change|values?|numbers?)\b.*"
    r"\b(?:reported|reporting|QDR|quarterly data)\b",
    re.I,
)


def _references_multiple_tables(question: str) -> bool:
    """Distinct workbook tables are independent evidence tasks."""
    numbers: set[str] = set()
    for reference in _TABLE_REF_RE.findall(question):
        numbers.update(re.findall(r"\d+", reference))
    return len(numbers) >= 2


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str


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
    requirement_ids: tuple[str, ...] = ()

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
    requirements: tuple[Requirement, ...] = ()


def supports_multistep_generation(plan: RetrievalPlan) -> bool:
    """Return whether a plan is safe to answer step-by-step then synthesize.

    Fallback, single-branch, and truncated plans use the normal merged-evidence
    answer path.  A second model pass cannot add coverage in those cases and
    may incorrectly weaken an insufficient-context decision.
    """
    return (
        plan.source == "model"
        and 2 <= len(plan.steps) <= DEFAULT_MAX_INITIAL_BRANCHES
        and plan.dropped_task_count == 0
        and bool(plan.requirements)
        and {item.id for item in plan.requirements}
        == {
            requirement_id
            for step in plan.steps
            for requirement_id in step.requirement_ids
        }
    )


def planning_reason(question: str) -> str | None:
    """Return a conservative reason based on independent evidence tasks."""
    normalized = " ".join(question.split())
    # Introductory source phrases do not create a second factual task. Without
    # stripping them, ", how many" is mistaken for a compound question.
    structural = re.sub(
        r"^(?:based on|according to)\s+[^,]+,\s*",
        "",
        normalized,
        count=1,
        flags=re.I,
    )
    if normalized.count("?") >= 2:
        return "multiple_questions"
    if _COMPARE_RE.search(normalized):
        return "comparison"
    if _AUDIT_RE.search(normalized):
        return "review_or_compliance"
    if _COMPOUND_RE.search(structural):
        return "compound_tasks"
    if _MULTI_SCOPE_RE.search(normalized):
        return "multi_document_scope"
    if _SIDE_BY_SIDE_RE.search(normalized):
        return "side_by_side_report"
    if _RECONCILE_RE.search(normalized):
        return "cross_source_reconciliation"
    # A single sentence can still require structured longitudinal execution.
    # Keep ordinary historical facts simple; planning is reserved for an
    # explicit reporting cue plus a numeric operation over a year interval.
    if _YEAR_RANGE_RE.search(normalized) and _REPORTED_METRIC_RE.search(normalized):
        return "longitudinal_reported_metric"
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
        requirement_ids=tuple(dict.fromkeys(
            cleaned
            for value in item.get("requirement_ids", [])
            if (cleaned := _clean_optional(value)) is not None
        )),
    )


def _requirements_from_payload(
    payload: dict[str, Any], candidates: list[RetrievalStep]
) -> tuple[Requirement, ...]:
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for item in payload.get("requirements", []):
        if not isinstance(item, dict):
            continue
        requirement_id = _clean_optional(item.get("id"))
        text = _clean_optional(item.get("text"))
        if requirement_id and text and requirement_id not in seen:
            seen.add(requirement_id)
            requirements.append(Requirement(requirement_id, text))
    if requirements:
        return tuple(requirements)
    return tuple(
        Requirement(f"R{index}", step.question)
        for index, step in enumerate(candidates, start=1)
    )


def _attach_legacy_requirement_ids(
    candidates: list[RetrievalStep], requirements: tuple[Requirement, ...]
) -> list[RetrievalStep]:
    if any(step.requirement_ids for step in candidates):
        return candidates
    return [
        replace(step, requirement_ids=(requirements[index].id,))
        for index, step in enumerate(candidates)
        if index < len(requirements)
    ]


def _merge_steps(left: RetrievalStep, right: RetrievalStep) -> RetrievalStep:
    def shared(a, b):
        return a if a == b else None

    return RetrievalStep(
        question=f"{left.question} Also retrieve evidence for: {right.question}",
        content_types=tuple(dict.fromkeys((*left.content_types, *right.content_types))),
        source=left.source if left.source == right.source else "both",
        document_role=shared(left.document_role, right.document_role),
        table_role=shared(left.table_role, right.table_role),
        entity=shared(left.entity, right.entity),
        metric=shared(left.metric, right.metric),
        period=shared(left.period, right.period),
        requirement_ids=tuple(dict.fromkeys(
            (*left.requirement_ids, *right.requirement_ids)
        )),
    )


def _bounded_merge(
    steps: list[RetrievalStep], max_branches: int
) -> list[RetrievalStep]:
    """Merge overflow tasks while preserving every requirement mapping."""
    selected = list(steps[:max_branches])
    for overflow in steps[max_branches:]:
        compatible = [
            index for index, current in enumerate(selected)
            if current.source == overflow.source
            and current.content_types == overflow.content_types
        ]
        same_source = [
            index for index, current in enumerate(selected)
            if current.source == overflow.source
        ]
        target = (compatible or same_source or list(range(len(selected))))[0]
        selected[target] = _merge_steps(selected[target], overflow)
    return selected


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
    if reason == "longitudinal_reported_metric":
        # One reported metric over a year range is one deterministic workbook
        # execution, not a decomposition problem. Avoid a model planner that
        # can invent extra tasks or rename the metric before Excel binding.
        requirement = Requirement("R1", question)
        return RetrievalPlan(
            (
                RetrievalStep(
                    question,
                    ("excel_card",),
                    source="excel",
                    requirement_ids=(requirement.id,),
                ),
            ),
            source="rules",
            trigger_reason=reason,
            requirements=(requirement,),
        )

    exact_shape = {
        "requirements": [
            {"id": "R1", "text": "one factual requirement from the user"}
        ],
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
                "requirement_ids": ["R1"],
            }
        ]
    }
    prompt = f"""Create a typed evidence plan for this California utility regulatory-data question.
List every factual requirement before creating at most {max_tasks} retrieval
tasks. Requirements must not be dropped merely because retrieval is limited to
{max_branches} branches. A task may serve multiple requirement IDs. Return
exactly this JSON shape:
{json.dumps(exact_shape, separators=(",", ":"))}

Project vocabulary:
- WMP means Wildfire Mitigation Plan. It never means Water Management Plan.
- OEIS means the California Office of Energy Infrastructure Safety (Energy Safety),
  the regulator that reviews utility WMP filings.
- QDR means Quarterly Data Report.
- SDG&E means San Diego Gas & Electric, the regulated utility.
- A WMP cycle such as 2023-2025 or 2026-2028 is the filing period.

Keep project acronyms and regulatory terms unchanged. Never replace them with a
different domain expansion. Distinguish evidence needed to support a requested
review or recommendation from the recommendation itself. A future period named
as the intended use of the analysis is not automatically a request for forecast
data or resource allocations for that period.

Create only tasks that retrieve evidence directly needed by an explicit user
requirement. Do not create background tasks to rediscover an entity, acronym,
filing period, or date already stated in the question. Do not introduce a QDR,
spreadsheet, forecast, or other document type unless the question requires facts
from it. For a review or recommendation, retrieve the applicable requirements,
the filing content being reviewed, and regulator findings or past criticisms;
do not retrieve the recommendation itself. Treat phrases such as "for a future
cycle" or "to help future development" as the intended use, not an evidence
period, unless the user explicitly asks for facts or projections from that cycle.
Prefer 2-4 necessary tasks and combine closely related facts that use the same
source role. Use 5-6 tasks only when the original question has that many truly
independent factual requirements. Do not add tasks for facts that merely might
be useful. Omit optional keys when their value would only be N/A or unknown.

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
Every requirement MUST have a stable ID such as R1, R2, and every ID MUST
appear in at least one task's requirement_ids. Closely related requirements
that use the same source, document scope, entity, period, or table should share
one retrieval task rather than being omitted.

Treat every explicitly requested output as a requirement, including status,
comparisons, percentages, totals, unit conversions, scope exclusions, and a
request not to make a causal claim. A derived calculation maps to the task that
retrieves its operands; it must not be silently omitted just because it is not
a separately stored fact. Before returning JSON, check the original question
clause by clause and confirm every requested output or constraint appears in a
requirement and maps to at least one task.

An Excel execution task must resolve to one workbook table. When a question
spans multiple workbook tables, create one task per table and name the table in
each task question. Keep fields together when they use the same table, entity,
period, and scope. Treat alternative reporting scopes as mutually exclusive
unless the question explicitly asks to combine them.

For longitudinal performance questions, including repeated outcomes across
years, target-versus-actual comparisons, completion status, or a complete list
of delayed, cancelled, or missed activities, create an Excel task for the
reported records. Add a separate PDF task only when narrative findings or
reasons are also required. For reconciliation questions asking whether a
filing, proposal, approval, or change appeared in later reported results,
create one PDF task for the filing or change and one Excel task for the
reported result. For document comparisons, populate document_role with the
specific document and period required by each task; do not give separate PDF
tasks the same broad document scope.

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
        requirements = _requirements_from_payload(payload, candidates)
        candidates = _attach_legacy_requirement_ids(candidates, requirements)
        known_ids = {requirement.id for requirement in requirements}
        mapped_ids = {
            requirement_id
            for step in candidates
            for requirement_id in step.requirement_ids
        }
        if not known_ids or known_ids != mapped_ids:
            raise ValueError("every requirement must map to a retrieval task")
        deduplicated: list[RetrievalStep] = []
        seen: dict[tuple[str, ...], int] = {}
        for step in candidates:
            signature = step.branch_signature
            if signature in seen:
                index = seen[signature]
                deduplicated[index] = _merge_steps(deduplicated[index], step)
                continue
            seen[signature] = len(deduplicated)
            deduplicated.append(step)
        selected = _bounded_merge(deduplicated, max_branches)
        return RetrievalPlan(
            tuple(selected),
            "model",
            reason,
            atomic_task_count=len(candidates),
            dropped_task_count=0,
            requirements=requirements,
        )
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
        return fallback_plan(question)
