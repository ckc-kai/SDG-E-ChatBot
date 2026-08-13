"""Provider-independent planning for complex retrieval questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from generation.providers.base import ModelProvider, ProviderError


CONTENT_TYPES = ("narrative", "table", "figure", "excel_card")
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subquestions"],
    "additionalProperties": False,
}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|comparing|versus|vs\.?|differences?|evolv(?:e|ed))\b", re.I)
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
_NUMERIC_RE = re.compile(
    r"\b(how many|how much|number|numbers|percent(?:age)?|rate|rates|target|targets|"
    r"completion|reported|quarter|Q[1-4]|20\d{2})\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalStep:
    question: str
    content_types: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalPlan:
    steps: tuple[RetrievalStep, ...]
    source: str = "simple"
    trigger_reason: str = "single_task"


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
    return None


def needs_planning(question: str) -> bool:
    return planning_reason(question) is not None


def fallback_plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        (RetrievalStep(question.strip(), CONTENT_TYPES),), "fallback", "planner_failed"
    )


def build_retrieval_plan(
    question: str,
    provider: ModelProvider,
    *,
    max_subquestions: int = 2,
) -> RetrievalPlan:
    """Ask the configured answer provider for a soft retrieval plan.

    Invalid output never narrows retrieval: it falls back to the original
    question across every evidence type.
    """
    question = question.strip()
    reason = planning_reason(question)
    if reason is None:
        return RetrievalPlan((RetrievalStep(question, CONTENT_TYPES),), "simple")
    prompt = f"""Plan evidence retrieval for this regulatory-document question.
Break it into at most {max_subquestions} independently searchable factual subquestions.
Preserve domain terminology and scope exactly. WMP means Wildfire Mitigation Plan.
Do not introduce agencies, methods, metrics, document cycles, or comparison subjects
that are absent from the original question. Keep open questions neutral: never assume
which targets were missed, why they were missed, or whether a cause was internal or
external. Combine closely related requirements.
Return JSON only in this exact shape:
{{"subquestions":[{{"question":"..."}}]}}

Original question:
{question}"""
    try:
        structured = getattr(provider, "generate_structured", None)
        raw = structured(prompt, PLAN_SCHEMA) if callable(structured) else provider.generate(prompt)
        payload = json.loads(_FENCE_RE.sub("", raw.strip()).strip())
        items = payload.get("subquestions")
        if not isinstance(items, list) or not items:
            raise ValueError("subquestions must be a non-empty list")
        steps: list[RetrievalStep] = []
        seen: set[str] = set()
        for item in items[:max_subquestions]:
            if not isinstance(item, dict):
                continue
            subquestion = " ".join(str(item.get("question", "")).split())
            signature = subquestion.casefold()
            if subquestion and signature not in seen:
                seen.add(signature)
                steps.append(RetrievalStep(subquestion, CONTENT_TYPES))
        if not steps:
            raise ValueError("plan contained no valid retrieval steps")
        return RetrievalPlan(tuple(steps), "model", reason)
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
        return fallback_plan(question)
