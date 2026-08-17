"""High-recall two-resource and PDF-support routing.

Deterministic cues handle obvious questions.  Only ambiguous support-evidence
questions may use a structured local judge, and judge failure deliberately
fails open to a small PDF support search.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from generation.providers.base import ModelProvider, ProviderError


ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "need_narrative": {"type": "boolean"},
        "need_table": {"type": "boolean"},
        "need_figure": {"type": "boolean"},
        "need_excel": {"type": "boolean"},
        "uncertain": {"type": "boolean"},
    },
    "required": [
        "need_narrative",
        "need_table",
        "need_figure",
        "need_excel",
        "uncertain",
    ],
    "additionalProperties": False,
}

_EXCEL_RE = re.compile(
    r"\b(workbook|spreadsheet|quarterly data report|QDR|cleaned (?:table|data)|"
    r"reporting quarter|Q[1-4])\b",
    re.I,
)
_PDF_RE = re.compile(r"\b(filing|guideline|decision|document|PDF)\b", re.I)
_TABLE_RE = re.compile(
    r"\b(table|row|column|tabular|matrix|cell|listed in)\b", re.I
)
_FIGURE_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|axis|axes|plotted|map|diagram|legend)\b", re.I
)
_AMBIGUOUS_SUPPORT_RE = re.compile(
    r"\b(exact values?|categories|series|as shown|displayed|breakdown|"
    r"how many|how much|percent(?:age)?|targets?|actuals?|reported value)\b",
    re.I,
)


@dataclass(frozen=True)
class RouteDecision:
    need_pdf: bool
    need_excel: bool
    need_narrative: bool
    need_table: bool = False
    need_figure: bool = False
    uncertain: bool = False
    source: str = "rules"

    @property
    def pdf_content_types(self) -> tuple[str, ...]:
        if not self.need_pdf:
            return ()
        selected = []
        if self.need_narrative:
            selected.append("narrative")
        if self.need_table:
            selected.append("table")
        if self.need_figure:
            selected.append("figure")
        return tuple(selected)

    @property
    def content_types(self) -> tuple[str, ...]:
        selected = list(self.pdf_content_types)
        if self.need_excel:
            selected.append("excel_card")
        return tuple(selected)


def _rule_route(question: str) -> RouteDecision:
    normalized = " ".join(question.split())
    excel = bool(_EXCEL_RE.search(normalized))
    explicit_pdf = bool(_PDF_RE.search(normalized))
    table = bool(_TABLE_RE.search(normalized))
    figure = bool(_FIGURE_RE.search(normalized))
    ambiguous = bool(_AMBIGUOUS_SUPPORT_RE.search(normalized)) and not (table or figure)

    # An explicit workbook/QDR request stays in Excel unless the question also
    # names a PDF source.  Every other question is a PDF question by default.
    need_pdf = not excel or explicit_pdf
    return RouteDecision(
        need_pdf=need_pdf,
        need_excel=excel,
        need_narrative=need_pdf,
        need_table=need_pdf and table,
        need_figure=need_pdf and figure,
        uncertain=need_pdf and ambiguous,
    )


def _judge_route(question: str, judge: ModelProvider) -> RouteDecision:
    prompt = f"""Classify which evidence is needed to answer this question.
PDF narrative is the default. Select table or figure only when structured
support is plausibly required. Select Excel only for quarterly workbook data.
Return the requested JSON booleans only.

Question: {question}"""
    structured = getattr(judge, "generate_structured", None)
    raw = structured(prompt, ROUTE_SCHEMA) if callable(structured) else judge.generate(prompt)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("route response must be an object")
    values = {key: payload.get(key) for key in ROUTE_SCHEMA["required"]}
    if not all(isinstance(value, bool) for value in values.values()):
        raise ValueError("route fields must be booleans")
    need_pdf = bool(
        values["need_narrative"] or values["need_table"] or values["need_figure"]
    )
    return RouteDecision(
        need_pdf=need_pdf,
        need_excel=values["need_excel"],
        need_narrative=values["need_narrative"],
        need_table=values["need_table"],
        need_figure=values["need_figure"],
        uncertain=values["uncertain"],
        source="judge",
    )


def route_question(
    question: str,
    *,
    judge: ModelProvider | None = None,
    judge_enabled: bool = True,
) -> RouteDecision:
    """Route one question with zero or one model call.

    Strong rules always win.  The judge is consulted only for a genuinely
    ambiguous support-evidence request; errors broaden PDF support instead of
    excluding potentially necessary evidence.
    """
    question = " ".join(question.strip().split())
    if not question:
        raise ValueError("question must not be empty")
    ruled = _rule_route(question)
    if not ruled.uncertain or judge is None or not judge_enabled:
        return ruled
    try:
        judged = _judge_route(question, judge)
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
        return RouteDecision(
            need_pdf=True,
            need_excel=ruled.need_excel,
            need_narrative=True,
            need_table=True,
            need_figure=True,
            uncertain=True,
            source="fail_open",
        )
    # A judge may add evidence but cannot remove a deterministic resource cue.
    return RouteDecision(
        need_pdf=ruled.need_pdf or judged.need_pdf,
        need_excel=ruled.need_excel or judged.need_excel,
        need_narrative=ruled.need_narrative or judged.need_narrative,
        need_table=ruled.need_table or judged.need_table,
        need_figure=ruled.need_figure or judged.need_figure,
        uncertain=judged.uncertain,
        source="judge",
    )
