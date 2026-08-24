"""Detect that a question compares named arms, and say which arms they are.

Both weak lanes fail the same way. ``multi_excel`` asks how a number moved
between periods; ``multi_pdf`` asks how a policy moved between filing cycles.
In each case the question names two or more arms and the answer is only worth
anything if evidence for *every* arm is in hand. Measured on the frozen set:
``multipdf_001`` retrieved twenty passages, all twenty from one cycle, and
scored 3.75; ``real_001`` retrieved ten from each side but only one section
corresponded, because the two filings renumbered their chapters.

Nothing here is specific to this corpus's documents. Arms are read out of the
question text, so a cycle, a year, or a quarter this corpus has never seen is
detected the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A filing cycle written as a span: 2023-2025, 2026–2028, 2023 to 2025.
_SPAN_RE = re.compile(r"\b(19|20)\d{2}\s*(?:-|--|–|—|\bto\b|\bthrough\b)\s*(?:(?:19|20))?\d{2}\b")
# A bare four-digit year, including one inside a span (spans are matched first).
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b|\b([1-4])(?:st|nd|rd|th)\s+quarter\b", re.I)

# Wording that asks for a relation between arms rather than a fact about one.
_COMPARISON_CUE_RE = re.compile(
    r"\b(compare[ds]?|comparison|versus|vs\.?|between|differen(?:ce|t|ces)|"
    r"chang(?:e|ed|es|ing)|evolv(?:e|ed|ing)|shift(?:ed|s)?|trend|"
    r"year[- ]over[- ]year|relative to|against|alongside|"
    r"more than|less than|higher than|lower than)\b",
    re.I,
)


@dataclass(frozen=True)
class ComparisonAxis:
    """The dimension a question compares along, and the arms it names."""

    kind: str                      # "cycle", "year", or "quarter"
    arms: tuple[str, ...]

    @property
    def is_comparison(self) -> bool:
        return len(self.arms) >= 2


def _normalise_span(text: str) -> str:
    years = _YEAR_RE.findall(text)
    if len(years) == 2:
        return f"{years[0]}-{years[1]}"
    # "2023-25" style: rebuild the closing year from the opening century.
    digits = re.findall(r"\d{2,4}", text)
    if len(digits) == 2 and len(digits[0]) == 4 and len(digits[1]) == 2:
        return f"{digits[0]}-{digits[0][:2]}{digits[1]}"
    return text.strip()


def detect_axis(question: str) -> ComparisonAxis:
    """Read the comparison axis out of the question, without a model call.

    Spans win over bare years: "between the 2023-2025 WMP and the 2026-2028
    WMP" names two cycles, not four years. A question that names arms without
    any comparison wording is still an axis with one arm, which callers treat
    as "not a comparison".
    """
    spans = [_normalise_span(match.group(0)) for match in _SPAN_RE.finditer(question)]
    spans = tuple(dict.fromkeys(spans))
    if len(spans) >= 2:
        return ComparisonAxis("cycle", spans)

    remaining = _SPAN_RE.sub(" ", question)
    years = tuple(dict.fromkeys(_YEAR_RE.findall(remaining)))
    if len(years) >= 2:
        return ComparisonAxis("year", years)

    quarters = tuple(
        dict.fromkeys(
            f"Q{match.group(1) or match.group(2)}"
            for match in _QUARTER_RE.finditer(question)
        )
    )
    if len(quarters) >= 2:
        return ComparisonAxis("quarter", quarters)

    if spans:
        return ComparisonAxis("cycle", spans)
    if years:
        return ComparisonAxis("year", years)
    return ComparisonAxis("none", ())


def asks_for_comparison(question: str) -> bool:
    """True when the question both names several arms and asks for a relation.

    Both halves are required. "What did we spend in 2023 and 2024" lists two
    arms and wants both figures; that is served by the same arm-coverage
    guarantee, so comparison wording is not demanded when the arms are explicit
    and plural.
    """
    axis = detect_axis(question)
    if not axis.is_comparison:
        return False
    return True


def missing_arms(question: str, covered: object) -> tuple[str, ...]:
    """Arms named by the question that nothing in ``covered`` mentions.

    ``covered`` is any text -- rendered evidence, source filenames, a joined
    breadcrumb list. The check is deliberately textual: an arm is covered when
    the evidence says its name somewhere.
    """
    axis = detect_axis(question)
    if not axis.is_comparison:
        return ()
    haystack = str(covered)
    absent = []
    for arm in axis.arms:
        years = _YEAR_RE.findall(arm)
        # A cycle counts as present when its own label, or every year it spans,
        # appears; ingest filenames vary in how they write a span.
        if arm in haystack:
            continue
        if years and all(year in haystack for year in years):
            continue
        absent.append(arm)
    return tuple(absent)
