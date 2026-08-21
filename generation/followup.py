"""Conservative conversion of contextual follow-ups into standalone questions.

Conversation history is used only to resolve references such as "what about
2024?".  It is never evidence: the returned standalone question still goes
through the normal retrieval and citation pipeline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from generation.providers.base import ModelProvider, ProviderError


FOLLOWUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "resolved": {"type": "boolean"},
    },
    "required": ["standalone_question", "resolved"],
    "additionalProperties": False,
}

_OPENING_RE = re.compile(r"^\s*(?:(?:and|also)\s+)?(?:what|how)\s+about\b", re.I)
_YEAR_FRAGMENT_RE = re.compile(
    r"^\s*(?:(?:and|also)\s+)?(?:(?:in|for|during)\s+)?20\d{2}\s*[?.!]*\s*$",
    re.I,
)
_LEADING_REFERENCE_RE = re.compile(
    r"^\s*(?:and|also|it|this|these|those|they|them|same)\b", re.I
)
_SHORT_REFERENCE_RE = re.compile(
    r"\b(?:it|this|these|those|they|them|the same|the former|the latter)\b",
    re.I,
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_MAX_HISTORY_TURNS = 2
_MAX_HISTORY_CHARS = 1_200


def is_followup_candidate(question: str) -> bool:
    """Return true only for questions that visibly depend on prior context."""
    normalized = " ".join(question.split())
    if not normalized:
        return False
    words = normalized.split()
    return bool(
        _OPENING_RE.search(normalized)
        or _YEAR_FRAGMENT_RE.fullmatch(normalized)
        or _LEADING_REFERENCE_RE.search(normalized)
        or (len(words) <= 14 and _SHORT_REFERENCE_RE.search(normalized))
    )


def resolve_followup_question(
    question: str,
    history: Iterable[Any],
    provider: ModelProvider,
) -> str:
    """Resolve one clear follow-up, otherwise preserve the original question.

    Provider errors, malformed output, and unresolved references all fail closed
    to the user's exact text.  Complete questions never call the provider.
    """
    original = " ".join(question.split())
    turns = _recent_turns(history)
    if not turns or not is_followup_candidate(original):
        return original

    history_text = "\n".join(
        f"{role}: {content}" for role, content in turns
    )
    prompt = f"""Rewrite the follow-up as one standalone retrieval question.

Use the recent conversation only to resolve omitted subjects, entities, time
periods, and references. Do not answer the question. Do not copy factual claims
from an assistant response into the question. Preserve the user's requested
metric, scope, comparison, and terminology. If the reference has more than one
reasonable interpretation, set resolved=false and return the follow-up unchanged.

Recent conversation:
{history_text}

Follow-up:
{original}

Return only the requested JSON object."""
    try:
        structured = getattr(provider, "generate_structured", None)
        raw = (
            structured(prompt, FOLLOWUP_SCHEMA)
            if callable(structured)
            else provider.generate(prompt)
        )
        payload = json.loads(_FENCE_RE.sub("", raw.strip()).strip())
        standalone = payload.get("standalone_question")
        resolved = payload.get("resolved")
        if resolved is not True or not isinstance(standalone, str):
            return original
        standalone = " ".join(standalone.split())
        if not standalone or len(standalone) > 1_000:
            return original
        return standalone
    except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
        return original


def _recent_turns(history: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, Mapping):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = " ".join(content.split())[:_MAX_HISTORY_CHARS]
        if content:
            normalized.append((role, content))
    return tuple(normalized[-_MAX_HISTORY_TURNS:])
