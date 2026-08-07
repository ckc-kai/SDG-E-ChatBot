"""Deterministic mock provider; it never calls a network or model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class RecordingScriptedMockProvider:
    """Return a configured response and retain the exact prompt for assertions."""

    model_id = "mock-model"

    def __init__(self, response: str | Mapping[str, Any] | None = None) -> None:
        self.response = response or {
            "answer": "The provided evidence is insufficient to answer the question.",
            "cited_chunk_ids": [],
            "insufficient_context": True,
        }
        self.last_prompt: str | None = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)
