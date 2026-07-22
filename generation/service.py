"""Task 3 orchestration independent of FastAPI and any specific model vendor."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from generation.citation_validation import validate_and_hydrate_citations
from generation.prompting import build_prompt
from generation.providers.base import ModelProvider, ProviderError
from generation.schemas import AnswerRequest, AnswerResponse, ErrorResponse, ModelAnswer


logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ModelOutputError(ValueError):
    """Raised when a provider does not honor the structured-output contract."""


def parse_model_answer(raw: str) -> ModelAnswer:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"Model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ModelOutputError("Model output must be a JSON object")

    answer = payload.get("answer")
    cited_ids = payload.get("cited_chunk_ids")
    insufficient = payload.get("insufficient_context")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("Model field 'answer' must be a non-empty string")
    if not isinstance(cited_ids, list) or not all(isinstance(item, (str, int)) for item in cited_ids):
        raise ModelOutputError("Model field 'cited_chunk_ids' must be a list of strings")
    if not isinstance(insufficient, bool):
        raise ModelOutputError("Model field 'insufficient_context' must be boolean")
    return ModelAnswer(
        answer=answer.strip(),
        cited_chunk_ids=tuple(str(item) for item in cited_ids),
        insufficient_context=insufficient,
    )


class AnswerService:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def answer(self, request: AnswerRequest) -> AnswerResponse | ErrorResponse:
        started = time.perf_counter()
        if not request.chunks:
            latency_ms = round((time.perf_counter() - started) * 1000)
            return AnswerResponse(
                request_id=request.request_id,
                answer="The provided evidence is insufficient to answer the question.",
                cited_chunk_ids=(),
                citations=(),
                insufficient_context=True,
                model_id=self.provider.model_id,
                latency_ms=latency_ms,
                warnings=("No evidence chunks were provided",),
            )

        prompt = build_prompt(request)
        try:
            model_answer = parse_model_answer(self.provider.generate(prompt))
        except (ModelOutputError, ProviderError, TimeoutError, ConnectionError):
            # Keep provider/model details in server logs. Task 4 receives only
            # the stable public error contract and never raw exception text.
            logger.exception("Task 3 answer generation failed for request_id=%s", request.request_id)
            return ErrorResponse(request_id=request.request_id)
        valid_ids, citations, warnings = validate_and_hydrate_citations(request, model_answer)
        latency_ms = round((time.perf_counter() - started) * 1000)
        return AnswerResponse(
            request_id=request.request_id,
            answer=model_answer.answer,
            cited_chunk_ids=valid_ids,
            citations=citations,
            insufficient_context=model_answer.insufficient_context,
            model_id=self.provider.model_id,
            latency_ms=latency_ms,
            warnings=warnings,
        )
