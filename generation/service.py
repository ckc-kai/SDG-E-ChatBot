"""Task 3 orchestration independent of FastAPI and any specific model vendor."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from generation.citation_validation import CitationValidationError, validate_and_hydrate_citations
from generation.prompting import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_OUTPUT_TOKEN_RESERVE,
    DEFAULT_TOKEN_SAFETY_FACTOR,
    PromptBudgetError,
    prepare_prompt,
)
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
    def __init__(
        self,
        provider: ModelProvider,
        *,
        prompt_token_budget: int | None = None,
        token_safety_factor: float | None = None,
    ) -> None:
        self.provider = provider
        context_tokens = getattr(
            provider, "context_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS
        )
        output_reserve = getattr(
            provider, "max_tokens", DEFAULT_OUTPUT_TOKEN_RESERVE
        )
        calculated_budget = context_tokens - output_reserve
        self.prompt_token_budget = (
            calculated_budget if prompt_token_budget is None else prompt_token_budget
        )
        if self.prompt_token_budget <= 0:
            raise ValueError("prompt_token_budget must be positive")
        provider_safety_factor = getattr(
            provider, "token_safety_factor", DEFAULT_TOKEN_SAFETY_FACTOR
        )
        self.token_safety_factor = (
            provider_safety_factor
            if token_safety_factor is None
            else token_safety_factor
        )
        if self.token_safety_factor < 1:
            raise ValueError("token_safety_factor must be at least 1")

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

        try:
            prepared_prompt = prepare_prompt(
                request,
                prompt_token_budget=self.prompt_token_budget,
                token_safety_factor=self.token_safety_factor,
            )
            prompt_request = AnswerRequest(
                request_id=request.request_id,
                question=request.question,
                chunks=prepared_prompt.chunks,
            )
            raw_model_answer = self.provider.generate(prepared_prompt.text)
            usage = getattr(self.provider, "last_usage", None)
            actual_input_tokens = getattr(usage, "input_tokens", None)
            if (
                isinstance(actual_input_tokens, int)
                and actual_input_tokens > self.prompt_token_budget
            ):
                logger.warning(
                    "Provider input tokens exceeded Task 3 prompt budget for "
                    "request_id=%s: actual=%d budget=%d estimated=%d "
                    "safety_adjusted=%d",
                    request.request_id,
                    actual_input_tokens,
                    self.prompt_token_budget,
                    prepared_prompt.estimated_tokens,
                    prepared_prompt.safety_adjusted_tokens,
                )
            model_answer = parse_model_answer(raw_model_answer)
            valid_ids, citations, warnings = validate_and_hydrate_citations(
                prompt_request, model_answer
            )
        except (
            ModelOutputError,
            CitationValidationError,
            ProviderError,
            TimeoutError,
            ConnectionError,
            PromptBudgetError,
        ):
            # Keep provider/model details in server logs. Task 4 receives only
            # the stable public error contract and never raw exception text.
            logger.exception("Task 3 answer generation failed for request_id=%s", request.request_id)
            return ErrorResponse(request_id=request.request_id)
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
