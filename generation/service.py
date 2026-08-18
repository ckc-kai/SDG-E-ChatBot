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
    NO_EVIDENCE_ANSWER,
    PromptBudgetError,
    prepare_prompt,
)
from generation.providers.base import ModelProvider, ProviderError
from generation.providers.ollama import ANSWER_SCHEMA
from generation.schemas import (
    AnswerRequest,
    AnswerResponse,
    AnswerTimings,
    ErrorResponse,
    ModelAnswer,
)


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
    answered_requirements = payload.get("answered_requirements", [])
    missing_requirements = payload.get("missing_requirements", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("Model field 'answer' must be a non-empty string")
    if not isinstance(cited_ids, list) or not all(isinstance(item, (str, int)) for item in cited_ids):
        raise ModelOutputError("Model field 'cited_chunk_ids' must be a list of strings")
    if not isinstance(insufficient, bool):
        raise ModelOutputError("Model field 'insufficient_context' must be boolean")
    if not isinstance(answered_requirements, list) or not all(
        isinstance(item, str) and item.strip() for item in answered_requirements
    ):
        raise ModelOutputError("Model field 'answered_requirements' must be a string array")
    if not isinstance(missing_requirements, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_requirements
    ):
        raise ModelOutputError("Model field 'missing_requirements' must be a string array")
    return ModelAnswer(
        answer=answer.strip(),
        cited_chunk_ids=tuple(str(item) for item in cited_ids),
        insufficient_context=insufficient or bool(missing_requirements),
        answered_requirements=tuple(item.strip() for item in answered_requirements),
        missing_requirements=tuple(item.strip() for item in missing_requirements),
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
        self._prompt_token_budget_override = prompt_token_budget
        capabilities = getattr(provider, "capabilities", None)
        context_tokens = getattr(
            capabilities,
            "context_window",
            getattr(provider, "context_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS),
        )
        output_reserve = getattr(
            provider, "max_tokens", DEFAULT_OUTPUT_TOKEN_RESERVE
        )
        calculated_budget = getattr(
            capabilities or provider,
            "prompt_token_budget",
            context_tokens - output_reserve,
        )
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
        prompt_build_ms = 0
        model_call_ms = 0
        parse_ms = 0
        citation_validation_ms = 0
        usage = None
        refresh_capabilities = getattr(self.provider, "refresh_capabilities", None)
        if callable(refresh_capabilities):
            refresh_capabilities()
        capability_budget = getattr(
            getattr(self.provider, "capabilities", None),
            "prompt_token_budget",
            self.prompt_token_budget,
        )
        active_prompt_budget = (
            capability_budget
            if self._prompt_token_budget_override is None
            else min(self._prompt_token_budget_override, capability_budget)
        )
        if not request.chunks:
            latency_ms = round((time.perf_counter() - started) * 1000)
            return AnswerResponse(
                request_id=request.request_id,
                answer=NO_EVIDENCE_ANSWER,
                cited_chunk_ids=(),
                citations=(),
                insufficient_context=True,
                model_id=self.provider.model_id,
                latency_ms=latency_ms,
                warnings=("No evidence chunks were provided",),
                timings=AnswerTimings(answer_service_total_ms=latency_ms),
            )

        try:
            prompt_started = time.perf_counter()
            prepared_prompt = prepare_prompt(
                request,
                prompt_token_budget=active_prompt_budget,
                token_safety_factor=self.token_safety_factor,
            )
            acquire_prompt_budget = getattr(
                self.provider, "acquire_prompt_token_budget", None
            )
            if callable(acquire_prompt_budget):
                quota_budget = acquire_prompt_budget(
                    prepared_prompt.safety_adjusted_tokens
                )
                final_budget = min(active_prompt_budget, quota_budget)
                if final_budget < active_prompt_budget:
                    prepared_prompt = prepare_prompt(
                        request,
                        prompt_token_budget=final_budget,
                        token_safety_factor=self.token_safety_factor,
                    )
            prompt_build_ms = round((time.perf_counter() - prompt_started) * 1000)
            prompt_request = AnswerRequest(
                request_id=request.request_id,
                question=request.question,
                chunks=prepared_prompt.chunks,
            )
            model_started = time.perf_counter()
            try:
                structured = getattr(self.provider, "generate_structured", None)
                if callable(structured):
                    # Schema-constrained output cannot be cut into invalid
                    # JSON by a mid-string token limit.
                    raw_model_answer = structured(prepared_prompt.text, ANSWER_SCHEMA)
                else:
                    raw_model_answer = self.provider.generate(prepared_prompt.text)
            finally:
                model_call_ms = round((time.perf_counter() - model_started) * 1000)
            usage = getattr(self.provider, "last_usage", None)
            actual_input_tokens = getattr(usage, "input_tokens", None)
            if (
                isinstance(actual_input_tokens, int)
                and actual_input_tokens > active_prompt_budget
            ):
                logger.warning(
                    "Provider input tokens exceeded Task 3 prompt budget for "
                    "request_id=%s: actual=%d budget=%d estimated=%d "
                    "safety_adjusted=%d",
                    request.request_id,
                    actual_input_tokens,
                    active_prompt_budget,
                    prepared_prompt.estimated_tokens,
                    prepared_prompt.safety_adjusted_tokens,
                )
            parse_started = time.perf_counter()
            model_answer = parse_model_answer(raw_model_answer)
            parse_ms = round((time.perf_counter() - parse_started) * 1000)
            validation_started = time.perf_counter()
            valid_ids, citations, warnings = validate_and_hydrate_citations(
                prompt_request, model_answer
            )
            citation_validation_ms = round(
                (time.perf_counter() - validation_started) * 1000
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
            total_ms = round((time.perf_counter() - started) * 1000)
            return ErrorResponse(
                request_id=request.request_id,
                timings=_answer_timings(
                    usage,
                    prompt_build_ms=prompt_build_ms,
                    model_call_ms=model_call_ms,
                    parse_ms=parse_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=total_ms,
                ),
            )
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
            timings=_answer_timings(
                usage,
                prompt_build_ms=prompt_build_ms,
                model_call_ms=model_call_ms,
                parse_ms=parse_ms,
                citation_validation_ms=citation_validation_ms,
                total_ms=latency_ms,
            ),
            answered_requirements=model_answer.answered_requirements,
            missing_requirements=model_answer.missing_requirements,
        )


def _answer_timings(
    usage,
    *,
    prompt_build_ms: int,
    model_call_ms: int,
    parse_ms: int,
    citation_validation_ms: int,
    total_ms: int,
) -> AnswerTimings:
    return AnswerTimings(
        prompt_build_ms=prompt_build_ms,
        model_call_ms=model_call_ms,
        model_reported_ms=getattr(usage, "latency_ms", None),
        parse_ms=parse_ms,
        citation_validation_ms=citation_validation_ms,
        answer_service_total_ms=total_ms,
        model_input_tokens=getattr(usage, "input_tokens", None),
        model_output_tokens=getattr(usage, "output_tokens", None),
    )
