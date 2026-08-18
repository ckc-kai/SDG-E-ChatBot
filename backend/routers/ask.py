"""POST /api/ask orchestration for retrieval, planning, and generation."""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from generation.planning import needs_planning
from generation.features import feature_enabled
from generation.routing import RouteDecision
from generation.schemas import ErrorResponse as GenerationErrorResponse
from models.schemas import AskRequest, AskResponse
from retrieval.query.excel.channel import is_entity_history_question
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService, RetrievalTimings


logger = logging.getLogger("uvicorn.error")
router = APIRouter()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@lru_cache
def get_generation_service() -> GenerationService:
    return GenerationService()


@router.post("/api/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
):
    pipeline_started = time.perf_counter()
    request_id = payload.request_id or f"req_{uuid4().hex}"
    try:
        filters = payload.filters
        single_type = filters.content_type if filters else None
        multiple_types = (
            tuple(filters.content_types)
            if filters and filters.content_types
            else None
        )
        if single_type and multiple_types:
            return JSONResponse(
                status_code=422,
                content={"request_id": request_id, "error": "invalid_filters"},
            )
        retrieval_kwargs = {
            "embedding_mode": payload.embedding_mode,
            # Planning owns decomposition. Keep Task 2's optional Anthropic
            # rewrite disabled so one request cannot trigger both planners.
            "rewrite_mode": "off",
        }
        if single_type or multiple_types:
            bundle = retrieval_service.retrieve(
                payload.question,
                content_type=single_type,
                content_types=multiple_types,
                **retrieval_kwargs,
            )
        elif is_entity_history_question(payload.question) and not needs_planning(
            payload.question
        ):
            # Preserve the exact, validated Excel fast path for single-intent
            # history questions; multi-part questions still decompose so their
            # non-history parts retrieve evidence too.
            bundle = retrieval_service.retrieve(payload.question, **retrieval_kwargs)
        elif payload.rewrite_mode == "off" or (
            payload.rewrite_mode != "always"
            and not needs_planning(payload.question)
        ):
            if feature_enabled("two_resource_router"):
                route = generation_service.route_retrieval(payload.question)
            else:
                route = None
            if isinstance(route, RouteDecision):
                bundle = retrieval_service.retrieve(
                    payload.question,
                    content_types=route.content_types,
                    **retrieval_kwargs,
                )
            else:
                bundle = retrieval_service.retrieve(payload.question, **retrieval_kwargs)
        else:
            if feature_enabled("typed_planner"):
                plan = generation_service.plan_retrieval(payload.question)
                bundle = retrieval_service.retrieve_plan(
                    payload.question, plan, **retrieval_kwargs
                )
            else:
                bundle = retrieval_service.retrieve(
                    payload.question, **retrieval_kwargs
                )
    except Exception:
        logger.exception(
            "Task 2 retrieval failed for request_id=%s total_ms=%d",
            request_id,
            round((time.perf_counter() - pipeline_started) * 1000),
        )
        return JSONResponse(
            status_code=503,
            content={"request_id": request_id, "error": "retrieval_failed"},
        )

    result = generation_service.generate(request_id, payload.question, bundle)
    if isinstance(bundle.plan_diagnostics, dict):
        logger.info(
            "retrieval_plan request_id=%s diagnostics=%s",
            request_id,
            json.dumps(
                bundle.plan_diagnostics,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    _log_pipeline_timing(
        request_id,
        bundle,
        result,
        round((time.perf_counter() - pipeline_started) * 1000),
    )
    if isinstance(result, GenerationErrorResponse):
        return JSONResponse(status_code=502, content=result.to_public_dict())
    return result.to_public_dict()


def _log_pipeline_timing(request_id, bundle, result, total_ms: int) -> None:
    retrieval = getattr(bundle, "timings", None)
    if not isinstance(retrieval, RetrievalTimings):
        retrieval = RetrievalTimings()
    generation = result.timings
    logger.info(
        "pipeline_timing request_id=%s db_connect_ms=%d retrieval_ms=%d "
        "excel_verification_ms=%d retrieval_total_ms=%d adapter_ms=%d "
        "prompt_build_ms=%d model_call_ms=%d model_reported_ms=%s "
        "parse_ms=%d citation_validation_ms=%d generation_total_ms=%d "
        "input_tokens=%s output_tokens=%s total_ms=%d",
        request_id,
        retrieval.connection_ms,
        retrieval.grouped_retrieval_ms,
        retrieval.excel_verification_ms,
        retrieval.total_ms,
        generation.adapter_ms if generation else 0,
        generation.prompt_build_ms if generation else 0,
        generation.model_call_ms if generation else 0,
        generation.model_reported_ms if generation else None,
        generation.parse_ms if generation else 0,
        generation.citation_validation_ms if generation else 0,
        generation.generation_total_ms if generation else 0,
        generation.model_input_tokens if generation else None,
        generation.model_output_tokens if generation else None,
        total_ms,
    )
