
import logging
import uuid
from functools import lru_cache
from typing import Union

from fastapi import APIRouter, Depends, Response

from models.schemas import AskErrorResponse, AskRequest, AskResponse
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@lru_cache
def get_generation_service() -> GenerationService:
    return GenerationService()


@router.post("/api/ask", response_model=Union[AskResponse, AskErrorResponse])
def ask(
    payload: AskRequest,
    response: Response,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
):
    request_id = str(uuid.uuid4())

    if payload.filters is not None or payload.rewrite_mode is not None:
        logger.info(
            "filters/rewrite_mode provided but not yet wired into retrieve(): "
            "filters=%s rewrite_mode=%s",
            payload.filters,
            payload.rewrite_mode,
        )

    ranked_results = retrieval_service.retrieve_ranked_results(payload.question)
    result = generation_service.answer(request_id, payload.question, ranked_results)

    if "error" in result:
        response.status_code = 502
        return AskErrorResponse(**result)

    return AskResponse(**result)