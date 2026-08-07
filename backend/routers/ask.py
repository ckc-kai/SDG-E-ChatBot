"""
Role: defines POST /api/ask.
"""
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends

from models.schemas import AskRequest, AskResponse
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


@router.post("/api/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
):
    if payload.filters is not None:
        # retrieve() doesn't support filtering yet
        # currently logging rather than silently dropping
        logger.info(
            "Filters provided but not yet supported by retrieve(): %s",
            payload.filters,
        )

    sources = retrieval_service.retrieve(
        payload.question,
        embedding_mode=payload.embedding_mode,
        rewrite_mode=payload.rewrite_mode,
    )
    answer = generation_service.generate(payload.question, sources)
    return AskResponse(answer=answer, sources=sources)