"""Load local retrieval and generation models before the first question."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

from retrieval.utils import get_embedding_model, get_reranker_model
from services.generation_service import GenerationService


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class WarmupReport:
    embedding_ms: int = 0
    reranker_ms: int = 0
    answer_model_ms: int = 0
    total_ms: int = 0
    ready: bool = False
    errors: tuple[str, ...] = ()


def warm_application_models(generation_service: GenerationService) -> WarmupReport:
    started = time.perf_counter()
    embedding_ms = 0
    reranker_ms = 0
    answer_model_ms = 0
    errors: list[str] = []

    stage_started = time.perf_counter()
    try:
        get_embedding_model()
    except Exception:
        errors.append("embedding")
        logger.exception("Embedding model warmup failed")
    finally:
        embedding_ms = round((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    try:
        get_reranker_model()
    except Exception:
        errors.append("reranker")
        logger.exception("Reranker model warmup failed")
    finally:
        reranker_ms = round((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    try:
        generation_service.warmup()
    except Exception:
        errors.append("answer_model")
        logger.exception("Answer model warmup failed")
    finally:
        answer_model_ms = round((time.perf_counter() - stage_started) * 1000)

    report = WarmupReport(
        embedding_ms=embedding_ms,
        reranker_ms=reranker_ms,
        answer_model_ms=answer_model_ms,
        total_ms=round((time.perf_counter() - started) * 1000),
        ready=not errors,
        errors=tuple(errors),
    )
    logger.info(
        "model_warmup ready=%s embedding_ms=%d reranker_ms=%d "
        "answer_model_ms=%d total_ms=%d errors=%s",
        report.ready,
        report.embedding_ms,
        report.reranker_ms,
        report.answer_model_ms,
        report.total_ms,
        ",".join(report.errors) or "none",
    )
    return report
