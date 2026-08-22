"""Structured, provider-neutral model call traces for debugging."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)

DEFAULT_MODEL_SEED = 42
_MAX_SEED = 2_147_483_647
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def configured_seed(values: Mapping[str, str], provider: str) -> int:
    """Return one reproducible seed, with an optional provider override."""
    raw = values.get(
        f"{provider.upper()}_SEED",
        values.get("MODEL_SEED", str(DEFAULT_MODEL_SEED)),
    )
    seed = int(raw)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"{provider} seed must be between 0 and {_MAX_SEED}")
    return seed


def configured_trace_dir(values: Mapping[str, str]) -> str | None:
    """Return the opt-in directory used for full prompt/result artifacts."""
    return values.get("MODEL_TRACE_DIR", "").strip() or None


class ModelCallTrace:
    """Record one actual generation request without changing provider behavior."""

    def __init__(
        self,
        *,
        model_id: str,
        prompt: str,
        request_payload: Mapping[str, Any],
        trace_dir: str | None,
        seed: int | None,
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.call_id = uuid4().hex
        self.model_id = model_id
        self.prompt = prompt
        self.request_payload = dict(request_payload)
        self.trace_dir = trace_dir
        self.seed = seed
        self.schema = dict(schema) if schema is not None else None
        self.started_at = datetime.now(timezone.utc)
        self.started = time.perf_counter()
        self.raw_output: str | None = None
        self.usage: Any = None
        self.response_metadata: dict[str, Any] = {}
        self.outcome = "started"

    def __enter__(self) -> "ModelCallTrace":
        return self

    def succeed(
        self,
        *,
        raw_output: str,
        usage: Any,
    ) -> None:
        self.raw_output = raw_output
        self.usage = usage
        self.outcome = "success"

    def capture_response(self, response: Mapping[str, Any]) -> None:
        self.response_metadata = _response_metadata(response)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed_ms = round((time.perf_counter() - self.started) * 1000)
        record: dict[str, Any] = {
            "schema_version": 1,
            "call_id": self.call_id,
            "timestamp_utc": self.started_at.isoformat(),
            "model_id": self.model_id,
            "call_type": "structured" if self.schema is not None else "plain",
            "seed": self.seed,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(
                self.prompt.encode("utf-8")
            ).hexdigest(),
            "schema": self.schema,
            "request_payload": self.request_payload,
            "raw_output": self.raw_output,
            "usage": self.usage,
            "response_metadata": self.response_metadata,
            "elapsed_ms": elapsed_ms,
            "outcome": self.outcome if exc is None else "error",
            "error": (
                None
                if exc is None
                else {"type": type(exc).__name__, "message": str(exc)}
            ),
        }
        path = _write_trace(record, self.trace_dir)
        logger.info(
            "model_call call_id=%s model_id=%s outcome=%s elapsed_ms=%d trace=%s",
            self.call_id,
            self.model_id,
            record["outcome"],
            elapsed_ms,
            path,
        )
        return False


def _write_trace(record: dict[str, Any], trace_dir: str | None) -> str | None:
    if trace_dir is None:
        return None
    try:
        directory = Path(trace_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_model = _SAFE_NAME_RE.sub("-", str(record["model_id"])).strip("-")
        path = directory / (
            f"model-{timestamp}-{safe_model}-{str(record['call_id'])[:8]}.json"
        )
        path.write_text(
            json.dumps(_jsonable(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)
    except (OSError, TypeError, ValueError):
        # Observability must never take down a model request.
        logger.exception("model_trace_write_failed")
        return None


def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: response.get(key)
        for key in (
            "id",
            "model",
            "created",
            "system_fingerprint",
            "done",
            "done_reason",
            "stopReason",
        )
        if response.get(key) is not None
    }
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
    return metadata


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
