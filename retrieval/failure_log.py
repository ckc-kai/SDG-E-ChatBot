from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"

_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log_path(component: str) -> Path:
    directory = LOGS_ROOT / component
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"failures_{_RUN_ID}.jsonl"


def get_failure_logger(component: str) -> Callable[..., None]:
    """Return a `log_failure(stage, source, error, **context)` bound to `component`."""

    def log_failure(stage: str, source: str, error: BaseException, **context: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "source": source,
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        with _log_path(component).open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    return log_failure
