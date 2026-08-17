"""Reproducible runtime metadata attached to evaluation artifacts."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata
from typing import Any


def _machine_model() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.node() or "unknown"


def _total_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _versions() -> dict[str, str | None]:
    packages = ("torch", "sentence-transformers", "psycopg2-binary", "fastapi")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def evaluation_environment(
    *,
    model_name: str | None = None,
    backend: str | None = None,
    quantization: str | None = None,
    context_limit: int | None = None,
    batch_or_concurrency: int | str | None = None,
) -> dict[str, Any]:
    """Return the minimum environment block required by architecture evals."""
    selected_backend = backend or os.getenv("SDGE_EVAL_BACKEND", "mps")
    return {
        "environment_id": "aws_cuda" if selected_backend == "cuda" else "local_mps",
        "machine_model": _machine_model(),
        "chip_or_gpu": platform.processor() or platform.machine() or "unknown",
        "total_memory": _total_memory(),
        "backend": selected_backend,
        "model_name": model_name or os.getenv("TASK3_PLANNER_MODEL") or os.getenv("OLLAMA_MODEL"),
        "quantization": quantization or os.getenv("SDGE_MODEL_QUANTIZATION"),
        "context_limit": context_limit,
        "batch_or_concurrency": batch_or_concurrency,
        "software_versions": _versions(),
    }
