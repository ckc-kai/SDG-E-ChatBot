"""DeepSeek Harness-backed structured planner provider.

The application keeps its existing answer provider and UI contract. This
adapter is used only after ``needs_planning`` has selected the typed-planning
path, and delegates that one structured planning turn to DSH.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation.providers.base import ProviderError
from generation.providers.capabilities import ModelCapabilities


DEFAULT_MODEL = "qwen3:14b"
DEFAULT_MAX_TOKENS = 1800
DEFAULT_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class DshUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class DshPlannerProvider:
    """Run a schema-directed planning turn through the DSH SDK runtime."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        project_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("DSH limits must be positive")
        self.model = model.strip()
        self.model_id = f"dsh/{self.model}"
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.project_root = (
            project_root or Path(__file__).resolve().parents[2]
        ).resolve()
        self.environ = dict(os.environ if environ is None else environ)
        self.last_usage: DshUsage | None = None
        self.last_finish_reason: str | None = None
        self.capabilities = ModelCapabilities(
            context_window=8192,
            max_output_tokens=max_tokens,
            requested_output_tokens=max_tokens,
            prompt_token_budget=8192 - max_tokens,
        )

    @classmethod
    def from_env(
        cls, *, environ: Mapping[str, str] | None = None
    ) -> "DshPlannerProvider":
        values = os.environ if environ is None else environ
        try:
            return cls(
                model=values.get("DSH_PLANNER_MODEL", DEFAULT_MODEL),
                max_tokens=int(
                    values.get("DSH_PLANNER_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
                ),
                timeout_seconds=float(
                    values.get(
                        "DSH_PLANNER_TIMEOUT_SECONDS",
                        str(DEFAULT_TIMEOUT_SECONDS),
                    )
                ),
                environ=values,
            )
        except ValueError as exc:
            raise ProviderError("Invalid DSH planner configuration") from exc

    def generate(self, prompt: str) -> str:
        return self.generate_structured(prompt, {"type": "object"})

    def generate_structured(
        self, prompt: str, schema: Mapping[str, Any]
    ) -> str:
        if not prompt.strip():
            raise ProviderError("DSH planner prompt must not be empty")
        runner = self.project_root / "scripts" / "run_dsh_planner.py"
        sdk_path = Path(
            self.environ.get(
                "DSH_SDK_PATH", str(self.project_root / ".dsh-python")
            )
        )
        if not sdk_path.is_absolute():
            sdk_path = self.project_root / sdk_path
        sdk_path = sdk_path.resolve()
        if not runner.is_file():
            raise ProviderError(f"DSH planner runner is missing: {runner}")
        if not sdk_path.is_dir():
            raise ProviderError(
                "DSH SDK is not installed; see docs/dsh-planner.md"
            )

        env = dict(self.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(sdk_path), existing_pythonpath) if value
        )
        env.setdefault("DSH_BASE_URL", "http://127.0.0.1:11434/v1")
        env.setdefault("DSH_API_KEY", "ollama")
        env.setdefault(
            "DSH_CORDIS_CONFIG", str(self.project_root / "dsh" / "cordis.yml")
        )
        payload = json.dumps(
            {
                "prompt": prompt,
                "schema": dict(schema),
                "model": self.model,
                "max_tokens": self.max_tokens,
            },
            ensure_ascii=False,
        )
        try:
            completed = subprocess.run(
                [env.get("DSH_PYTHON", "python3"), str(runner)],
                input=payload,
                text=True,
                capture_output=True,
                cwd=self.project_root,
                env=env,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderError("DSH planner process failed") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1200:]
            raise ProviderError(f"DSH planner failed: {detail}")
        try:
            envelope = json.loads(completed.stdout)
            output = envelope["output"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError("DSH planner returned an invalid envelope") from exc
        if not isinstance(output, str) or not output.strip():
            raise ProviderError("DSH planner returned no structured output")
        self.last_finish_reason = str(envelope.get("finish_reason") or "") or None
        self.last_usage = DshUsage(
            input_tokens=_optional_int(envelope.get("input_tokens")),
            output_tokens=_optional_int(envelope.get("output_tokens")),
            latency_ms=_optional_int(envelope.get("latency_ms")),
        )
        return output


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
