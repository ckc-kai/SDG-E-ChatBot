"""Evaluate Task 3 with benchmark gold evidence instead of retrieval output.

Dry-run mode is network-free and prints the exact prompt. Ollama mode uses a
local model; Bedrock mode requires approved account access and credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.evaluation import (
    aggregate_scores,
    load_jsonl,
    request_from_benchmark_row,
    score_response,
)
from generation.prompting import build_prompt
from generation.providers import (
    BedrockProvider,
    ModelProvider,
    OllamaProvider,
    ProviderError,
)
from generation.schemas import AnswerResponse
from generation.service import AnswerService


DEFAULT_EVAL_PATH = REPO_ROOT / "eval" / "pdf" / "evaluation.jsonl"


def load_suite(path: Path) -> tuple[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError("Suite must contain a string 'name'")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Suite must contain a non-empty 'cases' list")
    request_ids = {
        str(case["id"])
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }
    if len(request_ids) != len(cases):
        raise ValueError("Every suite case must have a unique string 'id'")
    return payload["name"], request_ids


def select_rows(
    rows: list[dict[str, Any]],
    *,
    request_ids: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    selected = rows
    if request_ids:
        selected = [row for row in rows if str(row.get("id")) in request_ids]
        missing = request_ids - {str(row.get("id")) for row in selected}
        if missing:
            raise ValueError(f"Unknown evaluation IDs: {', '.join(sorted(missing))}")
    return selected[:limit]


def build_dry_run_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for row in rows:
        request = request_from_benchmark_row(row)
        prompt = build_prompt(request) if request.chunks else None
        records.append(
            {
                "request_id": request.request_id,
                "question": request.question,
                "chunk_ids": [chunk.chunk_id for chunk in request.chunks],
                "prompt_chars": len(prompt) if prompt is not None else 0,
                "prompt": prompt,
            }
        )
    return {
        "mode": "dry-run",
        "uses_gold_evidence": True,
        "model_called": False,
        "count": len(records),
        "records": records,
    }


def run_model_report(
    rows: list[dict[str, Any]],
    provider: ModelProvider,
    *,
    mode: str,
) -> dict[str, Any]:
    service = AnswerService(provider)
    records = []
    scores = []
    errors = 0
    insufficient = 0

    for row in rows:
        request = request_from_benchmark_row(row)
        prompt = build_prompt(request) if request.chunks else None
        response = service.answer(request)
        record: dict[str, Any] = {
            "request_id": request.request_id,
            "question": request.question,
            "chunks": [asdict(chunk) for chunk in request.chunks],
            "prompt": prompt,
            "model_id": provider.model_id,
            "raw_model_output": getattr(provider, "last_raw_text", None),
            "response": response.to_public_dict(),
            "usage": _provider_usage(provider),
        }
        if isinstance(response, AnswerResponse):
            score = score_response(row, response)
            scores.append(score)
            record["score"] = asdict(score)
            insufficient += int(response.insufficient_context)
        else:
            errors += 1
        records.append(record)

    return {
        "mode": mode,
        "uses_gold_evidence": True,
        "model_called": True,
        "count": len(records),
        "errors": errors,
        "insufficient_context": insufficient,
        "score_summary": aggregate_scores(scores),
        "records": records,
    }


def _provider_usage(provider: ModelProvider) -> dict[str, Any] | None:
    usage = getattr(provider, "last_usage", None)
    return asdict(usage) if usage is not None else None


def run_bedrock_report(
    rows: list[dict[str, Any]], provider: BedrockProvider
) -> dict[str, Any]:
    """Backward-compatible wrapper retained for existing tests and callers."""
    return run_model_report(rows, provider, mode="bedrock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated Task 3 evaluation with benchmark gold evidence."
    )
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument(
        "--mode", choices=("dry-run", "ollama", "bedrock"), default="dry-run"
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--id", action="append", dest="request_ids")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.suite and args.request_ids:
        raise SystemExit("Use either --suite or --id, not both")
    suite_name = None
    request_ids = set(args.request_ids) if args.request_ids else None
    if args.suite:
        suite_name, request_ids = load_suite(args.suite)
    rows = select_rows(
        load_jsonl(args.eval),
        request_ids=request_ids,
        limit=args.limit,
    )
    if args.mode == "dry-run":
        report = build_dry_run_report(rows)
    elif args.mode == "ollama":
        try:
            report = run_model_report(rows, OllamaProvider.from_env(), mode="ollama")
        except ProviderError as exc:
            raise SystemExit(f"Ollama configuration failed: {exc}") from exc
    else:
        try:
            report = run_bedrock_report(rows, BedrockProvider.from_env())
        except ProviderError as exc:
            raise SystemExit(f"Bedrock configuration failed: {exc}") from exc
    if suite_name:
        report["suite"] = suite_name

    if args.output:
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        # ASCII-safe output avoids Windows consoles failing on regulatory text
        # characters that are not representable in the active code page.
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
