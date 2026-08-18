"""Run the cross-resource diagnostic without calling an answer model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.cross_resource_evaluation import (
    aggregate_cross_resource_scores,
    evaluate_cross_resource_rows,
)
from eval.environment import evaluation_environment
from generation.planning import build_retrieval_plan
from generation.providers import create_provider_from_env


DEFAULT_INPUT = Path("eval/architecture/dev/cross_resource_computation.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_dotenv()

    backend_path = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from services.retrieval_service import RetrievalService

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    values = dict(os.environ)
    values.update(OLLAMA_MODEL=args.model, OLLAMA_MAX_TOKENS="500")
    planner = create_provider_from_env("ollama", environ=values)
    retrieval = RetrievalService()
    completed = 0

    def plan_question(question: str):
        return build_retrieval_plan(question, planner)

    def retrieve_plan(question: str, plan):
        nonlocal completed
        bundle = retrieval.retrieve_plan(question, plan, rewrite_mode="off")
        completed += 1
        print(f"[{completed}/{len(rows)}] evaluated {rows[completed - 1]['id']}", flush=True)
        return bundle

    scores = evaluate_cross_resource_rows(rows, plan_question, retrieve_plan)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "answer_generation_attempted": False,
        "external_api_calls": 0,
        "environment": evaluation_environment(
            model_name=args.model,
            backend="metal_ollama",
            context_limit=int(values.get("OLLAMA_CONTEXT_TOKENS", "4096")),
            batch_or_concurrency=1,
        ),
        "summary": aggregate_cross_resource_scores(scores),
        "per_query": scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
