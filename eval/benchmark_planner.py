"""Benchmark the bounded planner without retrieval, generation, or Groq calls."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from eval.environment import evaluation_environment
from generation.planning import build_retrieval_plan
from generation.providers import create_provider_from_env


DEFAULT_QUESTIONS = Path("eval/architecture/blind/cross_resource_computation.jsonl")
DEFAULT_GOLD = Path("eval/architecture/dev/cross_resource_computation.jsonl")


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _score_required_sources(payload: dict, gold_path: Path) -> dict:
    gold = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    matched_facts = 0
    required_facts = 0
    for result in payload["per_question"]:
        reference = gold.get(result["id"], {})
        required = [fact["source"] for fact in reference.get("facts", [])]
        try:
            raw = json.loads(result.get("raw_output") or "{}")
            planned = [task.get("source") for task in raw.get("tasks", [])]
        except (json.JSONDecodeError, AttributeError):
            planned = []
        remaining = list(planned)
        matches = 0
        for source in required:
            if source in remaining:
                matches += 1
                remaining.remove(source)
        required_facts += len(required)
        matched_facts += matches
        result["required_sources"] = required
        result["planned_sources"] = planned
        result["required_source_fact_coverage"] = (
            matches / len(required) if required else None
        )
    payload["summary"]["required_source_facts_covered"] = matched_facts
    payload["summary"]["required_source_fact_count"] = required_facts
    payload["summary"]["required_source_fact_coverage"] = (
        matched_facts / required_facts if required_facts else None
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--quantization", default=None)
    parser.add_argument(
        "--resident-memory",
        default=None,
        help="Optional post-run Ollama residency value (for example, '5.6 GB').",
    )
    parser.add_argument(
        "--score-existing",
        type=Path,
        help="Re-score an existing benchmark artifact without calling a model.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    if args.score_existing:
        payload = json.loads(args.score_existing.read_text(encoding="utf-8"))
        payload = _score_required_sources(payload, args.gold)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2))
        return

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    values = dict(os.environ)
    values.update(
        OLLAMA_MODEL=args.model,
        OLLAMA_CONTEXT_TOKENS=str(args.context_limit),
        OLLAMA_MAX_TOKENS="500",
    )
    provider = create_provider_from_env("ollama", environ=values)
    results = []
    for row in rows:
        started = time.perf_counter()
        plan = build_retrieval_plan(row["question"], provider)
        wall_ms = round((time.perf_counter() - started) * 1000)
        usage = provider.last_usage
        results.append(
            {
                "id": row["id"],
                "valid_structured_plan": plan.source == "model",
                "plan_source": plan.source,
                "trigger_reason": plan.trigger_reason,
                "atomic_task_count": plan.atomic_task_count,
                "initial_branch_count": len(plan.steps),
                "dropped_task_count": plan.dropped_task_count,
                "wall_ms": wall_ms,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "model_reported_ms": usage.latency_ms if usage else None,
                "raw_output": provider.last_raw_text,
            }
        )
        print(
            f"{row['id']}: source={plan.source} branches={len(plan.steps)} "
            f"wall_ms={wall_ms}",
            flush=True,
        )

    latencies = [row["wall_ms"] for row in results]
    output_tokens = sum(row["output_tokens"] or 0 for row in results)
    reported_ms = sum(row["model_reported_ms"] or 0 for row in results)
    payload = {
        "environment": evaluation_environment(
            model_name=args.model,
            backend="metal_ollama",
            quantization=args.quantization,
            context_limit=args.context_limit,
            batch_or_concurrency=1,
        ),
        "summary": {
            "count": len(results),
            "valid_structured_plans": sum(
                row["valid_structured_plan"] for row in results
            ),
            "valid_structured_plan_rate": (
                sum(row["valid_structured_plan"] for row in results) / len(results)
                if results
                else 0.0
            ),
            "first_case_ms": latencies[0] if latencies else None,
            "cold_start_ms": None,
            "cold_start_note": "Model residency was not reset before this run",
            "p50_wall_ms": _percentile(latencies, 0.50),
            "p95_wall_ms": _percentile(latencies, 0.95),
            "output_tokens_per_second": (
                round(output_tokens / (reported_ms / 1000), 2)
                if reported_ms and output_tokens
                else None
            ),
            "groq_calls": 0,
            "peak_memory": args.resident_memory,
            "peak_memory_note": (
                "Post-run resident size from ollama ps; not a true peak"
                if args.resident_memory
                else "Ollama API does not expose peak unified memory"
            ),
        },
        "per_question": results,
    }
    payload = _score_required_sources(payload, args.gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
