"""Blind, resumable end-to-end answer capture for the SDG&E pipeline.

This module deliberately projects benchmark rows down to question-only fields
before any model call. Reference answers are consumed only by the separate
grading phase after a baseline capture has finished.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260817


def stratified_sample_indices(
    population_size: int,
    sample_size: int,
    *,
    seed: int,
) -> list[int]:
    """Choose one reproducible random index from each equal ordered bin."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if sample_size <= 0 or sample_size > population_size:
        raise ValueError("sample_size must be between 1 and population_size")

    generator = random.Random(seed)
    return [
        generator.randrange(
            bin_index * population_size // sample_size,
            (bin_index + 1) * population_size // sample_size,
        )
        for bin_index in range(sample_size)
    ]


def _question_only_case(
    row: dict[str, Any],
    *,
    source: str,
    source_index: int,
) -> dict[str, Any]:
    question = row.get("question")
    case_id = row.get("id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source} row {source_index} has no question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{source} row {source_index} has no id")
    category = row.get("category")
    return {
        "case_id": case_id,
        "source": source,
        "source_index": source_index,
        "category": category if isinstance(category, str) else None,
        "question": question.strip(),
    }


def build_blind_cases(
    user_questions_path: Path,
    beta_questions_path: Path,
    *,
    user_limit: int = 16,
    beta_sample_size: int = 10,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Return question-only cases without copying any reference-answer field."""
    user_payload = json.loads(user_questions_path.read_text(encoding="utf-8"))
    beta_payload = json.loads(beta_questions_path.read_text(encoding="utf-8"))
    if not isinstance(user_payload, list):
        raise ValueError("user questions must be a JSON array")
    if not isinstance(beta_payload, dict) or not isinstance(
        beta_payload.get("questions"), list
    ):
        raise ValueError("beta questions must contain a questions array")
    if user_limit <= 0 or user_limit > len(user_payload):
        raise ValueError("user_limit is outside the user question range")

    cases = [
        _question_only_case(row, source="user_questions", source_index=index)
        for index, row in enumerate(user_payload[:user_limit], start=1)
    ]
    beta_rows = beta_payload["questions"]
    selected = stratified_sample_indices(
        len(beta_rows), beta_sample_size, seed=seed
    )
    cases.extend(
        _question_only_case(
            beta_rows[index], source="beta_golden_questions", source_index=index + 1
        )
        for index in selected
    )
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case ids must be unique across the blind manifest")
    return cases


def pending_cases(
    cases: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    successful = {
        result.get("case_id")
        for result in results
        if result.get("status") == "success"
    }
    return [case for case in cases if case.get("case_id") not in successful]


def select_cases(
    cases: list[dict[str, Any]],
    source: str | None,
    case_ids: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    selected = (
        cases
        if source is None
        else [case for case in cases if case.get("source") == source]
    )
    if not case_ids:
        return selected
    requested = set(case_ids)
    known = {str(case.get("case_id")) for case in cases}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown case ids: {unknown}")
    return [case for case in selected if case.get("case_id") in requested]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"results file must contain a JSON array: {path}")
    return payload


def _ranked_result_to_dict(result: Any) -> dict[str, Any]:
    item = result.query_object
    return {
        "chunk_id": str(item.chunk_id),
        "source_file": item.source_pdf,
        "sub_document": item.sub_document,
        "breadcrumb": item.breadcrumb,
        "section_number": item.section_number,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "chunk_index": item.chunk_index,
        "content_type": item.content_type,
        "token_count": item.token_count,
        "distance": item.distance,
        "rerank_score": result.rerank_score,
        "content": item.content,
    }


def _safe_runtime_config() -> dict[str, Any]:
    import os

    names = (
        "TASK3_PROVIDER",
        "GROQ_MODEL",
        "GROQ_MAX_TOKENS",
        "GROQ_CONTEXT_TOKENS",
        "GROQ_REASONING_EFFORT",
        "GROQ_TEMPERATURE",
        "GROQ_TOKEN_SAFETY_FACTOR",
    )
    return {name: os.environ.get(name) for name in names}


def capture_answers(
    cases: list[dict[str, Any]],
    *,
    results_path: Path,
    limit: int | None,
    min_interval_seconds: float,
    pipeline_variant: str,
) -> int:
    """Run one pipeline variant, checkpointing every successful case."""
    from dotenv import load_dotenv

    load_dotenv()
    connection = None
    if pipeline_variant == "legacy_flat":
        from generation import AnswerRequest, AnswerService, adapt_ranked_results
        from generation.providers.factory import create_provider_from_env
        from retrieval.query.pdf import retrieve
        from retrieval.utils import connect_db

        provider = create_provider_from_env()
        answer_service = AnswerService(provider)
        connection = connect_db()
    elif pipeline_variant == "integrated_grouped":
        backend_path = str((Path(__file__).resolve().parents[1] / "backend"))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        from generation.planning import needs_planning
        from services.generation_service import (
            GenerationService,
            interleave_grouped_results,
        )
        from services.retrieval_service import RetrievalService

        retrieval_service = RetrievalService()
        generation_service = GenerationService()
    else:
        raise ValueError(f"unknown pipeline variant: {pipeline_variant}")

    results = _read_results(results_path)
    remaining = pending_cases(cases, results)
    if limit is not None:
        remaining = remaining[:limit]

    last_model_call_finished: float | None = None
    try:
        for position, case in enumerate(remaining, start=1):
            if last_model_call_finished is not None and min_interval_seconds > 0:
                elapsed = time.monotonic() - last_model_call_finished
                if elapsed < min_interval_seconds:
                    wait_seconds = min_interval_seconds - elapsed
                    print(
                        f"rate-limit pacing: waiting {wait_seconds:.1f}s before "
                        f"{case['case_id']}",
                        flush=True,
                    )
                    time.sleep(wait_seconds)

            print(
                f"[{position}/{len(remaining)}] retrieving {case['case_id']}",
                flush=True,
            )
            total_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            plan_diagnostics = None
            verified_excel = False
            excel_trace: list = []
            plan = None
            if pipeline_variant == "legacy_flat":
                ranked_results = retrieve(case["question"], connection)
                evidence_payload: Any = [
                    _ranked_result_to_dict(result) for result in ranked_results
                ]
                evidence_count = len(ranked_results)
            else:
                if needs_planning(case["question"]):
                    plan = generation_service.plan_retrieval(case["question"])
                    bundle = retrieval_service.retrieve_plan(
                        case["question"],
                        plan,
                        rewrite_mode="off",
                    )
                else:
                    bundle = retrieval_service.retrieve(
                        case["question"], rewrite_mode="off"
                    )
                ranked_results = interleave_grouped_results(bundle.evidence)
                evidence_payload = {
                    name: [
                        _ranked_result_to_dict(result)
                        for result in group.results
                    ]
                    for name, group in bundle.evidence.groups.items()
                }
                evidence_count = len(ranked_results)
                plan_diagnostics = bundle.plan_diagnostics
                verified_excel = bool(bundle.verified_excels or bundle.verified_excel)
                excel_trace = list(bundle.excel_trace)
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000)
            print(
                f"[{position}/{len(remaining)}] generating {case['case_id']} "
                f"with {evidence_count} ranked chunks",
                flush=True,
            )
            if pipeline_variant == "legacy_flat":
                chunks = adapt_ranked_results(ranked_results)
                request = AnswerRequest(
                    request_id=str(uuid.uuid4()),
                    question=case["question"],
                    chunks=chunks,
                )
                response = answer_service.answer(request)
            else:
                request_id = str(uuid.uuid4())
                response = (
                    generation_service.generate_mixed(
                        request_id, case["question"], bundle, plan
                    )
                    if plan is not None
                    else generation_service.generate(
                        request_id, case["question"], bundle
                    )
                )
            last_model_call_finished = time.monotonic()
            total_ms = round((time.perf_counter() - total_started) * 1000)
            if not hasattr(response, "to_dict"):
                print(
                    f"stopping after generation error for {case['case_id']}; "
                    "no failed result was checkpointed, so a later run can resume",
                    flush=True,
                )
                return 2

            response_payload = response.to_dict()
            results.append(
                {
                    **case,
                    "status": "success",
                    "captured_at": datetime.now(UTC).isoformat(),
                    "pipeline_variant": pipeline_variant,
                    "runtime_config": _safe_runtime_config(),
                    "retrieval_ms": retrieval_ms,
                    "total_ms": total_ms,
                    "retrieved_evidence": evidence_payload,
                    "verified_excel": verified_excel,
                    "excel_trace": excel_trace,
                    "plan_diagnostics": plan_diagnostics,
                    "response": response_payload,
                }
            )
            _write_json_atomic(results_path, results)
            print(
                f"[{position}/{len(remaining)}] saved {case['case_id']} "
                f"({total_ms / 1000:.1f}s)",
                flush=True,
            )
    finally:
        if connection is not None:
            connection.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and run a blind, resumable SDG&E answer evaluation."
    )
    parser.add_argument(
        "--user-questions",
        type=Path,
        default=Path("user_questions/user_questions.json"),
    )
    parser.add_argument(
        "--beta-questions",
        type=Path,
        default=Path("local/beta_golden_questions.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/evaluation_2026-08-17"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--user-limit", type=int, default=16)
    parser.add_argument("--beta-sample-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=60.0,
        help="Delay between Groq calls; 60s respects the published 8K free TPM.",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--pipeline-variant",
        choices=("legacy_flat", "integrated_grouped"),
        default="legacy_flat",
    )
    parser.add_argument(
        "--case-source",
        choices=("user_questions", "beta_golden_questions"),
        help="Optionally run only one source while keeping the full blind manifest.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this manifest case id; repeat to select multiple cases.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases = build_blind_cases(
        args.user_questions,
        args.beta_questions,
        user_limit=args.user_limit,
        beta_sample_size=args.beta_sample_size,
        seed=args.seed,
    )
    selected_beta_indices = [
        case["source_index"]
        for case in cases
        if case["source"] == "beta_golden_questions"
    ]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "blind": True,
        "references_exposed_to_generation": False,
        "selection": {
            "user_range": [1, args.user_limit],
            "beta_method": "one seeded random choice from each equal ordered bin",
            "beta_seed": args.seed,
            "beta_source_indices_1_based": selected_beta_indices,
        },
        "cases": cases,
    }
    manifest_path = args.out_dir / "blind_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("cases") != cases or existing.get("selection") != manifest["selection"]:
            raise ValueError("existing blind manifest does not match this selection")
        print(f"reused blind manifest: {manifest_path}", flush=True)
    else:
        _write_json_atomic(manifest_path, manifest)
        print(f"wrote blind manifest: {manifest_path}", flush=True)
    if args.manifest_only:
        return 0
    selected_cases = select_cases(
        cases,
        args.case_source,
        tuple(args.case_ids) if args.case_ids else None,
    )
    results_name = (
        "baseline_answers.json"
        if args.pipeline_variant == "legacy_flat"
        else "optimized_answers.json"
    )
    return capture_answers(
        selected_cases,
        results_path=args.out_dir / results_name,
        limit=args.limit,
        min_interval_seconds=args.min_interval_seconds,
        pipeline_variant=args.pipeline_variant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
