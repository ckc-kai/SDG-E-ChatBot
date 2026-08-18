"""Blind, resumable capture of every beta golden question end to end.

Unlike ``eval/blind_pipeline_eval.py`` (user questions plus a beta sample),
this runner takes the complete beta golden set. Only question-only fields are
projected before any model call; golden answers never reach the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.blind_pipeline_eval import capture_answers


def build_cases(beta_questions_path: Path) -> list[dict]:
    payload = json.loads(beta_questions_path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("beta questions must contain a non-empty questions array")
    cases = []
    for index, row in enumerate(questions, start=1):
        question = row.get("question")
        case_id = row.get("id")
        if not isinstance(question, str) or not isinstance(case_id, str):
            raise ValueError(f"beta row {index} is missing id or question")
        cases.append(
            {
                "case_id": case_id,
                "source": "beta_golden_questions",
                "source_index": index,
                "category": row.get("category"),
                "question": question.strip(),
            }
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("beta case ids must be unique")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beta-questions",
        type=Path,
        default=Path("local/beta_golden_questions.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=10.0,
        help="Pacing between answer-model calls.",
    )
    args = parser.parse_args()

    cases = build_cases(args.beta_questions)
    if args.case_ids:
        requested = set(args.case_ids)
        unknown = requested - {case["case_id"] for case in cases}
        if unknown:
            raise SystemExit(f"unknown case ids: {sorted(unknown)}")
        cases = [case for case in cases if case["case_id"] in requested]
    return capture_answers(
        cases,
        results_path=args.out_dir / "answers.json",
        limit=args.limit,
        min_interval_seconds=args.min_interval_seconds,
        pipeline_variant="integrated_grouped",
    )


if __name__ == "__main__":
    raise SystemExit(main())
