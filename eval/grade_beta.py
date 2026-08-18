"""Assemble grading packets and compute rubric scores for a beta run.

Grading itself is a model-assisted manual review against the frozen golden
answers (the same methodology that produced ``optimized_scores.json``). This
tool has two subcommands:

- ``packets``: render question, golden answer, and captured answer side by
  side for review, without exposing golds to any answering model.
- ``score``: combine a reviewed per-case dimension file with the rubric
  weights into the final summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUBRIC_WEIGHTS = {
    "correctness": 0.25,
    "completeness": 0.20,
    "groundedness": 0.15,
    "citation_quality": 0.15,
    "task_fulfillment": 0.10,
    "uncertainty_calibration": 0.10,
    "clarity": 0.05,
}
SCALE_MAX = 4


def weighted_score_100(scores: dict[str, int]) -> float:
    missing = set(RUBRIC_WEIGHTS) - set(scores)
    if missing:
        raise ValueError(f"missing rubric dimensions: {sorted(missing)}")
    total = sum(RUBRIC_WEIGHTS[name] * scores[name] for name in RUBRIC_WEIGHTS)
    return round(total / SCALE_MAX * 100, 3)


def write_packets(answers_path: Path, beta_path: Path, out_path: Path) -> None:
    answers = {
        row["case_id"]: row
        for row in json.loads(answers_path.read_text(encoding="utf-8"))
    }
    beta = json.loads(beta_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for question in beta["questions"]:
        case_id = question["id"]
        captured = answers.get(case_id)
        lines.append("=" * 100)
        lines.append(
            f"CASE {case_id} | {question.get('category')} | expected: "
            f"{question.get('expected_response_behavior')}"
        )
        lines.append(f"QUESTION: {question['question']}")
        lines.append(f"GOLDEN: {question.get('golden_answer', '')}")
        if captured is None:
            lines.append("ANSWER: <not captured>")
            continue
        response = captured.get("response", {})
        lines.append(
            f"ANSWER (insufficient_context={response.get('insufficient_context')}, "
            f"cited={len(response.get('cited_chunk_ids', []))}, "
            f"verified_excel={captured.get('verified_excel')}):"
        )
        lines.append(response.get("answer", ""))
        citations = response.get("citations", [])
        if citations:
            rendered = "; ".join(
                str(
                    citation.get("source_file")
                    or citation.get("source_id")
                    or citation.get("chunk_id")
                )
                for citation in citations[:8]
            )
            lines.append(f"CITED SOURCES: {rendered}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def score(reviews_path: Path, out_path: Path) -> None:
    payload = json.loads(reviews_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    scored = []
    for case in cases:
        scores = case["scores"]
        scored.append(
            {
                "case_id": case["case_id"],
                "score_100": weighted_score_100(scores),
                "scores": scores,
                "diagnosis": case.get("diagnosis", ""),
            }
        )
    count = len(scored)
    mean = round(sum(case["score_100"] for case in scored) / count, 3)
    dimension_means = {
        name: round(sum(case["scores"][name] for case in scored) / count, 3)
        for name in RUBRIC_WEIGHTS
    }
    summary = {
        "rubric_version": "1.0",
        "grader": payload.get(
            "grader",
            "manual model-assisted review against frozen beta golden answers "
            "after blind generation completed",
        ),
        "scale": "0=failed, 1=poor, 2=partial, 3=good, 4=excellent",
        "weights": RUBRIC_WEIGHTS,
        "summary": {
            "case_count": count,
            "mean_score_100": mean,
            "cases_at_or_above_75": sum(
                1 for case in scored if case["score_100"] >= 75
            ),
            "cases_at_or_above_50": sum(
                1 for case in scored if case["score_100"] >= 50
            ),
            "dimension_means_0_to_4": dimension_means,
        },
        "cases": scored,
    }
    out_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"mean_score_100={mean} cases>=75="
          f"{summary['summary']['cases_at_or_above_75']} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packets = sub.add_parser("packets")
    packets.add_argument("--answers", type=Path, required=True)
    packets.add_argument(
        "--beta", type=Path, default=Path("local/beta_golden_questions.json")
    )
    packets.add_argument("--out", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--reviews", type=Path, required=True)
    scorer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "packets":
        write_packets(args.answers, args.beta, args.out)
    else:
        score(args.reviews, args.out)


if __name__ == "__main__":
    main()
