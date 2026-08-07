"""Measure the Excel channel's routing precision on questions it must decline.

The Excel channel is gated by execution, not by a classifier, so its risk is not
a bad ranking — it is answering a *narrative* question with a confident-looking
number. This command runs the channel over the two PDF suites, where the correct
behaviour is always to decline, and reports the false-positive rate.

It also runs the Excel suite, where declining is the failure.

    uv run python -m eval.run_gate_routing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from retrieval.query.excel.channel import ExcelAnswer, answer_from_excel
from retrieval.utils import connect_db

SUITES = {
    "narrative": ("eval/pdf/evaluation_natural.jsonl", "decline"),
    "structured": ("eval/pdf/evaluation_structured.jsonl", "decline"),
    "excel": ("eval/excel/evaluation_excel.jsonl", "answer"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample the first N questions of each suite.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    conn = connect_db()
    report: dict[str, dict] = {}
    try:
        for suite, (path, expected) in SUITES.items():
            rows = [json.loads(line) for line in open(path) if line.strip()]
            if args.limit:
                rows = rows[: args.limit]
            answered = 0
            declines: Counter[str] = Counter()
            examples: list[dict] = []
            for row in rows:
                outcome = answer_from_excel(row["question"], conn)
                if isinstance(outcome, ExcelAnswer):
                    answered += 1
                    if expected == "decline":
                        examples.append(
                            {
                                "id": row.get("id"),
                                "question": row["question"][:110],
                                "table": outcome.table_number,
                                "value": str(outcome.result.rows[:1]),
                            }
                        )
                else:
                    # Group reasons by their leading clause.
                    declines[outcome.reason.split(":")[0][:52]] += 1
            n = len(rows)
            report[suite] = {
                "questions": n,
                "expected": expected,
                "answered": answered,
                "declined": n - answered,
                "answer_rate": answered / n if n else 0.0,
                "decline_reasons": dict(declines.most_common(6)),
                "false_positives": examples[:8],
            }
    finally:
        conn.close()

    print("=" * 72)
    print("EXCEL CHANNEL ROUTING PRECISION")
    print("=" * 72)
    for suite, data in report.items():
        verdict = (
            f"false-positive rate {data['answer_rate']:.3f}"
            if data["expected"] == "decline"
            else f"answer rate {data['answer_rate']:.3f}"
        )
        print(
            f"\n{suite:11s} n={data['questions']:3d}  expected={data['expected']:8s}  "
            f"answered={data['answered']:3d}  {verdict}"
        )
        for reason, count in data["decline_reasons"].items():
            print(f"    declined {count:3d}x  {reason}")
        for example in data["false_positives"]:
            print(f"    [FP] {example['id']}: {example['question']}")
            print(f"         -> T{example['table']} {example['value']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
