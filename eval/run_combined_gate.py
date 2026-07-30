"""Combined regression gate across all three retrieval suites.

The 32-point narrative regression shipped unnoticed because each suite was run
and judged alone, so a change that traded one corpus for another looked fine.
This command runs narrative, structured, and Excel together and fails if any of
them falls below its floor.

It also reports the **routing gap**: each PDF suite measured under oracle routing
(its own lane only) minus the same suite under real routing (every lane live).
That difference is the cost of the gate, and it is the number to drive down.

    uv run python -m eval.run_combined_gate            # oracle + real routing
    uv run python -m eval.run_combined_gate --real-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

NARRATIVE_EVAL = "eval/pdf/evaluation_natural.jsonl"
STRUCTURED_EVAL = "eval/pdf/evaluation_structured.jsonl"
EXCEL_EVAL = "eval/excel/evaluation_excel.jsonl"

COMMON = [
    "--rewrite-mode", "off",
    "--retrieval-top-k", "30",
    "--rerank-top-k", "10",
]

# Floors are the measured lane-separated results, minus a small tolerance.
FLOORS = {
    "narrative_hit@1": 0.80,
    "structured_hit@1": 0.60,
    "excel_answer_accuracy": 0.95,
}


def _run(argv: list[str], out: Path) -> dict:
    print(f"  $ {' '.join(argv[-6:])}")
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-3000:] + result.stderr[-3000:])
        raise SystemExit(f"eval failed: {' '.join(argv)}")
    return json.loads(out.read_text())


def narrative(out_dir: Path, lanes: list[str]) -> dict:
    out = out_dir / f"narrative-{'_'.join(lanes)}.json"
    payload = _run(
        [
            sys.executable, "-m", "eval.run_eval",
            "--eval", NARRATIVE_EVAL,
            "--embedding-mode", "hybrid", "--hybrid-pool-mode", "union",
            "--metric-k", "10", *COMMON,
            "--lanes", *lanes, "--out", str(out),
        ],
        out,
    )
    return payload["summary"]


def structured(out_dir: Path, lanes: list[str]) -> dict:
    out = out_dir / f"structured-{'_'.join(lanes)}.json"
    payload = _run(
        [
            sys.executable, "-m", "eval.run_structured_eval",
            "--eval", STRUCTURED_EVAL,
            "--metric-k", "5", *COMMON,
            "--lanes", *lanes, "--out", str(out),
        ],
        out,
    )
    return payload["summary"]


def excel(out_dir: Path, lanes: list[str]) -> dict:
    out = out_dir / f"excel-{'_'.join(lanes)}.json"
    payload = _run(
        [
            sys.executable, "-m", "eval.run_excel_eval",
            "--eval", EXCEL_EVAL, "--metric-k", "5",
            "--lanes", *lanes, "--out", str(out),
        ],
        out,
    )
    rows = payload["per_query"]
    answered = [r for r in rows if r["answer_correct"] is not None]
    return {
        "count": len(rows),
        "card_hit@1": sum(r["card_hit@1"] for r in rows) / len(rows),
        "card_recall@5": sum(r["card_recall@5"] for r in rows) / len(rows),
        "excel_at_rank_1": sum(r["excel_at_rank_1"] for r in rows) / len(rows),
        "answer_accuracy": (
            sum(1 for r in answered if r["answer_correct"]) / len(answered)
            if answered else 0.0
        ),
    }


def _get(summary: dict, *keys: str) -> float:
    for key in keys:
        if key in summary:
            return float(summary[key])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/results/gate"))
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Skip the oracle-routing runs; report real routing only.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_lanes = ["narrative", "structured", "excel"]
    report: dict[str, dict] = {}

    if not args.real_only:
        print("Oracle routing (each suite sees only its own lane):")
        report["narrative_oracle"] = narrative(args.out_dir, ["narrative"])
        report["structured_oracle"] = structured(args.out_dir, ["structured"])
        report["excel_oracle"] = excel(args.out_dir, ["excel"])

    print("\nReal routing (every lane live):")
    report["narrative_real"] = narrative(args.out_dir, all_lanes)
    report["structured_real"] = structured(args.out_dir, all_lanes)
    report["excel_real"] = excel(args.out_dir, all_lanes)

    print("\n" + "=" * 74)
    print("COMBINED RETRIEVAL GATE")
    print("=" * 74)
    print(f"{'suite':14s} {'metric':18s} {'oracle':>9s} {'real':>9s} {'gap':>9s}")
    print("-" * 74)

    comparisons = [
        ("narrative", "hit@1", "hit@1"),
        ("narrative", "recall@10", "recall@10"),
        ("structured", "hit@1", "hit@1"),
        ("structured", "recall@5", "recall@5"),
        ("excel", "answer_accuracy", "answer_accuracy"),
        ("excel", "card_hit@1", "card_hit@1"),
    ]
    for suite, label, key in comparisons:
        real = _get(report[f"{suite}_real"], key)
        if f"{suite}_oracle" in report:
            oracle = _get(report[f"{suite}_oracle"], key)
            print(
                f"{suite:14s} {label:18s} {oracle:9.3f} {real:9.3f} "
                f"{real - oracle:+9.3f}"
            )
        else:
            print(f"{suite:14s} {label:18s} {'-':>9s} {real:9.3f} {'-':>9s}")

    print("-" * 74)
    failures = []
    checks = {
        "narrative_hit@1": _get(report["narrative_real"], "hit@1"),
        "structured_hit@1": _get(report["structured_real"], "hit@1"),
        "excel_answer_accuracy": _get(report["excel_real"], "answer_accuracy"),
    }
    for name, value in checks.items():
        floor = FLOORS[name]
        status = "PASS" if value >= floor else "FAIL"
        if status == "FAIL":
            failures.append(f"{name}={value:.3f} < floor {floor:.2f}")
        print(f"[{status}] {name:26s} {value:.3f}  (floor {floor:.2f})")

    (args.out_dir / "combined_gate.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(f"\nWrote {args.out_dir / 'combined_gate.json'}")
    if failures:
        raise SystemExit("GATE FAILED: " + "; ".join(failures))
    print("Combined gate passed.")


if __name__ == "__main__":
    main()
