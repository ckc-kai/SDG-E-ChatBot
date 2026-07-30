"""Fail-fast validation for the versioned Excel evaluation suites.

This command is intentionally retrieval-free. It validates schema invariants,
the pinned active-revision manifest, and every stored gold plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.generate_excel_eval import _validate_rows
from eval.run_eval.excel import _gold_plan_score, validate_manifest
from retrieval.utils import connect_db

DEFAULT_SUITES = (
    Path("eval/excel/evaluation_excel.jsonl"),
    Path("eval/excel/evaluation_excel_challenge.jsonl"),
    Path("eval/excel/evaluation_excel_cross_corpus.jsonl"),
)


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", type=Path, default=list(DEFAULT_SUITES))
    parser.add_argument("--allow-corpus-drift", action="store_true")
    args = parser.parse_args()

    conn = connect_db()
    try:
        total = 0
        for path in args.suites:
            rows = _load(path)
            if path.name == "evaluation_excel.jsonl":
                suite, expected = "release", 80
            elif path.name == "evaluation_excel_challenge.jsonl":
                suite, expected = "challenge", 24
            elif path.name == "evaluation_excel_cross_corpus.jsonl":
                suite, expected = "cross", 20
            else:
                raise SystemExit(f"Unknown Excel evaluation suite: {path}")
            _validate_rows(rows, suite=suite, expected_size=expected)
            validate_manifest(
                path,
                conn,
                allow_corpus_drift=args.allow_corpus_drift,
            )
            failures: list[str] = []
            for row in rows:
                correct, status = _gold_plan_score(row, conn)
                if correct is not True:
                    failures.append(f"{row['id']}: {status}")
            if failures:
                raise SystemExit(
                    f"{path}: {len(failures)} invalid gold plans\n"
                    + "\n".join(failures[:12])
                )
            if suite == "cross" and any(
                not row.get("pdf_evidence") for row in rows
            ):
                raise SystemExit(f"{path}: cross-corpus row lacks PDF evidence")
            print(f"PASS {path}: {len(rows)} rows")
            total += len(rows)
    finally:
        conn.close()
    print(f"PASS all Excel evaluation suites: {total} rows")


if __name__ == "__main__":
    main()
