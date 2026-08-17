"""Build frozen, non-spatial architecture diagnostic suites.

Gold-bearing development files and question-only blind projections are written
to separate directories. The builder is deterministic and makes no model/API
calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eval/architecture"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _blind(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"id": str(row["id"]), "question": str(row["question"])} for row in rows]


def _period(row: dict[str, Any]) -> str:
    filters = row.get("plan", {}).get("filters", [])
    values = {
        item.get("field"): item.get("value")
        for item in filters
        if isinstance(item, dict)
    }
    year = values.get("reporting_year")
    quarter = values.get("reporting_quarter")
    return f"{year}-Q{quarter}" if quarter else str(year or "unspecified")


def _metric(row: dict[str, Any]) -> str:
    plan = row.get("plan", {})
    if plan.get("semantic_metric_key"):
        return str(plan["semantic_metric_key"])
    for item in plan.get("filters", []):
        if item.get("field") == "entity_key":
            return str(item.get("value"))
    return str(row.get("question_type", "metric"))


def _cross_resource_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(source_rows) < 20:
        raise ValueError("cross-corpus source suite must contain at least 20 rows")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if not row.get("pdf_evidence"):
            continue
        # This frozen source suite contains scalar numeric attributes.
        Decimal(str(row["expected_answer"]))
        groups[(str(row.get("unit")), _period(row))].append(row)
    eligible = [row for values in groups.values() for row in values]
    if len(eligible) < 20:
        raise ValueError("cross-corpus suite lacks 20 aligned scalar facts")

    output: list[dict[str, Any]] = []
    for index in range(24):
        left = eligible[index % len(eligible)]
        peers = groups[(str(left.get("unit")), _period(left))]
        left_position = peers.index(left)
        right = peers[(left_position + 1) % len(peers)]
        left_value = Decimal(str(left["expected_answer"]))
        right_value = Decimal(str(right["expected_answer"]))
        operation = "ratio_percent" if index < 20 else "difference"
        if operation == "ratio_percent":
            if right_value == 0:
                right = left
                right_value = left_value
            expected = (left_value / right_value * 100).quantize(Decimal("0.01"))
            unit = "percent"
            wording = "What percentage is operand A of operand B?"
        else:
            expected = left_value - right_value
            unit = left.get("unit")
            wording = "What is operand A minus operand B?"

        output.append(
            {
                "id": f"cross_resource_{index + 1:04d}",
                "question": (
                    f"{wording} Operand A must use the cleaned QDR workbook fact: "
                    f"{left['question']} Operand B must use the WMP filing fact: "
                    f"{right['question']}"
                ),
                "required_sources": ["excel", "pdf"],
                "facts": [
                    {
                        "id": "operand_a",
                        "source": "excel",
                        "metric": _metric(left),
                        "period": _period(left),
                        "expected_value": str(left_value),
                        "unit": left.get("unit"),
                        "provenance": left.get("provenance", []),
                    },
                    {
                        "id": "operand_b",
                        "source": "pdf",
                        "metric": _metric(right),
                        "period": _period(right),
                        "expected_value": str(right_value),
                        "unit": right.get("unit"),
                        "provenance": right.get("pdf_evidence", []),
                    },
                ],
                "formula": {
                    "operation": operation,
                    "left_ref": "operand_a",
                    "right_ref": "operand_b",
                },
                "expected_value": str(expected),
                "unit": unit,
                "time_scope": _period(left),
                "abstention_required": False,
                "generator": "deterministic_frozen_gold_no_external_api",
            }
        )
    return output


def _modality_rows(
    narrative_rows: list[dict[str, Any]], structured_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tables = [row for row in structured_rows if row.get("question_type") == "table"][:8]
    figures = [row for row in structured_rows if row.get("question_type") == "figure"][:8]
    selected = [
        *(dict(row, required_modality="narrative") for row in narrative_rows[:8]),
        *(dict(row, required_modality="table") for row in tables),
        *(dict(row, required_modality="figure") for row in figures),
    ]
    if len(selected) != 24:
        raise ValueError("modality source suites must provide 8 rows per modality")
    for index, row in enumerate(selected, start=1):
        row["id"] = f"modality_gate_{index:04d}"
    return selected


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_diagnostics(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    excel = _read_jsonl(ROOT / "eval/excel/evaluation_excel_challenge.jsonl")[:24]
    cross_source = _read_jsonl(ROOT / "eval/excel/evaluation_excel_cross_corpus.jsonl")
    narrative = _read_jsonl(ROOT / "eval/pdf/evaluation_natural.jsonl")
    structured = _read_jsonl(ROOT / "eval/pdf/evaluation_structured.jsonl")
    suites = {
        "excel_execution": excel,
        "cross_resource_computation": _cross_resource_rows(cross_source),
        "modality_gating": _modality_rows(narrative, structured),
    }
    for name, rows in suites.items():
        if len(rows) != 24:
            raise ValueError(f"{name} must contain exactly 24 rows")
        _write_jsonl(output / f"dev/{name}.jsonl", rows)
        _write_jsonl(output / f"blind/{name}.jsonl", _blind(rows))

    manifest = {
        "schema_version": "architecture-diagnostics-v1",
        "counts": {name: len(rows) for name, rows in suites.items()},
        "gold_visibility": "dev files contain gold; blind files contain id and question only",
        "external_api_calls": 0,
        "sha256": {
            f"dev/{name}.jsonl": _digest(output / f"dev/{name}.jsonl")
            for name in suites
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_diagnostics(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
