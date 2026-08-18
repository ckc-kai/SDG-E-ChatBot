"""Evaluate Excel retrieval, gold-plan execution, and the live Excel channel.

The three paths are reported independently: a relevant card can be retrieved
while a plan is wrong, and a correct gold plan does not prove that the live
deterministic channel can formulate it from a natural-language question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from eval.environment import evaluation_environment
from retrieval.query.excel.channel import ExcelAnswer, answer_from_excel
from retrieval.query.excel.query import (
    ExcelQueryPlan,
    Filter,
    PlanError,
    execute_plan,
)
from retrieval.query.lanes import ALL_LANES, EXCEL, lane_of
from retrieval.query.pdf.query import CAPTION_RERANK_WEIGHT, retrieve_with_diagnostics
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = "eval/excel/evaluation_excel.jsonl"
DEFAULT_RELATIVE_TOLERANCE = Decimal("1e-9")


def plan_from_dict(payload: dict[str, Any]) -> ExcelQueryPlan:
    filters = tuple(Filter(**flt) for flt in payload.get("filters", []))
    fields = {key: value for key, value in payload.items() if key != "filters"}
    for tuple_field in ("group_by", "select_json_keys"):
        if tuple_field in fields:
            fields[tuple_field] = tuple(fields[tuple_field])
    return ExcelQueryPlan(filters=filters, **fields)


def _normalise_scalar(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _values_match(expected: Any, actual: Any, tolerance: str | None = None) -> bool:
    if actual is None:
        return False
    try:
        want = Decimal(str(expected))
        got = Decimal(str(actual))
    except (InvalidOperation, ValueError, TypeError):
        return _normalise_scalar(expected) == _normalise_scalar(actual)
    if want == got:
        return True
    relative = Decimal(str(tolerance or DEFAULT_RELATIVE_TOLERANCE))
    if relative == 0:
        return False
    scale = max(abs(want), abs(got), Decimal(1))
    return abs(want - got) / scale <= relative


def _rows_match(expected: Any, actual: list[tuple]) -> bool:
    if not isinstance(expected, list) or len(expected) != len(actual):
        return False
    for expected_row, actual_row in zip(expected, actual):
        if len(expected_row) != len(actual_row):
            return False
        if not all(
            _values_match(expected_cell, actual_cell)
            for expected_cell, actual_cell in zip(expected_row, actual_row)
        ):
            return False
    return True


def _gold_plan_score(row: dict[str, Any], conn) -> tuple[bool | None, str]:
    if not row.get("plan"):
        return None, "not_attempted"
    plan = plan_from_dict(row["plan"])
    kind = row.get("expected_answer_kind", "numeric")
    try:
        result = execute_plan(plan, conn)
    except PlanError as exc:
        if kind == "clarification":
            return True, f"clarification_requested: {exc}"
        return False, f"refused: {exc}"

    if kind == "clarification":
        return False, "unexpectedly_executed"
    if kind == "empty":
        correct = result.contributing_facts == 0 and (
            not result.rows or result.rows[0][-1] is None
        )
        return correct, "empty_verified" if correct else "unexpected_value"
    if kind == "rows":
        correct = _rows_match(row["expected_answer"], result.rows)
        return correct, "executed"
    if not result.rows:
        return False, "executed_empty"
    if kind == "label":
        actual = result.rows[0][0]
    else:
        # Numeric, text, and attribute gold answers are all the final selected
        # value. Attribute rows must be compared, not merely be non-empty.
        actual = result.rows[0][-1]
    correct = _values_match(
        row.get("expected_answer"),
        actual,
        row.get("answer_tolerance"),
    )
    return correct, "executed"


def _live_channel_score(
    row: dict[str, Any],
    conn,
) -> tuple[bool | None, str, str | None]:
    behavior = row.get("expected_channel_behavior", "answer")
    outcome = answer_from_excel(row["question"], conn)
    if isinstance(outcome, ExcelAnswer):
        if behavior != "answer":
            return (
                False,
                "answered_when_expected_to_decline",
                str(outcome.result.rows[:1]),
            )
        kind = row.get("expected_answer_kind", "numeric")
        if kind in {"numeric", "label", "text", "attribute"}:
            if not outcome.result.rows:
                return False, "live_empty", None
            actual = outcome.result.rows[0][-1]
            correct = _values_match(
                row.get("expected_answer"),
                actual,
                row.get("answer_tolerance"),
            )
            return correct, "answered", str(actual)
        # Multi-row target behavior is intentionally challenge-only until the
        # production channel can formulate comparisons and trends.
        return (
            False,
            "live_channel_does_not_support_answer_shape",
            str(outcome.result.rows[:3]),
        )

    if behavior == "decline":
        return True, f"declined: {outcome.reason}", None
    if behavior == "clarify" and "plan refused" in outcome.reason:
        return True, f"clarified: {outcome.reason}", None
    return False, f"declined: {outcome.reason}", None


def _retrieval_score(
    row: dict[str, Any],
    conn,
    lanes: tuple[str, ...] | None,
    k: int,
    *,
    isolate_excel: bool = False,
) -> dict[str, Any]:
    expected_table = row.get("table_number")
    expected_key = row.get("semantic_metric_key")
    ranked, diagnostics = retrieve_with_diagnostics(
        row["question"],
        conn,
        rewrite_mode="off",
        lanes=lanes,
        content_types=("excel_card",) if isolate_excel else None,
    )
    top = ranked[:k]

    def is_target_card(result) -> bool:
        query_object = result.query_object
        if query_object.content_type != "excel_card":
            return False
        data = query_object.structured_data or {}
        if data.get("table_number") != expected_table:
            return False
        if expected_key and data.get("semantic_metric_key"):
            return data["semantic_metric_key"] == expected_key
        return True

    card_ranks = [index for index, item in enumerate(top, 1) if is_target_card(item)]
    top_lane = lane_of(top[0].query_object.content_type) if top else None
    preferred = row.get("preferred_lane")
    if preferred == "pdf":
        preferred_lane_correct = top_lane in {"narrative", "structured"}
    elif preferred:
        preferred_lane_correct = top_lane == preferred
    else:
        preferred_lane_correct = None
    return {
        "card_hit@1": bool(card_ranks and card_ranks[0] == 1),
        f"card_recall@{k}": bool(card_ranks),
        "any_table_card": any(
            (item.query_object.structured_data or {}).get("table_number")
            == expected_table
            for item in top
            if item.query_object.content_type == "excel_card"
        ),
        "excel_at_rank_1": top_lane == EXCEL,
        "top_lane": top_lane,
        "preferred_lane_correct": preferred_lane_correct,
        "reranked_candidates": [
            {
                "position": position,
                "chunk_id": item.query_object.chunk_id,
                "content_type": item.query_object.content_type,
                "rerank_score": item.rerank_score,
                "is_gold": is_target_card(item),
            }
            for position, item in enumerate(diagnostics.reranked_candidates, start=1)
        ],
    }


def score_row(
    row: dict[str, Any],
    conn,
    lanes: tuple[str, ...] | None,
    k: int,
    *,
    skip_retrieval: bool = False,
    skip_live_channel: bool = False,
    isolate_excel: bool = False,
) -> dict[str, Any]:
    retrieval = (
        {}
        if skip_retrieval
        else _retrieval_score(row, conn, lanes, k, isolate_excel=isolate_excel)
    )
    gold_correct, execution_status = _gold_plan_score(row, conn)
    if skip_live_channel:
        live_correct, live_status, live_value = None, "not_attempted", None
    else:
        live_correct, live_status, live_value = _live_channel_score(row, conn)
    return {
        "id": row["id"],
        "question": row["question"],
        "question_type": row["question_type"],
        "difficulty": row.get("difficulty", "unknown"),
        "source_scope": row.get("source_scope", "unknown"),
        "expected_channel_behavior": row.get("expected_channel_behavior", "answer"),
        "table_number": row.get("table_number"),
        **retrieval,
        # Backward-compatible name consumed by run_combined_gate.
        "answer_correct": gold_correct,
        "gold_plan_correct": gold_correct,
        "execution_status": execution_status,
        "live_channel_correct": live_correct,
        "live_channel_status": live_status,
        "live_channel_value": live_value,
    }


def _active_revision_hashes(conn) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.table_number, r.source_hash
            FROM excel_sources s
            JOIN excel_revisions r ON r.id=s.active_revision_id
            ORDER BY s.table_number
            """
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def validate_manifest(
    eval_path: Path,
    conn,
    *,
    allow_corpus_drift: bool,
) -> None:
    manifest_path = eval_path.parent / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No Excel evaluation manifest found at %s", manifest_path)
        return
    manifest = json.loads(manifest_path.read_text())
    suite = manifest.get("suites", {}).get(eval_path.name)
    if not suite:
        raise SystemExit(f"{eval_path.name} is not registered in {manifest_path}")
    actual_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    if actual_sha != suite["sha256"]:
        raise SystemExit(f"{eval_path} differs from its registered suite hash")
    expected = {
        row["table_number"]: row["source_hash"] for row in manifest["active_revisions"]
    }
    active = _active_revision_hashes(conn)
    if expected != active:
        message = (
            "Active Excel revisions differ from the evaluation manifest; "
            "regenerate and audit the suites before scoring."
        )
        if not allow_corpus_drift:
            raise SystemExit(message)
        logger.warning(message)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(bool(value) for value in values) / len(values) if values else None


def _summary_block(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    return {
        "count": len(rows),
        "card_hit@1": _rate(rows, "card_hit@1"),
        f"card_recall@{k}": _rate(rows, f"card_recall@{k}"),
        "excel_at_rank_1": _rate(rows, "excel_at_rank_1"),
        "preferred_lane_accuracy": _rate(rows, "preferred_lane_correct"),
        "gold_plan_accuracy": _rate(rows, "gold_plan_correct"),
        "live_channel_accuracy": _rate(rows, "live_channel_correct"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path(DEFAULT_EVAL_PATH))
    parser.add_argument("--metric-k", dest="k", type=int, default=5)
    parser.add_argument("--lanes", nargs="*", choices=list(ALL_LANES), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-live-channel", action="store_true")
    parser.add_argument("--allow-corpus-drift", action="store_true")
    parser.add_argument(
        "--isolate-excel",
        action="store_true",
        help="Score Excel cards as an independent evidence group.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    rows = [
        json.loads(line) for line in args.eval.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    lanes = tuple(args.lanes) if args.lanes else None
    if lanes and args.isolate_excel:
        parser.error("--lanes and --isolate-excel are mutually exclusive")
    conn = connect_db()
    try:
        validate_manifest(args.eval, conn, allow_corpus_drift=args.allow_corpus_drift)
        scores = [
            score_row(
                row,
                conn,
                lanes,
                args.k,
                skip_retrieval=args.skip_retrieval,
                skip_live_channel=args.skip_live_channel,
                isolate_excel=args.isolate_excel,
            )
            for row in rows
        ]
    finally:
        conn.close()

    summary = _summary_block(scores, args.k)
    route = (
        "excel-group" if args.isolate_excel else "+".join(lanes) if lanes else "global"
    )
    print("=" * 74)
    print(f"EXCEL EVAL  questions={len(scores)}  " f"route={route}")
    print("=" * 74)
    for label, key in (
        ("card hit@1", "card_hit@1"),
        (f"card recall@{args.k}", f"card_recall@{args.k}"),
        ("Excel lane at rank 1", "excel_at_rank_1"),
        ("preferred-lane accuracy", "preferred_lane_accuracy"),
        ("gold-plan accuracy", "gold_plan_accuracy"),
        ("live-channel accuracy", "live_channel_accuracy"),
    ):
        value = summary.get(key)
        if value is not None:
            print(f"  {label:27s} {value:.3f}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        groups[("type", score["question_type"])].append(score)
        groups[("difficulty", score["difficulty"])].append(score)
        groups[("scope", score["source_scope"])].append(score)
    print("\nBREAKDOWN")
    for (dimension, name), subset in sorted(groups.items()):
        block = _summary_block(subset, args.k)
        gold = block["gold_plan_accuracy"]
        live = block["live_channel_accuracy"]
        print(
            f"  {dimension:10s} {name:25s} n={len(subset):3d} "
            f"gold={gold if gold is not None else float('nan'):.3f} "
            f"live={live if live is not None else float('nan'):.3f}"
        )

    failures = [
        score
        for score in scores
        if score["gold_plan_correct"] is False or score["live_channel_correct"] is False
    ]
    if failures:
        print(f"\nFAILURES ({len(failures)}; first 16)")
        for score in failures[:16]:
            print(
                f"  {score['id']} gold={score['execution_status']} "
                f"live={score['live_channel_status']}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "environment": evaluation_environment(batch_or_concurrency=1),
                    "summary": summary,
                    "run_config": {
                        "caption_rerank_weight": CAPTION_RERANK_WEIGHT,
                        "lanes": args.lanes,
                        "isolate_excel": args.isolate_excel,
                        "skip_retrieval": args.skip_retrieval,
                        "skip_live_channel": args.skip_live_channel,
                    },
                    "per_query": scores,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
