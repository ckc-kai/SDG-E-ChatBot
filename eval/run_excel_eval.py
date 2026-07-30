"""Evaluate the Excel lane: card retrieval, routing, and answer correctness.

Three independent things are measured, because they fail differently:

1. **card recall** — did retrieval surface a card for the right table/metric?
2. **routing** — with every lane live, does an Excel card actually win, or does a
   PDF chunk take the slot? This is the gate's cost.
3. **answer accuracy** — does the stored, validated query plan execute to the
   expected value? This exercises the execution-verified path that makes Excel
   routing possible without a classifier.

    uv run python -m eval.run_excel_eval --lanes narrative structured excel
"""

from __future__ import annotations

import argparse
import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path

from retrieval.query.excel.query import (
    ExcelQueryPlan,
    Filter,
    PlanError,
    execute_plan,
)
from retrieval.query.lanes import ALL_LANES, EXCEL, lane_of
from retrieval.query.pdf.query import retrieve_with_diagnostics
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = "eval/excel/evaluation_excel.jsonl"
# Values are compared at a relative tolerance; the executor returns exact
# numerics, so this only absorbs formatting of the stored expectation.
RELATIVE_TOLERANCE = Decimal("1e-9")


def plan_from_dict(payload: dict) -> ExcelQueryPlan:
    filters = tuple(Filter(**f) for f in payload.get("filters", []))
    fields = {k: v for k, v in payload.items() if k != "filters"}
    return ExcelQueryPlan(filters=filters, **fields)


def _values_match(expected: str, actual) -> bool:
    if actual is None:
        return False
    try:
        want = Decimal(str(expected))
        got = Decimal(str(actual))
    except (InvalidOperation, ValueError):
        return str(expected).strip().lower() == str(actual).strip().lower()
    if want == got:
        return True
    scale = max(abs(want), abs(got), Decimal(1))
    return abs(want - got) / scale <= RELATIVE_TOLERANCE


def score_row(row: dict, conn, lanes: tuple[str, ...] | None, k: int) -> dict:
    question = row["question"]
    expected_table = row.get("table_number")
    expected_key = row.get("semantic_metric_key")

    ranked, diagnostics = retrieve_with_diagnostics(
        question, conn, rewrite_mode="off", lanes=lanes
    )
    top = ranked[:k]

    def is_target_card(result) -> bool:
        qo = result.query_object
        if qo.content_type != "excel_card":
            return False
        data = qo.structured_data or {}
        if data.get("table_number") != expected_table:
            return False
        if expected_key and data.get("semantic_metric_key"):
            return data["semantic_metric_key"] == expected_key
        return True

    card_ranks = [i for i, r in enumerate(top, 1) if is_target_card(r)]
    card_hit_1 = bool(card_ranks and card_ranks[0] == 1)
    card_recall_k = bool(card_ranks)
    excel_at_1 = bool(top) and lane_of(top[0].query_object.content_type) == EXCEL
    top_lane = lane_of(top[0].query_object.content_type) if top else None

    # Any-card recall: a card for the right table, ignoring the metric concept.
    any_table_card = any(
        (r.query_object.structured_data or {}).get("table_number") == expected_table
        for r in top
        if r.query_object.content_type == "excel_card"
    )

    # Execution-verified answer path.
    answer_correct = None
    execution_status = "not_attempted"
    if row.get("plan"):
        plan = plan_from_dict(row["plan"])
        try:
            result = execute_plan(plan, conn)
            if row.get("expected_clarification"):
                execution_status = "unexpectedly_executed"
                answer_correct = False
            elif row.get("expected_answer_kind") == "label":
                execution_status = "executed"
                answer_correct = bool(result.rows) and str(
                    result.rows[0][0]
                ) == str(row["expected_answer"])
            elif row.get("expected_answer_kind") == "attribute":
                execution_status = "executed"
                answer_correct = bool(result.rows)
            else:
                execution_status = "executed"
                answer_correct = bool(result.rows) and _values_match(
                    row["expected_answer"], result.rows[0][-1]
                )
        except PlanError as exc:
            if row.get("expected_clarification"):
                execution_status = "clarification_requested"
                answer_correct = True
            else:
                execution_status = f"refused: {exc}"
                answer_correct = False

    return {
        "id": row["id"],
        "question_type": row["question_type"],
        "table_number": expected_table,
        "card_hit@1": card_hit_1,
        f"card_recall@{k}": card_recall_k,
        "any_table_card": any_table_card,
        "excel_at_rank_1": excel_at_1,
        "top_lane": top_lane,
        "answer_correct": answer_correct,
        "execution_status": execution_status,
        "reranked_candidates": [
            {
                "position": position,
                "chunk_id": r.query_object.chunk_id,
                "content_type": r.query_object.content_type,
                "rerank_score": r.rerank_score,
                "is_gold": is_target_card(r),
            }
            for position, r in enumerate(
                diagnostics.reranked_candidates, start=1
            )
        ],
        "lane_confidence": [
            outcome.confidence.as_dict()
            for outcome in diagnostics.lane_outcomes
            if outcome.confidence
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--metric-k", dest="k", type=int, default=5)
    parser.add_argument("--lanes", nargs="*", choices=list(ALL_LANES), default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    rows = [json.loads(line) for line in open(args.eval) if line.strip()]
    lanes = tuple(args.lanes) if args.lanes else None
    conn = connect_db()
    try:
        scores = [score_row(row, conn, lanes, args.k) for row in rows]
    finally:
        conn.close()

    n = len(scores)
    answered = [s for s in scores if s["answer_correct"] is not None]

    def rate(key: str, subset=None) -> float:
        pool = subset if subset is not None else scores
        return sum(1 for s in pool if s[key]) / len(pool) if pool else 0.0

    print("=" * 70)
    print(f"EXCEL RETRIEVAL EVAL  (k={args.k}, questions={n}, "
          f"lanes={'+'.join(lanes) if lanes else 'global'})")
    print("=" * 70)
    print(f"  card hit@1                {rate('card_hit@1'):.3f}")
    print(f"  card recall@{args.k}             {rate(f'card_recall@{args.k}'):.3f}")
    print(f"  any card for right table  {rate('any_table_card'):.3f}")
    print(f"  Excel lane wins rank 1    {rate('excel_at_rank_1'):.3f}")
    print(f"  answer accuracy           "
          f"{sum(1 for s in answered if s['answer_correct'])/len(answered):.3f}"
          f"  ({len(answered)} executable)")

    lane_counts: dict[str, int] = {}
    for s in scores:
        lane_counts[s["top_lane"] or "none"] = lane_counts.get(s["top_lane"] or "none", 0) + 1
    print(f"\n  rank-1 lane distribution: {dict(sorted(lane_counts.items()))}")

    by_type: dict[str, list] = {}
    for s in scores:
        by_type.setdefault(s["question_type"], []).append(s)
    print("\nBY QUESTION TYPE")
    for qtype, subset in sorted(by_type.items()):
        sub_answered = [s for s in subset if s["answer_correct"] is not None]
        acc = (
            sum(1 for s in sub_answered if s["answer_correct"]) / len(sub_answered)
            if sub_answered else 0.0
        )
        print(
            f"  {qtype:24s} n={len(subset):3d}  card_hit@1={rate('card_hit@1', subset):.3f}  "
            f"excel@1={rate('excel_at_rank_1', subset):.3f}  answer={acc:.3f}"
        )

    failures = [s for s in scores if s["answer_correct"] is False]
    if failures:
        print(f"\nANSWER FAILURES ({len(failures)}):")
        for s in failures[:12]:
            print(f"  {s['id']}  {s['execution_status'][:70]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_query": scores}, indent=1, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
