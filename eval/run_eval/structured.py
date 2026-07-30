"""Retrieval evaluation for structured (table/figure) chunks.

This harness runs table/figure questions through the same unified
``retrieval.query.pdf.query`` path used for narrative questions and scores the
reranked
top-k against gold table/figure chunks in ``chunks``.

Gold chunks are resolved by the stable key
(source_pdf, content_type, page_start, chunk_index) against ``chunks``, so the
eval survives re-extraction that renumbers serial
ids. Duplicate-aware matching mirrors run_eval: the corpus contains
byte-identical duplicate documents, so a retrieved candidate counts as gold if
its key matches OR its stored content is identical to a gold chunk's content.

Row format (eval/pdf/evaluation_structured.jsonl):
    {"id": ..., "difficulty": ..., "question_type": "table" | "figure",
     "question": ..., "expected_answer": ...,
     "evidence": [{"source_pdf": ..., "content_type": ...,
                   "page_start_db": ..., "chunk_index": ...}, ...]}
Each evidence item is a required gold element (usually one per row).

Usage:
    python -m eval.run_eval --input PATH --pdf structured --output FEATURE
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from eval.run_eval.narrative import normalize
from retrieval.query.lanes import ALL_LANES
from retrieval.query.pdf.query import (
    QueryObject,
    RankedResult,
    retrieve_with_diagnostics,
)
from retrieval.utils import connect_db

DEFAULT_EVAL_PATH = "eval/pdf/evaluation_structured.jsonl"
DEFAULT_K = 5

GoldGroup = tuple[set[int], set[str]]  # (chunk ids, normalized contents)


class GoldNotFoundError(Exception):
    """A row's stable-key gold reference does not resolve in unified chunks."""


@dataclass(frozen=True)
class StructuredQueryScore:
    id: str
    difficulty: str
    question_type: str
    num_gold: int
    hit_at_1: float
    first_gold_rank: int | None
    retrieved_keys: list[str]
    pooled_gold_rank: int | None  # rank of first gold in the pre-rerank pool
    structured_in_top_k: int  # structured candidates among reranked top-k
    cutoff_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    # Full reranked list with gold flags, so score calibration can be fitted
    # from stored runs without re-retrieving.
    reranked_candidates: list[dict] = field(default_factory=list)


def load_eval(path: str) -> list[dict]:
    with open(path, "r") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    return rows


def gold_signatures(conn, row: dict) -> list[GoldGroup]:
    groups: list[GoldGroup] = []
    with conn.cursor() as cur:
        for evidence in row["evidence"]:
            cur.execute(
                """
                SELECT c.id, c.content
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.filename = %s AND c.content_type = %s
                  AND c.page_start = %s AND c.chunk_index = %s
                """,
                (
                    evidence["source_pdf"],
                    evidence["content_type"],
                    evidence["page_start_db"],
                    evidence["chunk_index"],
                ),
            )
            resolved = cur.fetchone()
            if resolved is None:
                raise GoldNotFoundError(
                    "gold structured chunk not found by stable key "
                    f"(source_pdf={evidence['source_pdf']!r}, "
                    f"content_type={evidence['content_type']!r}, "
                    f"page_start={evidence['page_start_db']}, "
                    f"chunk_index={evidence['chunk_index']})"
                )
            groups.append(({resolved[0]}, {normalize(resolved[1])}))
    return groups


def _query_object(
    candidate: QueryObject | RankedResult,
) -> QueryObject:
    if isinstance(candidate, RankedResult):
        return candidate.query_object
    return candidate


def _matches_group(
    candidate: QueryObject | RankedResult, group: GoldGroup
) -> bool:
    qo = _query_object(candidate)
    ids, contents = group
    return qo.chunk_id in ids or normalize(qo.content) in contents


def _distinct_gold_hits(ranked, gold_groups: list[GoldGroup]) -> list[bool]:
    credited: set[int] = set()
    flags: list[bool] = []
    for result in ranked:
        match = next(
            (
                index
                for index, group in enumerate(gold_groups)
                if index not in credited and _matches_group(result, group)
            ),
            None,
        )
        if match is not None:
            credited.add(match)
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _ndcg(flags: list[bool], num_gold: int, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, flag in enumerate(flags[:k], start=1)
        if flag
    )
    ideal_hits = min(num_gold, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _metrics_at_cutoff(flags: list[bool], num_gold: int, k: int) -> dict[str, float]:
    cutoff_flags = flags[:k]
    found = sum(cutoff_flags)
    first_rank = next(
        (rank for rank, flag in enumerate(cutoff_flags, start=1) if flag), None
    )
    return {
        "recall": found / num_gold if num_gold else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": _ndcg(flags, num_gold, k),
    }


def score_row(
    row: dict,
    conn,
    k: int,
    *,
    metric_cutoffs: list[int] | None = None,
    rewrite_mode: str = "off",
    retrieval_top_k: int | None = None,
    rerank_top_k: int | None = None,
    lanes: tuple[str, ...] | None = None,
) -> StructuredQueryScore:
    gold_groups = gold_signatures(conn, row)
    num_gold = len(gold_groups)

    kwargs: dict = {"rewrite_mode": rewrite_mode}
    if lanes:
        kwargs["lanes"] = tuple(lanes)
    if retrieval_top_k is not None:
        kwargs["retrieval_top_k"] = retrieval_top_k
    if rerank_top_k is not None:
        kwargs["rerank_top_k"] = rerank_top_k
    ranked, diagnostics = retrieve_with_diagnostics(
        row["question"], conn, **kwargs
    )

    cutoffs = sorted(set(metric_cutoffs or []) | {k})
    ranked = ranked[: max(cutoffs)]
    flags = _distinct_gold_hits(ranked, gold_groups)
    cutoff_metrics = {
        cutoff: _metrics_at_cutoff(flags, num_gold, cutoff) for cutoff in cutoffs
    }

    pooled_gold_rank = next(
        (
            rank
            for rank, candidate in enumerate(diagnostics.pooled_candidates, start=1)
            if any(_matches_group(candidate, group) for group in gold_groups)
        ),
        None,
    )
    first_rank = next(
        (rank for rank, flag in enumerate(flags[:k], start=1) if flag), None
    )

    return StructuredQueryScore(
        id=row["id"],
        difficulty=row.get("difficulty", "unknown"),
        question_type=row.get("question_type", "unknown"),
        num_gold=num_gold,
        hit_at_1=1.0 if flags and flags[0] else 0.0,
        first_gold_rank=first_rank,
        retrieved_keys=[str(result.query_object.chunk_id) for result in ranked[:k]],
        reranked_candidates=[
            {
                "position": position,
                "chunk_id": result.query_object.chunk_id,
                "content_type": result.query_object.content_type,
                "rerank_score": result.rerank_score,
                "is_gold": any(
                    _matches_group(result, group) for group in gold_groups
                ),
            }
            for position, result in enumerate(
                diagnostics.reranked_candidates, start=1
            )
        ],
        pooled_gold_rank=pooled_gold_rank,
        structured_in_top_k=sum(
            1
            for result in ranked[:k]
            if result.query_object.content_type in ("table", "figure")
        ),
        cutoff_metrics=cutoff_metrics,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[StructuredQueryScore], k: int, cutoffs: list[int]) -> dict:
    def block(subset: list[StructuredQueryScore]) -> dict:
        metrics = {
            "count": len(subset),
            "hit@1": round(_mean([s.hit_at_1 for s in subset]), 4),
        }
        for cutoff in cutoffs:
            for name in ("recall", "mrr", "ndcg"):
                metrics[f"{name}@{cutoff}"] = round(
                    _mean([s.cutoff_metrics[cutoff][name] for s in subset]), 4
                )
        return metrics

    by_type = defaultdict(list)
    by_difficulty = defaultdict(list)
    for score in scores:
        by_type[score.question_type].append(score)
        by_difficulty[score.difficulty].append(score)

    return {
        "k": k,
        "metric_cutoffs": cutoffs,
        "overall": block(scores),
        "by_question_type": {name: block(rows) for name, rows in sorted(by_type.items())},
        "by_difficulty": {
            name: block(rows) for name, rows in sorted(by_difficulty.items())
        },
    }


def print_report(
    summary: dict,
    scores: list[StructuredQueryScore],
    k: int,
    show_misses: bool,
    skipped: list[dict],
) -> None:
    def line(name: str, b: dict) -> str:
        return (
            f"  {name:<16} n={b['count']:<4} "
            f"hit@1={b['hit@1']:.3f}  recall@{k}={b[f'recall@{k}']:.3f}  "
            f"mrr@{k}={b[f'mrr@{k}']:.3f}  ndcg@{k}={b[f'ndcg@{k}']:.3f}"
        )

    print("\n" + "=" * 78)
    print(
        f"STRUCTURED RETRIEVAL EVAL  (k={k}, "
        f"questions scored={summary['overall']['count']}"
        f"{f', skipped={len(skipped)}' if skipped else ''})"
    )
    print("=" * 78)
    if skipped:
        print(f"\nSKIPPED (stale gold reference): {len(skipped)}")
        for item in skipped:
            print(f"  {item['id']}: {item['reason']}")
    print("OVERALL")
    print(line("all", summary["overall"]))
    print("\nBY QUESTION TYPE")
    for name, b in summary["by_question_type"].items():
        print(line(name, b))
    print("\nBY DIFFICULTY")
    for name, b in summary["by_difficulty"].items():
        print(line(name, b))

    pool_misses = [s for s in scores if s.pooled_gold_rank is None]
    rerank_misses = [
        s for s in scores if s.pooled_gold_rank is not None and s.first_gold_rank is None
    ]
    print(
        f"\nDIAGNOSTICS: candidate-generation misses={len(pool_misses)}, "
        f"reranker top-{k} misses={len(rerank_misses)}, "
        f"mean structured chunks in top-{k}="
        f"{_mean([s.structured_in_top_k for s in scores]):.2f}"
    )
    if show_misses:
        for label, subset in (("POOL MISS", pool_misses), ("RERANK MISS", rerank_misses)):
            for s in subset:
                print(
                    f"  [{label}] {s.id} [{s.difficulty}/{s.question_type}] "
                    f"pooled_gold_rank={s.pooled_gold_rank} retrieved={s.retrieved_keys}"
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score structured retrieval against an evaluation JSONL."
    )
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH, help="Path to evaluation JSONL.")
    parser.add_argument("--metric-k", dest="k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--misses", action="store_true", help="List gold misses.")
    parser.add_argument(
        "--rewrite-mode", choices=("auto", "off", "always"), default="off"
    )
    parser.add_argument(
        "--lanes", nargs="*", choices=list(ALL_LANES), default=None,
        help="Retrieve and rerank within these lanes before merging.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--metric-k must be positive")

    metric_cutoffs = sorted({DEFAULT_K, 10, args.k})
    rows = load_eval(args.eval)
    conn = connect_db()
    scores: list[StructuredQueryScore] = []
    skipped: list[dict] = []
    try:
        for row in rows:
            try:
                scores.append(
                    score_row(
                        row,
                        conn,
                        args.k,
                        metric_cutoffs=metric_cutoffs,
                        rewrite_mode=args.rewrite_mode,
                        retrieval_top_k=args.retrieval_top_k,
                        rerank_top_k=args.rerank_top_k
                        if args.rerank_top_k is not None
                        else max(metric_cutoffs),
                        lanes=tuple(args.lanes) if args.lanes else None,
                    )
                )
            except GoldNotFoundError as exc:
                skipped.append({"id": row["id"], "reason": str(exc)})
                print(f"  [SKIP] {row['id']}: {exc}")
    finally:
        conn.close()

    if not scores:
        print("No rows scored.")
        return

    summary = aggregate(scores, args.k, metric_cutoffs)
    print_report(summary, scores, args.k, args.misses, skipped)

    if args.out:
        payload = {
            "summary": summary,
            "skipped": skipped,
            "per_query": [
                {
                    "id": s.id,
                    "difficulty": s.difficulty,
                    "question_type": s.question_type,
                    "num_gold": s.num_gold,
                    "hit@1": s.hit_at_1,
                    **{
                        f"{name}@{cutoff}": s.cutoff_metrics[cutoff][name]
                        for cutoff in metric_cutoffs
                        for name in ("recall", "mrr", "ndcg")
                    },
                    "first_gold_rank": s.first_gold_rank,
                    "pooled_gold_rank": s.pooled_gold_rank,
                    "structured_in_top_k": s.structured_in_top_k,
                    "retrieved_keys": s.retrieved_keys,
                    "reranked_candidates": s.reranked_candidates,
                }
                for s in scores
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
