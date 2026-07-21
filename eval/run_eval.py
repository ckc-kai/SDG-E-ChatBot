"""Retrieval evaluation harness for the WMP knowledge base.

Runs every question in an evaluation JSONL through the real retrieval pipeline
(`retrieval.query.retrieve`: rewrite -> embed -> pgvector search -> rerank) and
scores the reranked top-k against each question's gold chunks.

Metrics (averaged over all questions, and broken down by difficulty / type):
  Hit@1     1 if the top reranked chunk is gold, else 0.
  Recall@k  (distinct gold chunks found in top-k) / (distinct gold chunks).
  MRR       1 / rank of the first gold hit (0 if none in top-k).
  nDCG@k    position-discounted gold gain, normalized to [0, 1].

Gold chunks are resolved by the stable key (source_pdf, page_start, chunk_index)
carried in each row's `evidence`, NOT by the serial `gold_chunk_ids`, so the eval
survives a re-ingest that renumbers ids.

Duplicate-aware matching ("credit identical twins"): the corpus contains
byte-identical duplicate documents (e.g. the two SDG&E WMP Decision PDFs), so a
retrieved chunk counts as gold if EITHER its chunk id is a resolved gold id OR its
stored content is identical to a gold chunk's content.

Usage:
    python -m eval.run_eval                         # scores eval/pdf/evaluation.jsonl
    python -m eval.run_eval --eval PATH --k 5 --out results.json --misses
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from retrieval.query import get_last_rewrite_trace, get_rewrite_call_count, reset_rewrite_call_count, retrieve
from retrieval.utils import connect_db

DEFAULT_EVAL_PATH = "eval/pdf/evaluation.jsonl"
DEFAULT_K = 5


def normalize(text: str) -> str:
    """Whitespace-collapsed, lowercased signature for content-identity matching."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


@dataclass(frozen=True)
class QueryScore:
    id: str
    difficulty: str
    question_type: str
    num_gold: int
    hit_at_1: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    first_gold_rank: int | None  # 1-indexed rank of first gold hit, None if missed
    retrieved_ids: list[int]
    api_rewrite_calls: int  # Claude API calls query_rewrite() made for this question
    rewrite_source: str  # "simple" | "api" | "fallback" -- see query.get_last_rewrite_trace
    sub_questions: list[str]  # the actual sub-question(s) retrieval was run against


def load_eval(path: str) -> list[dict]:
    with open(path, "r") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    return rows


class GoldNotFoundError(Exception):
    """A row's stable-key gold reference no longer resolves against the current DB.

    The (source_pdf, page_start, chunk_index) key survives re-ingests that only
    change chunk CONTENT, but not ones that change chunk BOUNDARIES -- e.g. a
    leaf-section extraction fix that trims leading text can shrink a section below
    the token threshold that used to require 2 chunks, collapsing chunk_index=1.
    Raised so the caller can skip and report the row instead of losing the whole run.
    """


def gold_signatures(conn, evidence: list[dict]) -> tuple[set[int], set[str]]:
    """Resolve gold chunks by the stable key (source_pdf, page_start, chunk_index).

    Keyed on the UNIQUE (document, page_start, chunk_index) tuple rather than the
    serial ``gold_chunk_ids`` so the eval survives a re-ingest that renumbers ids.
    Returns the resolved gold chunk id set plus the set of normalized gold contents
    (the latter powers credit-twins matching against byte-identical duplicate docs).
    """
    ids: set[int] = set()
    contents: set[str] = set()
    with conn.cursor() as cur:
        for e in evidence:
            cur.execute(
                """
                SELECT c.id, c.content
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.filename = %s AND c.page_start = %s AND c.chunk_index = %s
                """,
                (e["source_pdf"], e["page_start_db"], e["chunk_index"]),
            )
            row = cur.fetchone()
            if row is None:
                raise GoldNotFoundError(
                    "gold chunk not found by stable key "
                    f"(source_pdf={e['source_pdf']!r}, page_start={e['page_start_db']}, "
                    f"chunk_index={e['chunk_index']})"
                )
            ids.add(row[0])
            contents.add(normalize(row[1]))
    return ids, contents


def _distinct_gold_hits(ranked, gold_ids: set[int], gold_contents: set[str]) -> list[bool]:
    """Per-rank flag that is True only the FIRST time each distinct gold item is seen.

    Prevents a duplicate twin (identical content, different id) from being counted
    as a second, separate gold hit.
    """
    credited: set[str] = set()
    flags: list[bool] = []
    for result in ranked:
        qo = result.query_object
        signature = normalize(qo.content)
        is_gold = qo.chunk_id in gold_ids or signature in gold_contents
        if is_gold and signature not in credited:
            credited.add(signature)
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _ndcg(flags: list[bool], num_gold: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(flags[:k], start=1) if flag)
    ideal_hits = min(num_gold, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_row(row: dict, conn, k: int) -> QueryScore:
    gold_ids, gold_contents = gold_signatures(conn, row["evidence"])
    num_gold = len(gold_contents)  # distinct gold items (twins collapse to one)

    calls_before = get_rewrite_call_count()
    ranked = retrieve(row["question"], conn)[:k]
    api_rewrite_calls = get_rewrite_call_count() - calls_before
    trace = get_last_rewrite_trace() or {"source": "unknown", "sub_questions": [row["question"]]}
    if trace["source"] == "api":
        print(f"  [rewrite] {row['id']}: -> {trace['sub_questions']}")
    elif trace["source"] == "fallback":
        print(f"  [rewrite] {row['id']}: API call FAILED ({trace.get('error')}), used original question")

    flags = _distinct_gold_hits(ranked, gold_ids, gold_contents)

    found = sum(flags)
    first_rank = next((i for i, flag in enumerate(flags, start=1) if flag), None)

    return QueryScore(
        id=row["id"],
        difficulty=row.get("difficulty", "unknown"),
        question_type=row.get("question_type", "unknown"),
        num_gold=num_gold,
        hit_at_1=1.0 if flags and flags[0] else 0.0,
        recall_at_k=found / num_gold if num_gold else 0.0,
        mrr=1.0 / first_rank if first_rank else 0.0,
        ndcg_at_k=_ndcg(flags, num_gold, k),
        first_gold_rank=first_rank,
        retrieved_ids=[r.query_object.chunk_id for r in ranked],
        api_rewrite_calls=api_rewrite_calls,
        rewrite_source=trace["source"],
        sub_questions=trace["sub_questions"],
    )


def _mean(scores: list[QueryScore], attr: str) -> float:
    return sum(getattr(s, attr) for s in scores) / len(scores) if scores else 0.0


def aggregate(scores: list[QueryScore], k: int) -> dict:
    def block(subset: list[QueryScore]) -> dict:
        return {
            "count": len(subset),
            "hit@1": round(_mean(subset, "hit_at_1"), 4),
            f"recall@{k}": round(_mean(subset, "recall_at_k"), 4),
            "mrr": round(_mean(subset, "mrr"), 4),
            f"ndcg@{k}": round(_mean(subset, "ndcg_at_k"), 4),
        }

    by_difficulty = defaultdict(list)
    by_type = defaultdict(list)
    for score in scores:
        by_difficulty[score.difficulty].append(score)
        by_type[score.question_type].append(score)

    return {
        "k": k,
        "overall": block(scores),
        "by_difficulty": {name: block(rows) for name, rows in sorted(by_difficulty.items())},
        "by_question_type": {name: block(rows) for name, rows in sorted(by_type.items())},
    }


def print_report(
    summary: dict, scores: list[QueryScore], k: int, show_misses: bool, skipped: list[dict]
) -> None:
    def line(name: str, b: dict) -> str:
        return (f"  {name:<24} n={b['count']:<4} "
                f"hit@1={b['hit@1']:.3f}  recall@{k}={b[f'recall@{k}']:.3f}  "
                f"mrr={b['mrr']:.3f}  ndcg@{k}={b[f'ndcg@{k}']:.3f}")

    print("\n" + "=" * 78)
    print(f"RETRIEVAL EVAL  (k={k}, questions scored={summary['overall']['count']}"
          f"{f', skipped={len(skipped)}' if skipped else ''})")
    print("=" * 78)

    if skipped:
        print(f"\nSKIPPED (stale gold reference -- likely chunk boundaries shifted on "
              f"re-ingest): {len(skipped)}")
        for s in skipped:
            print(f"  {s['id']}: {s['reason']}")
    print("OVERALL")
    print(line("all", summary["overall"]))
    print("\nBY DIFFICULTY")
    for name, b in summary["by_difficulty"].items():
        print(line(name, b))
    print("\nBY QUESTION TYPE")
    for name, b in summary["by_question_type"].items():
        print(line(name, b))

    if show_misses:
        misses = [s for s in scores if s.first_gold_rank is None]
        print(f"\nMISSES (no gold chunk in top-{k}): {len(misses)}")
        for s in misses:
            print(f"  {s.id} [{s.difficulty}/{s.question_type}] retrieved={s.retrieved_ids}")

    by_source = defaultdict(list)
    for s in scores:
        by_source[s.rewrite_source].append(s)
    total_calls = sum(s.api_rewrite_calls for s in scores)
    print(f"\nCLAUDE API REWRITE CALLS: {total_calls} call(s) across {len(scores)} question(s)")
    for source in ("api", "fallback", "simple"):
        subset = by_source.get(source, [])
        if subset:
            print(f"  {source:<9} n={len(subset)}: " + ", ".join(s.id for s in subset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval against an evaluation JSONL.")
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH, help="Path to evaluation JSONL.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Cutoff for @k metrics.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON results output path.")
    parser.add_argument("--misses", action="store_true", help="List questions with no gold in top-k.")
    args = parser.parse_args()

    rows = load_eval(args.eval)
    reset_rewrite_call_count()  # so the totals reflect only this run
    conn = connect_db()
    scores: list[QueryScore] = []
    skipped: list[dict] = []
    try:
        for row in rows:
            try:
                scores.append(score_row(row, conn, args.k))
            except GoldNotFoundError as exc:
                skipped.append({"id": row["id"], "reason": str(exc)})
                print(f"  [SKIP] {row['id']}: {exc}")
    finally:
        conn.close()

    summary = aggregate(scores, args.k)
    print_report(summary, scores, args.k, args.misses, skipped)

    if args.out:
        payload = {
            "summary": summary,
            "skipped": skipped,
            "api_rewrite_usage": {
                "total_calls": sum(s.api_rewrite_calls for s in scores),
                "questions_rewritten": sum(1 for s in scores if s.rewrite_source == "api"),
                "questions_fallback": sum(1 for s in scores if s.rewrite_source == "fallback"),
                "questions_simple": sum(1 for s in scores if s.rewrite_source == "simple"),
                "questions_total": len(scores),
            },
            "per_query": [
                {
                    "id": s.id,
                    "difficulty": s.difficulty,
                    "question_type": s.question_type,
                    "num_gold": s.num_gold,
                    "hit@1": s.hit_at_1,
                    f"recall@{args.k}": s.recall_at_k,
                    "mrr": s.mrr,
                    f"ndcg@{args.k}": s.ndcg_at_k,
                    "first_gold_rank": s.first_gold_rank,
                    "retrieved_ids": s.retrieved_ids,
                    "api_rewrite_calls": s.api_rewrite_calls,
                    "rewrite_source": s.rewrite_source,
                    "sub_questions": s.sub_questions,
                }
                for s in scores
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
