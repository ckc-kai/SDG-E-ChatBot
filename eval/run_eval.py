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
survives a re-ingest that renumbers ids. Rows may additionally define
`evidence_groups`: each outer item is required evidence, while the chunks inside
that item are interchangeable alternatives. Legacy `evidence` rows are treated
as one required group per evidence item.

Duplicate-aware matching ("credit identical twins"): the corpus contains
byte-identical duplicate documents (e.g. the two SDG&E WMP Decision PDFs), so a
retrieved chunk counts as gold if EITHER its chunk id is a resolved gold id OR its
stored content is identical to a gold chunk's content.

Usage:
    python -m eval.run_eval                         # scores eval/pdf/evaluation.jsonl
    python -m eval.run_eval --eval PATH --metric-k 5 --out results.json --misses
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.query import (
    DEFAULT_EMBEDDING_MODE,
    DEFAULT_HYBRID_POOL_MODE,
    EMBEDDING_MODES,
    HYBRID_POOL_MODES,
    QUERY_REWRITE_MODE,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    QueryObject,
    RankedResult,
    get_last_rewrite_trace,
    get_rewrite_call_count,
    reset_rewrite_call_count,
    retrieve_with_diagnostics,
)
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
    rewrite_source: str  # "simple" | "api" | "fallback" | "disabled"
    rewrite_reason: str
    search_queries: list[str]
    vector_candidate_ids: list[list[int]]
    channel_candidate_ids: list[dict[str, list[int]]]
    pooled_candidate_ids: list[int]
    gold_best_vector_rank: int | None
    gold_full_rerank_rank: int | None
    cutoff_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    retrieval_candidates: list[dict] = field(default_factory=list)
    reranked_candidates: list[dict] = field(default_factory=list)
    gold_positions: list[dict] = field(default_factory=list)


def load_eval(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
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


GoldGroup = tuple[set[int], set[str]]


def _row_evidence_groups(row: dict) -> list[list[dict]]:
    """Return required evidence groups, preserving legacy row semantics."""
    if "evidence_groups" not in row:
        return [[evidence] for evidence in row["evidence"]]

    groups = row["evidence_groups"]
    if not groups or any(not isinstance(group, list) or not group for group in groups):
        raise ValueError(
            f"{row.get('id', '<unknown>')}: evidence_groups must contain non-empty lists"
        )
    return groups


def gold_signatures(conn, row: dict) -> list[GoldGroup]:
    """Resolve required gold groups by stable chunk keys.

    Keyed on the UNIQUE (document, page_start, chunk_index) tuple rather than the
    serial ``gold_chunk_ids`` so the eval survives a re-ingest that renumbers ids.
    Each returned group contains the ids and normalized contents of interchangeable
    alternatives. Normalized contents also power matching against byte-identical
    duplicate documents.
    """
    resolved_groups: list[GoldGroup] = []
    with conn.cursor() as cur:
        for group in _row_evidence_groups(row):
            ids: set[int] = set()
            contents: set[str] = set()
            for evidence in group:
                cur.execute(
                    """
                    SELECT c.id, c.content
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE d.filename = %s AND c.page_start = %s AND c.chunk_index = %s
                    """,
                    (
                        evidence["source_pdf"],
                        evidence["page_start_db"],
                        evidence["chunk_index"],
                    ),
                )
                resolved = cur.fetchone()
                if resolved is None:
                    raise GoldNotFoundError(
                        "gold chunk not found by stable key "
                        f"(source_pdf={evidence['source_pdf']!r}, "
                        f"page_start={evidence['page_start_db']}, "
                        f"chunk_index={evidence['chunk_index']})"
                    )
                ids.add(resolved[0])
                contents.add(normalize(resolved[1]))
            resolved_groups.append((ids, contents))
    return resolved_groups


def _matches_gold_group(candidate: QueryObject | RankedResult, group: GoldGroup) -> bool:
    query_object = _candidate_query_object(candidate)
    ids, contents = group
    return query_object.chunk_id in ids or normalize(query_object.content) in contents


def _distinct_gold_hits(ranked, gold_groups: list[GoldGroup]) -> list[bool]:
    """Flag the first ranked result that satisfies each required evidence group.

    Alternative chunks and byte-identical twins satisfy the same group only once.
    """
    credited_groups: set[int] = set()
    flags: list[bool] = []
    for result in ranked:
        matching_group = next(
            (
                index
                for index, group in enumerate(gold_groups)
                if index not in credited_groups and _matches_gold_group(result, group)
            ),
            None,
        )
        if matching_group is not None:
            credited_groups.add(matching_group)
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _candidate_query_object(candidate: QueryObject | RankedResult) -> QueryObject:
    return candidate.query_object if isinstance(candidate, RankedResult) else candidate


def _first_gold_rank_in_candidates(
    candidates: list[QueryObject] | list[RankedResult],
    gold_ids: set[int],
    gold_contents: set[str],
) -> int | None:
    """Return the first gold rank, including duplicate-content twin credit."""
    for rank, candidate in enumerate(candidates, start=1):
        query_object = _candidate_query_object(candidate)
        if query_object.chunk_id in gold_ids or normalize(query_object.content) in gold_contents:
            return rank
    return None


def _ndcg(flags: list[bool], num_gold: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(flags[:k], start=1) if flag)
    ideal_hits = min(num_gold, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _metrics_at_cutoff(flags: list[bool], num_gold: int, k: int) -> dict[str, float]:
    cutoff_flags = flags[:k]
    found = sum(cutoff_flags)
    first_rank = next((rank for rank, flag in enumerate(cutoff_flags, start=1) if flag), None)
    return {
        "recall": found / num_gold if num_gold else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": _ndcg(flags, num_gold, k),
    }


def _is_gold_candidate(
    candidate: QueryObject | RankedResult,
    gold_ids: set[int],
    gold_contents: set[str],
) -> bool:
    query_object = _candidate_query_object(candidate)
    return query_object.chunk_id in gold_ids or normalize(query_object.content) in gold_contents


def _position_for_gold_group(
    candidates: list[QueryObject] | list[RankedResult], gold_group: GoldGroup
) -> int | str:
    for position, candidate in enumerate(candidates, start=1):
        if _matches_gold_group(candidate, gold_group):
            return position
    return "not_found"


def _candidate_position_details(
    diagnostics,
    gold_groups: list[GoldGroup],
    gold_ids: set[int],
    gold_contents: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    retrieval_candidates = [
        {
            "search_query_index": query_index,
            "search_query": diagnostics.search_queries[query_index - 1],
            "candidates": [
                {
                    "position": position,
                    "chunk_id": candidate.chunk_id,
                    "source_pdf": candidate.source_pdf,
                    "breadcrumb": candidate.breadcrumb,
                    "section_number": candidate.section_number,
                    "chunk_index": candidate.chunk_index,
                    "distance": float(candidate.distance),
                    "retrieval_score": candidate.retrieval_score,
                    "is_gold": _is_gold_candidate(candidate, gold_ids, gold_contents),
                }
                for position, candidate in enumerate(candidates, start=1)
            ],
        }
        for query_index, candidates in enumerate(diagnostics.candidate_sets, start=1)
    ]
    reranked_candidates = [
        {
            "position": position,
            "chunk_id": result.query_object.chunk_id,
            "source_pdf": result.query_object.source_pdf,
            "breadcrumb": result.query_object.breadcrumb,
            "section_number": result.query_object.section_number,
            "chunk_index": result.query_object.chunk_index,
            "rerank_score": float(result.rerank_score),
            "retrieval_score": result.query_object.retrieval_score,
            "is_gold": _is_gold_candidate(result, gold_ids, gold_contents),
        }
        for position, result in enumerate(diagnostics.reranked_candidates, start=1)
    ]

    gold_positions: list[dict] = []
    for gold_index, gold_group in enumerate(gold_groups, start=1):
        retrieval_positions = [
            {
                "search_query_index": query_index,
                "position": _position_for_gold_group(candidates, gold_group),
            }
            for query_index, candidates in enumerate(diagnostics.candidate_sets, start=1)
        ]
        found_positions = [
            item["position"]
            for item in retrieval_positions
            if isinstance(item["position"], int)
        ]
        gold_positions.append(
            {
                "gold_item": gold_index,
                "alternative_count": len(gold_group[0]),
                "retrieval_positions": retrieval_positions,
                "best_retrieval_position": min(found_positions) if found_positions else "not_found",
                "rerank_position": _position_for_gold_group(
                    diagnostics.reranked_candidates, gold_group
                ),
            }
        )
    return retrieval_candidates, reranked_candidates, gold_positions


def score_row(
    row: dict,
    conn,
    k: int,
    *,
    metric_cutoffs: list[int] | None = None,
    rewrite_mode: str = QUERY_REWRITE_MODE,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    rrf_k: int = RRF_K,
    hybrid_pool_mode: str = DEFAULT_HYBRID_POOL_MODE,
) -> QueryScore:
    gold_groups = gold_signatures(conn, row)
    gold_ids = set().union(*(ids for ids, _ in gold_groups))
    gold_contents = set().union(*(contents for _, contents in gold_groups))
    num_gold = len(gold_groups)

    calls_before = get_rewrite_call_count()
    ranked, diagnostics = retrieve_with_diagnostics(
        row["question"],
        conn,
        rewrite_mode=rewrite_mode,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        embedding_mode=embedding_mode,
        rrf_k=rrf_k,
        hybrid_pool_mode=hybrid_pool_mode,
    )
    cutoffs = sorted(set(metric_cutoffs or []) | {k})
    ranked = ranked[:max(cutoffs)]
    api_rewrite_calls = get_rewrite_call_count() - calls_before
    trace = get_last_rewrite_trace() or {
        "source": "unknown",
        "reason": "unknown",
        "sub_questions": [row["question"]],
    }
    if trace["source"] == "api":
        print(f"  [rewrite] {row['id']}: -> {trace['sub_questions']}")
    elif trace["source"] == "fallback":
        print(f"  [rewrite] {row['id']}: API call FAILED ({trace.get('error')}), used original question")

    flags = _distinct_gold_hits(ranked, gold_groups)
    cutoff_metrics = {
        cutoff: _metrics_at_cutoff(flags, num_gold, cutoff)
        for cutoff in cutoffs
    }

    vector_gold_ranks = [
        rank
        for candidates in diagnostics.candidate_sets
        if (rank := _first_gold_rank_in_candidates(candidates, gold_ids, gold_contents))
        is not None
    ]
    gold_best_vector_rank = min(vector_gold_ranks) if vector_gold_ranks else None
    gold_full_rerank_rank = _first_gold_rank_in_candidates(
        diagnostics.reranked_candidates, gold_ids, gold_contents
    )

    primary_flags = flags[:k]
    first_rank = next((i for i, flag in enumerate(primary_flags, start=1) if flag), None)
    primary_metrics = cutoff_metrics[k]
    retrieval_candidates, reranked_candidates, gold_positions = _candidate_position_details(
        diagnostics, gold_groups, gold_ids, gold_contents
    )

    return QueryScore(
        id=row["id"],
        difficulty=row.get("difficulty", "unknown"),
        question_type=row.get("question_type", "unknown"),
        num_gold=num_gold,
        hit_at_1=1.0 if flags and flags[0] else 0.0,
        recall_at_k=primary_metrics["recall"],
        mrr=primary_metrics["mrr"],
        ndcg_at_k=primary_metrics["ndcg"],
        first_gold_rank=first_rank,
        retrieved_ids=[r.query_object.chunk_id for r in ranked[:k]],
        api_rewrite_calls=api_rewrite_calls,
        rewrite_source=trace["source"],
        rewrite_reason=trace.get("reason", "unknown"),
        search_queries=diagnostics.search_queries,
        vector_candidate_ids=[
            [candidate.chunk_id for candidate in candidates]
            for candidates in diagnostics.candidate_sets
        ],
        channel_candidate_ids=[
            {
                channel: [candidate.chunk_id for candidate in candidates]
                for channel, candidates in channels.items()
            }
            for channels in diagnostics.channel_candidate_sets
        ],
        pooled_candidate_ids=[candidate.chunk_id for candidate in diagnostics.pooled_candidates],
        gold_best_vector_rank=gold_best_vector_rank,
        gold_full_rerank_rank=gold_full_rerank_rank,
        cutoff_metrics=cutoff_metrics,
        retrieval_candidates=retrieval_candidates,
        reranked_candidates=reranked_candidates,
        gold_positions=gold_positions,
    )


def _mean(scores: list[QueryScore], attr: str) -> float:
    return sum(getattr(s, attr) for s in scores) / len(scores) if scores else 0.0


def aggregate(
    scores: list[QueryScore], k: int, metric_cutoffs: list[int] | None = None
) -> dict:
    cutoffs = sorted(set(metric_cutoffs or []) | {k})

    def cutoff_mean(subset: list[QueryScore], cutoff: int, metric: str) -> float:
        values = []
        for score in subset:
            if cutoff in score.cutoff_metrics:
                values.append(score.cutoff_metrics[cutoff][metric])
            elif cutoff == k:
                fallback = {
                    "recall": score.recall_at_k,
                    "mrr": score.mrr,
                    "ndcg": score.ndcg_at_k,
                }
                values.append(fallback[metric])
        return sum(values) / len(values) if values else 0.0

    def block(subset: list[QueryScore]) -> dict:
        metrics = {
            "count": len(subset),
            "hit@1": round(_mean(subset, "hit_at_1"), 4),
        }
        for cutoff in cutoffs:
            metrics[f"recall@{cutoff}"] = round(cutoff_mean(subset, cutoff, "recall"), 4)
            metrics[f"mrr@{cutoff}"] = round(cutoff_mean(subset, cutoff, "mrr"), 4)
            metrics[f"ndcg@{cutoff}"] = round(cutoff_mean(subset, cutoff, "ndcg"), 4)
        metrics["mrr"] = metrics[f"mrr@{k}"]
        return metrics

    by_difficulty = defaultdict(list)
    by_type = defaultdict(list)
    for score in scores:
        by_difficulty[score.difficulty].append(score)
        by_type[score.question_type].append(score)

    return {
        "k": k,
        "metric_cutoffs": cutoffs,
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
    for source in ("api", "fallback", "simple", "disabled"):
        subset = by_source.get(source, [])
        if subset:
            print(f"  {source:<9} n={len(subset)}: " + ", ".join(s.id for s in subset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval against an evaluation JSONL.")
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH, help="Path to evaluation JSONL.")
    parser.add_argument(
        "--metric-k",
        "--k",
        dest="k",
        type=int,
        default=DEFAULT_K,
        help=(
            "Primary final-ranking cutoff. Reports metrics at both the default cutoff 5 "
            "and this requested cutoff (--k remains an alias)."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON results output path.")
    parser.add_argument("--misses", action="store_true", help="List questions with no gold in top-k.")
    parser.add_argument(
        "--rewrite-mode",
        choices=("auto", "off", "always"),
        default=QUERY_REWRITE_MODE,
        help="Internal experiment control; production defaults to automatic heuristic routing.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=(
            "Vector candidates retrieved per search query before pooling; "
            "this does not change the metric cutoff."
        ),
    )
    parser.add_argument(
        "--embedding-mode",
        choices=EMBEDDING_MODES,
        default=DEFAULT_EMBEDDING_MODE,
        help="Dense retrieval strategy: raw, contextual, or hybrid.",
    )
    parser.add_argument(
        "--hybrid-pool-mode",
        choices=HYBRID_POOL_MODES,
        default=DEFAULT_HYBRID_POOL_MODE,
        help=(
            "How hybrid mode builds the reranker pool: fixed-size RRF or a "
            "raw-preserving deduplicated union (default: rrf)."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help="Reciprocal-rank-fusion constant used by hybrid mode (default: 60).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=(
            "Candidates retained after reranking. Defaults to the larger of the configured "
            "value and --metric-k."
        ),
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--metric-k must be positive")
    if args.retrieval_top_k < 1:
        parser.error("--retrieval-top-k must be positive")
    if args.rrf_k < 1:
        parser.error("--rrf-k must be positive")
    metric_cutoffs = sorted({DEFAULT_K, args.k})
    rerank_top_k = (
        max(RERANK_TOP_K, max(metric_cutoffs))
        if args.rerank_top_k is None
        else args.rerank_top_k
    )
    if rerank_top_k < max(metric_cutoffs):
        parser.error(
            "--rerank-top-k must be greater than or equal to every reported metric cutoff"
        )

    print(
        "EVAL CUTOFFS: "
        f"embedding_mode={args.embedding_mode}, "
        f"hybrid_pool_mode={args.hybrid_pool_mode}, "
        f"metric_cutoffs={metric_cutoffs}, "
        f"retrieval_top_k={args.retrieval_top_k}, "
        f"rerank_top_k={rerank_top_k}, rrf_k={args.rrf_k}"
    )

    rows = load_eval(args.eval)
    reset_rewrite_call_count()  # so the totals reflect only this run
    conn = connect_db()
    scores: list[QueryScore] = []
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
                        rerank_top_k=rerank_top_k,
                        embedding_mode=args.embedding_mode,
                        rrf_k=args.rrf_k,
                        hybrid_pool_mode=args.hybrid_pool_mode,
                    )
                )
            except GoldNotFoundError as exc:
                skipped.append({"id": row["id"], "reason": str(exc)})
                print(f"  [SKIP] {row['id']}: {exc}")
    finally:
        conn.close()

    summary = aggregate(scores, args.k, metric_cutoffs)
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
                "questions_disabled": sum(1 for s in scores if s.rewrite_source == "disabled"),
                "questions_total": len(scores),
            },
            "retrieval_diagnostics": {
                "metric_k": args.k,
                "metric_cutoffs": metric_cutoffs,
                "embedding_mode": args.embedding_mode,
                "hybrid_pool_mode": args.hybrid_pool_mode,
                "rrf_k": args.rrf_k,
                "rewrite_mode": args.rewrite_mode,
                "retrieval_top_k": args.retrieval_top_k,
                "rerank_top_k": rerank_top_k,
                "candidate_pool_size_min": min(
                    (len(s.pooled_candidate_ids) for s in scores),
                    default=0,
                ),
                "candidate_pool_size_max": max(
                    (len(s.pooled_candidate_ids) for s in scores),
                    default=0,
                ),
                "candidate_pool_size_mean": round(
                    (
                        sum(len(s.pooled_candidate_ids) for s in scores)
                        / len(scores)
                        if scores
                        else 0.0
                    ),
                    2,
                ),
                "candidate_generation_misses": sum(
                    1 for s in scores if s.gold_best_vector_rank is None
                ),
                "reranker_top_k_misses": sum(
                    1
                    for s in scores
                    if s.gold_best_vector_rank is not None and s.first_gold_rank is None
                ),
            },
            "per_query": [
                {
                    "id": s.id,
                    "difficulty": s.difficulty,
                    "question_type": s.question_type,
                    "num_gold": s.num_gold,
                    "hit@1": s.hit_at_1,
                    "mrr": s.mrr,
                    **{
                        metric_name: value
                        for cutoff in metric_cutoffs
                        for metric_name, value in (
                            (
                                f"recall@{cutoff}",
                                s.cutoff_metrics.get(cutoff, {}).get(
                                    "recall", s.recall_at_k if cutoff == args.k else 0.0
                                ),
                            ),
                            (
                                f"mrr@{cutoff}",
                                s.cutoff_metrics.get(cutoff, {}).get(
                                    "mrr", s.mrr if cutoff == args.k else 0.0
                                ),
                            ),
                            (
                                f"ndcg@{cutoff}",
                                s.cutoff_metrics.get(cutoff, {}).get(
                                    "ndcg", s.ndcg_at_k if cutoff == args.k else 0.0
                                ),
                            ),
                        )
                    },
                    "first_gold_rank": s.first_gold_rank,
                    "retrieved_ids": s.retrieved_ids,
                    "api_rewrite_calls": s.api_rewrite_calls,
                    "rewrite_source": s.rewrite_source,
                    "rewrite_reason": s.rewrite_reason,
                    "sub_questions": s.search_queries,
                    "search_queries": s.search_queries,
                    "vector_candidate_ids": s.vector_candidate_ids,
                    "channel_candidate_ids": s.channel_candidate_ids,
                    "pooled_candidate_ids": s.pooled_candidate_ids,
                    "gold_best_vector_rank": (
                        s.gold_best_vector_rank
                        if s.gold_best_vector_rank is not None
                        else "not_found"
                    ),
                    "gold_full_rerank_rank": (
                        s.gold_full_rerank_rank
                        if s.gold_full_rerank_rank is not None
                        else "not_found"
                    ),
                    "retrieval_candidates": s.retrieval_candidates,
                    "reranked_candidates": s.reranked_candidates,
                    "gold_positions": s.gold_positions,
                }
                for s in scores
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
