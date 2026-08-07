"""Fit per-content-type score calibration from existing eval runs.

Training pairs are harvested from eval result JSON, which already records each
candidate's ``rerank_score`` and an ``is_gold`` flag — so no new retrieval runs
are needed. Feed it runs produced with ``caption_rerank_weight: 0.0`` so the
curves are fitted on raw cross-encoder scores rather than on scores that already
carry the additive prior being retired.

Reported AUC is always out-of-fold. Folds are grouped by question, so a
question's candidates never appear in both the fit and the held-out split.

    uv run python -m eval.fit_calibration \\
        --runs run_a.json run_b.json --out config/score_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from retrieval.query.calibration import Calibrator, PlattCurve, fit_platt, roc_auc
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)

N_FOLDS = 5


def content_type_map(conn) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, content_type FROM chunks")
        return {row[0]: row[1] for row in cur.fetchall()}


def harvest(
    runs: list[Path], types: dict[int, str]
) -> dict[str, list[tuple[str, float, int]]]:
    """Return {content_type: [(question_id, score, label), ...]}."""
    pairs: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for path in runs:
        payload = json.loads(path.read_text())
        for row in payload.get("per_query", []):
            question_id = row.get("id", "?")
            for candidate in row.get("reranked_candidates") or []:
                chunk_id = candidate.get("chunk_id")
                score = candidate.get("rerank_score")
                if chunk_id is None or score is None:
                    continue
                content_type = types.get(chunk_id)
                if content_type is None:
                    continue
                pairs[content_type].append(
                    (question_id, float(score), int(bool(candidate.get("is_gold"))))
                )
    return pairs


def cross_validated_auc(rows: list[tuple[str, float, int]]) -> float | None:
    """Mean out-of-fold AUC, with folds grouped by question id."""
    questions = sorted({question for question, _, _ in rows})
    if len(questions) < N_FOLDS:
        return None
    fold_of = {q: i % N_FOLDS for i, q in enumerate(questions)}
    aucs: list[float] = []
    for fold in range(N_FOLDS):
        train = [(s, y) for q, s, y in rows if fold_of[q] != fold]
        test = [(s, y) for q, s, y in rows if fold_of[q] == fold]
        if not train or not test:
            continue
        if len({y for _, y in train}) < 2 or len({y for _, y in test}) < 2:
            continue
        curve = fit_platt([s for s, _ in train], [y for _, y in train])
        probabilities = [curve.probability(s) for s, _ in test]
        auc = roc_auc(probabilities, [y for _, y in test])
        if auc is not None:
            aucs.append(auc)
    return sum(aucs) / len(aucs) if aucs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("config/score_calibration.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = connect_db()
    try:
        types = content_type_map(conn)
        pairs = harvest(args.runs, types)
    finally:
        conn.close()

    if not pairs:
        raise SystemExit("No (score, label) pairs harvested from the supplied runs")

    curves: dict[str, PlattCurve] = {}
    print(f"{'content_type':14s} {'pairs':>7s} {'pos':>5s} {'holdout AUC':>12s}")
    print("-" * 44)
    for content_type, rows in sorted(pairs.items()):
        positives = sum(y for _, _, y in rows)
        if positives == 0:
            print(f"{content_type:14s} {len(rows):7d} {0:5d} {'skipped':>12s}")
            logger.warning(
                "%s has no positive examples; leaving it uncalibrated", content_type
            )
            continue
        auc = cross_validated_auc(rows)
        curve = fit_platt([s for _, s, _ in rows], [y for _, _, y in rows])
        curves[content_type] = PlattCurve(
            a=curve.a,
            b=curve.b,
            n_positive=curve.n_positive,
            n_negative=curve.n_negative,
            holdout_auc=auc,
        )
        auc_text = f"{auc:.3f}" if auc is not None else "n/a"
        print(f"{content_type:14s} {len(rows):7d} {positives:5d} {auc_text:>12s}")

    calibrator = Calibrator(curves)
    calibrator.save(
        args.out,
        metadata={
            "runs": [str(path) for path in args.runs],
            "folds": N_FOLDS,
            "note": (
                "AUC is out-of-fold, grouped by question. Fitted on raw "
                "cross-encoder scores (caption_rerank_weight 0.0)."
            ),
        },
    )
    print(f"\nWrote {len(curves)} calibration curve(s) to {args.out}")
    uncalibrated = sorted(set(pairs) - set(curves))
    if uncalibrated:
        print(
            "Uncalibrated content types (no positives in the label set): "
            + ", ".join(uncalibrated)
        )


if __name__ == "__main__":
    main()
