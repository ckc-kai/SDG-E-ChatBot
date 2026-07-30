"""Diagnostic per-content-type score calibration.

A cross-encoder returns an uncalibrated relevance logit. Its scale depends on how
much text the candidate carries, so comparing a 87-token figure against a
298-token narrative passage is invalid — which is exactly what the additive
caption prior was compensating for.

Calibration fits one Platt curve per content type,

    P(relevant | score, content_type) = sigmoid(a * score + b)

turning each lane's raw logits into probabilities that *are* comparable. Two
parameters per type keeps variance low on a 150-question training set; isotonic
regression would overfit here.

This module does not classify questions and does not route. It only makes
already-retrieved candidates mutually rankable.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PATH = "config/score_calibration.json"
CALIBRATION_PATH_ENV = "SDGE_CALIBRATION_PATH"

# Below this many candidates in a lane, calibration is unreliable; fall back to
# the raw score so a thin lane is never silently distorted.
MIN_LANE_CANDIDATES = 3


@dataclass(frozen=True)
class PlattCurve:
    a: float
    b: float
    n_positive: int = 0
    n_negative: int = 0
    holdout_auc: float | None = None

    @property
    def base_rate(self) -> float:
        total = self.n_positive + self.n_negative
        return self.n_positive / total if total else 0.5

    def _balanced_probability(self, score: float) -> float:
        z = self.a * score + self.b
        # Guard against overflow on extreme logits.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
        exp_z = math.exp(max(z, -60.0))
        return exp_z / (1.0 + exp_z)

    def probability(self, score: float) -> float:
        """Posterior relevance probability, corrected back to the true prior.

        The curve is fitted with balanced class weights so that a rare positive
        class does not collapse the fit — but balancing deliberately discards the
        base rate, and cross-lane merging *needs* it. Without this correction a
        table and a narrative chunk with identical cross-encoder scores receive
        0.823 vs 0.709, so the rarer type wins purely by being rare, which
        measured out at narrative hit@1 0.167.

        This maps the balanced posterior back onto the observed prior, which
        makes the values comparable across content types.
        """
        balanced = self._balanced_probability(score)
        prior = self.base_rate
        if not 0.0 < prior < 1.0:
            return balanced
        positive = balanced * prior
        negative = (1.0 - balanced) * (1.0 - prior)
        total = positive + negative
        return positive / total if total else balanced


IDENTITY = PlattCurve(a=1.0, b=0.0)


def fit_platt(
    scores: list[float],
    labels: list[int],
    *,
    iterations: int = 400,
    learning_rate: float = 0.05,
) -> PlattCurve:
    """Fit sigmoid(a*s + b) by class-weighted gradient descent.

    Positives are far rarer than negatives (roughly 1:40), so each class is
    weighted to its inverse frequency; otherwise the fit collapses toward
    predicting "irrelevant" for everything.
    """
    if not scores or len(scores) != len(labels):
        raise ValueError("scores and labels must be non-empty and equal length")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        logger.warning("Cannot fit calibration with a single class; using identity")
        return IDENTITY

    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    a, b = 1.0, 0.0
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for score, label in zip(scores, labels):
            z = a * score + b
            p = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
            weight = w_pos if label else w_neg
            error = (p - label) * weight
            grad_a += error * score
            grad_b += error
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return PlattCurve(a=a, b=b, n_positive=n_pos, n_negative=n_neg)


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUC; None when only one class is present."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Sum of ranks of positives, averaging ties.
    rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + (rank + (end - index))) / 2.0
        for position in range(index, end + 1):
            if pairs[position][1]:
                rank_sum += average_rank
        count = end - index + 1
        rank += count
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class Calibrator:
    """Loaded per-type curves, with graceful fallback to raw scores."""

    def __init__(self, curves: dict[str, PlattCurve] | None = None) -> None:
        self.curves = curves or {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Calibrator":
        resolved = Path(
            path or os.environ.get(CALIBRATION_PATH_ENV, DEFAULT_CALIBRATION_PATH)
        )
        if not resolved.exists():
            logger.info(
                "No score calibration at %s; using raw cross-encoder scores",
                resolved,
            )
            return cls({})
        with resolved.open() as fh:
            payload = json.load(fh)
        curves = {
            content_type: PlattCurve(**values)
            for content_type, values in (payload.get("curves") or {}).items()
        }
        return cls(curves)

    @property
    def is_fitted(self) -> bool:
        return bool(self.curves)

    def covers(self, content_types: set[str]) -> bool:
        """True only if every participating content type has a curve.

        Mixing calibrated probabilities (bounded by 1.0) with raw cross-encoder
        logits (routinely above 1.0) hands victory to whichever type lacks a
        curve, regardless of relevance. Partial calibration is therefore worse
        than none, so callers must fall back to raw scores for *all* types
        unless the whole merge is covered.
        """
        return bool(content_types) and content_types.issubset(self.curves)

    def score(self, content_type: str, raw_score: float, lane_size: int) -> float:
        """Calibrated probability, or the raw score when calibration is unsafe."""
        curve = self.curves.get(content_type)
        if curve is None or lane_size < MIN_LANE_CANDIDATES:
            return raw_score
        return curve.probability(raw_score)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "recipe": "platt-per-content-type-v1",
            "metadata": metadata or {},
            "curves": {
                content_type: {
                    "a": curve.a,
                    "b": curve.b,
                    "n_positive": curve.n_positive,
                    "n_negative": curve.n_negative,
                    "holdout_auc": curve.holdout_auc,
                }
                for content_type, curve in self.curves.items()
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
