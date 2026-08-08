"""Combining a panel's opinions.

The estimator is **log-odds pooling** -- the weighted arithmetic mean of the agents' logits,
which is the geometric mean of their odds. It is not a trimmed mean of probabilities, and the
difference is not cosmetic.

Averaging probabilities averages on the wrong scale. The arithmetic mean of probabilities is
mathematically guaranteed to sit no further from 0.5 than the most extreme member of the
panel, so a panel aggregated that way is **underconfident before anyone measures anything**.
Every subsequent complaint that "the forecaster is too compressed" is then a complaint about
the aggregator, not about the agents. Log-odds pooling has no such built-in pull: two agents
at 0.9 pool to 0.9, and two agents at 0.9 and 0.99 pool to 0.9676, not to 0.945.

Deliberately, this module does NOT extremise. A hand-chosen sharpening constant applied on
top of the pool is a calibration parameter wearing a disguise, and it cannot be validated
where it sits. All sharpening lives in `debate.calibration`, where it is fitted on a dev split
and scored on a held-out test split.
"""
from __future__ import annotations

import math

# Agents do emit 0.0 and 1.0, and logit(0) is -inf. Clipping is not a numerical nicety here:
# it is the statement that no panel member is allowed infinite influence over the pool.
EPS = 1e-3


def clip(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def pool(probs: list[float], weights: list[float] | None = None) -> float:
    """Weighted log-odds pool. Weights are normalised, so this never extremises by itself."""
    if not probs:
        raise ValueError("cannot pool an empty panel")
    if weights is None:
        weights = [1.0] * len(probs)
    if len(weights) != len(probs):
        raise ValueError("weights and probs differ in length")
    if any(w < 0 for w in weights):
        raise ValueError("negative weight")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights sum to zero")
    z = sum(w * logit(p) for w, p in zip(weights, probs, strict=True)) / total
    return sigmoid(z)


def spread(probs: list[float]) -> float:
    """Disagreement, measured in LOG-ODDS space and reported as a median absolute deviation.

    In probability space, 0.01 vs 0.02 looks like agreement (a gap of 0.01) while 0.45 vs 0.55
    looks like disagreement (a gap of 0.10). In odds terms the first pair differs by a factor
    of two and the second by a factor of 1.5, so probability-space spread reports the panel's
    disagreement backwards exactly where forecasts are most decision-relevant. The abstention
    gate is driven by this number, so getting the scale right decides which forecasts a human
    ever sees.
    """
    if len(probs) < 2:
        return 0.0
    zs = sorted(logit(p) for p in probs)
    med = _median(zs)
    return _median(sorted(abs(z - med) for z in zs))


def _median(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def movement(prev: float, cur: float) -> float:
    """How far the pooled estimate moved between two rounds, in log-odds. The adaptive
    stopping rule compares this against a threshold, so it has to be on the same scale as
    `spread` -- a fixed probability-point threshold would stop early on confident questions
    and never stop on uncertain ones."""
    return abs(logit(cur) - logit(prev))
