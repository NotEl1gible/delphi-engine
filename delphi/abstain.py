"""The right to refuse.

A forecaster that always returns a number has quietly decided that a confident answer and a
coin flip are the same product. They are not: one is acted on and the other should be routed
to a person. This engine returns `abstain` when the panel's disagreement is above a threshold,
and the threshold is read off a coverage/error curve rather than chosen by taste.

The curve reports two costs separately because different people pay them. **Coverage** is what
the product delivers automatically; **wrong-side rate** is the share of delivered forecasts
that pointed the wrong way past 0.5, which is what a decision made on the forecast actually
gets burned by. Collapsing them into one score hides the trade rather than pricing it.
"""
from __future__ import annotations

from .pooling import spread  # noqa: F401  (re-exported: the gate and the pool share one scale)

DEFAULT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99.0]


def curve(rows: list[dict], thresholds: list[float] | None = None) -> list[dict]:
    """`rows` carry p (calibrated), y (outcome), spread (log-odds MAD of the panel)."""
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    n = len(rows)
    out = []
    for tau in thresholds:
        kept = [r for r in rows if r["spread"] <= tau]
        k = len(kept)
        wrong = sum(1 for r in kept if (r["p"] > 0.5) != (r["y"] == 1))
        bs = sum((r["p"] - r["y"]) ** 2 for r in kept) / k if k else 0.0
        out.append({"tau": tau, "answered": k, "n": n,
                    "coverage": k / n if n else 0.0,
                    "wrong_side": wrong,
                    "wrong_rate": wrong / k if k else 0.0,
                    "brier_on_answered": bs})
    return out


def choose(rows: list[dict], max_wrong_rate: float = 0.15,
           thresholds: list[float] | None = None) -> float:
    """The widest gate whose delivered forecasts stay under the wrong-side budget.

    Widest, not tightest: tightening past the point where the error stops falling buys nothing
    and costs coverage. That plateau is visible in the curve and invisible in a single number,
    which is the whole reason the curve is printed.
    """
    rowsc = curve(rows, thresholds)
    ok = [r for r in rowsc if r["answered"] > 0 and r["wrong_rate"] <= max_wrong_rate]
    if not ok:
        return 0.0
    return max(r["tau"] for r in ok)
