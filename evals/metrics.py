"""Scoring, and the intervals that keep it honest.

Two rules run through this module.

**Paired data gets paired statistics.** Every arm forecasts the same questions, so the arms
are not independent samples. An independent interval per arm is much wider than the evidence
warrants and routinely leaves a real difference indistinguishable from nothing -- the classic
way an ablation ends in "the intervals overlap, we cannot say". The bootstrap here resamples
QUESTIONS and recomputes the difference inside each resample, which is where nearly all of the
power lives.

**A decomposition that does not close is a decomposition that is lying.** The Murphy identity
`Brier = reliability - resolution + uncertainty` holds exactly only for the binned forecast
means; with raw forecasts there is a within-bin residual. It is computed and returned rather
than dropped, and a test fails if it grows.
"""
from __future__ import annotations

import math
import random


def brier(ps: list[float], ys: list[int]) -> float:
    if not ps:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(ps, ys, strict=True)) / len(ps)


def log_score(ps: list[float], ys: list[int]) -> float:
    eps = 1e-3
    if not ps:
        return 0.0
    return -sum(y * math.log(min(max(p, eps), 1 - eps))
                + (1 - y) * math.log(1 - min(max(p, eps), 1 - eps))
                for p, y in zip(ps, ys, strict=True)) / len(ps)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{(k / n if n else 0):.2f} ({k}/{n}) [{lo:.2f}, {hi:.2f}]"


def reliability_bins(ps: list[float], ys: list[int], n_bins: int = 5) -> list[dict]:
    """Bins with COUNTS. A reliability table without counts hides that a bin holding two
    questions is driving the shape of the curve."""
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(ps)
               if (p >= lo and p < hi) or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            out.append({"lo": lo, "hi": hi, "n": 0, "mean_p": None, "obs": None})
            continue
        out.append({"lo": lo, "hi": hi, "n": len(idx),
                    "mean_p": sum(ps[i] for i in idx) / len(idx),
                    "obs": sum(ys[i] for i in idx) / len(idx)})
    return out


def murphy(ps: list[float], ys: list[int], n_bins: int = 5) -> dict:
    n = len(ps)
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0,
                "residual": 0.0}
    base = sum(ys) / n
    bins = reliability_bins(ps, ys, n_bins)
    rel = sum(b["n"] * (b["mean_p"] - b["obs"]) ** 2 for b in bins if b["n"]) / n
    res = sum(b["n"] * (b["obs"] - base) ** 2 for b in bins if b["n"]) / n
    unc = base * (1 - base)
    bs = brier(ps, ys)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
            "residual": bs - (rel - res + unc)}


def ece(ps: list[float], ys: list[int], n_bins: int = 5) -> float:
    """Reported, but never on its own. At these sample sizes ECE is dominated by binning
    noise: a perfectly calibrated forecaster scores well away from zero at n in the tens, so
    an ECE that looks bad is not evidence of miscalibration until it is compared against what
    a perfect forecaster would have scored on the same n. `ece_null` does that comparison."""
    n = len(ps)
    if n == 0:
        return 0.0
    return sum(b["n"] * abs(b["mean_p"] - b["obs"])
               for b in reliability_bins(ps, ys, n_bins) if b["n"]) / n


def ece_null(ps: list[float], n_bins: int = 5, trials: int = 400, seed: int = 0) -> float:
    """The ECE a PERFECTLY calibrated forecaster would score on this many questions, by
    simulation. This is the number that turns an ECE into a claim instead of a decoration."""
    rng = random.Random(seed)
    vals = []
    for _ in range(trials):
        ys = [1 if rng.random() < p else 0 for p in ps]
        vals.append(ece(ps, ys, n_bins))
    vals.sort()
    return vals[len(vals) // 2]


def paired_bootstrap(ps_a: list[float], ps_b: list[float], ys: list[int],
                     trials: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the DIFFERENCE in Brier over resampled questions.

    Returns delta = brier(a) - brier(b); negative means a is better. The interval is the
    percentile interval of the difference, not of either arm on its own.
    """
    n = len(ys)
    if n == 0 or len(ps_a) != n or len(ps_b) != n:
        raise ValueError("paired bootstrap needs three equal-length sequences")
    rng = random.Random(seed)
    point = brier(ps_a, ys) - brier(ps_b, ys)
    deltas = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum((ps_a[i] - ys[i]) ** 2 for i in idx) / n
        b = sum((ps_b[i] - ys[i]) ** 2 for i in idx) / n
        deltas.append(a - b)
    deltas.sort()
    lo = deltas[int(0.025 * trials)]
    hi = deltas[min(trials - 1, int(0.975 * trials))]
    below = sum(1 for d in deltas if d <= 0.0) / trials
    above = sum(1 for d in deltas if d >= 0.0) / trials
    return {"delta": point, "lo": lo, "hi": hi, "n": n,
            "p": min(1.0, 2 * min(below, above)),
            "verdict": "INCONCLUSIVE" if lo <= 0.0 <= hi
                       else ("b better" if point > 0 else "a better")}


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni: which of a FAMILY of comparisons survive.

    Not a formality. An ablation compares eight arms against one baseline at 95%, so roughly
    one spurious winner per run is the expected behaviour of the procedure -- and one duly
    appeared here, in an arm that was configurationally identical to the baseline and differed
    only by random seed. Reporting each interval on its own would have shipped it as a result.

    Holm is used rather than plain Bonferroni because it is uniformly more powerful at the
    same family-wise error rate; there is no reason to pay for the difference.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    out = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break                              # Holm stops at the first failure
    return out


def fmt_delta(d: dict, survives: bool | None = None) -> str:
    s = f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p']:.3f}  "
    if survives is None:
        return s + d["verdict"]
    if d["verdict"] == "INCONCLUSIVE":
        return s + "INCONCLUSIVE"
    return s + (d["verdict"] if survives else "n.s. after Holm")
