"""Calibration, fitted on a dev split and scored on a held-out test split.

This is where every bit of sharpening in the engine lives, and that placement is the point.
The usual pattern is a hand-chosen extremising constant applied to the pooled probability --
`p' = sigmoid(1.15 * logit(p))`. That constant is a fitted parameter that nobody fitted. It
cannot be validated where it sits, it is chosen by looking at the same numbers it is meant to
improve, and when it turns out to be too weak there is no procedure for finding a better one.

Here the same transform is `p' = sigmoid(A * logit(p) + B)`, with A and B fitted by minimising
log-loss on a dev split and reported on a test split the fit never saw. A > 1 sharpens, A < 1
compresses, B shifts. The number is the same shape; the difference is that it is measured and
that its generalisation gap is visible.

Isotonic regression is the non-parametric alternative, implemented here with pool-adjacent-
violators rather than pulled in from scikit-learn -- it is thirty lines, it removes a heavy
dependency, and at these sample sizes its tendency to overfit is exactly the thing the test
split is there to expose.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .pooling import EPS, clip, logit, sigmoid


def log_loss(ps: list[float], ys: list[int]) -> float:
    if not ps:
        return 0.0
    return -sum(y * math.log(clip(p)) + (1 - y) * math.log(1 - clip(p))
                for p, y in zip(ps, ys)) / len(ps)


class Calibrator:
    kind = "identity"

    def apply(self, p: float) -> float:
        return clip(p)

    def params(self) -> dict:
        return {"kind": self.kind}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.params(), indent=1), encoding="utf-8")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params()})"


class Platt(Calibrator):
    """p' = sigmoid(A * logit(p) + B). A is the extremising factor, fitted rather than chosen."""

    kind = "platt"

    A_MAX = 10.0
    B_MAX = 4.0
    # A weakly informative prior, and it is not decoration. Maximum likelihood on a dev split
    # that is close to separable has NO finite solution: the slope runs off to infinity and
    # the intercept follows it. Measured here, unregularised Nelder-Mead landed on a=10, b=+10
    # -- a combination that made the Brier score three times worse than doing nothing while
    # the log-likelihood kept improving. So this is a MAP fit: log-normal on the slope,
    # normal on the intercept, both centred on "change nothing", with the penalty divided by
    # n so that real data outvotes the prior as it accumulates.
    PRIOR_LOG_A_SD = 0.8
    PRIOR_B_SD = 1.0

    def __init__(self, a: float = 1.0, b: float = 0.0, clamped: bool = False):
        self.a, self.b, self.clamped = float(a), float(b), bool(clamped)

    def apply(self, p: float) -> float:
        return clip(sigmoid(self.a * logit(p) + self.b))

    def params(self) -> dict:
        return {"kind": self.kind, "a": self.a, "b": self.b, "clamped": self.clamped}

    @classmethod
    def fit(cls, ps: list[float], ys: list[int]) -> "Platt":
        if len(set(ys)) < 2:
            # One class only. Any A/B pair can drive the loss to zero by predicting the
            # constant, which is a fit to the split rather than to the world.
            return cls(1.0, 0.0)
        zs = [logit(p) for p in ps]
        n = len(ys)

        def objective(theta) -> float:
            log_a, b = float(theta[0]), float(theta[1])
            if not (math.isfinite(log_a) and math.isfinite(b)):
                return 1e18
            a = math.exp(log_a)
            nll = -sum(y * math.log(clip(sigmoid(a * z + b)))
                       + (1 - y) * math.log(1 - clip(sigmoid(a * z + b)))
                       for z, y in zip(zs, ys)) / n
            penalty = (log_a ** 2) / (2 * cls.PRIOR_LOG_A_SD ** 2) \
                + (b ** 2) / (2 * cls.PRIOR_B_SD ** 2)
            return nll + penalty / n

        # Coarse grid, then refine. Unregularised Nelder-Mead from a single start wandered
        # into the corner described above; a grid start makes the fit reproducible and cheap.
        best, best_val = (0.0, 0.0), objective((0.0, 0.0))
        steps_a = [math.log(0.1) + i * (math.log(cls.A_MAX) - math.log(0.1)) / 39
                   for i in range(40)]
        steps_b = [-cls.B_MAX + i * (2 * cls.B_MAX) / 32 for i in range(33)]
        for la in steps_a:
            for b0 in steps_b:
                v = objective((la, b0))
                if v < best_val:
                    best, best_val = (la, b0), v
        try:
            from scipy.optimize import minimize
            res = minimize(objective, list(best), method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000})
            if float(res.fun) <= best_val:
                best = (float(res.x[0]), float(res.x[1]))
        except Exception:                       # scipy absent: the grid answer stands
            pass
        a, b = math.exp(best[0]), best[1]

        # A non-positive slope would INVERT the forecaster -- map confidence in YES onto
        # confidence in NO. On a small unlucky split that can genuinely lower dev loss, and
        # it is refused outright: a calibrator may rescale a signal, never reverse it.
        if a <= 0 or not math.isfinite(a) or not math.isfinite(b):
            return cls(1.0, 0.0)

        # The bound is a backstop the prior should almost always keep the fit away from. When
        # it does fire, that is recorded rather than presented as a converged fit -- a report
        # that cannot tell a fitted slope from a slope that hit its ceiling is not a report.
        clamped = not (1.0 / cls.A_MAX <= a <= cls.A_MAX) or abs(b) > cls.B_MAX
        a = min(max(a, 1.0 / cls.A_MAX), cls.A_MAX)
        b = min(max(b, -cls.B_MAX), cls.B_MAX)
        return cls(a, b, clamped)


class Isotonic(Calibrator):
    """Non-parametric monotone fit by pool-adjacent-violators."""

    kind = "isotonic"

    def __init__(self, xs: list[float], ys: list[float]):
        self.xs, self.ys = list(xs), list(ys)

    def apply(self, p: float) -> float:
        if not self.xs:
            return clip(p)
        p = clip(p)
        if p <= self.xs[0]:
            return clip(self.ys[0])
        if p >= self.xs[-1]:
            return clip(self.ys[-1])
        for i in range(1, len(self.xs)):
            if p <= self.xs[i]:
                x0, x1 = self.xs[i - 1], self.xs[i]
                y0, y1 = self.ys[i - 1], self.ys[i]
                t = 0.0 if x1 == x0 else (p - x0) / (x1 - x0)
                return clip(y0 + t * (y1 - y0))
        return clip(self.ys[-1])

    def params(self) -> dict:
        return {"kind": self.kind, "xs": self.xs, "ys": self.ys}

    @classmethod
    def fit(cls, ps: list[float], ys: list[int]) -> "Isotonic":
        if not ps:
            return cls([], [])
        pairs = sorted(zip((clip(p) for p in ps), (float(y) for y in ys)))
        xs = [p for p, _ in pairs]
        # pool-adjacent-violators: merge any block whose mean breaks monotonicity
        blocks: list[list[float]] = []          # [sum, count, right_edge_index]
        for i, (_, y) in enumerate(pairs):
            blocks.append([y, 1.0, i])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                s2, n2, _ = blocks.pop()
                blocks[-1][0] += s2
                blocks[-1][1] += n2
                blocks[-1][2] = i
        fitted: list[float] = []
        for s, n, _ in blocks:
            fitted.extend([s / n] * int(n))
        return cls(xs, fitted)


def load_calibrator(path) -> Calibrator:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["kind"] == "platt":
        return Platt(d["a"], d["b"], d.get("clamped", False))
    if d["kind"] == "isotonic":
        return Isotonic(d["xs"], d["ys"])
    return Calibrator()


def fit_calibrator(kind: str, ps: list[float], ys: list[int]) -> Calibrator:
    if kind == "platt":
        return Platt.fit(ps, ys)
    if kind == "isotonic":
        return Isotonic.fit(ps, ys)
    if kind == "identity":
        return Calibrator()
    raise ValueError(f"unknown calibrator {kind!r}")
