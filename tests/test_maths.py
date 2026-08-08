"""Property tests on the parts that fail silently.

A wrong aggregator does not crash. It returns a plausible number that is slightly too close to
0.5 forever, and every downstream complaint gets attributed to the agents. So the pooling and
calibration maths are tested by properties over generated inputs rather than by a handful of
examples that were chosen because they passed.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from debate import pooling as P
from debate.abstain import choose, curve
from debate.calibration import Isotonic, Platt, fit_calibrator, log_loss
from evals import metrics as M

probs = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
panels = st.lists(probs, min_size=1, max_size=12)
labels = st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=60)


# ---------------------------------------------------------------- pooling
@given(panels)
def test_pool_never_leaves_the_range_of_its_members(ps):
    out = P.pool(ps)
    lo, hi = min(P.clip(p) for p in ps), max(P.clip(p) for p in ps)
    assert lo - 1e-9 <= out <= hi + 1e-9


@given(probs, st.integers(min_value=1, max_value=8))
def test_pooling_agreement_changes_nothing(p, k):
    assert P.pool([p] * k) == pytest.approx(P.clip(p), abs=1e-9)


@given(panels)
def test_pool_is_never_exactly_certain(ps):
    """An agent that says 1.0 must not be able to make the panel certain. Clipping is the
    statement that no single member holds infinite influence -- without it one confident
    agent takes the pool to the boundary and no later calibration can pull it back."""
    out = P.pool(ps)
    assert 0.0 < out < 1.0


@given(panels, probs)
def test_pool_is_monotone_in_every_member(ps, extra):
    lower = P.pool(ps + [P.EPS])
    higher = P.pool(ps + [1 - P.EPS])
    assert lower <= P.pool(ps + [extra]) + 1e-9
    assert P.pool(ps + [extra]) <= higher + 1e-9


@given(panels)
def test_equal_weights_match_no_weights(ps):
    assert P.pool(ps, [2.0] * len(ps)) == pytest.approx(P.pool(ps), abs=1e-12)


def test_pooling_beats_the_arithmetic_mean_at_keeping_information():
    """The claim the README leads with, asserted so it cannot quietly stop being true: the
    arithmetic mean of probabilities can never sit further from 0.5 than its most extreme
    member, so a panel aggregated that way is underconfident by construction."""
    ps = [0.90, 0.95, 0.99]
    arithmetic = sum(ps) / len(ps)
    assert abs(arithmetic - 0.5) <= abs(max(ps) - 0.5)
    assert P.pool(ps) > arithmetic


def test_spread_is_measured_in_odds_not_in_probability_points():
    """0.01 vs 0.02 is a factor of two; 0.45 vs 0.55 is a factor of 1.5. Probability-space
    spread ranks them backwards, and the abstention gate reads this number."""
    assert abs(0.02 - 0.01) < abs(0.55 - 0.45)
    assert P.spread([0.01, 0.02]) > P.spread([0.45, 0.55])


@given(panels)
def test_spread_is_non_negative_and_zero_on_agreement(ps):
    assert P.spread(ps) >= 0.0
    assert P.spread([ps[0]] * len(ps)) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- calibration
@given(probs)
def test_identity_platt_is_the_identity(p):
    assert Platt(1.0, 0.0).apply(p) == pytest.approx(P.clip(p), abs=1e-9)


@hyp_settings(deadline=None, max_examples=30)
@given(st.lists(probs, min_size=2, max_size=40), st.data())
def test_platt_fit_always_lands_in_bounds(ps, data):
    ys = data.draw(st.lists(st.integers(0, 1), min_size=len(ps), max_size=len(ps)))
    pl = Platt.fit(ps, ys)
    assert 1 / Platt.A_MAX <= pl.a <= Platt.A_MAX and abs(pl.b) <= Platt.B_MAX


def test_platt_refuses_to_invert_the_forecaster():
    """A negative slope maps confidence in YES onto confidence in NO. On a small unlucky
    split that can genuinely lower dev loss, which is exactly why it is refused: a calibrator
    may rescale a signal, never reverse it."""
    ps = [0.9, 0.8, 0.7, 0.2, 0.1]
    ys = [0, 0, 0, 1, 1]                       # perfectly anti-correlated
    pl = Platt.fit(ps, ys)
    assert pl.a > 0


def test_platt_recovers_a_known_shrinkage():
    """A forecaster that reports exactly 0.4x its own true log-odds must fit a slope near
    1/0.4 = 2.5. This is the case a hand-chosen extremising constant is imagining."""
    import random
    rng = random.Random(0)
    ps, ys = [], []
    for _ in range(1500):
        t = rng.gauss(0, 2.0)                  # the question's true log-odds
        ys.append(1 if rng.random() < P.sigmoid(t) else 0)
        ps.append(P.sigmoid(0.4 * t))          # reported, shrunk by a known factor
    pl = Platt.fit(ps, ys)
    assert 2.0 < pl.a < 3.1, f"expected a slope near 2.5, got {pl.a}"
    assert not pl.clamped
    assert M.brier([pl.apply(p) for p in ps], ys) < M.brier(ps, ys)


def test_platt_refuses_to_sharpen_when_the_compression_is_noise():
    """The case that makes a hand-chosen extremising constant dangerous, and the one that
    caught a wrong expectation in this very suite.

    "The pool looks compressed, so sharpen it" is only right when the compression is
    SHRINKAGE. When it is NOISE, the optimal slope is below 1 and sharpening makes the
    forecaster worse. Here the reported log-odds are 0.8*s + N(0, 1.4), for which the Bayes-
    optimal slope is 2k/sigma^2 = 2*0.8/1.96 = 0.816 -- and the fit must find that, not 1/0.4.
    """
    import random
    rng = random.Random(0)
    ps, ys = [], []
    for _ in range(1500):
        y = 1 if rng.random() < 0.5 else 0
        ps.append(P.sigmoid(0.8 * (1 if y else -1) + rng.gauss(0, 1.4)))
        ys.append(y)
    pl = Platt.fit(ps, ys)
    assert pl.a == pytest.approx(0.816, abs=0.15), f"analytic optimum is 0.816, got {pl.a}"
    assert pl.a < 1.0, "sharpening a noise-compressed pool makes it worse"


def test_near_separable_dev_data_does_not_blow_the_fit_up():
    """The bug this suite caught, kept as a regression test.

    On a near-separable dev split the maximum-likelihood slope has no finite solution. An
    unregularised fit landed on a=10 with b=+10, which made the Brier score three times WORSE
    than doing nothing while the log-likelihood kept improving -- a calibrator that scored
    better on its own objective and destroyed the product. The prior is what stops that, so
    the assertion is about the outcome and not about a bound: the fit must still improve the
    score it is supposed to improve.
    """
    import random
    rng = random.Random(0)
    ps, ys = [], []
    for _ in range(400):
        y = 1 if rng.random() < 0.5 else 0
        ps.append(P.sigmoid(0.4 * (2.0 if y else -2.0) + rng.gauss(0, 0.3)))
        ys.append(y)
    pl = Platt.fit(ps, ys)
    assert abs(pl.b) < 1.0, f"the intercept ran away: b={pl.b}"
    assert M.brier([pl.apply(p) for p in ps], ys) < M.brier(ps, ys)
    assert "clamped" in pl.params()


@given(st.lists(probs, min_size=1, max_size=40), st.data())
def test_isotonic_is_monotone(ps, data):
    ys = data.draw(st.lists(st.integers(0, 1), min_size=len(ps), max_size=len(ps)))
    iso = Isotonic.fit(ps, ys)
    grid = [i / 40 for i in range(41)]
    out = [iso.apply(g) for g in grid]
    assert all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))


def test_isotonic_overfits_where_platt_does_not():
    """Not a defect -- the reason the calibrator is scored on a split it never saw. Isotonic
    wins on dev and gives it back on test, which is what a two-parameter family buys you at
    these sample sizes."""
    import random
    rng = random.Random(3)
    ps, ys = [], []
    for _ in range(120):
        y = 1 if rng.random() < 0.5 else 0
        ps.append(P.sigmoid(0.35 * (2.2 if y else -2.2) + rng.gauss(0, 0.5)))
        ys.append(y)
    dp, dy, tp, ty = ps[:60], ys[:60], ps[60:], ys[60:]
    iso, pl = Isotonic.fit(dp, dy), Platt.fit(dp, dy)
    iso_gap = M.brier([iso.apply(p) for p in tp], ty) - M.brier([iso.apply(p) for p in dp], dy)
    pl_gap = M.brier([pl.apply(p) for p in tp], ty) - M.brier([pl.apply(p) for p in dp], dy)
    assert iso_gap > pl_gap


def test_unknown_calibrator_is_an_error_not_a_silent_identity():
    with pytest.raises(ValueError):
        fit_calibrator("magic", [0.5], [1])


@given(st.lists(probs, min_size=1, max_size=30), st.data())
def test_log_loss_is_finite_even_at_the_boundaries(ps, data):
    ys = data.draw(st.lists(st.integers(0, 1), min_size=len(ps), max_size=len(ps)))
    assert math.isfinite(log_loss(ps, ys))


# ---------------------------------------------------------------- metrics
@given(st.lists(probs, min_size=1, max_size=40), st.data())
def test_brier_is_bounded(ps, data):
    ys = data.draw(st.lists(st.integers(0, 1), min_size=len(ps), max_size=len(ps)))
    assert 0.0 <= M.brier(ps, ys) <= 1.0


@hyp_settings(max_examples=25)
@given(st.lists(probs, min_size=8, max_size=60), st.data())
def test_murphy_decomposition_closes(ps, data):
    ys = data.draw(st.lists(st.integers(0, 1), min_size=len(ps), max_size=len(ps)))
    d = M.murphy(ps, ys, n_bins=5)
    assert abs(d["residual"]) < 0.06, "the decomposition no longer accounts for the Brier"


def test_paired_bootstrap_reports_no_difference_between_an_arm_and_itself():
    ps = [0.2, 0.8, 0.55, 0.1, 0.9]
    ys = [0, 1, 1, 0, 1]
    d = M.paired_bootstrap(ps, ps, ys, trials=500)
    assert d["delta"] == 0.0 and d["verdict"] == "INCONCLUSIVE"


def test_paired_bootstrap_needs_equal_lengths():
    with pytest.raises(ValueError):
        M.paired_bootstrap([0.5, 0.5], [0.5], [1, 0])


def test_ece_is_meaningless_at_small_n_and_the_null_says_so():
    """Measured, and it is why ECE is never reported alone here: a PERFECTLY calibrated
    forecaster scores an ECE far from zero when n is small, so a bare ECE cannot distinguish
    a miscalibrated system from a small sample."""
    import random
    rng = random.Random(5)
    ps_small = [rng.uniform(0.05, 0.95) for _ in range(32)]
    ps_big = [rng.uniform(0.05, 0.95) for _ in range(1000)]
    assert M.ece_null(ps_small) > 3 * M.ece_null(ps_big)
    assert M.ece_null(ps_small) > 0.05


# ---------------------------------------------------------------- abstention
def test_coverage_rises_monotonically_with_the_gate():
    rows = [{"p": 0.6, "y": 1, "spread": s / 4} for s in range(12)]
    cov = [r["coverage"] for r in curve(rows, [0.0, 0.5, 1.0, 2.0, 99.0])]
    assert cov == sorted(cov)
    assert cov[-1] == 1.0


def test_the_gate_stays_open_when_disagreement_carries_no_signal():
    """If spread does not predict error, abstaining buys nothing and the chooser must not
    invent a threshold. A gate that fires on noise is pure lost coverage."""
    import random
    rng = random.Random(11)
    rows = [{"p": 0.7, "y": 1 if rng.random() < 0.7 else 0, "spread": rng.uniform(0, 3)}
            for _ in range(300)]
    assert choose(rows, max_wrong_rate=0.4) == 99.0


def test_the_gate_closes_when_disagreement_does_predict_error():
    rows = ([{"p": 0.9, "y": 1, "spread": 0.2} for _ in range(40)]
            + [{"p": 0.9, "y": 0, "spread": 2.5} for _ in range(40)])
    tau = choose(rows, max_wrong_rate=0.1)
    assert tau < 2.5
    kept = [r for r in rows if r["spread"] <= tau]
    assert kept and all(r["y"] == 1 for r in kept)
