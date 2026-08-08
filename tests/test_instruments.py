"""The instruments, pointed at a system whose behaviour is known by construction.

Every instrument here claims to measure one quantity. The deterministic provider has that
quantity as a configuration value, so each test asks the instrument to recover a number it was
never told. An instrument that cannot do that on a system it fully understands is not
measuring what its name says on a real model either -- it is measuring something else and
reporting it under the wrong name.
"""
from __future__ import annotations

import pytest

from debate.config import get_settings
from evals import instruments as I
from evals import metrics as M

QS = I.load_questions(limit=10)


def _settings(**kw):
    base = {"provider": "mock", "cache_enabled": False, "stop_movement": 0.0}
    base.update(kw)
    return get_settings(**base)


# ---------------------------------------------------------------- statistics
def test_holm_is_stricter_than_a_bare_interval_and_stops_at_the_first_failure():
    """Eight arms against one baseline at 95% produce roughly one spurious winner per run by
    design. Without a family correction that winner is published."""
    assert M.holm([0.001, 0.04, 0.20]) == [True, False, False]
    assert M.holm([0.001, 0.002, 0.003]) == [True, True, True]
    assert M.holm([0.04]) == [True]            # a single test is not a family
    assert M.holm([0.9, 0.9]) == [False, False]


def test_the_bootstrap_p_value_agrees_with_its_own_interval():
    ps_a = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    ps_b = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    ys = [1, 1, 1, 1, 1, 1, 1, 1]
    d = M.paired_bootstrap(ps_a, ps_b, ys, trials=2000)
    assert d["verdict"] == "a better" and d["p"] < 0.05
    same = M.paired_bootstrap(ps_a, ps_a, ys, trials=2000)
    assert same["verdict"] == "INCONCLUSIVE" and same["p"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- anchoring
def test_anchoring_recovers_the_configured_coefficient():
    s = _settings(max_rounds=1, mock_anchoring=0.4)
    none_ = I.run_arm(QS, s, "an_none", quiet=True)
    low = I.run_arm(QS, s, "an_low", anchor=I.ANCHOR_LOW, quiet=True)
    high = I.run_arm(QS, s, "an_high", anchor=I.ANCHOR_HIGH, quiet=True)
    beta = I.anchoring(none_, low, high)["beta"]
    assert beta == pytest.approx(0.4, abs=0.12), f"configured 0.40, instrument read {beta}"


def test_a_panel_that_ignores_its_anchor_reads_zero():
    """The other end of the scale. If beta only ever came out positive, it would be measuring
    the stimulus rather than the response."""
    s = _settings(max_rounds=1, mock_anchoring=0.0)
    none_ = I.run_arm(QS, s, "an0_none", quiet=True)
    low = I.run_arm(QS, s, "an0_low", anchor=I.ANCHOR_LOW, quiet=True)
    high = I.run_arm(QS, s, "an0_high", anchor=I.ANCHOR_HIGH, quiet=True)
    assert abs(I.anchoring(none_, low, high)["beta"]) < 0.12


# ---------------------------------------------------------------- conformity
def test_conformity_recovers_the_configured_herding():
    s = _settings(max_rounds=1, mock_herding=0.35)
    normal = I.run_arm(QS, s, "cf_normal", quiet=True)
    planted = I.run_arm(QS, s, "cf_planted", planted=True, quiet=True)
    r = I.conformity(normal, planted)
    assert r["herding"] == pytest.approx(0.35, abs=0.12), (
        f"configured 0.35, instrument read {r['herding']}")
    # A planted member is one of six, so movement toward IT is far smaller than movement
    # toward the group. Confusing the two would report a calm panel as a conformist one.
    assert r["pull"] < r["herding"]


def test_a_panel_that_never_revises_reads_zero_herding():
    s = _settings(max_rounds=1, mock_herding=0.0)
    normal = I.run_arm(QS, s, "cf0_normal", quiet=True)
    planted = I.run_arm(QS, s, "cf0_planted", planted=True, quiet=True)
    assert abs(I.conformity(normal, planted)["herding"]) < 0.12


def test_the_slope_estimator_is_not_the_mean_of_ratios():
    """Kept as a regression test. The mean-of-ratios estimator read 0.250 against a
    configured 0.35 because its denominator is noisy and agents starting close to the target
    produce enormous ratios in both directions."""
    pairs = [(0.35 * g + n, g) for g, n in
             [(2.0, 0.1), (-1.5, -0.05), (0.02, 0.4), (-0.03, -0.35), (1.0, 0.0)]]
    den = sum(g * g for _, g in pairs)
    slope = sum(m * g for m, g in pairs) / den
    mean_ratio = sum(m / g for m, g in pairs) / len(pairs)
    assert slope == pytest.approx(0.35, abs=0.05)
    assert abs(mean_ratio - 0.35) > abs(slope - 0.35)


# ---------------------------------------------------------------- the ablation
def test_every_round_arm_comes_from_one_run():
    """Snapshots, not separate executions. Re-running per arm would make the arms independent
    draws, and the A/A check below shows how large that noise is relative to the effects."""
    s = _settings(max_rounds=5)
    full = I.run_arm(QS[:4], s, "snap", quiet=True)
    f = full[0]
    assert len([sn for sn in f.snapshots if not sn.premortem]) == 6
    assert I.snapshot_at(f, 0) != I.snapshot_at(f, 5)
    # round 0 is the blind pool; asking for a round beyond the run returns the last one
    assert I.snapshot_at(f, 99) == I.snapshot_at(f, 5)


def test_the_adaptive_arm_uses_the_product_threshold_not_the_ablation_one():
    """The ablation disables early stopping so that every round arm exists on every question.
    Reading the adaptive arm off that disabled value made it byte-identical to the last fixed
    round -- a feature that never fired, reported as if it had."""
    s = _settings(max_rounds=5)
    full = I.run_arm(QS[:6], s, "adapt", quiet=True)
    arms = {a.name: a for a in I.build_arms(full, full, full, s, stop_movement=0.15)}
    assert arms["adaptive"].rounds < arms["round5"].rounds
    never = {a.name: a for a in I.build_arms(full, full, full, s, stop_movement=0.0)}
    assert never["adaptive"].rounds == pytest.approx(float(s.max_rounds))


def test_the_offline_evidence_stub_changes_nothing():
    """A stub that helped would win in CI and lose in production, and it would look like a
    measured result rather than a fixture."""
    s = _settings(max_rounds=0)
    plain = I.run_arm(QS[:6], s, "same", quiet=True)
    withev = I.run_arm(QS[:6], s, "same", evidence=True, quiet=True)
    for a, b in zip(plain, withev, strict=True):
        assert a.p_raw == pytest.approx(b.p_raw, abs=1e-9)


def test_the_aa_arm_is_configurationally_identical_to_the_baseline():
    """It must differ only by seed. If it ever differs by configuration it stops being an A/A
    check and becomes another arm nobody asked for."""
    s0 = _settings(max_rounds=0, seed=7)
    s1 = _settings(max_rounds=0, seed=1007)
    a = s0.model_dump()
    b = s1.model_dump()
    assert {k: v for k, v in a.items() if k != "seed"} == \
           {k: v for k, v in b.items() if k != "seed"}
    ra = I.run_arm(QS[:6], s0, "aa_a", quiet=True)
    rb = I.run_arm(QS[:6], s1, "aa_b", quiet=True)
    assert any(x.p_raw != pytest.approx(y.p_raw) for x, y in zip(ra, rb, strict=True)), (
        "the A/A arm produced identical numbers; it cannot measure seed variance")


def test_correlated_member_errors_stop_the_panel_scoring_perfectly():
    """Without a shared error term, pooling six independent agents drove the Brier to 0.0000
    -- a panel nobody has operated, and one that exercises none of the gates."""
    qs = I.load_questions()
    ys = [q.outcome for q in qs]
    indep = I.run_arm(qs, _settings(max_rounds=0, mock_common_bias=0.0), "iid", quiet=True)
    corr = I.run_arm(qs, _settings(max_rounds=0), "corr", quiet=True)
    b_indep = M.brier([f.p_raw for f in indep], ys)
    b_corr = M.brier([f.p_raw for f in corr], ys)
    assert b_corr > b_indep
    assert b_corr > 0.05, "the deterministic panel is still implausibly good"
