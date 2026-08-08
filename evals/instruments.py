"""The eval suite the product ships with.

Four instruments and a probe. Two of them need outcomes; two do not, and those two are the
ones that stay informative at n=32.

Every arm forecasts the same questions, so every comparison is paired and every interval is a
bootstrap of the DIFFERENCE. The alternative -- an independent interval per arm -- throws away
most of the power and is the standard way an ablation ends in "the intervals overlap, we
cannot say".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from delphi.calibration import Calibrator, fit_calibrator
from delphi.config import Settings
from delphi.evidence import offline_evidence, search
from delphi.panel import forecast as run_forecast
from delphi.panel import snapshot_at
from delphi.pooling import logit, sigmoid
from delphi.providers import build_provider
from delphi.schemas import Forecast, Question

from . import metrics as M

RUNS = Path("runs")
QUESTIONS = Path(__file__).with_name("questions.jsonl")

# Fixed for every question so the coefficient is comparable across them, and far enough apart
# that a reading of zero means indifference rather than a weak stimulus.
ANCHOR_LOW, ANCHOR_HIGH = 0.15, 0.85


def load_questions(split: str | None = None, limit: int = 0) -> list[Question]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    qs, extra = [], {}
    for r in rows:
        extra[r["id"]] = {"market_p": r.get("market_p")}
        qs.append(Question(**{k: v for k, v in r.items() if k in Question.model_fields}))
    if split:
        qs = [q for q in qs if q.split == split]
    if limit:
        qs = qs[:limit]
    for q in qs:
        MARKET[q.id] = extra[q.id]["market_p"]
    return qs


MARKET: dict[str, float] = {}


# ----------------------------------------------------------------------------
# Running arms
# ----------------------------------------------------------------------------
def run_arm(questions: list[Question], s: Settings, arm: str, *, evidence: bool = False,
            roster_variant: str = "designed", planted: bool = False,
            anchor: float | None = None, cal: Calibrator | None = None,
            quiet: bool = False) -> list[Forecast]:
    provider = build_provider(s)
    out = []
    for i, q in enumerate(questions):
        ev = None
        if evidence:
            ev = (search(q.text, api_key=s.tavily_api_key) if s.provider == "litellm"
                  else offline_evidence(q.text))
        f = run_forecast(q, settings=s, provider=provider, calibrator=cal,
                         roster_variant=roster_variant, planted=planted, anchor=anchor,
                         evidence=ev, arm=arm)
        out.append(f)
        if not quiet:
            print(f"  {arm:<12}{q.id:<14}{i + 1:>3}/{len(questions)}  "
                  f"raw {f.p_raw:.3f}  spread {f.spread:.2f}  rounds {f.rounds_used}  "
                  f"${f.usd:.4f}", flush=True)
    return out


def adaptive_snapshot(f: Forecast, threshold: float) -> tuple[float, int]:
    """Where the adaptive rule WOULD have stopped, read off a run that went all the way.

    The rule is a policy over rounds that already happened, so it needs no extra calls. That
    is worth about a third of the bill, and it also makes the adaptive arm perfectly paired
    with the fixed-round arms instead of being a separate execution with its own noise.
    """
    snaps = [s for s in f.snapshots if not s.premortem]
    for snp in snaps:
        if snp.movement is not None and snp.movement < threshold:
            return snp.pooled, snp.round
    return snaps[-1].pooled, snaps[-1].round


def save(rows: list[Forecast], name: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"{name}.jsonl"
    p.write_text("".join(json.dumps(f.model_dump()) + "\n" for f in rows), encoding="utf-8")
    return p


def load(name: str) -> list[Forecast]:
    p = RUNS / f"{name}.jsonl"
    return [Forecast(**json.loads(line))
            for line in p.read_text(encoding="utf-8").splitlines() if line]


# ----------------------------------------------------------------------------
# Instrument 1 -- the round ablation
# ----------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    p: dict[str, float]          # question id -> raw pooled probability
    usd: float = 0.0
    rounds: float = 0.0
    note: str = ""


def build_arms(full: list[Forecast], single: list[Forecast], evid: list[Forecast],
               s: Settings, stop_movement: float | None = None,
               aa: list[Forecast] | None = None) -> list[Arm]:
    """`stop_movement` is the PRODUCT's threshold, passed in separately.

    The ablation runs with early stopping disabled so that every round arm exists on every
    question. Reading the adaptive arm off `s.stop_movement` therefore read the disabled
    value, and the adaptive arm came out byte-identical to the last fixed round -- a feature
    that never fired, reported as if it had.
    """
    stop = s.stop_movement if stop_movement is None else stop_movement
    arms: list[Arm] = []
    arms.append(Arm("market", {q: MARKET[q] for q in MARKET if MARKET.get(q) is not None},
                    note="the crowd's closing price; zero model calls"))
    arms.append(Arm("single", {f.question_id: f.p_raw for f in single},
                    usd=sum(f.usd for f in single), rounds=0.0,
                    note="one agent, blind"))
    for r in sorted(set(s.snapshot_rounds)):
        arms.append(Arm(f"round{r}", {f.question_id: snapshot_at(f, r) for f in full},
                        usd=sum(sum(sn.usd for sn in f.snapshots
                                    if not sn.premortem and sn.round <= r) for f in full),
                        rounds=float(r),
                        note="no feedback at all" if r == 0 else f"{r} feedback round(s)"))
    arms.append(Arm("premortem",
                    {f.question_id: [sn for sn in f.snapshots if sn.premortem][-1].pooled
                     if any(sn.premortem for sn in f.snapshots) else f.p_raw
                     for f in full},
                    usd=sum(f.usd for f in full), rounds=float(s.max_rounds),
                    note="every round plus the premortem pass"))
    ad = {f.question_id: adaptive_snapshot(f, stop) for f in full}
    arms.append(Arm("adaptive", {k: v[0] for k, v in ad.items()},
                    usd=sum(sum(sn.usd for sn in f.snapshots
                                if not sn.premortem and sn.round <= ad[f.question_id][1])
                            for f in full),
                    rounds=sum(v[1] for v in ad.values()) / max(len(ad), 1),
                    note="stops when the pool settles"))
    arms.append(Arm("evidence", {f.question_id: f.p_raw for f in evid},
                    usd=sum(f.usd for f in evid), rounds=0.0,
                    note="retrieval, no feedback"))
    if aa is not None:
        # An A/A arm: byte-identical configuration to round0, different random seed. There is
        # nothing to find here by construction, so whatever the harness reports about it is
        # the harness's own false-positive rate. It earns its place because one arm in this
        # very run WAS declared a winner over the baseline while being configurationally
        # identical to it -- an arm that can only produce noise is the cheapest way to make
        # that visible instead of publishable.
        arms.append(Arm("aa", {f.question_id: f.p_raw for f in aa},
                        usd=sum(f.usd for f in aa), rounds=0.0,
                        note="same config as round0, different seed; must find nothing"))
    return arms


def score_arms(arms: list[Arm], qs: list[Question], cal: Calibrator,
               baseline: str = "round0") -> dict:
    ids = [q.id for q in qs]
    ys = [q.outcome for q in qs]
    base = None
    for a in arms:
        if a.name == baseline:
            base = [cal.apply(a.p[i]) for i in ids if i in a.p]

    scored, tests = [], []
    for a in arms:
        if any(i not in a.p for i in ids):
            continue
        # The crowd price is already a probability from a different process; calibrating it
        # with a map fitted on the panel would be scoring a hybrid nobody built.
        ps = [a.p[i] if a.name == "market" else cal.apply(a.p[i]) for i in ids]
        row = {"arm": a, "ps": ps, "brier": M.brier(ps, ys)}
        if base is not None and a.name != baseline:
            row["test"] = M.paired_bootstrap(ps, base, ys, trials=4000)
            tests.append(row["test"]["p"])
        scored.append(row)
    survives = M.holm(tests) if tests else []

    print(f"\n{'arm':<11}{'brier':>8}{'log':>8}{'ece':>7}{'null':>7}{'rounds':>8}"
          f"{'USD':>9}   vs {baseline} (paired bootstrap, Holm-corrected)")
    k = 0
    for row in scored:
        a, ps = row["arm"], row["ps"]
        line = (f"{a.name:<11}{row['brier']:>8.4f}{M.log_score(ps, ys):>8.4f}"
                f"{M.ece(ps, ys):>7.3f}{M.ece_null(ps):>7.3f}{a.rounds:>8.2f}"
                f"{a.usd:>9.4f}")
        if "test" in row:
            line += "   " + M.fmt_delta(row["test"], survives[k])
            k += 1
        print(line)
    print("\n  brier: lower is better. null: the ECE a PERFECTLY calibrated forecaster would")
    print("  score at this n -- an observed ECE below it is noise, not calibration.")
    print("  The bootstrap resamples QUESTIONS and recomputes the difference inside each")
    print("  resample. Every arm is compared against the same baseline, so these are a")
    print("  FAMILY of tests: at 95% each, roughly one spurious winner per run is what the")
    print("  procedure is designed to produce. Holm is applied across the family and a")
    print("  result that does not survive it is reported as n.s. rather than as a win.")

    # The A/A read-out. Two arms with the same configuration and a different seed have
    # nothing to find, so the size of THEIR gap is the yardstick every other gap in the table
    # has to clear. If re-seeding moves the score by more than the best arm does, the table is
    # measuring luck and says so here rather than in someone else's post-mortem.
    aa = next((r for r in scored if r["arm"].name == "aa"), None)
    base_row = next((r for r in scored if r["arm"].name == baseline), None)
    if aa is not None and base_row is not None:
        noise = abs(aa["brier"] - base_row["brier"])
        effects = [abs(r["brier"] - base_row["brier"]) for r in scored
                   if r["arm"].name not in ("aa", baseline, "market", "single")]
        best = max(effects) if effects else 0.0
        print(f"\n  A/A CHECK: re-seeding the identical configuration moved Brier by "
              f"{noise:.4f}.")
        print(f"  The largest panel-arm effect in this table is {best:.4f}.")
        if noise >= best:
            print("  Seed variance is at least as large as every effect measured. Treat the")
            print("  ranking above as underdetermined at this n; the two label-free")
            print("  instruments do not depend on it.")
        else:
            print("  Effects above this floor are the only ones worth reading.")
    return {"scored": scored, "survives": survives}


# ----------------------------------------------------------------------------
# Instrument 2 -- anchoring (needs no outcomes)
# ----------------------------------------------------------------------------
def anchoring(none_: list[Forecast], low: list[Forecast],
              high: list[Forecast]) -> dict:
    """beta = (z_high - z_low) / (logit(0.85) - logit(0.15)).

    A slope, not a correlation: 0 means the anchor moved nothing, 1 means the forecast IS the
    anchor. Both conditions are the same distance apart on every question, so the coefficient
    is comparable across them and a reading of zero means indifference rather than a stimulus
    too weak to detect.
    """
    span = logit(ANCHOR_HIGH) - logit(ANCHOR_LOW)
    by_low = {f.question_id: f for f in low}
    by_high = {f.question_id: f for f in high}
    betas, shifts = [], []
    for f in none_:
        if f.question_id not in by_low or f.question_id not in by_high:
            continue
        zl = logit(by_low[f.question_id].p_raw)
        zh = logit(by_high[f.question_id].p_raw)
        betas.append((zh - zl) / span)
        shifts.append(abs(sigmoid((zl + zh) / 2) - f.p_raw))
    betas.sort()
    n = len(betas)
    return {"n": n, "beta": sum(betas) / n if n else 0.0,
            "beta_median": betas[n // 2] if n else 0.0,
            "beta_lo": betas[max(0, int(0.05 * n))] if n else 0.0,
            "beta_hi": betas[min(n - 1, int(0.95 * n))] if n else 0.0,
            "mean_abs_shift": sum(shifts) / n if n else 0.0}


# ----------------------------------------------------------------------------
# Instrument 3 -- conformity (needs no outcomes)
# ----------------------------------------------------------------------------
def conformity(normal: list[Forecast], planted: list[Forecast]) -> dict:
    """How far the OTHER members move toward a confidently wrong newcomer.

    Measured on the members themselves, not on the pool. Moving the pool is arithmetic -- an
    extra extreme number drags any average. The question is whether the other agents CHANGE
    THEIR OWN ANSWERS after seeing it, which is the difference between an aggregation artefact
    and social conformity, and only the second one is a property of the protocol.
    """
    # Least-squares slopes, not means of per-agent ratios. The ratio (moved / gap) has a
    # noisy denominator, so agents who started close to the target produce enormous ratios in
    # both directions and the average of them is biased toward zero: with herding set to 0.35
    # the mean-of-ratios estimator read 0.250. The slope sum(moved*gap)/sum(gap^2) is the
    # least-squares estimate of the same coefficient and is not distorted by a small gap.
    def slope(pairs: list[tuple[float, float]]) -> float:
        den = sum(g * g for _, g in pairs)
        return sum(m * g for m, g in pairs) / den if den > 1e-12 else 0.0

    by_id = {f.question_id: f for f in normal}
    pulls: list[tuple[float, float]] = []
    herds: list[tuple[float, float]] = []
    pool_shift: list[float] = []
    firm = 0
    for f in planted:
        base = by_id.get(f.question_id)
        if base is None:
            continue
        r0 = {t.agent_id: t.verdict.probability for t in f.turns if t.round == 0}
        r1 = {t.agent_id: t.verdict.probability for t in f.turns if t.round == 1}
        planted_p = r0.get("a0")
        if planted_p is None or not r1:
            continue
        target = logit(planted_p)
        peer_mean = sum(logit(p) for p in r0.values()) / len(r0)
        for aid, p0 in r0.items():
            if aid == "a0" or aid not in r1:
                continue
            z0, z1 = logit(p0), logit(r1[aid])
            moved = z1 - z0
            pulls.append((moved, target - z0))       # 0 = held firm, 1 = fully adopted
            herds.append((moved, peer_mean - z0))
            firm += abs(moved) < 0.05
        s0 = [sn for sn in base.snapshots if not sn.premortem][-1].pooled
        s1 = [sn for sn in f.snapshots if not sn.premortem][-1].pooled
        pool_shift.append(s1 - s0)
    n = len(pulls)
    return {"n_agent_moves": n, "n_questions": len(pool_shift),
            # Two different quantities, and confusing them is easy. `herding` is movement
            # toward the whole group; `pull` is movement toward the one loud outlier. A
            # planted member is 1 of N, so pull is roughly herding/N when agents respond to
            # the group average and much larger when they respond to confidence itself --
            # which is exactly the distinction worth measuring.
            "herding": slope(herds),
            "pull": slope(pulls),
            "held_firm": firm / n if n else 0.0,
            "mean_pool_shift": sum(pool_shift) / len(pool_shift) if pool_shift else 0.0}


# ----------------------------------------------------------------------------
# Instrument 4 -- do the personas do anything
# ----------------------------------------------------------------------------
def persona_table(runs: dict[str, list[Forecast]], qs: list[Question],
                  cal: Calibrator) -> None:
    ids = [q.id for q in qs]
    ys = [q.outcome for q in qs]
    print(f"\n{'roster':<12}{'brier':>9}{'spread':>9}{'|diff| vs designed':>22}"
          f"   paired vs designed")
    base = None
    for name in ("designed", "identical", "shuffled"):
        rows = runs.get(name)
        if not rows:
            continue
        by = {f.question_id: f for f in rows}
        ps = [cal.apply(by[i].p_raw) for i in ids]
        spread = sum(by[i].spread for i in ids) / len(ids)
        if name == "designed":
            base = ps
            print(f"{name:<12}{M.brier(ps, ys):>9.4f}{spread:>9.3f}{'--':>22}")
            continue
        diff = sum(abs(a - b) for a, b in zip(ps, base, strict=True)) / len(ps)
        print(f"{name:<12}{M.brier(ps, ys):>9.4f}{spread:>9.3f}{diff:>22.4f}   "
              + M.fmt_delta(M.paired_bootstrap(ps, base, ys, trials=4000)))


# ----------------------------------------------------------------------------
# Calibration -- fitted on dev, scored on test
# ----------------------------------------------------------------------------
def fit_on_dev(full: list[Forecast], qs: list[Question], kind: str,
               round_for_fit: int) -> Calibrator:
    dev = {q.id: q.outcome for q in qs if q.split == "dev"}
    by = {f.question_id: f for f in full}
    ps = [snapshot_at(by[i], round_for_fit) for i in dev if i in by]
    ys = [dev[i] for i in dev if i in by]
    return fit_calibrator(kind, ps, ys)


def calibration_table(full: list[Forecast], qs: list[Question], round_for_fit: int) -> dict:
    dev = [q for q in qs if q.split == "dev"]
    test = [q for q in qs if q.split == "test"]
    by = {f.question_id: f for f in full}
    out = {}
    print(f"\n{'calibrator':<12}{'params':<34}{'dev brier':>11}{'test brier':>12}"
          f"{'gap':>8}")
    for kind in ("identity", "platt", "isotonic"):
        cal = fit_on_dev(full, qs, kind, round_for_fit)
        dp = [cal.apply(snapshot_at(by[q.id], round_for_fit)) for q in dev if q.id in by]
        tp = [cal.apply(snapshot_at(by[q.id], round_for_fit)) for q in test if q.id in by]
        db = M.brier(dp, [q.outcome for q in dev if q.id in by])
        tb = M.brier(tp, [q.outcome for q in test if q.id in by])
        params = {k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in cal.params().items() if k != "xs" and k != "ys"}
        print(f"{kind:<12}{str(params):<34}{db:>11.4f}{tb:>12.4f}{tb - db:>+8.4f}")
        out[kind] = {"cal": cal, "dev": db, "test": tb}
    print("\n  The gap column is the point of the split. A calibrator that wins on dev and")
    print("  gives it back on test has fitted the split, not the forecaster.")
    return out
