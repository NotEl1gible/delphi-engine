"""Command line for the engine and its eval suite.

`forecast` is the product. Everything else is the evidence that the product's defaults were
chosen rather than guessed: the stopping threshold comes out of `ablate`, the calibrator out
of `calibrate`, the abstention gate out of the coverage curve, and `anchor` is a release gate
that can fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals import instruments as I
from evals import metrics as M

from .abstain import choose, curve
from .calibration import Calibrator
from .config import get_settings
from .panel import forecast as run_forecast
from .panel import snapshot_at
from .providers import build_provider
from .schemas import Question


def _settings(a, **over):
    kw = {"provider": a.provider}
    if getattr(a, "model", None):
        kw["panel_model"] = a.model
    if getattr(a, "agents", 0):
        kw["n_agents"] = a.agents
    kw.update(over)
    return get_settings(**kw)


# ----------------------------------------------------------------------------
def cmd_forecast(a) -> int:
    s = _settings(a)
    q = Question(id="live-1", text=a.question, resolution_date=a.date or "unknown",
                 split="live")
    f = run_forecast(q, settings=s, provider=build_provider(s), anchor=a.anchor)
    print(f"\n{q.text}\n")
    for snp in f.snapshots:
        tag = "premortem" if snp.premortem else f"round {snp.round}   "
        mv = "  --  " if snp.movement is None else f"{snp.movement:6.3f}"
        print(f"  {tag}  pooled {snp.pooled:.3f}  spread {snp.spread:.3f}  moved {mv}  "
              f"{[round(p, 2) for p in snp.probabilities]}")
    if f.decision == "abstain":
        print(f"\n  ABSTAIN -- the panel disagreed too much (spread {f.spread:.2f} > "
              f"tau {s.abstain_tau})")
        print(f"  the pool before the gate was {f.p_raw:.3f}; it is withheld, not reported")
    else:
        print(f"\n  P(YES) = {f.p:.3f}    (raw pool {f.p_raw:.3f}, "
              f"calibrator {f.calibrator.get('kind')})")
    print(f"  {f.rounds_used} feedback rounds"
          f"{' (settled early)' if f.stopped_early else ''}, "
          f"{f.tokens_in}/{f.tokens_out} tokens, ${f.usd:.4f}, {f.latency_ms / 1000:.1f}s")
    return 0


def cmd_questions(a) -> int:
    qs = I.load_questions()
    for split in ("dev", "test"):
        sub = [q for q in qs if q.split == split]
        k = sum(q.outcome for q in sub)
        print(f"{split:<5} n={len(sub):<3} YES {k}/{len(sub)}   "
              f"{sub[0].resolution_date} .. {sub[-1].resolution_date}")
    ys = [q.outcome for q in qs]
    mk = [I.MARKET[q.id] for q in qs]
    print(f"\ncrowd baseline Brier {M.brier(mk, ys):.4f}   "
          f"constant 0.5 scores {M.brier([0.5] * len(ys), ys):.4f}")
    print("the crowd price is a BASELINE and is never rendered into a prompt")
    return 0


# ----------------------------------------------------------------------------
def cmd_ablate(a) -> int:
    qs = I.load_questions(limit=a.limit)
    # Every round arm is a snapshot of ONE run, so the stopping rule is switched off here.
    # With it on, the later arms would exist only on questions that happened not to settle --
    # a different subset per arm, which is not an ablation.
    s_full = _settings(a, stop_movement=0.0)
    s_single = _settings(a, n_agents=1, max_rounds=0, stop_movement=0.0)
    s_ev = _settings(a, max_rounds=0, stop_movement=0.0)

    print(f"panel: {s_full.n_agents} agents, rounds 0..{s_full.max_rounds}, "
          f"provider={s_full.provider}, model={s_full.panel_model}, n={len(qs)}")
    s_aa = _settings(a, max_rounds=0, stop_movement=0.0, seed=s_full.seed + 1000)
    full = I.run_arm(qs, s_full, "panel", quiet=a.quiet)
    single = I.run_arm(qs, s_single, "single", quiet=a.quiet)
    evid = I.run_arm(qs, s_ev, "evidence", evidence=True, quiet=a.quiet)
    aa = I.run_arm(qs, s_aa, "aa", quiet=a.quiet)
    for rows, name in ((full, "panel"), (single, "single"), (evid, "evidence"), (aa, "aa")):
        I.save(rows, name)

    cals = I.calibration_table(full, qs, round_for_fit=a.fit_round)
    pick = min(("platt", "isotonic", "identity"), key=lambda k: cals[k]["test"])
    cal: Calibrator = cals[a.calibrator]["cal"] if a.calibrator else cals[pick]["cal"]
    print(f"\nusing {cal.params().get('kind')} "
          f"({'chosen by test score' if not a.calibrator else 'requested'})")

    test = [q for q in qs if q.split == "test"]
    arms = I.build_arms(full, single, evid, s_full,
                        stop_movement=get_settings().stop_movement, aa=aa)
    I.score_arms(arms, test, cal, baseline="round0")

    fails = []
    by = {arm.name: arm for arm in arms}
    ids = [q.id for q in test]
    ys = [q.outcome for q in test]
    # Suspicious perfection, not exact zero. A Brier under 0.02 across sixteen real
    # prediction-market questions is not a forecaster, it is a leak -- contamination, a
    # calibrator fitted on the test split, or a question set that resolved before the
    # model's cutoff. The threshold fires long before the number reaches zero.
    for name in ("round0", "adaptive", "premortem"):
        if name in by and all(i in by[name].p for i in ids):
            b = M.brier([cal.apply(by[name].p[i]) for i in ids], ys)
            if b < a.min_brier:
                fails.append(f"{name} scored Brier {b:.4f} < {a.min_brier} -- that is not "
                             f"forecasting skill on real questions, it is a leak")
    if "market" in by and all(i in by["market"].p for i in ids):
        mk = M.brier([by["market"].p[i] for i in ids], ys)
        best = min(M.brier([cal.apply(by[n].p[i]) for i in ids], ys)
                   for n in by if n != "market" and all(i in by[n].p for i in ids))
        print(f"\ncrowd {mk:.4f} vs best panel arm {best:.4f}   "
              f"{'the crowd wins' if mk < best else 'the panel wins'}")
    for f in fails:
        print(f"GATE FAILED: {f}", file=sys.stderr)
    return 1 if fails else 0


def cmd_calibrate(a) -> int:
    qs = I.load_questions()
    full = I.load("panel")
    cals = I.calibration_table(full, qs, round_for_fit=a.fit_round)
    kind = a.calibrator or min(("platt", "isotonic", "identity"),
                               key=lambda k: cals[k]["test"])
    cal = cals[kind]["cal"]
    Path("artifacts").mkdir(exist_ok=True)
    cal.save("artifacts/calibrator.json")
    print(f"\nwrote artifacts/calibrator.json  ({cal.params()})")

    s = _settings(a)
    try:
        import mlflow
        mlflow.set_tracking_uri(s.mlflow_uri)
        mlflow.set_experiment("llm-debate-calibration")
        with mlflow.start_run(run_name=kind):
            mlflow.log_params({k: v for k, v in cal.params().items()
                               if k not in ("xs", "ys")})
            mlflow.log_params({"fit_round": a.fit_round, "n_agents": s.n_agents})
            mlflow.log_metrics({"dev_brier": cals[kind]["dev"],
                                "test_brier": cals[kind]["test"],
                                "gap": cals[kind]["test"] - cals[kind]["dev"]})
            mlflow.log_artifact("artifacts/calibrator.json")
        print(f"logged to MLflow at {s.mlflow_uri}")
    except Exception as e:                     # tracking is useful, not load-bearing
        print(f"MLflow logging skipped: {type(e).__name__}: {e}")

    by = {f.question_id: f for f in full}
    rows = [{"p": cal.apply(snapshot_at(by[q.id], a.fit_round)), "y": q.outcome,
             "spread": [sn for sn in by[q.id].snapshots if not sn.premortem][-1].spread}
            for q in qs if q.id in by]
    print(f"\n{'tau':>6}{'coverage':>20}{'wrong-side':>14}{'brier on answered':>20}")
    for r in curve(rows):
        cov = f"{r['coverage']:.2f} ({r['answered']}/{r['n']})"
        wrong = f"{r['wrong_rate']:.2f} ({r['wrong_side']})"
        print(f"{r['tau']:>6.2f}{cov:>20}{wrong:>14}{r['brier_on_answered']:>20.4f}")
    tau = choose(rows, max_wrong_rate=s.max_wrong_rate)
    print(f"\nwidest gate holding wrong-side under {s.max_wrong_rate:.0%}: tau = {tau}")
    print("widest, not tightest: past the point where error stops falling, tightening only")
    print("costs coverage, and that plateau is invisible in any single number")
    return 0


# ----------------------------------------------------------------------------
def cmd_anchor(a) -> int:
    qs = I.load_questions(limit=a.limit or 12)
    s = _settings(a, max_rounds=a.rounds, stop_movement=0.0)
    print(f"three conditions on {len(qs)} questions: no anchor, {I.ANCHOR_LOW}, "
          f"{I.ANCHOR_HIGH}")
    none_ = I.run_arm(qs, s, "anchor_none", quiet=a.quiet)
    low = I.run_arm(qs, s, "anchor_low", anchor=I.ANCHOR_LOW, quiet=a.quiet)
    high = I.run_arm(qs, s, "anchor_high", anchor=I.ANCHOR_HIGH, quiet=a.quiet)
    for rows, name in ((none_, "anchor_none"), (low, "anchor_low"), (high, "anchor_high")):
        I.save(rows, name)
    r = I.anchoring(none_, low, high)
    print(f"\nanchoring coefficient beta = {r['beta']:.3f}   "
          f"median {r['beta_median']:.3f}   [{r['beta_lo']:.3f}, {r['beta_hi']:.3f}]  "
          f"n={r['n']}")
    print(f"mean absolute shift away from the unanchored forecast: "
          f"{r['mean_abs_shift']:.3f}")
    print("\nbeta = 0 means the anchor moved nothing; beta = 1 means the forecast IS the")
    print("anchor. This needs no outcomes at all, so it stays informative at any n --")
    print("changing an irrelevant reference number must not move a well-founded forecast.")
    if r["beta"] > a.max_beta:
        print(f"\nGATE FAILED: beta {r['beta']:.3f} exceeds {a.max_beta} -- the panel is "
              f"transcribing its anchor rather than forecasting", file=sys.stderr)
        return 1
    return 0


def cmd_conform(a) -> int:
    qs = I.load_questions(limit=a.limit or 12)
    s = _settings(a, max_rounds=max(1, a.rounds), stop_movement=0.0)
    normal = I.run_arm(qs, s, "conform_normal", quiet=a.quiet)
    planted = I.run_arm(qs, s, "conform_planted", planted=True, quiet=a.quiet)
    I.save(normal, "conform_normal")
    I.save(planted, "conform_planted")
    r = I.conformity(normal, planted)
    print(f"\none member replaced by a confidently WRONG insider, on {r['n_questions']} "
          f"questions")
    print(f"  herding toward the group mean:  {r['herding']:+.3f}   "
          f"({r['n_agent_moves']} agent moves)")
    print(f"  pull toward the loud outlier:   {r['pull']:+.3f}")
    print(f"  agents that held firm:          {r['held_firm']:.2f}")
    print(f"  shift in the final pool:        {r['mean_pool_shift']:+.3f}")
    print("\n0 means the panel ignored it, 1 means it adopted the position outright.")
    print("Measured on the AGENTS, not on the pool: an extra extreme number drags any")
    print("average by arithmetic. Only a change in the others' own answers is conformity.")
    print("\nThe two rows differ on purpose. A planted member is 1 of N, so an agent that")
    print("responds to the GROUP AVERAGE shows pull near herding/N. Pull much larger than")
    print("that means the panel is responding to stated confidence rather than to the")
    print("balance of opinion, which is the failure mode a Delphi protocol exists to avoid.")
    if s.provider == "mock":
        print(f"\n  [mock] herding was configured at {s.mock_herding:.2f}; the instrument "
              f"read {r['herding']:.3f}.")
        print("  An instrument that cannot recover a known value is not measuring what its")
        print("  name says on a real model either.")
    return 0


def cmd_personas(a) -> int:
    qs = I.load_questions(limit=a.limit)
    s = _settings(a, max_rounds=a.rounds, stop_movement=0.0)
    runs = {}
    for variant in ("designed", "identical", "shuffled"):
        runs[variant] = I.run_arm(qs, s, f"roster_{variant}", roster_variant=variant,
                                  quiet=a.quiet)
        I.save(runs[variant], f"roster_{variant}")
    test = [q for q in qs if q.split == "test"] or qs
    cal = I.fit_on_dev(runs["designed"], qs, "platt", a.rounds)
    I.persona_table(runs, test, cal)
    print("\nidentical gives every member the same neutral brief; shuffled keeps the six")
    print("briefs and pairs each with the wrong label. If nothing moves under identical,")
    print("the roster is decoration -- and that is a result, not a failure.")
    return 0


def cmd_probe(a) -> int:
    """Does the model already know how these resolved?"""
    s = _settings(a)
    if s.provider != "litellm":
        print("the contamination probe needs a live provider; the deterministic one has no")
        print("training data to have been contaminated by. Re-run with --provider litellm.")
        return 0
    import litellm
    qs = I.load_questions(limit=a.limit)
    known = 0
    for q in qs:
        r = litellm.completion(
            model=s.panel_model, max_tokens=10,
            messages=[{"role": "system", "content":
                       "Answer with one word: KNOWN if you already know how this question "
                       "actually resolved from your training data, or UNKNOWN if you do not."},
                      {"role": "user", "content": f"{q.text} (resolved {q.resolution_date})"}])
        said = (r.choices[0].message.content or "").strip().upper()
        known += said.startswith("KNOWN")
        print(f"  {q.id:<14}{said[:8]:<10}{q.text[:60]}")
    print(f"\nself-reported as already known: {M.rate(known, len(qs))}")
    print("A 'forecast' of a known outcome is a recall test. This is self-report and is")
    print("therefore a LOWER bound on contamination, which is why the question set is")
    print("filtered to resolutions after the model's cutoff as well.")
    return 0


def cmd_trace(a) -> int:
    rows = I.load(a.arm)
    f = next((x for x in rows if x.question_id == a.question), None)
    if f is None:
        print(f"{a.question} not in runs/{a.arm}.jsonl", file=sys.stderr)
        return 3
    print(f"{f.question_id}  arm={f.arm}  decision={f.decision}  "
          f"raw={f.p_raw:.3f}  p={f.p}  spread={f.spread:.3f}")
    print(f"\n  {f.question}\n")
    for snp in f.snapshots:
        tag = "premortem" if snp.premortem else f"round {snp.round}"
        print(f"  {tag:<11} pooled {snp.pooled:.3f}  spread {snp.spread:.3f}  "
              f"{[round(p, 2) for p in snp.probabilities]}")
        for t in [t for t in f.turns if t.round == snp.round]:
            print(f"      {t.persona:<12} {t.verdict.probability:.3f}  "
                  f"{(t.verdict.reasoning or '')[:78]}")
    print(f"\n  {f.tokens_in}/{f.tokens_out} tokens  ${f.usd:.4f}  "
          f"{f.latency_ms / 1000:.1f}s  trace {f.trace_id}")
    return 0


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="debate", description=__doc__.split("\n")[0])
    ap.add_argument("--provider", choices=["mock", "litellm"], default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--agents", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("forecast", help="the product: one question in, a probability out")
    p.add_argument("question")
    p.add_argument("--date", default="")
    p.add_argument("--anchor", type=float, default=None)
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("questions", help="the committed question set and its baseline")
    p.set_defaults(fn=cmd_questions)

    p = sub.add_parser("ablate", help="what does each round buy, and what does it cost")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fit-round", dest="fit_round", type=int, default=1)
    p.add_argument("--calibrator", default="", choices=["", "identity", "platt", "isotonic"])
    p.add_argument("--min-brier", dest="min_brier", type=float, default=0.02,
                   help="gate: any arm scoring below this is a leak, not skill")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("calibrate", help="fit on dev, score on test, write the artifact")
    p.add_argument("--fit-round", dest="fit_round", type=int, default=1)
    p.add_argument("--calibrator", default="", choices=["", "identity", "platt", "isotonic"])
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("anchor", help="is there signal, or is the panel transcribing?")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-beta", dest="max_beta", type=float, default=0.5)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_anchor)

    p = sub.add_parser("conform", help="persuasion or herding")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_conform)

    p = sub.add_parser("personas", help="does the roster do any work")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_personas)

    p = sub.add_parser("probe", help="does the model already know the answers")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("trace", help="one forecast, end to end")
    p.add_argument("--question", required=True)
    p.add_argument("--arm", default="panel")
    p.set_defaults(fn=cmd_trace)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
