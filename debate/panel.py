"""The product: a question in, a calibrated probability or an abstention out.

Four steps, and the order is the design.

1. Run the panel graph. It returns one snapshot per round.
2. Pool the final round in log-odds. Nothing here sharpens.
3. Apply the fitted calibrator. Every bit of sharpening in the engine happens on this line,
   with parameters fitted on a split the engine was never scored on.
4. Gate on disagreement. If the panel did not agree, the product says so instead of averaging
   its way to a confident-looking number.

`p_raw` is kept next to `p` so that a bad forecast can be attributed rather than argued about:
if `p_raw` was already wrong, the panel failed; if `p_raw` was right and `p` is wrong, the
calibrator did.
"""
from __future__ import annotations

import time
import uuid

from .abstain import spread as _spread  # noqa: F401  (kept in one place, one scale)
from .calibration import Calibrator, load_calibrator
from .config import Settings
from .graph import build_graph
from .personas import PLANTED, Persona, roster
from .pooling import pool
from .schemas import Forecast, Question
from .tracing import Tracing


def make_personas(s: Settings, variant: str = "designed",
                  planted: bool = False) -> list[Persona]:
    people = roster(variant, n=s.n_agents)
    if planted:
        # The conformity probe replaces ONE member, it does not add one. Adding an agent would
        # change the panel size at the same time as its composition, and the drift measured
        # afterwards could then be either effect.
        people = list(people)
        people[0] = PLANTED
    return people


def load_or_identity(path: str) -> Calibrator:
    try:
        return load_calibrator(path)
    except Exception:
        return Calibrator()


def forecast(question: Question, *, settings: Settings, provider,
             calibrator: Calibrator | None = None, tracing: Tracing | None = None,
             cache=None, roster_variant: str = "designed", planted: bool = False,
             anchor: float | None = None, evidence: str | None = None,
             arm: str = "panel") -> Forecast:
    s = settings
    cal = calibrator if calibrator is not None else load_or_identity(s.calibrator_path)
    people = make_personas(s, roster_variant, planted)
    graph = build_graph(settings=s, provider=provider, personas=people,
                        tracing=tracing, cache=cache)

    t0 = time.perf_counter()
    state = graph.invoke({"question": question, "anchor": anchor, "evidence": evidence,
                          "arm": arm},
                         {"recursion_limit": 4 * (s.max_rounds + 4)})
    ms = (time.perf_counter() - t0) * 1000

    snaps = list(state.get("snapshots", []))
    turns = list(state.get("turns", []))
    final = snaps[-1]
    feedback_rounds = sum(1 for snp in snaps if not snp.premortem)
    p_raw = final.pooled
    p_cal = cal.apply(p_raw)
    gated = final.spread > s.abstain_tau

    return Forecast(
        question_id=question.id, question=question.text,
        decision="abstain" if gated else "forecast",
        p=None if gated else p_cal, p_raw=p_raw, spread=final.spread,
        # Feedback rounds only. Counting the premortem as a round would make the adaptive
        # rule look like it spends one more round than it does.
        rounds_used=feedback_rounds, stopped_early=bool(state.get("stopped_early")),
        snapshots=snaps, turns=turns, calibrator=cal.params(),
        evidence_used=evidence is not None, anchor=anchor,
        tokens_in=sum(t.tokens_in for t in turns),
        tokens_out=sum(t.tokens_out for t in turns),
        usd=sum(t.usd for t in turns), latency_ms=ms,
        trace_id=uuid.uuid4().hex[:16],
        model=getattr(provider, "model", s.panel_model), arm=arm)


def snapshot_at(f: Forecast, rnd: int, cal: Calibrator | None = None) -> float:
    """The pooled estimate as it stood after round `rnd`, calibrated.

    This is what makes the round ablation affordable AND perfectly paired: one panel run
    yields every round arm, so "no feedback at all", "one round" and "five rounds" are read
    off the same execution rather than compared across three of them.
    """
    usable = [snp for snp in f.snapshots if not snp.premortem and snp.round <= rnd]
    if not usable:
        usable = [snp for snp in f.snapshots if not snp.premortem][:1] or f.snapshots[:1]
    p = usable[-1].pooled
    return cal.apply(p) if cal is not None else p


def pool_of(probs: list[float]) -> float:
    return pool(probs)
