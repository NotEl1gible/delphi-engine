"""The panel as a LangGraph state machine.

The shape matters more than the framework. Two properties come from drawing the protocol as
a graph rather than writing a loop:

**The stopping rule is an EDGE.** "Keep going while the pool is still moving" lives in
`route_after_round`, a function whose whole job is to choose the next node. It is not an `if`
buried inside a round body, so the adaptive-rounds feature is a property of the topology and
can be reasoned about, drawn, and changed without touching how a round works.

**A round is a fan-out with a barrier.** `Send` dispatches one `agent` invocation per panel
member and LangGraph gathers them before `gather` runs. That is the actual Delphi protocol:
members answer independently, then and only then see a summary. A loop would still work, but
the barrier would be implicit and nothing would stop a later edit from leaking one agent's
answer into the next agent's prompt within the same round -- which would silently turn the
blind round into a sequential chain and destroy the one property the protocol exists for.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Settings
from .personas import Persona
from .pooling import movement, pool, spread
from .providers import Ask
from .schemas import AgentTurn, Question, RoundSnapshot
from .tracing import Tracing, record_usage


class PanelState(TypedDict, total=False):
    question: Question
    round: int
    turns: Annotated[list[AgentTurn], operator.add]
    snapshots: Annotated[list[RoundSnapshot], operator.add]
    stopped_early: bool
    anchor: float | None
    evidence: str | None
    arm: str
    _premortem: bool


def _turns_of(state: PanelState, rnd: int) -> list[AgentTurn]:
    return [t for t in state.get("turns", []) if t.round == rnd]


def build_graph(*, settings: Settings, provider, personas: list[Persona],
                tracing: Tracing | None = None, cache=None):
    """`personas` fixes the panel size; the roster variant is chosen by the caller so the
    `personas` instrument can swap it without the graph knowing."""
    s = settings

    def start(state: PanelState) -> dict:
        return {"round": 0, "stopped_early": False}

    def dispatch(state: PanelState) -> list[Send]:
        rnd = state["round"]
        prev = {t.agent_id: t.verdict.probability for t in _turns_of(state, rnd - 1)}
        # The blind round shows nothing. Later rounds show the peer estimates WITHOUT
        # identities: a Delphi panel is anonymous by definition, and attaching names is what
        # turns controlled feedback into a status contest.
        peers = sorted(prev.values()) if rnd > 0 else []
        return [Send("agent", {"question": state["question"], "round": rnd,
                               "persona": p, "agent_id": f"a{i}",
                               "prev": prev.get(f"a{i}"), "peers": peers,
                               "anchor": state.get("anchor"),
                               "evidence": state.get("evidence"),
                               "premortem": False, "arm": state.get("arm", "panel")})
                for i, p in enumerate(personas)]

    def dispatch_premortem(state: PanelState) -> list[Send]:
        rnd = state["round"]
        prev = {t.agent_id: t.verdict.probability for t in _turns_of(state, rnd)}
        peers = sorted(prev.values())
        return [Send("agent_pm", {"question": state["question"], "round": rnd + 1,
                               "persona": p, "agent_id": f"a{i}",
                               "prev": prev.get(f"a{i}"), "peers": peers,
                               "anchor": state.get("anchor"),
                               "evidence": state.get("evidence"),
                               "premortem": True, "arm": state.get("arm", "panel")})
                for i, p in enumerate(personas)]

    def agent(payload: dict[str, Any]) -> dict:
        ask = Ask(question=payload["question"], persona=payload["persona"],
                  agent_id=payload["agent_id"], round=payload["round"],
                  prev=payload["prev"], peers=payload["peers"], anchor=payload["anchor"],
                  evidence=payload["evidence"], premortem=payload["premortem"],
                  arm=payload["arm"])
        t0 = time.perf_counter()
        hit = cache.get(ask) if cache is not None else None
        if hit is not None:
            verdict, usage = hit
        else:
            if tracing is not None:
                with tracing.agent_span(name="debate.agent", model=getattr(
                        provider, "model", s.panel_model), system=provider.name,
                        round=ask.round, agent_id=ask.agent_id,
                        persona=ask.persona.id, arm=ask.arm) as span:
                    verdict, usage = provider.ask(ask)
                    record_usage(span, usage, verdict.probability)
            else:
                verdict, usage = provider.ask(ask)
            if cache is not None:
                cache.put(ask, verdict, usage)
        return {"turns": [AgentTurn(
            agent_id=ask.agent_id, persona=ask.persona.id,
            model=getattr(provider, "model", s.panel_model), round=ask.round,
            verdict=verdict, tokens_in=usage.tokens_in, tokens_out=usage.tokens_out,
            usd=usage.usd, cached=usage.cached, error=usage.error,
            latency_ms=(time.perf_counter() - t0) * 1000)]}

    def gather(state: PanelState) -> dict:
        # The round is derived from the turns that actually arrived, not from the counter.
        # An earlier version read `state["round"]` in both gather nodes; the premortem branch
        # dispatches at rnd+1, so its turns were collected under a round number that held
        # none, the pool fell back to the neutral 0.5, and the engine returned exactly 0.500
        # with zero disagreement for every question. Loud in a smoke test, invisible in a
        # summary table.
        all_turns = state.get("turns", [])
        rnd = max((t.round for t in all_turns), default=0)
        turns = [t for t in all_turns if t.round == rnd]
        is_pm = bool(turns) and rnd > 0 and all(
            t.round == rnd for t in turns) and state.get("_premortem", False)
        # A member whose call errored is ABSENT, not neutral. Substituting 0.5 would let an
        # outage widen the disagreement that drives the abstention gate.
        ps = [t.verdict.probability for t in turns if t.error is None]
        if not ps:
            ps = [0.5]
        prev = state.get("snapshots", [])
        p = pool(ps)
        snap = RoundSnapshot(
            round=rnd, pooled=p, spread=spread(ps), probabilities=ps, premortem=is_pm,
            movement=None if not prev else movement(prev[-1].pooled, p),
            tokens_in=sum(t.tokens_in for t in turns),
            tokens_out=sum(t.tokens_out for t in turns),
            usd=sum(t.usd for t in turns))
        return {"snapshots": [snap], "round": rnd + 1}

    def route_after_round(state: PanelState) -> str:
        """The adaptive stopping rule, as an edge.

        Stop when the last round moved the pooled estimate less than `stop_movement` in
        log-odds. The threshold is on the log-odds scale for the same reason `spread` is: a
        fixed probability-point rule stops instantly on a confident question and never stops
        on an uncertain one, so it would spend the budget in exactly the wrong places.
        """
        snaps = state.get("snapshots", [])
        rnd = state["round"]                      # already incremented by `gather`
        last = snaps[-1] if snaps else None
        settled = (last is not None and last.movement is not None
                   and last.movement < s.stop_movement)
        if rnd > s.max_rounds or settled:
            return "premortem" if s.premortem else END
        return "fanout"

    def fanout(state: PanelState) -> dict:
        return {}

    def premortem_anchor(state: PanelState) -> dict:
        # Whether the loop ended because the panel SETTLED or because it ran out of rounds is
        # recorded here. Budget exhaustion presented as convergence is exactly what makes an
        # adaptive system look better than it is.
        snaps = state.get("snapshots", [])
        settled = bool(snaps and snaps[-1].movement is not None
                       and snaps[-1].movement < s.stop_movement)
        return {"stopped_early": settled, "_premortem": True}

    g = StateGraph(PanelState)
    g.add_node("start", start)
    g.add_node("fanout", fanout)
    g.add_node("agent", agent)
    g.add_node("gather", gather)
    g.add_node("premortem", premortem_anchor)
    g.add_node("gather_pm", gather)

    g.add_edge(START, "start")
    g.add_conditional_edges("start", dispatch, ["agent"])
    g.add_conditional_edges("fanout", dispatch, ["agent"])
    g.add_edge("agent", "gather")
    g.add_conditional_edges("gather", route_after_round, ["fanout", "premortem", END])
    g.add_conditional_edges("premortem", dispatch_premortem, ["agent_pm"])
    g.add_node("agent_pm", agent)
    g.add_edge("agent_pm", "gather_pm")
    g.add_edge("gather_pm", END)
    return g.compile()
