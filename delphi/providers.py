"""Where a panel member's number comes from.

Two implementations behind one interface. `LiteLLMProvider` talks to Anthropic, OpenAI or
Groq through a single call. `MockProvider` is a deterministic simulator with **three knobs
whose true values are known**: separation (how learnable the question is), herding (how far an
agent moves toward what it can see its peers said) and anchoring (how far it moves toward a
number handed to it).

That last part is the point of the mock, and it is not a stub. Every instrument in the eval
suite claims to measure one of those quantities, so every instrument is first pointed at a
system whose value is known by construction. An instrument that cannot recover `mock_herding`
from the mock is not measuring herding on a real model either -- it is measuring something
else and reporting it under the wrong name.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field

from .config import PRICES, Settings
from .personas import Persona
from .pooling import clip, logit, sigmoid
from .schemas import AgentVerdict, Question


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None


@dataclass
class Ask:
    """Everything a panel member is allowed to know at this point in the protocol.

    The graph builds one of these per agent per round. A real provider renders it into a
    prompt; the mock reads the numbers. Because both see exactly the same object, the mock
    cannot accidentally be given information the real panel never gets.
    """

    question: Question
    persona: Persona
    agent_id: str
    round: int
    prev: float | None = None
    peers: list[float] = field(default_factory=list)   # empty in the blind round
    anchor: float | None = None
    evidence: str | None = None
    premortem: bool = False
    arm: str = "panel"


def price(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


class MockProvider:
    """Deterministic, offline, and built to be measured against."""

    name = "mock"

    def __init__(self, s: Settings):
        self.s = s

    def _skill(self, agent_id: str) -> float:
        r = random.Random(_seed("skill", agent_id, self.s.seed))
        return 0.65 + 0.7 * r.random()

    def ask(self, a: Ask) -> tuple[AgentVerdict, Usage]:
        s = self.s
        rng = random.Random(_seed(a.question.id, a.agent_id, a.round, a.arm, s.seed))

        if a.persona.id == "planted":
            # The confidently-wrong insider. Its position is the opposite of the truth, stated
            # at the edge of the scale, which is what makes the conformity measurement legible.
            y = a.question.outcome if a.question.outcome is not None else 1
            p = 0.03 if y == 1 else 0.97
            return (AgentVerdict(probability=p, reasoning="Settled by information I hold.",
                                 key_factor="insider", self_confidence=0.98),
                    Usage(tokens_in=180, tokens_out=60))

        if a.round == 0 or a.prev is None:
            if a.question.outcome is None:
                z = 0.0
            else:
                signal = 0.5 + (a.question.outcome - 0.5) * s.mock_separation
                z = logit(signal) * self._skill(a.agent_id)
            # Shared first, independent second. The shared term survives pooling and the
            # independent one does not, which is the whole reason a panel is worth less than
            # its member count suggests.
            common = random.Random(_seed("common", a.question.id, s.seed))
            z += common.gauss(0.0, s.mock_common_bias)
            z += rng.gauss(0.0, s.mock_noise)
            # Evidence deliberately does NOTHING here. An offline stub that sharpened the
            # estimate would make the retrieval arm win in CI and lose in production, which
            # is worse than having no arm: it would look like a measured result. The mock
            # cannot speak to retrieval, and the report says so instead of pretending.
            if a.anchor is not None:
                z = (1 - s.mock_anchoring) * z + s.mock_anchoring * logit(a.anchor)
        else:
            z = logit(a.prev)
            if a.peers:
                peer_z = sum(logit(p) for p in a.peers) / len(a.peers)
                z = (1 - s.mock_herding) * z + s.mock_herding * peer_z
            z += rng.gauss(0.0, s.mock_noise * 0.25)
            if a.premortem:
                z *= 0.92                       # a premortem pulls back from the extremes

        p = clip(sigmoid(z))
        return (AgentVerdict(probability=p, reasoning=f"round {a.round} estimate",
                             key_factor=a.persona.id, self_confidence=0.5),
                Usage(tokens_in=420 + 90 * len(a.peers), tokens_out=110))


SYSTEM = """You are one member of a forecasting panel using the Delphi method.

{brief}

Return ONE JSON object and nothing else:
{{"probability": <number between 0 and 1>, "reasoning": "<two sentences>",
  "key_factor": "<the single fact that moves your estimate most>",
  "self_confidence": <number between 0 and 1>}}

Give a real number, not a hedge. 0.5 means you genuinely have no information."""


def render(a: Ask) -> tuple[str, str]:
    system = SYSTEM.format(brief=a.persona.brief)
    parts = [f"Question: {a.question.text}",
             f"Resolves by: {a.question.resolution_date}"]
    if a.evidence:
        parts.append(f"\nRetrieved evidence:\n{a.evidence}")
    if a.anchor is not None:
        parts.append(f"\nA reference estimate from another source is {a.anchor:.2f}.")
    if a.peers:
        shown = ", ".join(f"{p:.2f}" for p in a.peers)
        parts.append(f"\nOther panel members estimated: {shown}. They cannot see who you are "
                     f"and you cannot see who they are. Revise only if their reasoning would "
                     f"change yours; holding your position is a valid answer.")
    if a.premortem:
        parts.append("\nBefore answering: assume your current estimate turns out badly wrong. "
                     "Write the single most likely reason, then give your revised estimate.")
    return system, "\n".join(parts)


class LiteLLMProvider:
    """One call for Anthropic, OpenAI and Groq. A heterogeneous panel is not heterogeneous
    if every member has to be wired up separately."""

    name = "litellm"

    def __init__(self, s: Settings, model: str | None = None):
        self.s = s
        self.model = model or s.panel_model

    def ask(self, a: Ask) -> tuple[AgentVerdict, Usage]:
        import litellm

        system, user = render(a)
        t0 = time.perf_counter()
        last = ""
        for attempt in range(3):
            try:
                resp = litellm.completion(
                    model=self.model, max_tokens=400,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user
                               + ("" if attempt == 0 else
                                  "\n\nYour previous reply was not valid JSON. Reply with the "
                                  "JSON object only.")}])
                raw = resp.choices[0].message.content or ""
                last = raw
                v = _parse(raw)
                u = resp.usage
                tin = int(getattr(u, "prompt_tokens", 0) or 0)
                tout = int(getattr(u, "completion_tokens", 0) or 0)
                return v, Usage(tin, tout, price(self.model, tin, tout),
                                (time.perf_counter() - t0) * 1000)
            except Exception as e:                       # transient API or parse failure
                last = f"{type(e).__name__}: {e}"
                if attempt == 2:
                    # A failed member is recorded as absent, never as 0.5. Substituting a
                    # neutral number would let an outage masquerade as panel disagreement and
                    # silently widen the very spread the abstention gate reads.
                    return (AgentVerdict(probability=0.5, reasoning=""),
                            Usage(error=last, latency_ms=(time.perf_counter() - t0) * 1000))
                time.sleep(1.5 * (attempt + 1))
        return AgentVerdict(probability=0.5), Usage(error=last)


def _parse(raw: str) -> AgentVerdict:
    txt = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {txt[:120]!r}")
    obj = json.loads(m.group(0))
    p = obj.get("probability")
    if isinstance(p, str):
        p = float(p.strip().rstrip("%")) / (100.0 if "%" in obj["probability"] else 1.0)
    obj["probability"] = min(max(float(p), 0.0), 1.0)
    return AgentVerdict(**obj)


def build_provider(s: Settings, model: str | None = None):
    if s.provider == "mock":
        return MockProvider(s)
    if s.provider == "litellm":
        return LiteLLMProvider(s, model)
    raise ValueError(f"unknown provider {s.provider!r}")
