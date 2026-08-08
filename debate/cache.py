"""Agent-call cache, keyed per arm.

The per-arm part of the key is the whole point and it costs real money to keep. The ablation
runs the same questions through several configurations, and a great many of those calls are
byte-identical: the blind round of the `evidence` arm asks exactly what the blind round of the
plain arm asks. Sharing them would be free.

It is still wrong. Once two arms draw from one cache they stop being independent: arm B's
latency, its cost and even its retry behaviour depend on whether arm A ran first, so the
measured difference between them is partly an artefact of execution order. That is a SUTVA
violation, and it is not hypothetical -- the same mechanism was measured widening a zero-effect
band by an order of magnitude in a sibling project. The arm goes in the key, and the report
says what that cost.

The cache is therefore only a saving WITHIN an arm: re-running one arm after a crash, or two
questions that happen to produce the same prompt.
"""
from __future__ import annotations

import hashlib
import json

from .providers import Ask, Usage
from .schemas import AgentVerdict


def key_of(ask: Ask, model: str, seed: int) -> str:
    payload = {
        "arm": ask.arm,                       # <- the line that keeps the arms independent
        "model": model, "seed": seed,
        "question": ask.question.id, "agent": ask.agent_id, "persona": ask.persona.id,
        "round": ask.round, "prev": ask.prev, "peers": ask.peers,
        "anchor": ask.anchor, "premortem": ask.premortem,
        "evidence": hashlib.sha256((ask.evidence or "").encode()).hexdigest()[:12],
    }
    return "debate:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class Cache:
    """Wraps any redis-like client. `fakeredis` in the unit tests, a real server in CI and
    compose, and `None` to disable."""

    def __init__(self, client, model: str, seed: int, ttl_seconds: int = 7 * 24 * 3600):
        self.client = client
        self.model = model
        self.seed = seed
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, ask: Ask):
        if self.client is None:
            return None
        raw = self.client.get(key_of(ask, self.model, self.seed))
        if not raw:
            self.misses += 1
            return None
        self.hits += 1
        d = json.loads(raw)
        u = d["usage"]
        # A cache hit is billed at zero and FLAGGED. Replaying the original token counts
        # would make a re-run look as expensive as the first run and quietly inflate every
        # cost column that the stopping rule is chosen from.
        return (AgentVerdict(**d["verdict"]),
                Usage(tokens_in=u["tokens_in"], tokens_out=u["tokens_out"],
                      usd=0.0, latency_ms=0.0, cached=True))

    def put(self, ask: Ask, verdict: AgentVerdict, usage: Usage) -> None:
        if self.client is None or usage.error:
            return                            # never cache a failure as if it were an answer
        self.client.set(
            key_of(ask, self.model, self.seed),
            json.dumps({"verdict": verdict.model_dump(),
                        "usage": {"tokens_in": usage.tokens_in,
                                  "tokens_out": usage.tokens_out}}),
            ex=self.ttl)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


def build_cache(settings, model: str):
    if not settings.cache_enabled:
        return None
    try:
        import redis
        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        client.ping()
    except Exception:
        return None                            # no server: run uncached rather than fail
    return Cache(client, model, settings.seed)
