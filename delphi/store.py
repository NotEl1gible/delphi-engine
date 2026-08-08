"""One schema, two databases.

SQLite locally and in the unit tests, PostgreSQL in the integration job and in compose. The
same `Table` definitions serve both through `.with_variant()`, so there is no second copy of
the schema to drift.

That is worth stating precisely, because it is easy to overclaim: SQLAlchemy will happily
COMPILE `BIGSERIAL`, `JSONB` and `TIMESTAMPTZ` with no Postgres anywhere in sight, so a test
that only checks the generated DDL proves nothing about whether Postgres accepts it. The CI
integration job runs these statements against a real server for that reason.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Float, Integer, JSON,
                        MetaData, String, Table, Text, create_engine, insert, select)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from .schemas import Forecast

metadata = MetaData()

BigId = BigInteger().with_variant(Integer, "sqlite")
Json = JSON().with_variant(JSONB, "postgresql")
Ts = DateTime(timezone=True).with_variant(TIMESTAMP(timezone=True), "postgresql")

runs = Table(
    "runs", metadata,
    Column("id", String(40), primary_key=True),
    Column("created_at", Ts, nullable=False),
    Column("arm", String(40), nullable=False),
    Column("provider", String(20), nullable=False),
    Column("model", String(80), nullable=False),
    Column("n_questions", Integer, nullable=False, default=0),
    Column("settings", Json, nullable=False),
    Column("notes", Text, default=""),
)

forecasts = Table(
    "forecasts", metadata,
    Column("id", BigId, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, index=True),
    Column("question_id", String(40), nullable=False, index=True),
    Column("decision", String(10), nullable=False),
    Column("p", Float),                       # null when the engine abstained
    Column("p_raw", Float, nullable=False),   # the pool before calibration
    Column("spread", Float, nullable=False),
    Column("rounds_used", Integer, nullable=False),
    Column("stopped_early", Boolean, nullable=False),
    Column("anchor", Float),
    Column("evidence_used", Boolean, nullable=False, default=False),
    Column("outcome", Integer),                # 1 / 0 / null while unresolved
    Column("tokens_in", Integer, nullable=False, default=0),
    Column("tokens_out", Integer, nullable=False, default=0),
    Column("usd", Float, nullable=False, default=0.0),
    Column("latency_ms", Float, nullable=False, default=0.0),
    Column("trace_id", String(40), default=""),
    Column("calibrator", Json),
    Column("snapshots", Json),
)

agent_turns = Table(
    "agent_turns", metadata,
    Column("id", BigId, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, index=True),
    Column("question_id", String(40), nullable=False, index=True),
    Column("agent_id", String(20), nullable=False),
    Column("persona", String(40), nullable=False),
    Column("model", String(80), nullable=False),
    Column("round", Integer, nullable=False),
    Column("probability", Float, nullable=False),
    Column("tokens_in", Integer, nullable=False, default=0),
    Column("tokens_out", Integer, nullable=False, default=0),
    Column("usd", Float, nullable=False, default=0.0),
    Column("cached", Boolean, nullable=False, default=False),
    Column("error", Text),
)


def build_engine(url: str, echo: bool = False):
    kw = {"future": True, "echo": echo}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kw)


def create_all(engine) -> None:
    metadata.create_all(engine)


def start_run(engine, *, run_id: str, arm: str, provider: str, model: str,
              settings: dict, n_questions: int, notes: str = "") -> str:
    with engine.begin() as conn:
        conn.execute(insert(runs).values(
            id=run_id, created_at=datetime.now(timezone.utc), arm=arm, provider=provider,
            model=model, n_questions=n_questions,
            settings=json.loads(json.dumps(settings, default=str)), notes=notes))
    return run_id


def save_forecast(engine, run_id: str, f: Forecast, outcome: int | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(insert(forecasts).values(
            run_id=run_id, question_id=f.question_id, decision=f.decision, p=f.p,
            p_raw=f.p_raw, spread=f.spread, rounds_used=f.rounds_used,
            stopped_early=f.stopped_early, anchor=f.anchor,
            evidence_used=f.evidence_used, outcome=outcome, tokens_in=f.tokens_in,
            tokens_out=f.tokens_out, usd=f.usd, latency_ms=f.latency_ms,
            trace_id=f.trace_id, calibrator=f.calibrator,
            snapshots=[s.model_dump() for s in f.snapshots]))
        if f.turns:
            conn.execute(insert(agent_turns), [
                {"run_id": run_id, "question_id": f.question_id, "agent_id": t.agent_id,
                 "persona": t.persona, "model": t.model, "round": t.round,
                 "probability": t.verdict.probability, "tokens_in": t.tokens_in,
                 "tokens_out": t.tokens_out, "usd": t.usd, "cached": t.cached,
                 "error": t.error} for t in f.turns])


def load_run(engine, run_id: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(forecasts).where(forecasts.c.run_id == run_id)).mappings().all()
    return [dict(r) for r in rows]


def token_totals(engine, run_id: str) -> tuple[int, int]:
    """Read back from the turns table. `compare` cross-checks this against the sum of the
    OTel spans -- two independent accounting paths that must agree, because a token count
    that only exists in one place is a number nobody can contradict."""
    with engine.connect() as conn:
        rows = conn.execute(select(agent_turns.c.tokens_in, agent_turns.c.tokens_out)
                            .where(agent_turns.c.run_id == run_id)).all()
    return sum(r[0] for r in rows), sum(r[1] for r in rows)
