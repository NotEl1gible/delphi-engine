"""The infrastructure layer, with no containers.

SQLite stands in for Postgres, `fakeredis` for Redis, Celery runs eager and the OTel exporter
is in memory. What that CANNOT prove is that Postgres accepts the DDL: SQLAlchemy will compile
BIGSERIAL, JSONB and TIMESTAMPTZ with no server anywhere in sight, so a passing test here says
nothing about a real database. The CI integration job runs the same statements against a real
Postgres and a real Redis with a live worker, and that is where those claims are settled.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from delphi import store, tracing
from delphi.cache import Cache, key_of
from delphi.config import get_settings
from delphi.panel import forecast, make_personas
from delphi.personas import PLANTED
from delphi.providers import Ask, build_provider
from delphi.schemas import Question

Q = Question(id="Q-test", text="Will the sample question resolve YES?",
             resolution_date="2026-09-01", outcome=1, split="dev")


@pytest.fixture
def settings():
    return get_settings(provider="mock", cache_enabled=False, celery_eager=True,
                        database_url="sqlite://", otlp_endpoint="")


# ---------------------------------------------------------------- store
def test_the_same_schema_compiles_for_both_databases():
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable
    pg = str(CreateTable(store.forecasts).compile(dialect=postgresql.dialect()))
    lite = str(CreateTable(store.forecasts).compile(dialect=sqlite.dialect()))
    assert "JSONB" in pg and "JSONB" not in lite
    assert "BIGSERIAL" in pg or "BIGINT" in pg
    assert "TIMESTAMP" in str(CreateTable(store.runs).compile(dialect=postgresql.dialect()))


def test_a_forecast_round_trips_through_the_store(settings):
    eng = store.build_engine("sqlite://")
    store.create_all(eng)
    store.start_run(eng, run_id="r1", arm="panel", provider="mock", model="mock",
                    settings={"n_agents": settings.n_agents}, n_questions=1)
    f = forecast(Q, settings=settings, provider=build_provider(settings))
    store.save_forecast(eng, "r1", f, outcome=Q.outcome)
    rows = store.load_run(eng, "r1")
    assert len(rows) == 1
    assert rows[0]["question_id"] == "Q-test"
    assert rows[0]["p_raw"] == pytest.approx(f.p_raw)
    assert rows[0]["outcome"] == 1
    tin, tout = store.token_totals(eng, "r1")
    assert tin == f.tokens_in and tout == f.tokens_out


def test_an_abstention_stores_a_null_probability_not_a_number(settings):
    """`p` must be NULL when the engine refused. Writing the raw pool there would let a
    downstream consumer read an answer the product declined to give."""
    s = get_settings(provider="mock", abstain_tau=-1.0, database_url="sqlite://",
                     cache_enabled=False)
    eng = store.build_engine("sqlite://")
    store.create_all(eng)
    store.start_run(eng, run_id="r2", arm="panel", provider="mock", model="mock",
                    settings={}, n_questions=1)
    f = forecast(Q, settings=s, provider=build_provider(s))
    assert f.decision == "abstain" and f.p is None
    store.save_forecast(eng, "r2", f)
    assert store.load_run(eng, "r2")[0]["p"] is None


# ---------------------------------------------------------------- cache
def test_the_cache_key_separates_the_arms(settings):
    """The line that keeps an ablation honest. Two arms asking a byte-identical question must
    still miss each other's cache, or their latency, cost and retry behaviour depend on which
    ran first and the measured difference between them is partly execution order."""
    people = make_personas(settings)
    base = {"question": Q, "persona": people[0], "agent_id": "a0", "round": 0,
            "prev": None, "peers": [], "anchor": None, "evidence": None,
            "premortem": False}
    k1 = key_of(Ask(**base, arm="panel"), "m", 7)
    k2 = key_of(Ask(**base, arm="evidence"), "m", 7)
    k3 = key_of(Ask(**base, arm="panel"), "m", 7)
    assert k1 != k2, "arms share a cache entry"
    assert k1 == k3


def test_a_cache_hit_is_billed_at_zero_and_flagged(settings):
    import fakeredis
    people = make_personas(settings)
    cache = Cache(fakeredis.FakeRedis(), "m", 7)
    ask = Ask(question=Q, persona=people[0], agent_id="a0", round=0, prev=None, peers=[],
              anchor=None, evidence=None, premortem=False, arm="panel")
    provider = build_provider(settings)
    verdict, usage = provider.ask(ask)
    usage.usd = 0.5
    cache.put(ask, verdict, usage)
    v2, u2 = cache.get(ask)
    assert v2.probability == verdict.probability
    assert u2.cached is True and u2.usd == 0.0, (
        "replaying the original cost would make a re-run look as expensive as the first")
    assert cache.stats()["hits"] == 1


def test_a_failed_call_is_never_cached(settings):
    import fakeredis

    from delphi.providers import Usage
    from delphi.schemas import AgentVerdict
    people = make_personas(settings)
    cache = Cache(fakeredis.FakeRedis(), "m", 7)
    ask = Ask(question=Q, persona=people[0], agent_id="a0", round=0, prev=None, peers=[],
              anchor=None, evidence=None, premortem=False, arm="panel")
    cache.put(ask, AgentVerdict(probability=0.5), Usage(error="429 rate limited"))
    assert cache.get(ask) is None


# ---------------------------------------------------------------- tracing
def test_every_agent_call_emits_one_genai_span(settings):
    exp = InMemorySpanExporter()
    tr = tracing.build(settings, exporter=exp)
    f = forecast(Q, settings=settings, provider=build_provider(settings), tracing=tr)
    spans = exp.get_finished_spans()
    assert len(spans) == len(f.turns) > 0
    a = spans[0].attributes
    for key in ("gen_ai.operation.name", "gen_ai.request.model",
                "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
        assert key in a, f"{key} missing: the trace is not GenAI-conventional"
    assert {"delphi.round", "delphi.agent_id", "delphi.persona", "delphi.arm"} <= set(a)


def test_span_tokens_reconcile_with_the_forecast(settings):
    """Two independent accounting paths. A token count that exists in only one place is a
    number nobody can contradict, and the stopping rule is chosen from it."""
    exp = InMemorySpanExporter()
    tr = tracing.build(settings, exporter=exp)
    f = forecast(Q, settings=settings, provider=build_provider(settings), tracing=tr)
    from_spans = sum(s.attributes["gen_ai.usage.input_tokens"]
                     for s in exp.get_finished_spans())
    assert from_spans == f.tokens_in


# ---------------------------------------------------------------- panel behaviour
def test_the_blind_round_shows_nobody_anything(settings):
    f = forecast(Q, settings=settings, provider=build_provider(settings))
    first = [snp for snp in f.snapshots if not snp.premortem][0]
    assert first.round == 0
    assert len(first.probabilities) == settings.n_agents


def test_the_premortem_is_a_stage_not_a_round(settings):
    f = forecast(Q, settings=settings, provider=build_provider(settings))
    pms = [snp for snp in f.snapshots if snp.premortem]
    assert len(pms) == 1
    assert f.rounds_used == len([s for s in f.snapshots if not s.premortem])


def test_disabling_the_stopping_rule_runs_every_round(settings):
    """The ablation needs all five rounds on every question. If the stopping rule could not
    be switched off, the round arms would exist only on questions that happened not to
    settle -- a different subset per arm, which is not an ablation."""
    s = get_settings(provider="mock", stop_movement=0.0, max_rounds=5, cache_enabled=False)
    f = forecast(Q, settings=s, provider=build_provider(s))
    assert f.rounds_used == 6, f"expected rounds 0..5, got {f.rounds_used}"
    assert not f.stopped_early


def test_planting_replaces_a_member_rather_than_adding_one(settings):
    """Adding an agent would change the panel SIZE at the same time as its composition, and
    the drift measured afterwards could then be either effect."""
    normal = make_personas(settings, planted=False)
    planted = make_personas(settings, planted=True)
    assert len(normal) == len(planted) == settings.n_agents
    assert planted[0].id == PLANTED.id and normal[0].id != PLANTED.id


# ---------------------------------------------------------------- api
def test_the_api_serves_a_forecast_and_counts_it(settings, monkeypatch):
    monkeypatch.setenv("DELPHI_PROVIDER", "mock")
    monkeypatch.setenv("DELPHI_CELERY_EAGER", "true")
    monkeypatch.setenv("DELPHI_CACHE_ENABLED", "false")
    from delphi import api as api_mod
    client = TestClient(api_mod.create_app())
    assert client.get("/health").json()["ok"] is True
    r = client.post("/forecast", json={"question": "Will this test pass?",
                                       "resolution_date": "2026-12-31"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "done"
    assert body["forecast"]["decision"] in ("forecast", "abstain")
    text = client.get("/metrics").text
    assert "delphi_forecasts_total" in text and "# TYPE" in text


def test_the_task_is_serialisable_json(settings):
    from delphi.tasks import forecast_task
    out = forecast_task.run(Q.model_dump(), "panel", None, False, "designed", False,
                            {"provider": "mock", "cache_enabled": False})
    d = json.loads(out)
    assert d["question_id"] == "Q-test" and "snapshots" in d
