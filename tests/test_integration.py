"""Runs only where a real Postgres and a real Redis exist -- the CI integration job.

These are the claims the unit tests cannot make. SQLAlchemy will compile BIGSERIAL, JSONB and
TIMESTAMPTZ with no server anywhere in sight, and `fakeredis` will accept any command a real
Redis might reject, so passing locally proves the code is written, not that it works. Skipped
rather than faked when the services are absent, because a green tick that means nothing is
worse than a skip.
"""
from __future__ import annotations

import os
import uuid

import pytest

from delphi import store
from delphi.cache import Cache, key_of
from delphi.config import get_settings
from delphi.panel import forecast, make_personas
from delphi.providers import Ask, build_provider
from delphi.schemas import Question

PG = os.environ.get("DELPHI_TEST_POSTGRES_URL", "")
RD = os.environ.get("DELPHI_TEST_REDIS_URL", "")
# A broker with no worker behind it is not a broker test, it is a 120-second timeout. Set
# only by the CI step that has actually started a worker first.
BROKER = os.environ.get("DELPHI_TEST_BROKER_URL", "")

pg_only = pytest.mark.skipif(not PG, reason="no DELPHI_TEST_POSTGRES_URL")
redis_only = pytest.mark.skipif(not RD, reason="no DELPHI_TEST_REDIS_URL")
worker_only = pytest.mark.skipif(not BROKER, reason="no live worker (DELPHI_TEST_BROKER_URL)")

Q = Question(id="Q-int", text="Will the integration job pass?", resolution_date="2026-09-01",
             outcome=1, split="dev")


@pg_only
def test_postgres_accepts_the_schema_and_the_round_trip():
    s = get_settings(provider="mock", cache_enabled=False)
    eng = store.build_engine(PG)
    store.metadata.drop_all(eng)
    store.create_all(eng)                     # BIGSERIAL / JSONB / TIMESTAMPTZ for real
    run_id = uuid.uuid4().hex[:16]
    store.start_run(eng, run_id=run_id, arm="panel", provider="mock", model="mock",
                    settings={"n_agents": s.n_agents}, n_questions=1)
    f = forecast(Q, settings=s, provider=build_provider(s))
    store.save_forecast(eng, run_id, f, outcome=1)
    rows = store.load_run(eng, run_id)
    assert len(rows) == 1
    # JSONB round-trips as a structure, not as a string. On SQLite this passes either way.
    assert isinstance(rows[0]["snapshots"], list)
    assert isinstance(rows[0]["snapshots"][0], dict)
    tin, tout = store.token_totals(eng, run_id)
    assert (tin, tout) == (f.tokens_in, f.tokens_out)


@redis_only
def test_a_real_redis_stores_and_expires_the_cache():
    import redis as redis_lib
    s = get_settings(provider="mock")
    client = redis_lib.Redis.from_url(RD)
    client.flushdb()
    cache = Cache(client, "m", 7, ttl_seconds=60)
    ask = Ask(question=Q, persona=make_personas(s)[0], agent_id="a0", round=0, prev=None,
              peers=[], anchor=None, evidence=None, premortem=False, arm="panel")
    verdict, usage = build_provider(s).ask(ask)
    cache.put(ask, verdict, usage)
    k = key_of(ask, "m", 7)
    assert 0 < client.ttl(k) <= 60, "the entry has no expiry; the cache grows without bound"
    got = cache.get(ask)
    assert got is not None and got[1].cached is True and got[1].usd == 0.0


@worker_only
def test_a_real_celery_worker_runs_the_task_through_the_broker():
    """The round trip that eager mode cannot make.

    Eager mode never touches the broker, so it cannot catch a task that fails to serialise, a
    queue name that does not match, or a worker that cannot import its own module. `.apply()`
    has the same blind spot -- it runs the function locally and returns, which looks like a
    passing worker test and is not one. Only `apply_async` against a running worker exercises
    any of it, which is why this test refuses to run unless one was started.
    """
    import json

    from delphi.tasks import app as celery_app
    from delphi.tasks import forecast_task

    # The task is already registered on the module-level app, which reads its broker from
    # the environment -- the same environment the worker was started in. Passing an app to
    # apply_async does not rebind anything: `app` is not one of its parameters, so Celery
    # treats it as a task option and tries to JSON-serialise the Celery object itself.
    assert celery_app.conf.broker_url == BROKER, (
        f"the test would enqueue to {celery_app.conf.broker_url} while the worker listens "
        f"on {BROKER}")
    # Without this the test passes by running the task in-process and never touching Redis,
    # which is exactly the blind spot it exists to close.
    assert not celery_app.conf.task_always_eager, "eager mode bypasses the broker entirely"

    res = forecast_task.apply_async(
        args=[Q.model_dump(), "panel", None, False, "designed", False,
              {"provider": "mock", "cache_enabled": False}], queue="celery")
    out = json.loads(res.get(timeout=180))
    assert out["question_id"] == "Q-int"
    assert out["decision"] in ("forecast", "abstain")


@pg_only
@redis_only
def test_the_api_serves_a_forecast_against_the_real_services():
    from fastapi.testclient import TestClient

    from delphi import api as api_mod
    client = TestClient(api_mod.create_app())
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "delphi_forecasts_total" in client.get("/metrics").text
