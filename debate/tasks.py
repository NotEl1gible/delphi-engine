"""Celery, because a forecast is minutes long and the API must not hold the connection.

Six agents over up to five rounds is thirty-six sequential provider calls under someone
else's rate limit. Three things follow, and none of them is decorative:

- `/forecast` enqueues and returns a job id. A request that blocks for four minutes is a
  request that times out at every proxy between the caller and the worker.
- Retries are bounded and **jittered**. Every worker retrying a rate-limit at the same
  backoff reproduces the burst that caused it.
- Concurrency is capped in one place. The cap is what keeps a sweep inside the provider's
  limit instead of discovering it.

`celery_eager` runs tasks inline, which is what the unit tests and single-machine runs use;
the integration job runs a real worker against a real broker.
"""
from __future__ import annotations

import json

from celery import Celery

from .config import Settings, get_settings


def build_app(s: Settings | None = None) -> Celery:
    s = s or get_settings()
    app = Celery("debate", broker=s.celery_broker_url, backend=s.celery_result_backend)
    app.conf.update(
        task_always_eager=s.celery_eager,
        task_eager_propagates=True,
        task_acks_late=True,                  # a worker that dies mid-forecast redelivers
        worker_prefetch_multiplier=1,         # long tasks: never hoard a queue
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        task_default_retry_delay=10,
        task_time_limit=60 * 20,
    )
    return app


app = build_app()


@app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_jitter=True,
          max_retries=3, name="debate.forecast")
def forecast_task(self, question: dict, arm: str = "panel", anchor: float | None = None,
                  evidence: bool = False, roster_variant: str = "designed",
                  planted: bool = False, overrides: dict | None = None) -> str:
    """Run one forecast. Returns the serialised Forecast.

    Imports live inside the task so that importing this module -- which the API does at
    start-up -- never drags in the model stack. A web process that cannot start because a
    worker dependency is missing is an outage caused by an import.
    """
    from . import tracing as tracing_mod
    from .cache import build_cache
    from .evidence import offline_evidence, search
    from .panel import forecast as run_forecast
    from .panel import load_or_identity
    from .providers import build_provider
    from .schemas import Question

    s = get_settings(**(overrides or {}))
    q = Question(**question)
    provider = build_provider(s)
    ev = None
    if evidence:
        ev = (search(q.text, api_key=s.tavily_api_key) if s.provider == "litellm"
              else offline_evidence(q.text))
    f = run_forecast(q, settings=s, provider=provider,
                     calibrator=load_or_identity(s.calibrator_path),
                     tracing=tracing_mod.build(s) if s.otlp_endpoint else None,
                     cache=build_cache(s, s.panel_model), roster_variant=roster_variant,
                     planted=planted, anchor=anchor, evidence=ev, arm=arm)
    return json.dumps(f.model_dump())
