"""The product surface.

`POST /forecast` enqueues and returns a job id, because a panel run takes minutes and a
request that blocks that long dies at the first proxy. `GET /forecast/{job_id}` collects it.

`/metrics` is hand-rolled Prometheus text rather than a client library. One fewer dependency,
and the four counters here are the four the engine actually has: forecasts served, abstentions,
tokens, dollars. A metrics endpoint that exports fifty numbers nobody reads is not
observability, it is furniture.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from .config import get_settings
from .schemas import Forecast, ForecastRequest, Question
from .tasks import build_app, forecast_task

COUNTERS: dict[str, float] = {
    "debate_forecasts_total": 0.0,
    "debate_abstentions_total": 0.0,
    "debate_tokens_in_total": 0.0,
    "debate_tokens_out_total": 0.0,
    "debate_cost_usd_total": 0.0,
    "debate_rounds_total": 0.0,
}


def record(f: Forecast) -> None:
    COUNTERS["debate_forecasts_total"] += 1
    COUNTERS["debate_abstentions_total"] += 1 if f.decision == "abstain" else 0
    COUNTERS["debate_tokens_in_total"] += f.tokens_in
    COUNTERS["debate_tokens_out_total"] += f.tokens_out
    COUNTERS["debate_cost_usd_total"] += f.usd
    COUNTERS["debate_rounds_total"] += f.rounds_used


def create_app() -> FastAPI:
    s = get_settings()
    celery = build_app(s)
    api = FastAPI(title="llm-debate-engine",
                  description="LLM Delphi panels with fitted calibration and the right to "
                              "abstain.",
                  version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "provider": s.provider, "model": s.panel_model,
                "n_agents": s.n_agents, "max_rounds": s.max_rounds,
                "abstain_tau": s.abstain_tau, "eager": s.celery_eager}

    @api.post("/forecast", status_code=202)
    def submit(req: ForecastRequest) -> dict[str, Any]:
        q = Question(id=f"live-{uuid.uuid4().hex[:8]}", text=req.question,
                     resolution_date=req.resolution_date or "unknown", split="live")
        overrides: dict[str, Any] = {}
        if req.n_agents:
            overrides["n_agents"] = req.n_agents
        if req.max_rounds:
            overrides["max_rounds"] = req.max_rounds
        async_result = forecast_task.delay(q.model_dump(), "panel", req.anchor,
                                           req.evidence, "designed", False, overrides)
        # Eager mode resolves inline, which is what makes the endpoint testable without a
        # broker; the status field says which path was taken so a caller is never guessing.
        if s.celery_eager:
            f = Forecast(**json.loads(async_result.get()))
            record(f)
            return {"job_id": async_result.id, "status": "done",
                    "forecast": f.model_dump()}
        return {"job_id": async_result.id, "status": "queued",
                "poll": f"/forecast/{async_result.id}"}

    @api.get("/forecast/{job_id}")
    def collect(job_id: str) -> dict[str, Any]:
        res = celery.AsyncResult(job_id)
        if not res.ready():
            return {"job_id": job_id, "status": res.status.lower()}
        if res.failed():
            raise HTTPException(status_code=500, detail=str(res.result))
        f = Forecast(**json.loads(res.get()))
        record(f)
        return {"job_id": job_id, "status": "done", "forecast": f.model_dump()}

    @api.get("/metrics")
    def metrics() -> Response:
        lines = []
        for name, value in COUNTERS.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return api


app = create_app()
