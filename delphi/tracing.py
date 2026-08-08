"""OpenTelemetry spans, one per agent per round, under the GenAI semantic conventions.

This is not observability decoration bolted on at the end. The engine's stopping rule is
"keep debating while a round still moves the estimate more than it costs", and the cost side
of that sentence is read straight off these spans. Remove the tracing and the product loses a
feature, not a dashboard.

The provider is held on an object rather than installed globally with
`set_tracer_provider`, which can only be called once per process. That single line is the
difference between a test suite that can assert on spans and one that cannot: every test gets
its own in-memory exporter instead of fighting over one global.
"""
from __future__ import annotations

from contextlib import contextmanager

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from .config import Settings

# GenAI semantic conventions, so the spans mean the same thing to Langfuse as they would to
# any other OTLP backend. A bespoke attribute schema would make the traces readable by exactly
# one tool, which is the opposite of why one adopts OpenTelemetry.
GEN_AI = {
    "operation": "gen_ai.operation.name",
    "system": "gen_ai.system",
    "model": "gen_ai.request.model",
    "tokens_in": "gen_ai.usage.input_tokens",
    "tokens_out": "gen_ai.usage.output_tokens",
}


class Tracing:
    def __init__(self, provider: TracerProvider):
        self.provider = provider

    def tracer(self, name: str = "delphi"):
        return self.provider.get_tracer(name)

    def shutdown(self) -> None:
        self.provider.shutdown()

    @contextmanager
    def agent_span(self, *, name: str, model: str, system: str, round: int,
                   agent_id: str, persona: str, arm: str):
        with self.tracer().start_as_current_span(name) as span:
            span.set_attribute(GEN_AI["operation"], "chat")
            span.set_attribute(GEN_AI["system"], system)
            span.set_attribute(GEN_AI["model"], model)
            span.set_attribute("delphi.round", round)
            span.set_attribute("delphi.agent_id", agent_id)
            span.set_attribute("delphi.persona", persona)
            span.set_attribute("delphi.arm", arm)
            yield span


def record_usage(span, usage, probability: float | None = None) -> None:
    span.set_attribute(GEN_AI["tokens_in"], usage.tokens_in)
    span.set_attribute(GEN_AI["tokens_out"], usage.tokens_out)
    span.set_attribute("delphi.usd", usage.usd)
    span.set_attribute("delphi.cached", usage.cached)
    if probability is not None:
        span.set_attribute("delphi.probability", probability)
    if usage.error:
        span.set_attribute("delphi.error", usage.error)


def build(s: Settings, exporter=None) -> Tracing:
    provider = TracerProvider(resource=Resource.create({"service.name": s.service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif s.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        headers = dict(kv.split("=", 1) for kv in s.otlp_headers.split(",") if "=" in kv)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otlp_endpoint, headers=headers)))
    return Tracing(provider)
