"""OpenTelemetry setup.

CARTWISE_TRACING=none     (default) no export
CARTWISE_TRACING=console  print spans to stderr
CARTWISE_TRACING=otlp     export over OTLP/HTTP; configure with the standard OTEL_EXPORTER_OTLP_*
                          variables. Any OTLP backend works, including Langfuse (see README).
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

_configured = False


def setup_tracing(mode: str | None = None, service_name: str = "cartwise") -> trace.Tracer:
    global _configured
    mode = (mode or os.environ.get("CARTWISE_TRACING", "none")).lower()
    if not _configured and mode != "none":
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if mode == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif mode == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            raise ValueError(f"Unknown CARTWISE_TRACING mode '{mode}'")
        trace.set_tracer_provider(provider)
        _configured = True
    return trace.get_tracer("cartwise")


def flush() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
