import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cartwise.agent import Agent
from cartwise.cart import Cart
from cartwise.catalog import Catalog
from cartwise.llm.scripted import ScriptedLLM
from cartwise.retrieval import BM25Retriever
from cartwise.tools import Toolbox


@pytest.fixture
def catalog():
    return Catalog.from_json()


@pytest.fixture
def toolbox(catalog):
    return Toolbox(catalog, BM25Retriever.from_policies(), Cart())


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def agent(toolbox, spans):
    provider, _ = spans
    return Agent(ScriptedLLM(), toolbox, tracer=provider.get_tracer("test"))
