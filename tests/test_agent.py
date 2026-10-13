from cartwise.agent import FALLBACK, Agent
from cartwise.llm.base import LLMResponse, ToolCall


def test_agent_answers_from_tools(agent):
    result = agent.run("Noise cancelling headphones under $100")
    assert [t.name for t in result.tool_calls] == ["search_products"]
    assert "$89.99" in result.answer


def test_multi_step_add_to_cart(agent):
    result = agent.run("Add 2 Nomad Sling Bags to my cart")
    assert [t.name for t in result.tool_calls] == ["search_products", "add_to_cart"]
    assert agent.toolbox.cart.snapshot()["subtotal"] == 70.0


def test_cart_persists_across_turns(agent):
    agent.run("Add the Morrow Pour-Over Kettle to my cart")
    result = agent.run("What's in my cart?")
    assert "Morrow Pour-Over Kettle" in result.answer and "$39.00" in result.answer


def test_tool_errors_are_relayed_not_raised(agent):
    result = agent.run("Add 8 Morrow Pour-Over Kettles to my cart")
    assert "Limit is 5" in result.answer


class LoopingLLM:
    """A misbehaving model that never stops calling tools."""
    provider, model = "test", "loop"

    def generate(self, system, messages, tools):
        return LLMResponse(tool_calls=[ToolCall("view_cart", {})])


def test_max_steps_guard(toolbox):
    result = Agent(LoopingLLM(), toolbox, max_steps=3).run("hello")
    assert result.answer == FALLBACK
    assert result.steps == 3 and "max_steps_exceeded" in result.flags


def test_spans_cover_turn_model_and_tools(agent, spans):
    _, exporter = spans
    agent.run("Compare the Kestrel Air 14 and the Kestrel Pro 16")
    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("agent.run") == 1
    assert "llm.generate" in names and "tool.compare_products" in names
    root = next(s for s in exporter.get_finished_spans() if s.name == "agent.run")
    assert root.attributes["cartwise.steps"] == 3
    model_span = next(s for s in exporter.get_finished_spans() if s.name == "llm.generate")
    assert model_span.attributes["gen_ai.system"] == "scripted"
