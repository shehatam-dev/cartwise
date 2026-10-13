"""Checks the Gemini request builder without calling the API."""

import pytest

pytest.importorskip("google.genai")

from cartwise.llm.base import Message, ToolCall  # noqa: E402
from cartwise.llm.gemini import GeminiClient  # noqa: E402


def test_tool_results_are_grouped_into_one_user_turn():
    client = GeminiClient(api_key="test-key")
    messages = [
        Message("user", "hi"),
        Message("assistant", "", tool_calls=[ToolCall("view_cart", {}), ToolCall("search_products", {"query": "x"})]),
        Message("tool", '{"cart": {}}', name="view_cart"),
        Message("tool", '{"results": []}', name="search_products"),
    ]
    contents = client._contents(messages)
    assert [(c.role, len(c.parts)) for c in contents] == [("user", 1), ("model", 2), ("user", 2)]
    assert contents[1].parts[0].function_call.name == "view_cart"
    assert contents[2].parts[1].function_response.name == "search_products"
