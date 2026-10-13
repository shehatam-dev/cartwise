from __future__ import annotations

import os

from .base import LLMClient, LLMResponse, Message, ToolCall


def get_client(provider: str | None = None) -> LLMClient:
    """`scripted` runs offline and deterministically; `gemini` needs GEMINI_API_KEY."""
    provider = provider or os.environ.get("CARTWISE_PROVIDER", "scripted")
    if provider == "scripted":
        from .scripted import ScriptedLLM
        return ScriptedLLM()
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient()
    raise ValueError(f"Unknown provider '{provider}'. Use 'scripted' or 'gemini'.")


__all__ = ["LLMClient", "LLMResponse", "Message", "ToolCall", "get_client"]
