"""Provider-neutral message types and the LLM client protocol."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # Opaque provider data that must be sent back with the call on the next request,
    # e.g. Gemini's thought signature. The agent never reads it.
    extra: dict = field(default_factory=dict)


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None          # tool name, for role == "tool"
    tool_call_id: str | None = None  # links a tool result to its call


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse: ...
