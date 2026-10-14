"""Google Gemini provider using the google-genai SDK's function calling."""

from __future__ import annotations

import json
import os

from .base import LLMResponse, Message, ToolCall


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.2):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the Gemini extra: pip install 'cartwise[gemini]'") from exc
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY to use the Gemini provider.")
        self._types = types
        self._client = genai.Client(api_key=key)
        self.model = model or os.environ.get("CARTWISE_MODEL", "gemini-2.5-flash")
        self.temperature = temperature

    def _contents(self, messages: list[Message]) -> list:
        t, contents, pending = self._types, [], []
        for m in messages:
            if m.role == "tool":
                # Gemini expects all results for one model turn together, sent as a user turn.
                pending.append(t.Part.from_function_response(name=m.name, response={"result": json.loads(m.content)}))
                continue
            if pending:
                contents.append(t.Content(role="user", parts=pending))
                pending = []
            if m.role == "user":
                contents.append(t.Content(role="user", parts=[t.Part(text=m.content)]))
            else:
                parts = [t.Part(text=m.content)] if m.content else []
                # Thinking models sign each call; replaying the signature keeps their reasoning
                # across steps, and Gemini 3 rejects function-call history without it.
                parts += [t.Part(function_call=t.FunctionCall(name=c.name, args=c.args),
                                 thought_signature=c.extra.get("thought_signature")) for c in m.tool_calls]
                contents.append(t.Content(role="model", parts=parts))
        if pending:
            contents.append(t.Content(role="user", parts=pending))
        return contents

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        t = self._types
        declarations = [t.FunctionDeclaration(name=s["name"], description=s["description"],
                                              parameters_json_schema=s["parameters"]) for s in tools]
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[t.Tool(function_declarations=declarations)] if declarations else None,
            # The agent loop runs tools itself, so it can validate, trace and guard every call.
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._client.models.generate_content(model=self.model, contents=self._contents(messages), config=config)
        return self._parse(resp)

    def _parse(self, resp) -> LLMResponse:
        # Read the parts rather than `resp.function_calls`, which drops each call's thought signature.
        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts if content else None) or []
        calls = [ToolCall(name=p.function_call.name, args=dict(p.function_call.args or {}),
                          extra={"thought_signature": p.thought_signature} if p.thought_signature else {})
                 for p in parts if p.function_call]
        usage = resp.usage_metadata
        return LLMResponse(
            text="" if calls else (resp.text or ""),
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
