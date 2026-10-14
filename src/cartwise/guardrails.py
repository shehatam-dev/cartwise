"""Guardrails for untrusted text.

Product descriptions, reviews and retrieved documents are data, not instructions. Before tool
output reaches the model, strings that look like prompt injection are removed and flagged.
User input is scanned too, and the agent refuses it in code, before any model call.
"""

from __future__ import annotations

import re

_ASK = r"\b(reveal|print|show|repeat|output|tell)\s+(me\s+)?"

# User input that matches is refused outright, so a pattern must not catch a shopper's ordinary words:
# "ignore my previous message, I meant headphones" retracts a message, "repeat the instructions" asks
# about a product. Each wording of "show me your prompt" therefore gets its own rule, ordered by how
# unambiguous it is. `\s+` so doubled spaces and newlines don't slip past.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?(previous|prior|above|earlier|system|initial)\s+"
    r"(instructions|rules|prompts?)",
    _ASK + r"(your|the)\s+(system|initial|hidden)\s+(prompt|instructions)",
    _ASK + r"your\s+original\s+(prompt|instructions)",
    _ASK + r"your\s+prompt\b(?!\s+\w+ing)",                              # not "your prompt pricing"
    _ASK + r"your\s+instructions\b(?!\s+(for|on|to|about|of)\s+(?!me\b|us\b))",  # not "...for returning a laptop"
    _ASK + r"the\s+prompt\b",
    _ASK + r"the\s+instructions\s+(above|you\s+were\s+given|verbatim)",  # not "the instructions in the manual"
    r"\byou\s+are\s+now\s+(in\s+\w+\s+mode|no\s+longer\b|an?\s+(\w+\s+)?(ai|assistant|bot|chatbot|model)\b|dan\b)",
    r"\btell\s+the\s+user\s+(this|that|the)\s+(item|product|order)\s+is\s+free",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_INVISIBLE = re.compile("[​-‍⁠﻿]")  # zero-width characters hidden inside a word
REDACTED = "[removed: possible prompt injection]"
REFUSAL = ("I can't help with that request. I'm here to help you shop: finding products, "
           "checking stock and answering questions about our policies.")


def find_injection(text: str) -> list[str]:
    text = _INVISIBLE.sub("", text)
    return [p.pattern for p in _COMPILED if p.search(text)]


def sanitize(payload):
    """Return (clean_payload, flags). Sentences containing injection text are redacted."""
    flags: list[str] = []

    def clean(value):
        if isinstance(value, str):
            if not find_injection(value):
                return value
            flags.append("tool_output_injection")
            sentences = re.split(r"(?<=[.!?])\s+", value)
            return " ".join(REDACTED if find_injection(s) else s for s in sentences)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(payload), sorted(set(flags))
