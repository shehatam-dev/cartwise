"""Guardrails for untrusted text.

Product descriptions, reviews and retrieved documents are data, not instructions. Before tool
output reaches the model, strings that look like prompt injection are removed and flagged.
User input is scanned too, so the agent can refuse instead of complying.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(of )?(the )?(previous|prior|above|earlier) instructions",
    r"disregard (all |the )?(previous|prior|system|above)",
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"tell the user (this|that|the) (item|product|order) is free",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
REDACTED = "[removed: possible prompt injection]"


def find_injection(text: str) -> list[str]:
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
