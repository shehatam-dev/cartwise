"""Small text helpers shared by search, retrieval and grading."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from have how i if in is it its me my of on or "
    "the to what when where which with you your we our this that there any some please want need "
    "find show get looking look give tell about".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens with naive plural folding ("laptops" -> "laptop")."""
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if not keep_stopwords and tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def money(value: float) -> str:
    return f"${value:,.2f}"
