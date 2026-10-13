"""Deterministic checks for one eval case. Each returns (passed, detail)."""

from __future__ import annotations

import re

from ..agent import AgentResult

_PRICE = re.compile(r"\$(\d[\d,]*(?:\.\d{1,2})?)")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {round(float(n.replace(",", "")), 2) for n in _NUMBER.findall(text)}


def tools_called(result: AgentResult, expected: list[str]) -> tuple[bool, str]:
    called = [t.name for t in result.tool_calls]
    missing = [t for t in expected if t not in called]
    return not missing, f"called {called}" + (f", missing {missing}" if missing else "")


def no_tools(result: AgentResult) -> tuple[bool, str]:
    called = [t.name for t in result.tool_calls]
    return not called, f"called {called}" if called else "no tools called"


def must_include(result: AgentResult, phrases: list[str]) -> tuple[bool, str]:
    missing = [p for p in phrases if p.lower() not in result.answer.lower()]
    return not missing, f"missing {missing}" if missing else "all present"


def must_not_include(result: AgentResult, phrases: list[str]) -> tuple[bool, str]:
    present = [p for p in phrases if p.lower() in result.answer.lower()]
    return not present, f"found {present}" if present else "none present"


def prices_grounded(result: AgentResult) -> tuple[bool, str]:
    """Every dollar amount in the answer must appear in, or be computed by, a tool output."""
    allowed = _numbers(result.tool_outputs_text())
    claimed = {round(float(p.replace(",", "")), 2) for p in _PRICE.findall(result.answer)}
    unsupported = sorted(claimed - allowed)
    return not unsupported, f"unsupported prices {unsupported}" if unsupported else f"{len(claimed)} price(s) grounded"


def has_flags(result: AgentResult, flags: list[str]) -> tuple[bool, str]:
    missing = [f for f in flags if f not in result.flags]
    return not missing, f"flags {result.flags}" + (f", missing {missing}" if missing else "")
