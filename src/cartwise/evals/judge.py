"""LLM-as-judge scoring, plus an offline heuristic judge with the same interface."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from ..llm.base import LLMClient, Message
from ..text import tokenize

JUDGE_PROMPT = """You grade a shopping assistant's answer.

Question: {question}

Tool outputs the assistant saw (the only allowed source of facts):
{evidence}

Answer:
{answer}

Score each from 1 to 5:
- groundedness: every product, price, stock and policy claim is supported by the tool outputs
  (5 = fully supported, 1 = invented facts). A refusal or a clarifying question with no factual
  claims is fully grounded.
- helpfulness: the answer addresses the question directly (5 = fully, 1 = not at all).
Reply with JSON only: {{"groundedness": n, "helpfulness": n, "reason": "<one sentence>"}}"""


@dataclass
class Verdict:
    groundedness: int
    helpfulness: int
    reason: str


class Judge(Protocol):
    def score(self, question: str, answer: str, evidence: str) -> Verdict: ...


class LLMJudge:
    """Asks a model (no tools) to grade the answer against the tool outputs."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, question: str, answer: str, evidence: str) -> Verdict:
        prompt = JUDGE_PROMPT.format(question=question, evidence=evidence or "(none)", answer=answer)
        raw = self.llm.generate("You are a strict, fair evaluator.", [Message("user", prompt)], []).text
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return Verdict(1, 1, f"Judge returned no JSON: {raw[:80]}")
        data = json.loads(match.group(0))
        return Verdict(int(data["groundedness"]), int(data["helpfulness"]), str(data.get("reason", "")))


class HeuristicJudge:
    """Offline approximation of groundedness: are the answer's names and numbers in the evidence?

    Claim tokens are numbers and capitalized words (product names, brands). Template wording
    ("Here are the best matches") makes no claims, so it is ignored.
    """

    _NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

    def _claims(self, answer: str) -> list[str]:
        text = re.sub(r"/5\b", "", answer)  # rating scale, not a claim
        claims = [str(round(float(n.replace(",", "")), 2)) for n in self._NUM.findall(text)]
        for sentence in re.split(r"(?<=[.!?:])\s+|\n", text):
            words = re.findall(r"[A-Za-z][\w-]*", sentence)
            claims += [w.lower() for w in words[1:] if w[0].isupper() and w.lower() not in {"i", "i'm"}]
        return claims

    def score(self, question: str, answer: str, evidence: str) -> Verdict:
        claims = self._claims(answer)
        support = {w.lower() for w in re.findall(r"[A-Za-z][\w-]*", evidence)}
        support |= {str(round(float(n.replace(",", "")), 2)) for n in self._NUM.findall(evidence)}
        if not claims:
            groundedness, detail = 5, "no factual claims"
        else:
            ratio = sum(c in support for c in claims) / len(claims)
            groundedness = 5 if ratio >= 0.9 else 4 if ratio >= 0.75 else 3 if ratio >= 0.5 else 2 if ratio >= 0.25 else 1
            detail = f"{ratio:.0%} of {len(claims)} names and numbers found in tool outputs"
        q_overlap = set(tokenize(question)) & set(tokenize(answer))
        helpfulness = 5 if len(q_overlap) >= 2 else 4 if q_overlap else 3
        return Verdict(groundedness, helpfulness, detail)
