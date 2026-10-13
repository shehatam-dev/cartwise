"""Run eval cases through a fresh agent each and grade them.

A case is one JSON object per line:
  {"id": "...", "input": "...", "expect_tools": [...], "no_tools": false,
   "must_include": [...], "must_not_include": [...], "expect_flags": [...],
   "min_groundedness": 4, "tags": [...], "status": "active" | "draft"}
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..agent import Agent
from . import graders
from .judge import Judge


@dataclass
class CaseResult:
    id: str
    passed: bool
    checks: dict[str, tuple[bool, str]]
    answer: str
    groundedness: int | None = None
    helpfulness: int | None = None
    tags: list[str] = field(default_factory=list)


def load_cases(*paths: str | Path, include_drafts: bool = False) -> list[dict]:
    cases = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("//"):
                case = json.loads(line)
                if include_drafts or case.get("status", "active") != "draft":
                    cases.append(case)
    return cases


def grade(case: dict, agent_factory: Callable[[], Agent], judge: Judge | None) -> CaseResult:
    result = agent_factory().run(case["input"])
    checks: dict[str, tuple[bool, str]] = {}
    if case.get("expect_tools"):
        checks["tools_called"] = graders.tools_called(result, case["expect_tools"])
    if case.get("no_tools"):
        checks["no_tools"] = graders.no_tools(result)
    if case.get("must_include"):
        checks["must_include"] = graders.must_include(result, case["must_include"])
    if case.get("must_not_include"):
        checks["must_not_include"] = graders.must_not_include(result, case["must_not_include"])
    if case.get("expect_flags"):
        checks["flags"] = graders.has_flags(result, case["expect_flags"])
    checks["prices_grounded"] = graders.prices_grounded(result)

    out = CaseResult(case["id"], True, checks, result.answer, tags=case.get("tags", []))
    if judge is not None:
        verdict = judge.score(case["input"], result.answer, result.tool_outputs_text())
        out.groundedness, out.helpfulness = verdict.groundedness, verdict.helpfulness
        minimum = case.get("min_groundedness", 4)
        checks["judge_groundedness"] = (verdict.groundedness >= minimum,
                                        f"{verdict.groundedness}/5 (min {minimum}): {verdict.reason}")
    out.passed = all(ok for ok, _ in checks.values())
    return out


def run_suite(cases: list[dict], agent_factory: Callable[[], Agent], judge: Judge | None = None,
              report_dir: str | Path | None = None) -> list[CaseResult]:
    results = [grade(c, agent_factory, judge) for c in cases]
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        path = Path(report_dir) / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return results


def summarize(results: list[CaseResult]) -> str:
    lines = []
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        judged = f"  g={r.groundedness} h={r.helpfulness}" if r.groundedness is not None else ""
        lines.append(f"{mark}  {r.id}{judged}")
        for name, (ok, detail) in r.checks.items():
            if not ok:
                lines.append(f"      {name}: {detail}")
    passed = sum(r.passed for r in results)
    rate = passed / len(results) if results else 0
    judged = [r for r in results if r.groundedness is not None]
    avg = f", mean groundedness {sum(r.groundedness for r in judged) / len(judged):.2f}" if judged else ""
    lines.append(f"\n{passed}/{len(results)} passed ({rate:.0%}){avg}")
    return "\n".join(lines)
