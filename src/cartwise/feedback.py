"""Feedback loop: log each run, collect thumbs up/down, and promote bad answers into eval cases.

Promoted cases start as drafts. A developer fills in what the right answer must contain,
flips status to "active", and from then on the case guards against that regression in CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .agent import AgentResult

RUNS_DIR = Path("runs")
TRANSCRIPTS = RUNS_DIR / "transcripts.jsonl"
FEEDBACK = RUNS_DIR / "feedback.jsonl"


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def log_run(question: str, result: AgentResult, path: Path = TRANSCRIPTS) -> None:
    _append(path, {"trace_id": result.trace_id, "ts": time.time(), "input": question, "answer": result.answer,
                   "tools": [t.name for t in result.tool_calls], "flags": result.flags})


def add_feedback(trace_id: str, rating: str, note: str = "", path: Path = FEEDBACK) -> dict:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    row = {"trace_id": trace_id, "rating": rating, "note": note, "ts": time.time()}
    _append(path, row)
    return row


def promote(transcripts: Path = TRANSCRIPTS, feedback: Path = FEEDBACK,
            out: Path = Path("evals/regressions.jsonl")) -> list[dict]:
    """Turn each thumbs-down into a draft regression case (once per trace)."""
    runs = {r["trace_id"]: r for r in _read(transcripts)}
    existing = {c.get("source_trace") for c in _read(out)}
    created = []
    for fb in _read(feedback):
        run = runs.get(fb["trace_id"])
        if fb["rating"] != "down" or not run or fb["trace_id"] in existing:
            continue
        case = {"id": f"regression-{fb['trace_id'][:8]}", "input": run["input"], "status": "draft",
                "source_trace": fb["trace_id"], "notes": fb["note"], "bad_answer": run["answer"],
                "must_include": [], "must_not_include": [], "tags": ["regression", "from-feedback"]}
        _append(out, case)
        existing.add(fb["trace_id"])
        created.append(case)
    return created
