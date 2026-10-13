import json

from cartwise import feedback
from cartwise.agent import AgentResult


def test_thumbs_down_becomes_one_draft_regression_case(tmp_path):
    transcripts, fb, out = tmp_path / "t.jsonl", tmp_path / "f.jsonl", tmp_path / "r.jsonl"
    feedback.log_run("express shipping?", AgentResult(answer="It's $5.99.", trace_id="abc12345ff"), transcripts)
    feedback.log_run("hi", AgentResult(answer="Hi!", trace_id="def67890aa"), transcripts)
    feedback.add_feedback("abc12345ff", "down", "Quoted the standard fee", fb)
    feedback.add_feedback("def67890aa", "up", "", fb)

    created = feedback.promote(transcripts, fb, out)
    assert len(created) == 1
    case = json.loads(out.read_text().splitlines()[0])
    assert case["status"] == "draft" and case["input"] == "express shipping?"
    assert case["bad_answer"] == "It's $5.99." and case["notes"] == "Quoted the standard fee"

    assert feedback.promote(transcripts, fb, out) == []  # idempotent
