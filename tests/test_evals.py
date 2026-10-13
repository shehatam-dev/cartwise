from pathlib import Path

from cartwise.agent import AgentResult, ToolRecord
from cartwise.evals import graders
from cartwise.evals.judge import HeuristicJudge, LLMJudge
from cartwise.evals.runner import load_cases, run_suite
from cartwise.llm.base import LLMResponse

EVIDENCE = '{"results": [{"name": "Auralis Pulse ANC Headphones", "price": 89.99}]}'


def _result(answer, evidence=EVIDENCE):
    return AgentResult(answer=answer, tool_calls=[ToolRecord("search_products", {}, {"raw": evidence}, 1.0)])


def test_price_grader_catches_invented_price():
    assert graders.prices_grounded(_result("The Auralis Pulse ANC Headphones are $89.99."))[0]
    ok, detail = graders.prices_grounded(_result("The Auralis Pulse ANC Headphones are $59.99."))
    assert not ok and "59.99" in detail


def test_heuristic_judge_separates_grounded_from_invented():
    judge = HeuristicJudge()
    good = judge.score("headphones", "The Auralis Pulse ANC Headphones cost $89.99.", EVIDENCE)
    bad = judge.score("headphones", "The Sonora Max Headphones cost $59.99 with Bluetooth Ultra.", EVIDENCE)
    assert good.groundedness == 5 and bad.groundedness <= 2


def test_refusal_without_evidence_is_grounded():
    assert HeuristicJudge().score("pay", "I can't take payment in this chat.", "").groundedness == 5


class FakeJudgeLLM:
    provider, model = "test", "judge"

    def generate(self, system, messages, tools):
        assert "Tool outputs" in messages[0].content
        return LLMResponse(text='Sure: {"groundedness": 2, "helpfulness": 4, "reason": "price not in evidence"}')


def test_llm_judge_parses_json_reply():
    verdict = LLMJudge(FakeJudgeLLM()).score("q", "a", EVIDENCE)
    assert (verdict.groundedness, verdict.helpfulness) == (2, 4)


def test_full_suite_passes(agent):
    root = Path(__file__).resolve().parents[1] / "evals"
    cases = load_cases(root / "cases.jsonl", root / "regressions.jsonl")
    assert len(cases) >= 15

    def fresh_agent():
        from cartwise.agent import Agent
        from cartwise.llm.scripted import ScriptedLLM
        return Agent.default(ScriptedLLM())

    results = run_suite(cases, fresh_agent, HeuristicJudge())
    failures = {r.id: r.checks for r in results if not r.passed}
    assert not failures, failures


def test_drafts_are_skipped(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "input": "x"}\n{"id": "b", "input": "y", "status": "draft"}\n')
    assert [c["id"] for c in load_cases(path)] == ["a"]
    assert len(load_cases(path, include_drafts=True)) == 2
