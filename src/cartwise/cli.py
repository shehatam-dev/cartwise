"""Command line: chat, ask, eval and feedback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import feedback
from .agent import Agent
from .evals.judge import HeuristicJudge, LLMJudge
from .evals.runner import load_cases, run_suite, summarize
from .llm import get_client
from .tracing import flush, setup_tracing


def _agent(provider: str | None) -> Agent:
    return Agent.default(get_client(provider), tracer=setup_tracing())


def _print_result(result) -> None:
    print(result.answer)
    tools = ", ".join(t.name for t in result.tool_calls) or "none"
    flags = f" | flags: {', '.join(result.flags)}" if result.flags else ""
    print(f"\n[tools: {tools}{flags} | trace {result.trace_id[:12]}]")


def cmd_ask(args) -> int:
    agent = _agent(args.provider)
    result = agent.run(args.question)
    feedback.log_run(args.question, result)
    _print_result(result)
    flush()
    return 1 if "llm_error" in result.flags else 0


def cmd_chat(args) -> int:
    agent = _agent(args.provider)
    print("Cartwise. Ask about products, stock or policies. Type 'quit' to exit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if question.lower() in ("quit", "exit"):
            break
        if question:
            result = agent.run(question)
            feedback.log_run(question, result)
            print("cartwise> ", end="")
            _print_result(result)
    flush()
    return 0


def cmd_eval(args) -> int:
    paths = args.cases or ["evals/cases.jsonl", "evals/regressions.jsonl"]
    cases = load_cases(*paths, include_drafts=args.include_drafts)
    if args.tag:
        cases = [c for c in cases if args.tag in c.get("tags", [])]
    if not cases:
        print(f"No eval cases found in {', '.join(map(str, paths))}"
              + (f" with tag '{args.tag}'" if args.tag else "")
              + ". Run from the repo root or pass case files.", file=sys.stderr)
        return 2
    judge = None
    if args.judge == "heuristic":
        judge = HeuristicJudge()
    elif args.judge == "llm":
        judge = LLMJudge(get_client(args.judge_provider))
    tracer = setup_tracing()
    results = run_suite(cases, lambda: Agent.default(get_client(args.provider), tracer=tracer), judge,
                        report_dir="runs")
    print(summarize(results))
    flush()
    rate = sum(r.passed for r in results) / len(results) if results else 0
    return 0 if rate >= args.min_pass else 1


def cmd_feedback(args) -> int:
    if args.action == "add":
        feedback.add_feedback(args.trace_id, args.rating, args.note or "")
        print("Feedback saved.")
    else:
        created = feedback.promote(out=Path(args.out))
        print(f"Created {len(created)} draft regression case(s) in {args.out}.")
        for case in created:
            print(f"  {case['id']}: {case['input']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cartwise", description=__doc__)
    p.add_argument("--provider", choices=["scripted", "gemini"], default=None,
                   help="Model provider (default: $CARTWISE_PROVIDER or scripted)")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="Interactive chat").set_defaults(func=cmd_chat)

    ev = sub.add_parser("eval", help="Run the eval suite")
    ev.add_argument("cases", nargs="*", help="Case files (default: evals/cases.jsonl evals/regressions.jsonl)")
    ev.add_argument("--judge", choices=["none", "heuristic", "llm"], default="heuristic")
    ev.add_argument("--judge-provider", choices=["scripted", "gemini"], default="gemini")
    ev.add_argument("--tag", help="Only run cases with this tag")
    ev.add_argument("--min-pass", type=float, default=1.0, help="Exit non-zero below this pass rate")
    ev.add_argument("--include-drafts", action="store_true")
    ev.set_defaults(func=cmd_eval)

    fb = sub.add_parser("feedback", help="Record feedback or promote it into regression cases")
    fb_sub = fb.add_subparsers(dest="action", required=True)
    add = fb_sub.add_parser("add")
    add.add_argument("trace_id")
    add.add_argument("rating", choices=["up", "down"])
    add.add_argument("--note")
    promote = fb_sub.add_parser("promote")
    promote.add_argument("--out", default="evals/regressions.jsonl")
    fb.set_defaults(func=cmd_feedback)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
