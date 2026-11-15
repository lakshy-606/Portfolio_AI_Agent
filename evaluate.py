"""Run the ground-truth questions through the agent and report accuracy.

Usage:
    python evaluate.py                     # all questions
    python evaluate.py --ids 1 7 9         # a subset
    python evaluate.py --output report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from portfolio_agent import config
from portfolio_agent.agent import PortfolioAgent
from portfolio_agent.db import create_connection
from portfolio_agent.evaluation import CaseResult, load_cases, run_evaluation, summarize, to_report

DEFAULT_REPORT_PATH = config.PROJECT_ROOT / "eval_report.json"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cases = load_cases(args.dataset)
    if args.ids:
        cases = [case for case in cases if case["id"] in set(args.ids)]
    if not cases:
        print("No matching questions.", file=sys.stderr)
        return 1

    conn = create_connection()
    try:
        agent = PortfolioAgent(conn=conn)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluating {len(cases)} question(s) with {config.OPENAI_MODEL}...\n")
    results = run_evaluation(agent, conn, cases)

    _print_table(results)
    _print_summary(summarize(results))
    args.output.write_text(json.dumps(to_report(results), indent=2, default=str))
    print(f"\nFull report written to {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the portfolio agent against ground truth.")
    parser.add_argument("--dataset", type=Path, default=config.GROUND_TRUTH_PATH)
    parser.add_argument("--ids", type=int, nargs="+", help="only run these question ids")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def _print_table(results: list[CaseResult]) -> None:
    header = f"{'ID':>3}  {'Type':<20} {'Difficulty':<10} {'Verdict':<8} {'Time':>6}  Reason"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.id:>3}  {r.type:<20} {r.difficulty:<10} {r.verdict.value.upper():<8} "
            f"{r.latency_s:>5.1f}s  {r.reason}"
        )


def _print_summary(summary: dict) -> None:
    print(
        f"\nPassed {summary['pass']}/{summary['total']} "
        f"(strict {summary['strict_accuracy']:.0%}, "
        f"lenient {summary['lenient_accuracy']:.0%} incl. {summary['partial']} partial); "
        f"{summary['fail']} failed, {summary['error']} errors; "
        f"mean latency {summary['mean_latency_s']:.1f}s"
    )


if __name__ == "__main__":
    sys.exit(main())
