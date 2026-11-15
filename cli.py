"""Command-line interface for the portfolio agent.

Usage:
    python cli.py "How many portfolios do we have?"   # answer one question
    python cli.py                                      # interactive session
    python cli.py -v ...                               # also show tool calls
"""

import argparse
import logging
import sys
import uuid

from portfolio_agent.agent import AgentResponse, PortfolioAgent

EXIT_COMMANDS = {"exit", "quit", "q"}
RESET_COMMAND = "/new"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        agent = PortfolioAgent()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.question:
        return _answer_once(agent, " ".join(args.question), args.verbose)
    return _interactive(agent, args.verbose)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about the portfolio database.")
    parser.add_argument("question", nargs="*", help="question to answer; omit for interactive mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the tool calls behind each answer")
    return parser.parse_args()


def _answer_once(agent: PortfolioAgent, question: str, verbose: bool) -> int:
    try:
        response = agent.ask(question)
    except Exception as exc:  # surface API/config failures as a clean CLI error
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_response(response, verbose)
    return 0


def _interactive(agent: PortfolioAgent, verbose: bool) -> int:
    print(f"Portfolio agent. Type a question, '{RESET_COMMAND}' to start over, or 'exit' to quit.")
    thread_id = str(uuid.uuid4())
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return 0
        if question == RESET_COMMAND:
            thread_id = str(uuid.uuid4())
            print("Started a new conversation.")
            continue

        try:
            response = agent.ask(question, thread_id=thread_id)
        except Exception as exc:  # keep the session alive after a failed request
            print(f"Error: {exc}", file=sys.stderr)
            continue
        _print_response(response, verbose)


def _print_response(response: AgentResponse, verbose: bool) -> None:
    if verbose:
        for call in response.tool_calls:
            print(f"\n[tool] {call.name}")
            for name, value in call.args.items():
                print(f"  {name}: {value}")
        print()
    print(response.answer)


if __name__ == "__main__":
    sys.exit(main())
