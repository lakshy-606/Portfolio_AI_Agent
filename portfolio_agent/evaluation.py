"""Score agent responses against the ground-truth dataset.

text2sql cases: the agent's SQL is not compared as text. Instead both queries
are executed and their result sets compared, ignoring row order, column order
and extra columns. An agent result that contains every expected row plus some
extra rows (e.g. portfolios with 0% tech exposure) is reported as "partial".

exposure_calculator cases: the agent must call the exposure tool, the tool
must resolve to the expected portfolio, and the final answer must name every
sector in the computed breakdown.
"""

import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from portfolio_agent.agent import AgentResponse, PortfolioAgent
from portfolio_agent.tools import SqlResult, calculate_sector_exposure, run_sql

FLOAT_PRECISION = 2


class Verdict(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    ERROR = "error"



@dataclass(frozen=True)
class CaseResult:
    id: int
    type: str
    difficulty: str
    question: str
    verdict: Verdict
    reason: str
    latency_s: float
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["questions"]


def evaluate_case(agent: PortfolioAgent, conn: sqlite3.Connection, case: dict[str, Any]) -> CaseResult:
    started = time.perf_counter()
    try:
        response = agent.ask(case["question"])
    except Exception as exc:  # one failing case must not abort the whole run
        return _result(case, Verdict.ERROR, f"{type(exc).__name__}: {exc}", started)

    if case["type"] == "text2sql":
        verdict, reason = _score_text2sql(conn, case["ground_truth"], response)
    elif case["type"] == "exposure_calculator":
        verdict, reason = _score_exposure(conn, case["ground_truth"], response)
    else:
        verdict, reason = Verdict.ERROR, f"Unknown case type {case['type']!r}"
    return _result(case, verdict, reason, started, response)


def run_evaluation(
    agent: PortfolioAgent,
    conn: sqlite3.Connection,
    cases: list[dict[str, Any]],
) -> list[CaseResult]:
    return [evaluate_case(agent, conn, case) for case in cases]


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    counts = Counter(r.verdict for r in results)
    total = len(results)
    return {
        "total": total,
        **{verdict.value: counts[verdict] for verdict in Verdict},
        "strict_accuracy": counts[Verdict.PASS] / total if total else 0.0,
        "lenient_accuracy": (counts[Verdict.PASS] + counts[Verdict.PARTIAL]) / total if total else 0.0,
        "mean_latency_s": sum(r.latency_s for r in results) / total if total else 0.0,
    }


def to_report(results: list[CaseResult]) -> dict[str, Any]:
    return {"summary": summarize(results), "cases": [asdict(r) for r in results]}


def _score_text2sql(
    conn: sqlite3.Connection, ground_truth: dict[str, Any], response: AgentResponse
) -> tuple[Verdict, str]:
    expected = run_sql(conn, ground_truth["sql_query"])
    if not expected.ok:
        return Verdict.ERROR, f"Ground-truth SQL failed: {expected.error}"

    queries = [c.args.get("query", "") for c in response.tool_calls if c.name == "run_sql"]
    if not queries:
        return Verdict.FAIL, "Agent did not run any SQL"

    # The agent sometimes explores before its final query, so credit its best attempt.
    verdicts = [compare_results(expected, run_sql(conn, query)) for query in queries]
    if Verdict.PASS in verdicts:
        return Verdict.PASS, "Result set matches ground truth"
    if Verdict.PARTIAL in verdicts:
        return Verdict.PARTIAL, "Contains all expected rows plus extra rows"
    return Verdict.FAIL, "No agent query reproduced the expected result"


def compare_results(expected: SqlResult, actual: SqlResult) -> Verdict:
    """Compare two result sets, ignoring row order, column order and extra columns."""
    if not actual.ok:
        return Verdict.FAIL

    expected_rows = [_normalize_row(row) for row in expected.rows]
    actual_rows = [_normalize_row(row) for row in actual.rows]

    # Match each expected column to an unused actual column holding the same values.
    mapping = []
    for j in range(len(expected.columns)):
        expected_values = Counter(row[j] for row in expected_rows)
        match = next(
            (
                k
                for k in range(len(actual.columns))
                if k not in mapping and not expected_values - Counter(row[k] for row in actual_rows)
            ),
            None,
        )
        if match is None:
            return Verdict.FAIL
        mapping.append(match)

    projected = Counter(tuple(row[k] for k in mapping) for row in actual_rows)
    if Counter(expected_rows) - projected:  # some expected row is missing
        return Verdict.FAIL
    return Verdict.PASS if len(actual_rows) == len(expected_rows) else Verdict.PARTIAL


def _normalize_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_normalize_value(value) for value in row)


def _normalize_value(value: Any) -> Any:
    # 13 and 13.0 compare equal; unrounded vs ROUND(..., 2) values compare equal.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), FLOAT_PRECISION)
    if isinstance(value, str):
        return value.strip()
    return value


def _score_exposure(
    conn: sqlite3.Connection, ground_truth: dict[str, Any], response: AgentResponse
) -> tuple[Verdict, str]:
    expected_tool = ground_truth["tool_name"]
    expected_name = ground_truth["parameters"]["portfolio_name"]

    calls = [c for c in response.tool_calls if c.name == expected_tool]
    if not calls:
        used = ", ".join(c.name for c in response.tool_calls) or "none"
        return Verdict.FAIL, f"Expected tool {expected_tool!r}, agent used: {used}"

    resolved = [_parse_json(c.output).get("portfolio_name") for c in calls]
    if expected_name not in resolved:
        return Verdict.FAIL, f"Expected portfolio {expected_name!r}, tool resolved {resolved}"

    reference = calculate_sector_exposure(conn, expected_name)
    answer = response.answer.lower()
    missing = [e.sector for e in reference.exposures if e.sector.lower() not in answer]
    if missing:
        return Verdict.PARTIAL, f"Correct tool call, but answer omits sectors: {', '.join(missing)}"
    return Verdict.PASS, "Correct tool, portfolio and sectors"


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _result(
    case: dict[str, Any],
    verdict: Verdict,
    reason: str,
    started: float,
    response: AgentResponse | None = None,
) -> CaseResult:
    return CaseResult(
        id=case["id"],
        type=case["type"],
        difficulty=case.get("difficulty", ""),
        question=case["question"],
        verdict=verdict,
        reason=reason,
        latency_s=round(time.perf_counter() - started, 2),
        tool_calls=[{"name": c.name, "args": c.args} for c in response.tool_calls] if response else [],
        answer=response.answer if response else "",
    )
