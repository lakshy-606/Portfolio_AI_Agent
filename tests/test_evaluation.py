import json

import pytest

from portfolio_agent import config
from portfolio_agent.agent import AgentResponse, ToolCall
from portfolio_agent.evaluation import (
    CaseResult,
    Verdict,
    _score_exposure,
    _score_text2sql,
    compare_results,
    load_cases,
    summarize,
)
from portfolio_agent.tools import SqlResult, calculate_sector_exposure, run_sql


def result(columns, rows):
    return SqlResult(query="", columns=columns, rows=rows)


EXPECTED = result(["name", "value"], [("A", 1.0), ("B", 2.5)])


@pytest.mark.parametrize(
    ("actual", "verdict"),
    [
        (result(["name", "value"], [("B", 2.5), ("A", 1.0)]), Verdict.PASS),  # row order
        (result(["value", "name"], [(1, "A"), (2.5, "B")]), Verdict.PASS),  # column order, int vs float
        (result(["id", "name", "value"], [(1, "A", 1.0), (2, "B", 2.5)]), Verdict.PASS),  # extra column
        (result(["name", "value"], [("A", 1.0012), ("B", 2.4999)]), Verdict.PASS),  # rounding
        (result(["name", "value"], [("A", 1.0), ("B", 2.5), ("C", 0.0)]), Verdict.PARTIAL),  # extra row
        (result(["name", "value"], [("A", 1.0)]), Verdict.FAIL),  # missing row
        (result(["name", "value"], [("A", 2.5), ("B", 1.0)]), Verdict.FAIL),  # values swapped between rows
        (result(["name"], [("A",), ("B",)]), Verdict.FAIL),  # missing column
    ],
)
def test_compare_results(actual, verdict):
    assert compare_results(EXPECTED, actual) == verdict


def sql_response(*queries):
    calls = [ToolCall("run_sql", {"query": q}, "") for q in queries]
    return AgentResponse(question="", answer="", tool_calls=calls)


def test_text2sql_equivalent_query_passes(conn):
    ground_truth = {"sql_query": "SELECT COUNT(*) FROM portfolios;"}
    response = sql_response("SELECT COUNT(portfolio_id) AS n FROM portfolios")
    assert _score_text2sql(conn, ground_truth, response)[0] == Verdict.PASS


def test_text2sql_credits_best_query(conn):
    ground_truth = {"sql_query": "SELECT COUNT(*) FROM portfolios;"}
    response = sql_response("SELECT nonsense", "SELECT COUNT(*) FROM portfolios")
    assert _score_text2sql(conn, ground_truth, response)[0] == Verdict.PASS


def test_text2sql_without_sql_fails(conn):
    ground_truth = {"sql_query": "SELECT 1"}
    assert _score_text2sql(conn, ground_truth, sql_response())[0] == Verdict.FAIL


def exposure_response(conn, portfolio_query, answer):
    output = json.dumps(calculate_sector_exposure(conn, portfolio_query).to_dict())
    call = ToolCall("exposure_calculator", {"portfolio_name": portfolio_query}, output)
    return AgentResponse(question="", answer=answer, tool_calls=[call])


EXPOSURE_GT = {
    "tool_name": "exposure_calculator",
    "parameters": {"portfolio_name": "International Equity Fund"},
}


def test_exposure_pass(conn):
    answer = "Energy 38%, Technology 36%, Automotive 15%, Consumer Staples 11%"
    response = exposure_response(conn, "international equity", answer)
    assert _score_exposure(conn, EXPOSURE_GT, response)[0] == Verdict.PASS


def test_exposure_answer_missing_sector_is_partial(conn):
    response = exposure_response(conn, "international equity", "Energy 38%, Technology 36%")
    assert _score_exposure(conn, EXPOSURE_GT, response)[0] == Verdict.PARTIAL


def test_exposure_wrong_portfolio_fails(conn):
    response = exposure_response(conn, "Tech Innovation Fund", "Technology 100%")
    assert _score_exposure(conn, EXPOSURE_GT, response)[0] == Verdict.FAIL


def test_ground_truth_queries_execute(conn):
    for case in load_cases(config.GROUND_TRUTH_PATH):
        if case["type"] == "text2sql":
            assert run_sql(conn, case["ground_truth"]["sql_query"]).ok, case["id"]


def test_summarize_counts_verdicts():
    results = [
        CaseResult(1, "text2sql", "easy", "q", Verdict.PASS, "", 1.0),
        CaseResult(2, "text2sql", "easy", "q", Verdict.PARTIAL, "", 3.0),
    ]
    summary = summarize(results)
    assert summary["strict_accuracy"] == 0.5
    assert summary["lenient_accuracy"] == 1.0
    assert summary["mean_latency_s"] == 2.0
