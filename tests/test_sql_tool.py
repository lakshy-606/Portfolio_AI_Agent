import pytest

from portfolio_agent.tools.sql_tool import UnsafeQueryError, run_sql, validate_query


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select 1;",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "-- count portfolios\n  SELECT COUNT(*) FROM portfolios",
        "/* note */ SELECT 1",
    ],
)
def test_validate_accepts_read_only_queries(query):
    assert validate_query(query)


@pytest.mark.parametrize(
    "query",
    ["", "   ;", "DROP TABLE portfolios", "UPDATE portfolios SET status = 'x'", "PRAGMA query_only = OFF"],
)
def test_validate_rejects_non_select(query):
    with pytest.raises(UnsafeQueryError):
        validate_query(query)


def test_run_sql_returns_rows_and_columns(conn):
    result = run_sql(conn, "SELECT COUNT(*) AS n FROM portfolios;")
    assert result.ok
    assert result.columns == ["n"]
    assert result.rows == [(13,)]


def test_run_sql_truncates_large_results(conn):
    result = run_sql(conn, "SELECT * FROM historical_prices", max_rows=10)
    assert result.ok
    assert len(result.rows) == 10
    assert result.truncated


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM no_such_table",
        "SELECT 1; DELETE FROM portfolios",
        "DELETE FROM portfolios",
    ],
)
def test_run_sql_reports_errors_instead_of_raising(conn, query):
    result = run_sql(conn, query)
    assert not result.ok
    assert "error" in result.to_dict()
    assert conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0] == 13
