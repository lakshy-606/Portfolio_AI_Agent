import sqlite3

import pytest

from portfolio_agent.db import TABLE_LOAD_ORDER, describe_schema


@pytest.mark.parametrize(
    ("table", "expected_rows"),
    [("portfolios", 13), ("securities", 30), ("sectors", 10), ("holdings", 95)],
)
def test_tables_are_loaded(conn, table, expected_rows):
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected_rows


def test_empty_csv_cells_are_null(conn):
    bonds_with_sector = conn.execute(
        "SELECT COUNT(*) FROM securities WHERE asset_type = 'Bond' AND sector_id IS NOT NULL"
    ).fetchone()[0]
    assert bonds_with_sector == 0


def test_numeric_columns_are_numeric(conn):
    weight_type = conn.execute("SELECT typeof(current_weight) FROM holdings LIMIT 1").fetchone()[0]
    assert weight_type == "real"


def test_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM portfolios")


def test_describe_schema_lists_tables_and_categorical_values(conn):
    description = describe_schema(conn)
    for table in TABLE_LOAD_ORDER:
        assert f"{table}(" in description
    assert "'Growth Equity Fund'" in description
    assert "portfolio_id -> portfolios.portfolio_id" in description
