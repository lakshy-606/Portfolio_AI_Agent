"""Build the in-memory SQLite database from the provided schema and CSV files."""

import csv
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from portfolio_agent import config

logger = logging.getLogger(__name__)

# Parents before children, so foreign keys resolve during load.
TABLE_LOAD_ORDER = (
    "sectors",
    "securities",
    "benchmarks",
    "portfolios",
    "holdings",
    "transactions",
    "historical_prices",
    "portfolio_performance",
    "risk_metrics",
)

# Low-cardinality columns whose distinct values are shown to the LLM, so it
# filters on literals that actually exist (e.g. 'High', not 'high').
CATEGORICAL_COLUMNS = {
    "portfolios": ("portfolio_name", "status", "target_risk_level", "strategy_type", "benchmark_index"),
    "sectors": ("sector_name",),
    "securities": ("asset_type", "exchange", "country"),
    "transactions": ("transaction_type",),
}


def create_connection(
    data_dir: Path = config.DATA_DIR,
    schema_path: Path = config.SCHEMA_PATH,
) -> sqlite3.Connection:
    """Return a read-only, in-memory connection populated with the portfolio data."""
    # check_same_thread=False: LangGraph may run tool calls on worker threads.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_path.read_text())

    for table in TABLE_LOAD_ORDER:
        row_count = _load_csv(conn, table, data_dir / f"{table}.csv")
        logger.debug("Loaded %d rows into %s", row_count, table)
    conn.commit()

    # Everything after this point is read-only; writes raise sqlite3.OperationalError.
    conn.execute("PRAGMA query_only = ON")
    return conn


def _load_csv(conn: sqlite3.Connection, table: str, csv_path: Path) -> int:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"{csv_path} has no header row")
        # Empty cells become NULL; SQLite's column affinity handles numeric conversion.
        rows = [[value if value != "" else None for value in row.values()] for row in reader]

    column_list = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})", rows)
    return len(rows)


def describe_schema(conn: sqlite3.Connection, tables: Sequence[str] | None = None) -> str:
    """Render tables, columns and known categorical values as compact text for a prompt."""
    lines = []
    for table in tables or TABLE_LOAD_ORDER:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
        column_defs = ", ".join(f"{name} {col_type}" for _, name, col_type, *_ in columns)
        lines.append(f"{table}({column_defs})")

        foreign_keys = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        for fk in foreign_keys:
            ref_table, from_col, to_col = fk[2], fk[3], fk[4]
            lines.append(f"  - {from_col} -> {ref_table}.{to_col}")

        for column in CATEGORICAL_COLUMNS.get(table, ()):
            values = [
                row[0]
                for row in conn.execute(
                    f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY 1"
                )
            ]
            lines.append(f"  - {column} values: {', '.join(map(repr, values))}")
    return "\n".join(lines)
