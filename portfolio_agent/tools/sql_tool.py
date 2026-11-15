"""Execute LLM-generated SQL safely against the portfolio database."""

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from portfolio_agent import config

_LEADING_COMMENTS = re.compile(r"^(?:\s*(?:--[^\n]*|/\*.*?\*/))*\s*", re.DOTALL)
_READ_ONLY_PREFIX = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE)


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only statement."""


@dataclass(frozen=True)
class SqlResult:
    query: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        if not self.ok:
            return {"query": self.query, "error": self.error}
        return {
            "query": self.query,
            "columns": self.columns,
            "rows": [list(row) for row in self.rows],
            "row_count": len(self.rows),
            "truncated": self.truncated,
        }


def validate_query(query: str) -> str:
    """Return the cleaned query, or raise UnsafeQueryError if it is not read-only.

    This is a first line of defence that gives the LLM a clear error message.
    The connection itself is opened with `PRAGMA query_only`, and sqlite3
    refuses to execute more than one statement per call.
    """
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Query is empty.")

    body = _LEADING_COMMENTS.sub("", cleaned)
    if not _READ_ONLY_PREFIX.match(body):
        raise UnsafeQueryError("Only SELECT queries (optionally starting with WITH) are allowed.")
    return cleaned


def run_sql(
    conn: sqlite3.Connection,
    query: str,
    max_rows: int = config.MAX_SQL_ROWS,
) -> SqlResult:
    """Run a read-only query and return its rows, or the error as data.

    Errors are returned rather than raised so the agent can read them and retry
    with a corrected query.
    """
    try:
        cleaned = validate_query(query)
        cursor = conn.execute(cleaned)
        rows = cursor.fetchmany(max_rows + 1)
    # sqlite3.Warning covers "You can only execute one statement at a time" on Python < 3.12.
    except (UnsafeQueryError, sqlite3.Error, sqlite3.Warning) as exc:
        return SqlResult(query=query, columns=[], rows=[], error=f"{type(exc).__name__}: {exc}")

    columns = [description[0] for description in cursor.description or ()]
    truncated = len(rows) > max_rows
    return SqlResult(query=cleaned, columns=columns, rows=rows[:max_rows], truncated=truncated)
