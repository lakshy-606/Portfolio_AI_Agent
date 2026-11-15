"""Sector exposure calculator for a single portfolio (equities only)."""

import difflib
import sqlite3
from dataclasses import dataclass
from typing import Any

# Weights are summed per sector over stock holdings only. Bonds have no sector
# and are excluded, as the exercise specifies.
_SECTOR_WEIGHTS_SQL = """
SELECT sec.sector_name, SUM(h.current_weight) AS weight
FROM holdings h
JOIN securities s ON s.security_id = h.security_id
JOIN sectors sec ON sec.sector_id = s.sector_id
WHERE h.portfolio_id = ? AND s.asset_type = 'Stock'
GROUP BY sec.sector_name
"""

_TOTAL_WEIGHT_SQL = "SELECT SUM(current_weight) FROM holdings WHERE portfolio_id = ?"


class PortfolioNotFoundError(ValueError):
    """Raised when a portfolio name cannot be resolved to exactly one portfolio."""


@dataclass(frozen=True)
class SectorExposure:
    sector: str
    exposure_pct: float  # share of the portfolio's equity holdings, 0-100


@dataclass(frozen=True)
class ExposureResult:
    portfolio_id: int
    portfolio_name: str
    equity_weight_pct: float  # share of the whole portfolio held in equities, 0-100
    exposures: list[SectorExposure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "equity_weight_pct": round(self.equity_weight_pct, 2),
            "sector_exposures_pct": {e.sector: round(e.exposure_pct, 2) for e in self.exposures},
        }


def resolve_portfolio(conn: sqlite3.Connection, name: str) -> tuple[int, str]:
    """Map a user-supplied name (e.g. "international equity") to (portfolio_id, portfolio_name).

    Tries, in order: case-insensitive exact match, unique substring match, and
    close fuzzy match. Raises PortfolioNotFoundError if none is unambiguous.
    """
    portfolios = conn.execute("SELECT portfolio_id, portfolio_name FROM portfolios").fetchall()
    by_lower_name = {portfolio_name.lower(): (pid, portfolio_name) for pid, portfolio_name in portfolios}
    query = " ".join(name.lower().split())

    if query in by_lower_name:
        return by_lower_name[query]

    substring_matches = [key for key in by_lower_name if query in key]
    if len(substring_matches) == 1:
        return by_lower_name[substring_matches[0]]

    fuzzy_matches = difflib.get_close_matches(query, list(by_lower_name), n=2, cutoff=0.6)
    if len(fuzzy_matches) == 1:
        return by_lower_name[fuzzy_matches[0]]

    candidates = substring_matches or fuzzy_matches
    if candidates:
        options = ", ".join(by_lower_name[key][1] for key in candidates)
        raise PortfolioNotFoundError(f"'{name}' is ambiguous; did you mean one of: {options}?")
    known = ", ".join(sorted(portfolio_name for _, portfolio_name in portfolios))
    raise PortfolioNotFoundError(f"No portfolio matches '{name}'. Known portfolios: {known}.")


def calculate_sector_exposure(conn: sqlite3.Connection, portfolio_name: str) -> ExposureResult:
    """Return each sector's share of the portfolio's equity holdings, largest first.

    Exposures are normalised over equity weight so they sum to 100%, which also
    corrects portfolios whose stored weights do not sum exactly to 1.
    """
    portfolio_id, resolved_name = resolve_portfolio(conn, portfolio_name)

    sector_weights = conn.execute(_SECTOR_WEIGHTS_SQL, (portfolio_id,)).fetchall()
    total_weight = conn.execute(_TOTAL_WEIGHT_SQL, (portfolio_id,)).fetchone()[0] or 0.0
    equity_weight = sum(weight for _, weight in sector_weights)

    exposures = (
        [SectorExposure(sector, weight / equity_weight * 100) for sector, weight in sector_weights]
        if equity_weight
        else []
    )
    exposures.sort(key=lambda e: (-e.exposure_pct, e.sector))

    return ExposureResult(
        portfolio_id=portfolio_id,
        portfolio_name=resolved_name,
        equity_weight_pct=equity_weight / total_weight * 100 if total_weight else 0.0,
        exposures=exposures,
    )
