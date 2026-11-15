import pytest

from portfolio_agent.tools.exposure_tool import (
    PortfolioNotFoundError,
    calculate_sector_exposure,
    resolve_portfolio,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Tech Innovation Fund", "Tech Innovation Fund"),
        ("tech innovation fund", "Tech Innovation Fund"),
        ("international equity", "International Equity Fund"),
        ("Balanced Portfolo", "Balanced Portfolio"),
    ],
)
def test_resolve_portfolio(conn, query, expected):
    assert resolve_portfolio(conn, query)[1] == expected


def test_resolve_portfolio_ambiguous(conn):
    # Matches both "International Equity Fund" and "Total International Index Fund".
    with pytest.raises(PortfolioNotFoundError, match="ambiguous"):
        resolve_portfolio(conn, "international")


def test_resolve_portfolio_unknown(conn):
    with pytest.raises(PortfolioNotFoundError, match="No portfolio matches"):
        resolve_portfolio(conn, "Crypto Moonshot")


def test_mixed_portfolio_excludes_bonds_and_normalises(conn):
    # Balanced Portfolio: 36% equities (Tech 0.22, Financials 0.08, Healthcare 0.06), rest bonds.
    result = calculate_sector_exposure(conn, "Balanced Portfolio")
    exposures = {e.sector: e.exposure_pct for e in result.exposures}

    assert result.equity_weight_pct == pytest.approx(36.0)
    assert exposures == pytest.approx(
        {"Technology": 22 / 36 * 100, "Financials": 8 / 36 * 100, "Healthcare": 6 / 36 * 100}
    )
    assert [e.sector for e in result.exposures] == ["Technology", "Financials", "Healthcare"]


def test_single_sector_portfolio(conn):
    result = calculate_sector_exposure(conn, "Tech Innovation Fund")
    assert result.to_dict()["sector_exposures_pct"] == {"Technology": 100.0}


def test_bond_only_portfolio_has_no_exposures(conn):
    result = calculate_sector_exposure(conn, "Fixed Income Plus")
    assert result.exposures == []
    assert result.equity_weight_pct == 0.0


def test_exposures_sum_to_100_for_every_equity_portfolio(conn):
    names = [row[0] for row in conn.execute("SELECT portfolio_name FROM portfolios")]
    for name in names:
        result = calculate_sector_exposure(conn, name)
        if result.exposures:
            assert sum(e.exposure_pct for e in result.exposures) == pytest.approx(100.0), name
