import pytest

from portfolio_agent.db import create_connection


@pytest.fixture(scope="session")
def conn():
    connection = create_connection()
    yield connection
    connection.close()
