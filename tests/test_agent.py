"""Graph wiring tests with a scripted model, so no API calls are made."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from portfolio_agent.agent import PortfolioAgent
from portfolio_agent.graph import build_tools


class ScriptedChatModel(GenericFakeChatModel):
    """Replays a fixed sequence of AI messages; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted_agent(conn, *responses: AIMessage) -> PortfolioAgent:
    return PortfolioAgent(llm=ScriptedChatModel(messages=iter(responses)), conn=conn)


def tool_call(name: str, call_id: str = "call_1", **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_sql_round_trip(conn):
    agent = scripted_agent(
        conn,
        tool_call("run_sql", query="SELECT COUNT(*) FROM portfolios"),
        AIMessage("There are 13 portfolios."),
    )
    response = agent.ask("How many portfolios do we have?")

    assert response.answer == "There are 13 portfolios."
    [call] = response.tool_calls
    assert call.name == "run_sql"
    assert json.loads(call.output)["rows"] == [[13]]


def test_exposure_round_trip(conn):
    agent = scripted_agent(
        conn,
        tool_call("exposure_calculator", portfolio_name="international equity"),
        AIMessage("Energy 38%, Technology 36%, ..."),
    )
    [call] = agent.ask("Sector exposure for international equity?").tool_calls

    assert json.loads(call.output)["portfolio_name"] == "International Equity Fund"


def test_tool_errors_are_returned_to_the_model(conn):
    tools = {t.name: t for t in build_tools(conn)}

    sql_output = json.loads(tools["run_sql"].invoke({"query": "DROP TABLE portfolios"}))
    exposure_output = json.loads(
        tools["exposure_calculator"].invoke({"portfolio_name": "Crypto Moonshot"})
    )

    assert "error" in sql_output
    assert "No portfolio matches" in exposure_output["error"]


def test_follow_up_reports_only_current_turn(conn):
    agent = scripted_agent(
        conn,
        tool_call("run_sql", query="SELECT COUNT(*) FROM portfolios"),
        AIMessage("13."),
        AIMessage("You're welcome."),
    )
    agent.ask("How many portfolios?", thread_id="t1")
    follow_up = agent.ask("Thanks!", thread_id="t1")

    assert follow_up.answer == "You're welcome."
    assert follow_up.tool_calls == []


def test_empty_question_is_rejected(conn):
    with pytest.raises(ValueError):
        scripted_agent(conn).ask("   ")
