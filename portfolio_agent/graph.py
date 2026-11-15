"""LangGraph wiring: expose the tools to the LLM and loop until it answers."""

import json
import sqlite3
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from portfolio_agent.tools import PortfolioNotFoundError, calculate_sector_exposure, run_sql


def build_tools(conn: sqlite3.Connection) -> list[BaseTool]:
    """Bind the plain tool functions to a connection and describe them for the LLM."""

    @tool("run_sql")
    def run_sql_tool(query: str) -> str:
        """Run one read-only SQLite SELECT query against the portfolio database.

        Returns the columns and rows as JSON, or an error message to correct and retry.
        """
        return _to_json(run_sql(conn, query).to_dict())

    @tool("exposure_calculator")
    def sector_exposure_tool(portfolio_name: str) -> str:
        """Calculate the sector exposure breakdown of one portfolio's equity holdings.

        Bonds are excluded. Exposures are percentages of the equity holdings and
        sum to 100. Accepts a full or partial portfolio name, e.g. "international equity".
        """
        try:
            return _to_json(calculate_sector_exposure(conn, portfolio_name).to_dict())
        except PortfolioNotFoundError as exc:
            return _to_json({"error": str(exc)})

    return [run_sql_tool, sector_exposure_tool]


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the agent loop: agent -> (tools -> agent)* -> END."""
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, Any]:
        # The system prompt is prepended per call rather than stored in state,
        # so the conversation history stays free of it.
        response = llm_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # routes to "tools" or END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)
