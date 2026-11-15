"""Public entry point: ask the portfolio agent a question and get a structured response."""

import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from portfolio_agent import config
from portfolio_agent.db import create_connection, describe_schema
from portfolio_agent.graph import build_graph, build_tools
from portfolio_agent.prompts import build_system_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    output: str


@dataclass(frozen=True)
class AgentResponse:
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class PortfolioAgent:
    """Answers natural-language questions about the portfolio database.

    Calls that share a `thread_id` share conversation history, so follow-up
    questions work. Omitting it starts a fresh conversation.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to a .env file in the project root.")
            llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
        self._conn = conn or create_connection()
        system_prompt = build_system_prompt(describe_schema(self._conn))
        self._graph = build_graph(
            llm,
            build_tools(self._conn),
            system_prompt,
            checkpointer=InMemorySaver(),
        )

    def ask(self, question: str, thread_id: str | None = None) -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        run_config = {
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
            "recursion_limit": config.MAX_AGENT_STEPS,
        }
        try:
            state = self._graph.invoke({"messages": [HumanMessage(question)]}, run_config)
        except GraphRecursionError:
            logger.warning("Step limit reached for question: %s", question)
            return AgentResponse(
                question=question,
                answer="Sorry, I couldn't reach an answer within the step limit. Try rephrasing the question.",
            )

        turn = _current_turn(state["messages"])
        tool_calls = _collect_tool_calls(turn)
        for call in tool_calls:
            logger.info("Tool %s called with %s", call.name, call.args)
        return AgentResponse(question=question, answer=turn[-1].text, tool_calls=tool_calls)


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages produced since the latest user question (history is kept per thread)."""
    last_question = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    return messages[last_question + 1 :]


def _collect_tool_calls(messages: list[BaseMessage]) -> list[ToolCall]:
    outputs = {m.tool_call_id: m.text for m in messages if isinstance(m, ToolMessage)}
    return [
        ToolCall(name=call["name"], args=call["args"], output=outputs.get(call["id"], ""))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
