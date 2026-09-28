"""The three scaffolds under test. Same model object, same two domain tools,
same task system prompt. Only the scaffold varies.

  langgraph      a hand-built LangGraph ReAct loop (StateGraph + ToolNode) —
                 what "plain LangGraph" means to most people
  deepagents     `create_deep_agent` with every default left on
  deepagents_no_offload
                 `create_deep_agent` with the main agent's large-tool-result
                 eviction switched off (`tool_token_limit_before_evict=None`);
                 everything else — base prompt, todos, subagents,
                 summarisation, the filesystem tools — unchanged
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from support_audit.data import accounts_payload, export_payload

ARMS = ("langgraph", "deepagents", "deepagents_no_offload")

SYSTEM_PROMPT = """You are a support-operations analyst for a B2B SaaS vendor. You answer audit \
questions about customer accounts using the tools provided. `export_tickets` returns an account's \
full ticket history as JSON Lines: a header line, then one ticket per line. Work only from the \
exported data; do not guess. Be exact when counting.

When you are done, end your final message with a single line of the form:
FINAL ANSWER: <answer>
Give just the value on that line: a number, a ticket_id, an account_id, or an agent's full name."""


def make_tools(size: str) -> list:
    @tool
    def list_accounts() -> str:
        """List every customer account (account_id and name)."""
        return accounts_payload()

    @tool
    def export_tickets(account_id: str) -> str:
        """Export the full support-ticket history for one account as JSON Lines
        (header line, then one ticket per line with ticket_id, created_at,
        priority, status, tags, assignee, resolved_by, error_codes,
        first_response_minutes, subject and the full transcript)."""
        return export_payload(account_id, size)

    return [list_accounts, export_tickets]


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def build_langgraph(model: BaseChatModel, tools: list) -> Any:
    bound = model.bind_tools(tools)

    def agent(state: _State) -> dict:
        return {"messages": [bound.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    g = StateGraph(_State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())


def build_deepagents(model: BaseChatModel, tools: list, *, offload: bool = True) -> Any:
    from deepagents import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemMiddleware

    middleware = [] if offload else [FilesystemMiddleware(tool_token_limit_before_evict=None)]
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
                             middleware=middleware, checkpointer=InMemorySaver())


def build(arm: str, model: BaseChatModel, size: str) -> Any:
    tools = make_tools(size)
    if arm == "langgraph":
        return build_langgraph(model, tools)
    if arm == "deepagents":
        return build_deepagents(model, tools, offload=True)
    if arm == "deepagents_no_offload":
        return build_deepagents(model, tools, offload=False)
    raise ValueError(arm)
