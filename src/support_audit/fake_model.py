"""A scripted, offline chat model for the no-key, no-spend test path.

It drives every scaffold through the same motions a real model would — call
`export_tickets` for each account the question names, then answer — so the
tests exercise the real LangGraph and deepagents graphs, the real eviction
middleware, the metering callbacks and the grader. It also simulates the
provider's per-request ceiling so the classifier can be validated before any
paid run.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult


class RequestTooLarge(RuntimeError):
    """Mimics the text of OpenAI's 429 for a single over-limit request."""


class ScriptedModel(BaseChatModel):
    mode: str = "oracle"  # oracle | wrong | silent
    ceiling_tokens: int | None = None
    gold: str = ""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kw: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        n_in = count_tokens_approximately(messages)
        if self.ceiling_tokens and n_in > self.ceiling_tokens:
            raise RequestTooLarge(f"Request too large for scripted-model: Limit {self.ceiling_tokens}, Requested {n_in}")
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        question = humans[0].text if humans else ""
        wanted = re.findall(r"ACC-\d{3}", question)
        done = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        issued = [tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        if not issued and wanted:
            calls = [{"name": "export_tickets", "args": {"account_id": a}, "id": f"call_{i}", "type": "tool_call"}
                     for i, a in enumerate(dict.fromkeys(wanted))]
            msg = AIMessage(content="", tool_calls=calls)
        elif issued and not all(tc["id"] in done for tc in issued):
            msg = AIMessage(content="waiting")
        else:
            answer = {"oracle": self.gold, "wrong": "999999", "silent": None}[self.mode]
            msg = AIMessage(content="Done." + (f"\nFINAL ANSWER: {answer}" if answer is not None else ""))
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": 10, "total_tokens": n_in + 10}
        return ChatResult(generations=[ChatGeneration(message=msg)])
