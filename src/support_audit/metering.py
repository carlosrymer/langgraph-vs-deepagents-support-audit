"""Per-run metering via LangChain callbacks. Callbacks propagate into
deepagents subagents, so a subagent's model calls and tool calls are counted
against the run that spawned them.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

MODEL_CALL_CAP = 40

# gpt-5.4-mini, Standard tier, USD per 1M tokens, from OpenAI's API pricing page
# (developers.openai.com/api/docs/pricing) as read on 2026-09-28.
PRICE_PER_M = {"input": 0.75, "cached_input": 0.075, "output": 4.50}


class ModelCallCapExceeded(RuntimeError):
    pass


class Meter(BaseCallbackHandler):
    raise_error = True  # so the cap below actually stops the run

    def __init__(self, cap: int = MODEL_CALL_CAP) -> None:
        self.cap = cap
        self.lock = threading.Lock()
        self.model_calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.max_request_input_tokens = 0
        self.per_call_input: list[int] = []
        self.tool_calls: Counter[str] = Counter()
        self.large_result_reads = 0  # read_file/grep calls that target an evicted tool result

    def on_chat_model_start(self, serialized: Any, messages: Any, **kw: Any) -> None:
        with self.lock:
            self.model_calls += 1
            if self.model_calls > self.cap:
                raise ModelCallCapExceeded(f"model call cap {self.cap} exceeded")

    def on_llm_end(self, response: Any, **kw: Any) -> None:
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                with self.lock:
                    inp = int(usage.get("input_tokens", 0))
                    self.input_tokens += inp
                    self.per_call_input.append(inp)
                    self.max_request_input_tokens = max(self.max_request_input_tokens, inp)
                    self.output_tokens += int(usage.get("output_tokens", 0))
                    self.cached_input_tokens += int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
                    self.reasoning_tokens += int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)

    def on_tool_start(self, serialized: Any, input_str: str, **kw: Any) -> None:
        name = (serialized or {}).get("name") or kw.get("name") or "?"
        with self.lock:
            self.tool_calls[name] += 1
            if name in ("read_file", "grep", "glob") and "large_tool_results" in str(kw.get("inputs") or input_str):
                self.large_result_reads += 1

    def cost_usd(self) -> float:
        return cost_usd(self.input_tokens, self.cached_input_tokens, self.output_tokens)

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "max_request_input_tokens": self.max_request_input_tokens,
            "per_call_input_tokens": list(self.per_call_input),
            "tool_calls": dict(self.tool_calls),
            "large_result_reads": self.large_result_reads,
            "cost_usd": round(self.cost_usd(), 6),
        }


def cost_usd(input_tokens: int, cached: int, output: int) -> float:
    uncached = input_tokens - cached
    return (uncached * PRICE_PER_M["input"] + cached * PRICE_PER_M["cached_input"]
            + output * PRICE_PER_M["output"]) / 1e6
