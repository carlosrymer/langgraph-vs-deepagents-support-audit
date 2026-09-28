"""Run the experiment: tasks x sizes x trials x arms.

    uv run python -m support_audit.runner --fake            # offline, no key
    uv run python -m support_audit.runner --budget-usd 40   # the paid run

Results append to artifacts/runs.jsonl (one JSON record per run). The runner
resumes: any run_id already recorded with an outcome other than harness_error
is skipped, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from support_audit.arms import ARMS, build
from support_audit.data import SIZES
from support_audit.grading import classify_exception, grade
from support_audit.metering import Meter
from support_audit.tasks import TASKS, TASKS_BY_ID, gold

MODEL = "gpt-5.4-mini-2026-03-17"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 8000
RECURSION_LIMIT = 250
ROOT = Path(__file__).resolve().parents[2]
EVICTION_MARKER = "/large_tool_results/"

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-<redacted>"),
    (re.compile(r"org-[A-Za-z0-9]{12,}"), "org-<redacted>"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}"), r"\1<redacted>"),
]


def scrub(text: str) -> str:
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY", "GH_TOKEN", "GITHUB_TOKEN",
                "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "BFL_API_KEY"):
        val = os.environ.get(var)
        if val and len(val) >= 8:
            text = text.replace(val, f"<{var}>")
    return text


def make_model(fake: bool = False, **fake_kw: Any):
    if fake:
        from support_audit.fake_model import ScriptedModel

        return ScriptedModel(**fake_kw)
    from langchain_openai import ChatOpenAI

    # Responses API: this model rejects function tools + reasoning_effort on /v1/chat/completions.
    return ChatOpenAI(model=MODEL, use_responses_api=True, reasoning={"effort": REASONING_EFFORT},
                      max_tokens=MAX_OUTPUT_TOKENS, max_retries=6, timeout=600)


def _compact(content: Any, limit: int = 2000) -> Any:
    """Keep trajectories committable: large blobs become a hash + length + head.
    Every tool payload is regenerable from the seed, so the hash is checkable."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return content
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text), "head": text[:300]}


def _trajectory(messages: list) -> list[dict]:
    out = []
    for m in messages:
        rec: dict[str, Any] = {"type": m.type}
        if isinstance(m, ToolMessage):
            rec["name"] = m.name
            rec["tool_call_id"] = m.tool_call_id
        if isinstance(m, AIMessage) and m.tool_calls:
            rec["tool_calls"] = [{"name": tc["name"], "args": {k: _compact(v, 500) for k, v in tc["args"].items()}}
                                 for tc in m.tool_calls]
        rec["content"] = _compact(m.text if isinstance(m, AIMessage) else m.content)
        out.append(rec)
    return out


def run_id(arm: str, size: str, task_id: str, trial: int) -> str:
    return f"{arm}|{size}|{task_id}|{trial}"


def run_one(arm: str, size: str, task_id: str, trial: int, model: Any) -> dict[str, Any]:
    task = TASKS_BY_ID[task_id]
    expected = gold(task_id)
    graph = build(arm, model, size)
    meter = Meter()
    thread = run_id(arm, size, task_id, trial)
    config = {"configurable": {"thread_id": thread}, "callbacks": [meter], "recursion_limit": RECURSION_LIMIT}
    t0 = time.time()
    error: str | None = None
    try:
        graph.invoke({"messages": [HumanMessage(task.question)]}, config)
    except Exception as exc:  # noqa: BLE001 — every failure is classified, none swallowed
        error = scrub(f"{type(exc).__name__}: {exc}")[:2000]
        outcome = classify_exception(exc)
    wall = time.time() - t0
    values = graph.get_state({"configurable": {"thread_id": thread}}).values or {}
    messages = values.get("messages", [])
    final_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    final_text = final_ai.text if final_ai is not None else None
    answer = None
    if error is None:
        outcome, answer = grade(task.kind, final_text, expected)
    evicted = sum(1 for m in messages if isinstance(m, ToolMessage) and m.name == "export_tickets"
                  and EVICTION_MARKER in str(m.content))
    exports = sum(1 for m in messages if isinstance(m, ToolMessage) and m.name == "export_tickets")
    return {
        "run_id": thread, "arm": arm, "size": size, "task_id": task_id, "trial": trial,
        "n_accounts": len(task.accounts), "kind": task.kind,
        "outcome": outcome, "gold": expected, "answer": answer,
        "final_text": scrub(final_text)[-1500:] if final_text else None,
        "error": error, "wall_s": round(wall, 2),
        "exports_in_main_context": exports, "exports_evicted_in_main_context": evicted,
        "todos": len(values.get("todos") or []),
        "files_in_state": sorted((values.get("files") or {}).keys()),
        "state_keys": sorted(values.keys()),
        **meter.snapshot(),
        "trajectory": _trajectory(messages),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def plan(arms: list[str], sizes: list[str], tasks: list[str], trials: int, seed: int = 7) -> list[tuple]:
    runs = [(a, s, t, k) for k in range(1, trials + 1) for s in sizes for t in tasks for a in arms]
    random.Random(seed).shuffle(runs)  # interleave arms so API conditions over time hit all arms alike
    return runs


def load_done(path: Path) -> tuple[set[str], float]:
    done, spent = set(), 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            spent += r.get("cost_usd", 0.0)
            if r["outcome"] != "harness_error":
                done.add(r["run_id"])
    return done, spent


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--sizes", nargs="+", default=list(SIZES))
    p.add_argument("--tasks", nargs="+", default=[t.task_id for t in TASKS])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--budget-usd", type=float, default=40.0)
    p.add_argument("--out", default=str(ROOT / "artifacts" / "runs.jsonl"))
    p.add_argument("--fake", action="store_true", help="offline scripted model; no key, no spend")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, spent = load_done(out)
    todo = [r for r in plan(a.arms, a.sizes, a.tasks, a.trials) if run_id(*r) not in done]
    print(f"{len(todo)} runs to do, ${spent:.2f} already spent, budget ${a.budget_usd:.2f}", flush=True)

    def work(r: tuple) -> dict:
        arm, size, task_id, trial = r
        model = make_model(a.fake, gold=gold(task_id)) if a.fake else shared_model
        return run_one(arm, size, task_id, trial, model)

    shared_model = None if a.fake else make_model()
    with ThreadPoolExecutor(a.concurrency) as pool, out.open("a") as fh:
        pending = iter(todo)
        futures = {}
        for r in pending:
            futures[pool.submit(work, r)] = r
            if len(futures) >= a.concurrency:
                break
        while futures:
            fut = next(as_completed(futures))
            futures.pop(fut)
            rec = fut.result()
            spent += rec["cost_usd"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{rec['run_id']:<40} {rec['outcome']:<16} calls={rec['model_calls']:<3} "
                  f"in={rec['input_tokens']:<8} max_req={rec['max_request_input_tokens']:<7} "
                  f"${rec['cost_usd']:.4f}  total ${spent:.2f}", flush=True)
            if spent >= a.budget_usd:
                print("budget reached; not scheduling more runs", flush=True)
                continue
            nxt = next(pending, None)
            if nxt is not None:
                futures[pool.submit(work, nxt)] = nxt


if __name__ == "__main__":
    main()
