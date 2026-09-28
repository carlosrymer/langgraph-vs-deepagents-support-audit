"""Build site/data/summary.json from artifacts/runs.jsonl. Pure function of the
committed artifacts — no key, no network."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from support_audit.arms import ARMS
from support_audit.data import SIZES, export_payload
from support_audit.grading import OUTCOMES
from support_audit.metering import PRICE_PER_M, cost_usd
from support_audit.runner import MAX_OUTPUT_TOKENS, MODEL, REASONING_EFFORT, ROOT
from support_audit.tasks import TASKS, gold

RUNS = ROOT / "artifacts" / "runs.jsonl"
PILOT = ROOT / "artifacts" / "pilot" / "runs.jsonl"
SUMMARY = ROOT / "site" / "data" / "summary.json"
VERSIONS = {"langgraph": "1.2.12", "deepagents": "0.7.19", "langchain": "1.4.2",
            "langchain-core": "1.6.5", "langchain-openai": "1.6.6"}


def load_all(path: Path = RUNS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_runs(path: Path = RUNS) -> list[dict[str, Any]]:
    """Latest record per run_id wins (a harness_error that was re-run is superseded)."""
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        latest[r["run_id"]] = r
    return list(latest.values())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


_REQUESTED = re.compile(r"Requested (\d+)")


def attempted_peak(r: dict[str, Any]) -> int:
    """Largest request the scaffold tried to send. A rejected request is never
    metered, so for context_ceiling runs I take the size OpenAI quotes back in the
    429 ("Requested N" — input plus the max-output reservation)."""
    m = _REQUESTED.search(r.get("error") or "")
    return max(r["max_request_input_tokens"], int(m.group(1)) if m else 0)


def read_strategy(r: dict[str, Any]) -> str:
    """How a run got at offloaded exports: searched them, paged through windows, or both."""
    reads = r["tool_calls"].get("read_file", 0)
    greps = r["tool_calls"].get("grep", 0)
    if greps and reads:
        return "grep_and_read_file"
    if greps:
        return "grep_only"
    if reads:
        return "read_file_windows_only"
    return "no_file_access"


def _cell(rs: list[dict]) -> dict[str, Any]:
    scored = [r for r in rs if r["outcome"] != "harness_error"]
    k = sum(r["outcome"] == "pass" for r in scored)
    n = len(scored)
    tokens_in = sum(r["input_tokens"] for r in rs)
    return {
        "runs": len(rs), "scored": n, "passed": k,
        "accuracy": round(k / n, 4) if n else None, "ci95": wilson(k, n),
        "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in OUTCOMES},
        "mean_input_tokens": round(tokens_in / len(rs)) if rs else 0,
        "mean_output_tokens": round(sum(r["output_tokens"] for r in rs) / len(rs)) if rs else 0,
        "mean_model_calls": round(sum(r["model_calls"] for r in rs) / len(rs), 2) if rs else 0,
        "max_request_input_tokens": max((r["max_request_input_tokens"] for r in rs), default=0),
        "mean_attempted_peak_tokens": round(sum(attempted_peak(r) for r in rs) / len(rs)) if rs else 0,
        "max_attempted_peak_tokens": max((attempted_peak(r) for r in rs), default=0),
        "cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
        "mean_cost_usd": round(sum(r["cost_usd"] for r in rs) / len(rs), 5) if rs else 0,
        "cost_per_pass_usd": round(sum(r["cost_usd"] for r in rs) / k, 5) if k else None,
    }


def build_summary(runs: list[dict[str, Any]], attempts: list[dict[str, Any]] | None = None,
                  pilot: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    attempts = attempts if attempts is not None else runs
    pilot = pilot or []
    by: dict[tuple, list] = defaultdict(list)
    for r in runs:
        by[(r["arm"], r["size"])].append(r)
        by[(r["arm"], r["size"], r["n_accounts"])].append(r)
        by[(r["arm"],)].append(r)
        by[(r["arm"], "evicting" if r["size"] in ("L", "XL") else "non_evicting")].append(r)

    grid = {a: {s: _cell(by[(a, s)]) for s in SIZES} for a in ARMS}
    by_accounts = {a: {s: {str(n): _cell(by[(a, s, n)]) for n in (1, 2, 3)} for s in SIZES} for a in ARMS}
    per_arm = {a: _cell(by[(a,)]) for a in ARMS}
    bands = {a: {b: _cell(by[(a, b)]) for b in ("non_evicting", "evicting")} for a in ARMS}

    tools = {a: dict(sum((Counter(r["tool_calls"]) for r in by[(a,)]), Counter())) for a in ARMS}
    evictions = {a: sum(r["exports_evicted_in_main_context"] for r in by[(a,)]) for a in ARMS}
    total_in = sum(r["input_tokens"] for r in runs)
    total_cached = sum(r["cached_input_tokens"] for r in runs)
    total_out = sum(r["output_tokens"] for r in runs)

    offloaded = [r for r in runs if r["exports_evicted_in_main_context"] > 0]
    strategies: dict[str, dict[str, int]] = {}
    for r in offloaded:
        st = strategies.setdefault(read_strategy(r), {"runs": 0, "passed": 0})
        st["runs"] += 1
        st["passed"] += r["outcome"] == "pass"
    da = by[("deepagents",)]

    g = grid
    money = lambda x: "n/a" if x is None else f"${x:.3f}"  # noqa: E731
    pct = lambda x: "n/a" if x is None else f"{round(100 * x)}%"  # noqa: E731
    headline = {
        "da_xl": pct(g["deepagents"]["XL"]["accuracy"]),
        "lg_xl": pct(g["langgraph"]["XL"]["accuracy"]),
        "nooff_xl": pct(g["deepagents_no_offload"]["XL"]["accuracy"]),
        "lg_ceiling_runs": f'{sum(g["langgraph"][s]["outcomes"]["context_ceiling"] for s in SIZES)}',
        "nooff_ceiling_runs": f'{sum(g["deepagents_no_offload"][s]["outcomes"]["context_ceiling"] for s in SIZES)}',
        "da_ceiling_runs": f'{sum(g["deepagents"][s]["outcomes"]["context_ceiling"] for s in SIZES)}',
        "total_cost": f"${cost_usd(total_in, total_cached, total_out):.2f}",
        "total_runs": f"{len(runs)}",
        "da_l_cost_run": money(g["deepagents"]["L"]["mean_cost_usd"]),
        "lg_l_cost_run": money(g["langgraph"]["L"]["mean_cost_usd"]),
        "da_xl_cost_pass": money(g["deepagents"]["XL"]["cost_per_pass_usd"]),
        "lg_xl_cost_pass": money(g["langgraph"]["XL"]["cost_per_pass_usd"]),
        "da_l_acc": pct(g["deepagents"]["L"]["accuracy"]),
        "lg_l_acc": pct(g["langgraph"]["L"]["accuracy"]),
        "nooff_l_acc": pct(g["deepagents_no_offload"]["L"]["accuracy"]),
        "s_input_overhead": pct(g["deepagents"]["S"]["mean_input_tokens"] / g["langgraph"]["S"]["mean_input_tokens"] - 1)
                            if g["langgraph"]["S"]["mean_input_tokens"] else "n/a",
        "grep_passes": f'{strategies.get("grep_only", {}).get("passed", 0) + strategies.get("grep_and_read_file", {}).get("passed", 0)}/'
                       f'{strategies.get("grep_only", {}).get("runs", 0) + strategies.get("grep_and_read_file", {}).get("runs", 0)}',
        "window_passes": f'{strategies.get("read_file_windows_only", {}).get("passed", 0)}/'
                         f'{strategies.get("read_file_windows_only", {}).get("runs", 0)}',
        "da_runs_with_todos": f'{sum(r["todos"] > 0 for r in da)}',
        "da_subagent_calls": f'{sum(r["tool_calls"].get("task", 0) for r in da)}',
        "all_spend": f"${round(sum(r['cost_usd'] for r in attempts) + sum(r['cost_usd'] for r in pilot), 2):.2f}",
    }
    return {
        "meta": {
            "model": MODEL, "reasoning_effort": REASONING_EFFORT, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "versions": VERSIONS, "price_per_m_usd": PRICE_PER_M, "arms": list(ARMS), "sizes": list(SIZES),
            "payload_chars": {s: len(export_payload("ACC-101", s)) for s in SIZES},
            # o200k_base token counts of the ACC-101 export, measured with tiktoken at build
            # time. Quoted, not verified: tiktoken needs a network download.
            "payload_tokens_o200k": {"S": 5395, "M": 14567, "L": 39275, "XL": 78370},
            "eviction_threshold_chars": 80000,
            "tasks": [{"task_id": t.task_id, "accounts": list(t.accounts), "kind": t.kind,
                       "question": t.question, "gold": gold(t.task_id)} for t in TASKS],
        },
        "totals": {"runs": len(runs), "input_tokens": total_in, "cached_input_tokens": total_cached,
                   "output_tokens": total_out, "model_calls": sum(r["model_calls"] for r in runs),
                   "cost_usd": round(cost_usd(total_in, total_cached, total_out), 4),
                   "harness_errors": sum(r["outcome"] == "harness_error" for r in runs),
                   # Every record ever written, including harness-error attempts later
                   # superseded by a re-run, plus the excluded pilot: what I actually spent.
                   "attempt_records": len(attempts),
                   "superseded_harness_error_attempts": len(attempts) - len(runs),
                   "attempts_cost_usd": round(sum(r["cost_usd"] for r in attempts), 4),
                   "pilot_runs": len(pilot),
                   "pilot_cost_usd": round(sum(r["cost_usd"] for r in pilot), 4),
                   "all_spend_usd": round(sum(r["cost_usd"] for r in attempts) + sum(r["cost_usd"] for r in pilot), 2)},
        "grid": grid, "by_accounts": by_accounts, "per_arm": per_arm, "bands": bands,
        "tool_calls": tools, "evictions_in_main_context": evictions,
        "offloaded_read_strategies": strategies,
        "headline": headline,
        "runs": [{k: r[k] for k in ("run_id", "arm", "size", "task_id", "trial", "n_accounts", "outcome",
                                    "answer", "gold", "model_calls", "input_tokens", "output_tokens",
                                    "max_request_input_tokens", "cost_usd", "wall_s",
                                    "exports_evicted_in_main_context", "tool_calls", "error")}
                 | {"attempted_peak_tokens": attempted_peak(r)}
                 for r in sorted(runs, key=lambda r: (r["task_id"], r["size"], r["arm"], r["trial"]))],
    }


def main() -> None:
    summary = build_summary(load_runs(), load_all(), load_all(PILOT))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"headline": summary["headline"], "totals": summary["totals"]}, indent=1))


if __name__ == "__main__":
    main()
