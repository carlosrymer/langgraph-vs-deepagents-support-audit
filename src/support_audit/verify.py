"""Re-derive every published number from the committed artifacts.

    uv run python -m support_audit.verify      # exit 0 = everything re-derives

No API key, no network, no spend. It checks, in order:

 1. Gold answers regenerate from the seeded corpus and match every run record.
 2. Every scored run re-grades to the same outcome under the pre-registered
    rule (runs that ended in an exception keep their recorded class, and the
    recorded error text must still classify the same way).
 3. Every export_tickets payload recorded in a trajectory — inline or as a
    sha256 — matches the payload regenerated from the seed for that
    account and size, so the tool outputs the model saw are checkable.
 4. site/data/summary.json equals a fresh rebuild from artifacts/runs.jsonl.
 5. Every headline figure in summary.json appears verbatim in README.md.

What it CANNOT check: that the model actually produced the recorded text and
token counts (that would need re-running against the API, which is
non-deterministic anyway); o200k token sizes of the payloads quoted in the
docs (tiktoken needs a network download, so I quote them, not verify them);
and anything that happened inside a deepagents subagent beyond the model-call,
token and tool-call counts the callbacks recorded — subagent message
histories are not part of the main graph state, so they are not committed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Any

from support_audit.data import export_payload
from support_audit.grading import classify_exception, grade
from support_audit.report import PILOT, RUNS, SUMMARY, build_summary, load_all, load_runs
from support_audit.runner import ROOT
from support_audit.tasks import TASKS_BY_ID, gold


class _Recorded(Exception):
    pass


_HEADER_ACCT = re.compile(r'"account_id": "(ACC-\d{3})"')


def _check_trajectory(r: dict[str, Any], problems: list[str]) -> int:
    """Every non-evicted export result must equal the seed payload for the
    account its own header line names, and that account must be one the run
    asked for. (Parallel tool calls can come back in any order, so results are
    matched by the header, not by position.)"""
    checked = 0
    called = {tc["args"].get("account_id") for m in r["trajectory"] if m["type"] == "ai"
              for tc in m.get("tool_calls", []) if tc["name"] == "export_tickets"}
    for msg in (m for m in r["trajectory"] if m["type"] == "tool" and m.get("name") == "export_tickets"):
        content = msg["content"]
        text = content.get("head", "") if isinstance(content, dict) else str(content)
        if "/large_tool_results/" in text:
            continue  # evicted: the model saw a pointer + preview, not the payload
        m = _HEADER_ACCT.search(text)
        if not m or m.group(1) not in called:
            problems.append(f"{r['run_id']}: export result names no requested account")
            continue
        expected = export_payload(m.group(1), r["size"])
        if isinstance(content, dict):
            ok = content["sha256"] == hashlib.sha256(expected.encode()).hexdigest()
        else:
            ok = content == expected
        checked += 1
        if not ok:
            problems.append(f"{r['run_id']}: export payload for {m.group(1)} does not match seed")
    return checked


def main() -> int:
    problems: list[str] = []
    runs = load_runs(RUNS)
    payloads = 0
    for r in runs:
        task = TASKS_BY_ID[r["task_id"]]
        if r["gold"] != gold(r["task_id"]):
            problems.append(f"{r['run_id']}: recorded gold {r['gold']!r} != regenerated {gold(r['task_id'])!r}")
        if r["error"] is None:
            final = next((m["content"] for m in reversed(r["trajectory"]) if m["type"] == "ai"), None)
            if isinstance(final, dict):
                final = r["final_text"]
            outcome, _ = grade(task.kind, final, r["gold"])
            if outcome != r["outcome"]:
                problems.append(f"{r['run_id']}: re-graded {outcome} != recorded {r['outcome']}")
        else:
            cls = classify_exception(_Recorded(r["error"]))
            if cls != r["outcome"]:
                problems.append(f"{r['run_id']}: error re-classifies as {cls} != recorded {r['outcome']}")
        payloads += _check_trajectory(r, problems)

    # Round-trip through JSON so tuples compare equal to the lists read from disk.
    fresh = json.loads(json.dumps(build_summary(runs, load_all(RUNS), load_all(PILOT)), ensure_ascii=False))
    published = json.loads(SUMMARY.read_text())
    if fresh != published:
        diff = [k for k in fresh if fresh[k] != published.get(k)]
        problems.append(f"site/data/summary.json differs from a fresh rebuild in: {diff}")

    readme = (ROOT / "README.md").read_text()
    for key, val in fresh["headline"].items():
        if val not in readme:
            problems.append(f"README.md does not contain headline figure {key} = {val!r}")

    print(f"runs: {len(runs)}  export payloads checked against seed: {payloads}  "
          f"headline figures: {len(fresh['headline'])}")
    for p in problems:
        print("FAIL", p)
    print("OK — every published number re-derives" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
