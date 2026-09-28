"""No key, no spend. Validates the data, the oracles, the grader, the failure
classifier and each scaffold's context behaviour before any paid run."""

import json

import pytest

from support_audit.arms import ARMS
from support_audit.data import ACCOUNTS, SIZES, export_payload, structured_corpus
from support_audit.fake_model import RequestTooLarge, ScriptedModel
from support_audit.grading import classify_exception, grade
from support_audit.metering import ModelCallCapExceeded
from support_audit.runner import run_one, scrub
from support_audit.tasks import TASKS, gold


def test_corpus_is_deterministic_and_size_independent():
    c = structured_corpus()
    assert set(c) == set(ACCOUNTS)
    for acct in ACCOUNTS:
        rows = {s: [json.loads(l) for l in export_payload(acct, s).splitlines()[1:]] for s in SIZES}
        strip = lambda r: {k: v for k, v in r.items() if k != "transcript"}  # noqa: E731
        assert all([strip(r) for r in rows[s]] == [strip(r) for r in rows["S"]] for s in SIZES)


def test_payload_sizes_straddle_the_eviction_threshold():
    threshold = 4 * 20000  # deepagents NUM_CHARS_PER_TOKEN * tool_token_limit_before_evict
    lens = {s: len(export_payload("ACC-101", s)) for s in SIZES}
    assert lens["S"] < lens["M"] < threshold < lens["L"] < lens["XL"]


def test_bodies_carry_no_answer_signal():
    for s in SIZES:
        for acct in ACCOUNTS:
            for line in export_payload(acct, s).splitlines()[1:]:
                body = json.loads(line)["transcript"]
                assert "E-" not in body and "P1" not in body


def test_gold_answers_are_stable():
    assert {t.task_id: gold(t.task_id) for t in TASKS} == {
        "T1": "3", "T2": "T-102-014", "T3": "Marcus Webb", "T4": "ACC-105",
        "T5": "8", "T6": "12", "T7": "ACC-106", "T8": "4",
    }


@pytest.mark.parametrize("kind,text,gold_,expected", [
    ("int", "blah\nFINAL ANSWER: 12", "12", "pass"),
    ("int", "FINAL ANSWER: **12** tickets", "12", "pass"),
    ("int", "FINAL ANSWER: 1,2", "12", "pass"),
    ("int", "FINAL ANSWER: 11", "12", "wrong_answer"),
    ("int", "I think 12.", "12", "no_answer"),
    ("id", "FINAL ANSWER: `acc-105`", "ACC-105", "pass"),
    ("id", "FINAL ANSWER: ACC-104", "ACC-105", "wrong_answer"),
    ("id", "FINAL ANSWER: T-102-014.", "T-102-014", "pass"),
    ("name", "FINAL ANSWER: **Marcus Webb**", "Marcus Webb", "pass"),
    ("name", "FINAL ANSWER: marcus  webb.", "Marcus Webb", "pass"),
    ("name", "FINAL ANSWER: Marcus", "Marcus Webb", "wrong_answer"),
    ("int", "FINAL ANSWER: 3\nwait, recount\nFINAL ANSWER: 12", "12", "pass"),
])
def test_grader_known_good_and_bad(kind, text, gold_, expected):
    assert grade(kind, text, gold_)[0] == expected


def test_classifier():
    assert classify_exception(RequestTooLarge("Request too large for gpt: Limit 200000")) == "context_ceiling"
    assert classify_exception(RuntimeError("Error code: 400 - context_length_exceeded")) == "context_ceiling"
    assert classify_exception(ModelCallCapExceeded("model call cap 40 exceeded")) == "step_limit"
    assert classify_exception(RuntimeError("Error code: 429 - Rate limit reached for requests")) == "harness_error"
    assert classify_exception(RuntimeError("Error code: 503 - upstream")) == "harness_error"


def test_scrub(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-abcdefghijklmnopqrstuvwxyz")
    s = scrub("key sk-live-abcdefghijklmnopqrstuvwxyz in org-5klKTy2fuBGVlYpzMZSVBjOQ")
    assert "abcdefghijkl" not in s and "5klKTy" not in s


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_passes_with_an_oracle_model(arm):
    rec = run_one(arm, "S", "T6", 1, ScriptedModel(gold=gold("T6")))
    assert rec["outcome"] == "pass", rec["error"]
    assert rec["tool_calls"].get("export_tickets") == 3
    assert rec["exports_in_main_context"] == 3


@pytest.mark.parametrize("mode,expected", [("wrong", "wrong_answer"), ("silent", "no_answer")])
def test_bad_models_are_caught(mode, expected):
    rec = run_one("langgraph", "S", "T1", 1, ScriptedModel(gold=gold("T1"), mode=mode))
    assert rec["outcome"] == expected


def test_offload_is_on_only_where_expected():
    """The core mechanism under test: at L, deepagents evicts each export to a
    file; with eviction off, and in plain LangGraph, the full payload stays in
    context."""
    got = {arm: run_one(arm, "L", "T6", 1, ScriptedModel(gold=gold("T6"))) for arm in ARMS}
    assert got["deepagents"]["exports_evicted_in_main_context"] == 3
    assert got["deepagents_no_offload"]["exports_evicted_in_main_context"] == 0
    assert got["langgraph"]["exports_evicted_in_main_context"] == 0
    assert got["deepagents"]["max_request_input_tokens"] < got["deepagents_no_offload"]["max_request_input_tokens"]


def test_no_eviction_below_threshold():
    rec = run_one("deepagents", "M", "T6", 1, ScriptedModel(gold=gold("T6")))
    assert rec["exports_evicted_in_main_context"] == 0


def test_simulated_ceiling_is_a_subject_failure_not_a_harness_error():
    rec = run_one("langgraph", "XL", "T6", 1, ScriptedModel(gold=gold("T6"), ceiling_tokens=200_000))
    assert rec["outcome"] == "context_ceiling"
    rec = run_one("deepagents", "XL", "T6", 1, ScriptedModel(gold=gold("T6"), ceiling_tokens=200_000))
    assert rec["outcome"] == "pass"


def test_report_and_verifier_run_end_to_end_on_fake_runs(tmp_path, monkeypatch):
    """The whole pipeline — runner, report, verifier checks — on scripted runs."""
    from support_audit import report, runner, verify

    out = tmp_path / "runs.jsonl"
    runner.main(["--fake", "--tasks", "T1", "T4", "--sizes", "S", "L", "--trials", "1",
                 "--concurrency", "2", "--out", str(out)])
    runs = report.load_runs(out)
    assert len(runs) == 2 * 2 * 3 and all(r["outcome"] == "pass" for r in runs)
    summary = report.build_summary(runs)
    assert summary["grid"]["deepagents"]["L"]["accuracy"] == 1.0
    problems: list[str] = []
    assert sum(verify._check_trajectory(r, problems) for r in runs) > 0 and not problems


def test_summarisation_cannot_rescue_one_oversized_step():
    """Documents a finding, offline. With eviction off, three XL exports land in
    a single step. Deep Agents' summarisation trigger (170k tokens for a model
    with no profile) is exceeded, but `keep=("messages", 6)` covers the whole
    conversation, so nothing is summarised and the oversized request goes out."""
    from langchain_openai import ChatOpenAI
    from deepagents.middleware.summarization import compute_summarization_defaults

    defaults = compute_summarization_defaults(ChatOpenAI(model="gpt-5.4-mini-2026-03-17", api_key="unused"))
    assert defaults["trigger"] == ("tokens", 170000) and defaults["keep"] == ("messages", 6)
    rec = run_one("deepagents_no_offload", "XL", "T6", 1, ScriptedModel(gold=gold("T6")))
    assert rec["model_calls"] == 2  # no summarisation call happened
    assert rec["per_call_input_tokens"][-1] > 170_000


def test_openai_request_too_large_is_not_a_context_overflow_error():
    """Why Deep Agents' ContextOverflowError fallback never engaged: OpenAI's
    per-request TPM rejection is a 429, which langchain-openai maps to a rate-limit
    error, not ContextOverflowError."""
    import httpx
    import openai
    from langchain_core.exceptions import ContextOverflowError
    from langchain_openai.chat_models.base import _handle_openai_api_error

    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    err = openai.RateLimitError("Request too large for gpt-5.4-mini: Limit 200000, Requested 238431",
                                response=httpx.Response(429, request=req), body=None)
    with pytest.raises(Exception) as got:
        _handle_openai_api_error(err)
    assert not isinstance(got.value, ContextOverflowError)
    assert classify_exception(got.value) == "context_ceiling"
