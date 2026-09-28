# PRD — LangGraph vs Deep Agents: context under pressure

## The claim under test

> Deep Agents "includes middleware that helps agents compress conversation history, offload large
> tool results, isolate context with subagents, and use prompt caching" — so that an agent built
> with it handles "complex, non-deterministic, and long running tasks" that a plain LangGraph agent
> does not handle out of the box.

Paraphrased from LangChain's Deep Agents product page and overview docs (read 2026-09-28). The
testable form I committed to before running anything: **with the model, tools and task prompt held
fixed, `create_deep_agent` answers multi-step account audits more accurately than a hand-built
LangGraph ReAct loop once tool outputs get large, and that advantage comes from offloading large
tool results specifically.**

LangChain publishes no head-to-head numbers against plain LangGraph, so this is my framing of a
qualitative claim, not a vendor benchmark I am re-checking.

## Problem statement

Support-ops and RevOps agents live on exports: a ticket history, a usage log, a CRM dump. Those
tool results are big, and a naive agent loop keeps every one of them in context for every later
model call. Deep Agents' pitch is that its harness handles this for you. I want to know whether it
does, what it costs when outputs are small, and which part of the harness is doing the work.

## Target user

An engineer choosing between writing a LangGraph agent by hand and adopting `deepagents` for an
agent that calls data-heavy tools.

## Experiment design

**Held fixed (control):** model `gpt-5.4-mini-2026-03-17` (Responses API, reasoning effort `low`,
max 8,000 output tokens); the two domain tools (`list_accounts`, `export_tickets`); the task system
prompt; the eight questions and their gold answers; a 40-model-call cap per run.

**Varied (the independent variable):** the scaffold, across three arms —
- `langgraph` — `StateGraph` + `ToolNode` ReAct loop, `langgraph==1.2.12`
- `deepagents` — `create_deep_agent`, all defaults, `deepagents==0.7.19`
- `deepagents_no_offload` — same, with the main agent's large-tool-result eviction switched off

…crossed with a **payload-size dial**: every ticket carries a support transcript whose length sets
the export size — S ≈ 5.4k, M ≈ 14.6k, L ≈ 39k, XL ≈ 78k tokens per account (o200k_base). M sits
under Deep Agents' 20k-token eviction threshold; L and XL sit over it. Transcripts carry no
answer-relevant signal and **gold answers are identical at every size**, so any accuracy change
across sizes is caused by context pressure alone.

**Prediction if the claim holds:** at S and M the three arms are within noise of each other; at L
and XL `deepagents` beats both `langgraph` and `deepagents_no_offload`, and `deepagents_no_offload`
tracks `langgraph` (showing offloading, not the rest of the harness, is what matters).

**What would show it false:** `deepagents` accuracy at L/XL is not higher than `langgraph`'s; or it
is higher only because `deepagents_no_offload` is also higher (the gain comes from the prompt /
todos / subagents, not offloading); or the gain costs more tokens than it saves.

**Scale:** 8 tasks × 4 sizes × 3 trials × 3 arms = 288 runs. Tasks cover 1, 2 or 3 accounts
(3 / 2 / 3 tasks), so a run makes 1–3 large exports.

A 12-run pilot (T1, T6 × S, XL × 3 arms × 1 trial) ran first to shake out the harness; it is
committed under `artifacts/pilot/` and is excluded from every published number.

## Pre-registered scoring rule

Fixed before the main run and implemented in `src/support_audit/grading.py`. Each run gets exactly
one outcome:

- **pass** — the text after the last `FINAL ANSWER:` in the final assistant message matches gold.
  int tasks compare the first integer; id tasks compare the first `T-###-###` / `ACC-###` token,
  upper-cased; name tasks compare the full string, case-folded, with quotes, backticks, markdown
  emphasis and trailing punctuation stripped and whitespace collapsed.
- **wrong_answer** — a `FINAL ANSWER:` line that does not match.
- **no_answer** — the run ended without a `FINAL ANSWER:` line.
- **context_ceiling** — the provider rejected a request as too large (OpenAI "Request too large" /
  `context_length_exceeded`). Deterministic — retrying cannot help — so it is a **failure of the
  scaffold under test** to manage its context, not a harness failure.
- **step_limit** — more than 40 model calls (subagent calls included) or the graph recursion limit.
- **harness_error** — anything else: transient rate limit after 6 retries, 5xx, network, a bug in my
  code. Excluded from accuracy denominators, reported separately, and re-run.

**Accuracy** = pass / (runs − harness_error), per arm × size. **Cost** = metered input, cached
input and output tokens from every model call in the run, including subagents and summarisation,
priced at OpenAI's published gpt-5.4-mini Standard rates ($0.75 / $0.075 / $4.50 per 1M).

The note about the account's rate limit belongs here too: the pilot showed this org has a 200,000
tokens-per-minute limit on gpt-5.4-mini, and OpenAI rejects any **single request** above it with a
429 "Request too large" — well below the model's 400k context window. I decided before the main
run that this counts as `context_ceiling`, and that the README must say the effective ceiling is a
property of the account tier.

## Goals

- Measure accuracy and token cost of the three scaffolds across the size dial.
- Attribute any gap to a specific Deep Agents mechanism via the ablation arm.
- Ship every raw run record and a verifier that re-derives every published number offline.

## Non-goals

- Comparing against a hand-tuned LangGraph agent with its own truncation or retrieval. The control
  is deliberately the naive loop; the question is what Deep Agents gives you out of the box.
- Ranking models. One model, held fixed.
- Testing Deep Agents' long-term memory, skills, sandboxes or human-in-the-loop.

## Scope (MVP)

Seeded corpus + oracle, three scaffolds, offline scripted model and test suite, metered paid run,
verifier, static results site on GitHub Pages.

## User stories

- As an engineer choosing a scaffold, I want to see where the naive loop breaks and what Deep
  Agents costs me when it doesn't, so that I pick based on my tools' output sizes.
- As a skeptical reader, I want to re-derive every number without an API key, so that I don't
  have to trust the write-up.

## Success criteria

The build succeeds if it answers the claim either way, with the pre-registered rule unchanged. The
claim **holds** if `deepagents` beats `langgraph` at L/XL and `deepagents_no_offload` does not;
**holds with caveats** if that is true only under a condition LangChain's framing leaves out;
**splits** if the gap exists but is not attributable to offloading; **doesn't hold** if there is
no gap.

## Budget

Hard stop at $40 of metered spend (`--budget-usd`), enforced by the runner before scheduling each
run; 40 model calls per run. Known quota: 200k TPM on gpt-5.4-mini for this org, which also rate
limits concurrent runs — handled with 6 client retries and 3 concurrent runs.

## Risks / open questions

- The 200k per-request ceiling is tier-specific; a higher tier moves where the naive loop breaks.
- `deepagents_no_offload` disables eviction on the main agent only; the default general-purpose
  subagent keeps its own filesystem middleware with eviction on.
- Counting over 40–120 tickets is error-prone for any scaffold at any size; S is the baseline for that.

## Timeline

Single session, 2026-09-28.
