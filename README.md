# LangGraph vs Deep Agents: what happens when the tool output gets big

**Try it live: [https://carlosrymer.github.io/langgraph-vs-deepagents-support-audit/](https://carlosrymer.github.io/langgraph-vs-deepagents-support-audit/)**

I held the model, the tools and the task prompt fixed and swapped only the agent scaffold. Then I ran
288 support-ops audits through a plain LangGraph loop, LangChain's Deep Agents, and Deep Agents
with its tool-result offloading switched off. I wanted to see whether Deep Agents' context
management actually holds up as tool outputs grow.

## Deep Agents held 88% where plain LangGraph fell to 50%, and turning off one middleware setting erased the gap

At the largest payload size (~78k tokens per account export), `create_deep_agent` with default
settings answered **88%** of audits correctly (21/24). A hand-built LangGraph ReAct loop answered
**50%** (12/24), and so did Deep Agents with large-tool-result offloading disabled: **50%**. The
plain loop hit the context ceiling on **9** of its 24 XL runs, the ablated Deep Agents also on
**9**, and default Deep Agents on **0**. Offloading also made it cheaper: **$0.047** per correct
answer at XL against **$0.105** for plain LangGraph.

**The caveat that most undercuts this:** the ceiling those two arms hit was **not** the model's
400k context window. It was my OpenAI org's 200,000 tokens-per-minute limit, which rejects any
*single request* above 200k tokens with a 429. On a higher usage tier, the three-account XL tasks
(~236k tokens requested in one call, counting the 8k output reservation OpenAI adds) would have been sent. Below that ceiling, at the L size (~39k
tokens per export), the accuracy differences are inside the noise: **88%** Deep Agents vs **79%**
LangGraph vs **83%** no-offload, with overlapping 95% intervals. What held there is cost: **$0.028**
per run for Deep Agents against **$0.061** for plain LangGraph.

## What this showcases

**Technology:** Deep Agents `deepagents==0.7.19` (LangChain) — an agent harness on top of LangGraph
that adds a virtual filesystem, large-tool-result offloading, summarisation, a `write_todos`
planner and `task` subagents. The control is `langgraph==1.2.12`: the same model with the same
tools, in a `StateGraph` + `ToolNode` loop.

The claim I tested, from LangChain's own framing: Deep Agents "includes middleware that helps
agents compress conversation history, offload large tool results, isolate context with subagents,
and use prompt caching", so it handles long, complex tasks that a plain agent doesn't. LangChain
publishes no numbers against plain LangGraph, so the testable form is mine. It's fixed in
[`PRD.md`](PRD.md), which I committed before the main run: with model, tools and prompt held
fixed, Deep Agents is more accurate than a plain LangGraph loop once tool outputs get large, **and
the gain comes from offloading specifically**.

- **Held fixed:** `gpt-5.4-mini-2026-03-17` (Responses API, reasoning effort `low`), two tools
  (`list_accounts`, `export_tickets`), one system prompt, eight questions with oracle answers, a
  40-model-call cap.
- **Varied:** the scaffold (three arms) × a payload-size dial. Each ticket carries a support
  transcript whose length sets the export size: S ≈ 5.4k, M ≈ 14.6k, L ≈ 39k, XL ≈ 78k tokens per
  account. Deep Agents offloads a tool result at 80,000 characters (~20k tokens), so M sits just
  under the threshold and L/XL sit over it.
- **Why this isolates context pressure:** transcripts carry no answer-relevant signal, and every
  gold answer is identical at every size. The only thing that changes across the dial is how much
  text lands in context.

### What surprised me

**1. Of everything in the harness, only offloading did any measurable work in this experiment.**
The ablation arm keeps Deep Agents' base prompt, planner, subagents, summarisation and filesystem
tools, and it tracked plain LangGraph almost exactly: 50% vs 50% at XL, and the same 9 ceiling
failures each. Default Deep Agents wrote a todo list in **0** of 96 runs and delegated to a
subagent **1** time in 96 runs. These tasks are short and data-heavy, and planning and delegation
might matter on longer tasks. Here, one middleware setting accounted for the whole gap.

**2. Summarisation can't save a single oversized step.** With offloading off, the three-account XL
tasks deliver ~236k tokens of tool results in one parallel step. Deep Agents' summarisation trigger
for this model is 170k tokens (the dated model string has no profile, so the fallback default
applies), and it *is* exceeded. But the default `keep=("messages", 6)` covers the whole
conversation at that point (system, question, one tool-calling turn and three results), so there's
nothing old enough to summarise, and the oversized request goes out unchanged. The middleware also
has a fallback that summarises and retries on `ContextOverflowError`. OpenAI's per-request
rejection is a 429 rate-limit error, which `langchain-openai` doesn't map to
`ContextOverflowError`, so that fallback never fires. Both behaviours are reproduced offline in the
test suite.

**3. Offloading swaps one failure mode for another.** After offloading, the model sees a pointer
and a preview, and has to go back to the file. At L and XL, runs where Deep Agents searched the
file with `grep` passed **38/38**. Runs where it only paged through `read_file` windows passed
**4/10**, and every Deep Agents failure at those sizes was one of them. The worst case was T8
("tickets tagged `sso` created in August"): the model guessed that August tickets would sit around
line 30 of a date-sorted file, read a 10–15-line window from each account, and missed the rest.
The plain loop can't make that mistake, because the whole export is in front of it until it
doesn't fit. This analysis is descriptive: I defined it after seeing the runs, and strategy is
confounded with task.

**4. The overhead at small sizes is real but modest.** At S, Deep Agents used **34%** more input
tokens per run than plain LangGraph, which is its own base prompt and tool schemas. Cost only rose
about 5% ($0.0101 vs $0.0096 per run), because that static prompt is cached across calls.

## The use case

B2B SaaS support operations. An analyst agent answers audit questions about customer accounts from
full ticket-history exports: "how many P1 tickets are unresolved across these three accounts", "who
resolved the most tickets", "which account has the most distinct error codes". Exports like these
are exactly the tool results that blow up naive agent loops. Six fictional accounts, 40 tickets
each, eight questions over one, two or three accounts. Every answer is an exact value from an oracle
over the structured fields, so no model grades anything.

The control is deliberately the naive loop. A LangGraph user who adds their own truncation, a
retrieval tool or a `grep`-style tool could close some or all of this gap. The question here is
what Deep Agents gives you **out of the box**, not whether LangGraph can be made to do the same.

## Docs

- [Architecture](ARCHITECTURE.md) — system design, components, data flow, deployment
- [PRD](PRD.md) — problem statement, scope, pre-registered success criteria

## Running locally

```bash
# 1. Install (Python 3.12, uv)
uv sync

# 2. The full test suite — no API key, no spend.
#    Drives all three real graphs with a scripted model, including the real eviction middleware.
uv run pytest -q

# 3. Re-derive every published number from the committed artifacts.
#    No API key, no spend. Exit 0 means every figure above re-derives.
uv run python -m support_audit.verify

# 4. View the site (reads the committed JSON over HTTP)
cd site && python -m http.server 8000   # then open http://localhost:8000

# 5. Optional: re-run the experiment. Costs money (~$10 at list price) and appends to artifacts/runs.jsonl.
#    Needs OPENAI_API_KEY. Move artifacts/runs.jsonl aside first for a clean run.
uv run python -m support_audit.runner --budget-usd 40
uv run python -m support_audit.report
```

## Committed artifacts

- `artifacts/runs.jsonl` — every run record ever written: 288 scored runs plus 61 earlier attempts
  that ended in a transient rate-limit `harness_error` and were re-run, as the pre-registered rule
  requires (the latest record per `run_id` wins). Each record has the outcome, answer, gold, every
  metered token count, per-call input sizes, tool-call counts, the error text (scrubbed), and a
  compacted trajectory.
- `artifacts/pilot/runs.jsonl` — the 12-run pilot that preceded the main run. Excluded from every
  published number; its cost is included in the spend below.
- `site/data/summary.json` — everything the site shows, built from the above by
  `support_audit.report`.

`support_audit.verify` regenerates gold answers from the seed, re-grades every run, re-classifies
every recorded error, checks every export payload recorded in a trajectory against the seed (by
sha256), rebuilds `summary.json` and diffs it, and checks every headline figure appears in this
README.

**What it can't check:** that the model actually produced the recorded text and token counts;
that needs the API, and the API isn't deterministic anyway. It also can't check the o200k token
sizes of the payloads quoted above, because tiktoken needs a network download, so I quote them
rather than verify them. And it can't see inside Deep Agents' subagent histories, which aren't
part of the main graph state. Their model calls, tokens and tool calls are counted via callbacks,
but their messages aren't committed. Strings longer than 2,000 characters in trajectories are
stored as `{sha256, chars, head}` to keep the repo small. Every export payload can be regenerated
from the seed, so those hashes are checkable.

## Stack

Python 3.12 + `uv`; `langgraph` 1.2.12, `deepagents` 0.7.19, `langchain` 1.4.2, `langchain-core`
1.6.5, `langchain-openai` 1.6.6; OpenAI `gpt-5.4-mini-2026-03-17` through the Responses API. A plain
HTML/CSS/JS site with inline SVG charts, no framework. GitHub Pages.

## What it cost

Metered from `usage_metadata` on every model response, including subagent and summarisation calls,
at OpenAI's published gpt-5.4-mini Standard rates ($0.75 / $0.075 cached / $4.50 per 1M
tokens). The 288 scored runs cost **$9.51** (13.8M input tokens, 1.9M of them cached, 94k output
tokens, 656 model calls). Counting the superseded rate-limited attempts and the pilot, I spent
**$9.80** in total. These are list-price calculations from metered tokens, not a figure read from
an OpenAI balance. The hard cap was $40.

## Honest limitations

- **The cliff is tier-specific.** Every ceiling failure was OpenAI's per-request TPM rejection at
  200k tokens, not the 400k context window. I decided before the main run to count it as the
  scaffold's failure. On a higher tier, the plain loop's XL three-account runs would have been
  sent, and I don't know how accurate they'd have been. There's a hint in the two-account XL
  tasks, which were sent at ~157k tokens: plain LangGraph got 3 of 6 right, and all three misses
  were wrong answers, not ceilings. That's too few runs to conclude anything.
- **Three trials per cell.** Wilson 95% intervals on 24 runs are up to about ±19 points. The XL gap
  clears them, but only just: Deep Agents' interval starts at 69%, and plain LangGraph's ends at
  68.6%. The S, M and L differences don't clear them.
- **One model.** A model with a larger per-request budget, or one that uses long contexts better or
  worse, moves every number.
- **The ablation is partial.** It turns eviction off on the main agent only. Deep Agents' default
  general-purpose subagent keeps its own eviction. That barely matters here, since there was one
  subagent call in 96 runs, but it isn't a clean all-off switch.
- **Post-hoc analyses are labelled.** The read-strategy breakdown and the "largest attempted
  request" metric were defined after I saw the runs. The scoring rule, the outcome classes and the
  accuracy/cost definitions were not changed.
- **The baseline is naive by design.** A better-engineered LangGraph agent is a different
  experiment.
- **My own harness bug:** the first pilot sent `reasoning_effort` with function tools to
  `/v1/chat/completions`, which gpt-5.4-mini rejects. All 12 runs were correctly classified as
  `harness_error`. I switched to the Responses API and re-ran the pilot before the main run.

## Did Deep Agents deliver?

**Held, with caveats.** When tool outputs are big enough that keeping them in context breaks the
request, Deep Agents' default offloading turned a 50% scaffold into an 88% one, at less than half
the cost per correct answer. The ablation pins that gain on offloading and nothing else. The
caveats LangChain's framing leaves out:

- The accuracy win only showed up past a hard per-request ceiling, and on my account that ceiling
  was a rate-limit tier, not the model.
- Below it, the durable benefit is cost, not correctness.
- The rest of the harness (planning, subagents, summarisation) contributed nothing measurable on
  these tasks.
- Offloading introduces its own failure: the model reads part of a file and answers as if it had
  read all of it.

## Deployed via

GitHub Pages (`gh-pages` branch, mirrored into `docs/` on `main`) via `scripts/deploy_pages.sh`.

---
Part of an ongoing series of small, real-world builds trialing frontier AI models, frameworks,
and tools as they ship.
