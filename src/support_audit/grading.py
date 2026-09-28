"""Pre-registered scoring rule (see PRD.md). Written before any paid run.

A run's outcome is exactly one of:

  pass             final answer matches gold under the normalisation below
  wrong_answer     a FINAL ANSWER line was produced and does not match
  no_answer        the run ended without a FINAL ANSWER line
  context_ceiling  the provider rejected a request as too large for the
                   context window or for the per-request token limit —
                   deterministic, retrying cannot help, so it is a failure of
                   the scaffold under test to manage its context
  step_limit       the run hit the 40-model-call cap or the graph recursion cap
  harness_error    anything else that is not the subject's fault: transient
                   rate limit after retries, 5xx, network, a bug in my code.
                   Excluded from accuracy denominators, reported separately,
                   and re-run.

Normalisation: take the text after the LAST "FINAL ANSWER:" in the final
assistant message. int tasks compare the first integer found; id tasks compare
the first T-###-### / ACC-### token, upper-cased; name tasks compare the full
string lower-cased with surrounding punctuation, quotes, backticks and markdown
emphasis stripped and whitespace collapsed.
"""

from __future__ import annotations

import re

OUTCOMES = ("pass", "wrong_answer", "no_answer", "context_ceiling", "step_limit", "harness_error")
SUBJECT_FAILURES = ("wrong_answer", "no_answer", "context_ceiling", "step_limit")

_FINAL = re.compile(r"FINAL ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
_ID = re.compile(r"\b(T-\d{3}-\d{3}|ACC-\d{3})\b", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def extract_final(text: str | None) -> str | None:
    if not text:
        return None
    hits = _FINAL.findall(text)
    return hits[-1].strip() if hits else None


def normalise(kind: str, answer: str) -> str | None:
    if kind == "int":
        m = _INT.search(answer.replace(",", ""))
        return str(int(m.group())) if m else None
    if kind == "id":
        m = _ID.search(answer)
        return m.group().upper() if m else None
    s = re.sub(r"[*_`\"'“”‘’]", "", answer).strip().strip(".,;:!").strip()
    return re.sub(r"\s+", " ", s).lower() or None


def grade(kind: str, final_text: str | None, gold: str) -> tuple[str, str | None]:
    """Return (outcome, extracted answer) for a run that ended normally."""
    ans = extract_final(final_text)
    if ans is None:
        return "no_answer", None
    return ("pass" if normalise(kind, ans) == normalise(kind, gold) else "wrong_answer"), ans


_CEILING = re.compile(r"request too large|context_length_exceeded|maximum context length|context window|ContextOverflow",
                      re.IGNORECASE)
_STEP = re.compile(r"GraphRecursionError|Recursion limit|ModelCallCapExceeded", re.IGNORECASE)


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if _CEILING.search(text):
        return "context_ceiling"
    if _STEP.search(text):
        return "step_limit"
    return "harness_error"
