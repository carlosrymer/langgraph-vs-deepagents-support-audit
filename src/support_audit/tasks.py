"""The eight account-audit tasks and their oracle answers.

Every answer is an exact value computed from the structured corpus, so grading
needs no model and no judgement. Gold answers do not depend on payload size.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

from support_audit.data import ACCOUNTS, RESOLVED, Ticket

Corpus = dict[str, list[Ticket]]
Kind = Literal["int", "id", "name"]


@dataclass(frozen=True)
class Task:
    task_id: str
    accounts: tuple[str, ...]
    kind: Kind
    question: str
    oracle: Callable[[Corpus], str]
    # Returns False when the corpus makes the task ambiguous (tied arg-max,
    # missing needle, trivial zero). The generator rejects such seeds.
    valid: Callable[[Corpus], bool] = lambda c: True


def _unresolved(t: Ticket) -> bool:
    return t.status not in RESOLVED


def _unique_max(counts: dict[str, int]) -> bool:
    top = sorted(counts.values(), reverse=True)
    return len(top) >= 2 and top[0] > top[1]


def _argmax(counts: dict[str, int]) -> str:
    return max(counts, key=lambda k: counts[k])


def _names(accts: tuple[str, ...]) -> str:
    return ", ".join(f"{ACCOUNTS[a]} ({a})" for a in accts)


# --- oracles -----------------------------------------------------------------

def _t1(c: Corpus) -> str:
    return str(sum(1 for t in c["ACC-101"] if t.priority == "P1" and _unresolved(t)))


def _t2_hits(c: Corpus) -> list[Ticket]:
    return [t for t in c["ACC-102"] if "E-4172" in t.error_codes]


def _t2(c: Corpus) -> str:
    return min(_t2_hits(c), key=lambda t: (t.created_at, t.ticket_id)).ticket_id


def _t2_valid(c: Corpus) -> bool:
    hits = _t2_hits(c)
    if len(hits) < 2:
        return False
    dates = sorted(t.created_at for t in hits)
    return dates[0] < dates[1]  # the earliest is unambiguous by date alone


def _t3_counts(c: Corpus) -> dict[str, int]:
    return dict(Counter(t.resolved_by for t in c["ACC-103"] if t.resolved_by))


def _t4_counts(c: Corpus) -> dict[str, int]:
    return {a: sum(1 for t in c[a] if "billing" in t.tags) for a in ("ACC-104", "ACC-105")}


def _t5(c: Corpus) -> str:
    return str(sum(1 for a in ("ACC-101", "ACC-106") for t in c[a]
                   if t.priority == "P1" and t.first_response_minutes > 240))


def _t6(c: Corpus) -> str:
    return str(sum(1 for a in ("ACC-102", "ACC-103", "ACC-104") for t in c[a]
                   if t.priority == "P1" and _unresolved(t)))


def _t7_counts(c: Corpus) -> dict[str, int]:
    return {a: len({code for t in c[a] if _unresolved(t) for code in t.error_codes})
            for a in ("ACC-101", "ACC-105", "ACC-106")}


def _t8(c: Corpus) -> str:
    return str(sum(1 for a in ("ACC-102", "ACC-104", "ACC-106") for t in c[a]
                   if "sso" in t.tags and t.created_at.startswith("2026-08")))


TASKS: list[Task] = [
    Task("T1", ("ACC-101",), "int",
         f"How many P1 tickets for {_names(('ACC-101',))} are currently unresolved "
         "(status `open` or `pending`)?", _t1, lambda c: int(_t1(c)) >= 2),
    Task("T2", ("ACC-102",), "id",
         f"What is the ticket_id of the earliest-created ticket for {_names(('ACC-102',))} "
         "whose error_codes include E-4172?", _t2, _t2_valid),
    Task("T3", ("ACC-103",), "name",
         f"Which support agent resolved the most tickets for {_names(('ACC-103',))}? "
         "Count tickets by the `resolved_by` field.", lambda c: _argmax(_t3_counts(c)),
         lambda c: _unique_max(_t3_counts(c))),
    Task("T4", ("ACC-104", "ACC-105"), "id",
         f"Of {_names(('ACC-104', 'ACC-105'))}, which account has more tickets tagged `billing`? "
         "Answer with the account_id.", lambda c: _argmax(_t4_counts(c)),
         lambda c: _unique_max(_t4_counts(c))),
    Task("T5", ("ACC-101", "ACC-106"), "int",
         f"Across {_names(('ACC-101', 'ACC-106'))}, how many P1 tickets had a first response time "
         "(`first_response_minutes`) greater than 240 minutes?", _t5, lambda c: int(_t5(c)) >= 2),
    Task("T6", ("ACC-102", "ACC-103", "ACC-104"), "int",
         f"Across {_names(('ACC-102', 'ACC-103', 'ACC-104'))}, how many P1 tickets are currently "
         "unresolved (status `open` or `pending`) in total?", _t6, lambda c: int(_t6(c)) >= 3),
    Task("T7", ("ACC-101", "ACC-105", "ACC-106"), "id",
         f"Of {_names(('ACC-101', 'ACC-105', 'ACC-106'))}, which account has the most distinct error "
         "codes across its unresolved tickets (status `open` or `pending`)? Answer with the account_id.",
         lambda c: _argmax(_t7_counts(c)), lambda c: _unique_max(_t7_counts(c))),
    Task("T8", ("ACC-102", "ACC-104", "ACC-106"), "int",
         f"Across {_names(('ACC-102', 'ACC-104', 'ACC-106'))}, how many tickets tagged `sso` were "
         "created in August 2026?", _t8, lambda c: int(_t8(c)) >= 2),
]
TASKS_BY_ID = {t.task_id: t for t in TASKS}


def constraints_hold(corpus: Corpus) -> bool:
    return all(t.valid(corpus) for t in TASKS)


def gold(task_id: str) -> str:
    from support_audit.data import structured_corpus

    return TASKS_BY_ID[task_id].oracle(structured_corpus())
