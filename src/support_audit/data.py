"""Seeded synthetic support-ops corpus for a fictional B2B SaaS vendor.

Two independent random streams:

- the *structured* stream (ticket id, dates, priority, status, tags, assignee,
  error codes, first-response time) depends only on the structured seed, so the
  gold answer to every task is identical at every payload size;
- the *body* stream (a verbose support transcript per ticket) is the only thing
  the size dial changes. Bodies never mention error codes, priorities or
  statuses, so they carry no answer-relevant signal — they are pure context
  pressure, which is exactly the variable under test.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache

ACCOUNTS: dict[str, str] = {
    "ACC-101": "Northwind Logistics",
    "ACC-102": "Brightpath Health",
    "ACC-103": "Kestrel Fintech",
    "ACC-104": "Lumen Retail",
    "ACC-105": "Orbital Media",
    "ACC-106": "Tandem HR",
}

TICKETS_PER_ACCOUNT = 40

AGENTS = [
    "Priya Raman", "Marcus Webb", "Elena Sokolova", "Tomás Herrera",
    "Aisha Bello", "Jonah Kline", "Mei Takahashi", "Rafael Duarte",
]
TAGS = ["billing", "sso", "api", "onboarding", "performance", "data-export", "integrations"]
ERROR_CODES = ["E-1043", "E-2210", "E-3307", "E-4172", "E-5520", "E-6618", "E-7731", "E-8804", "E-9150"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
PRIORITY_WEIGHTS = [0.2, 0.3, 0.3, 0.2]
STATUSES = ["open", "pending", "resolved", "closed"]
STATUS_WEIGHTS = [0.25, 0.15, 0.35, 0.25]
RESOLVED = {"resolved", "closed"}
START = date(2026, 5, 1)
DAYS = 138  # through 2026-09-15

SUBJECTS = [
    "Dashboard slow to load", "Question about invoice", "SAML login loop", "Webhook retries",
    "CSV export truncated", "New admin onboarding", "Rate limit on bulk API", "Sync gap with CRM",
    "Report totals mismatch", "User provisioning delay", "Seat count question", "Timeout on search",
]

# Size levels: target characters of body text per ticket. Chosen so M stays
# under deepagents' 20k-token eviction threshold (80k chars) and L/XL exceed it.
SIZES: dict[str, int] = {"S": 180, "M": 1300, "L": 4400, "XL": 9300}

_CUST = [
    "Thanks for getting back to me so quickly, I really appreciate the follow-up on this.",
    "I checked with our operations lead and she confirmed the behaviour started last week.",
    "Our team uses this workflow every morning, so any workaround would help in the meantime.",
    "I attached a screenshot earlier but I am not sure it came through on your side.",
    "We tried clearing the browser cache and logging in from a different machine as suggested.",
    "Could you let me know whether this affects other customers or only our workspace?",
    "Our finance team asked me to confirm the details before the end of the quarter.",
    "I will loop in our IT administrator so he can share the configuration details.",
    "Happy to jump on a call tomorrow afternoon if that makes it easier to debug.",
    "The issue seems intermittent, it happened twice yesterday and not at all this morning.",
]
_AGENT = [
    "Thanks for the detail, I have shared it with the engineering team for a closer look.",
    "I reproduced something similar in a test workspace and I am gathering logs now.",
    "Could you confirm the approximate time this happened so I can match it in our records?",
    "I have updated the internal notes on this case and will keep you posted on progress.",
    "Our documentation covers part of this, and I will send the relevant article separately.",
    "I appreciate your patience while we look into this, it is a fair question to raise.",
    "I checked the account configuration and nothing looks out of the ordinary so far.",
    "Let me confirm with a colleague who owns this area before I give you a firm answer.",
    "Thank you, that screenshot came through and it is very helpful for the investigation.",
    "I will follow up by the end of the day with whatever the team finds in the meantime.",
]


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    created_at: str
    priority: str
    status: str
    tags: tuple[str, ...]
    assignee: str
    resolved_by: str | None
    error_codes: tuple[str, ...]
    first_response_minutes: int
    subject: str


def _gen_structured(seed: int) -> dict[str, list[Ticket]]:
    rng = random.Random(seed)
    out: dict[str, list[Ticket]] = {}
    for acct in ACCOUNTS:
        tickets = []
        for i in range(TICKETS_PER_ACCOUNT):
            status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
            assignee = rng.choice(AGENTS)
            n_codes = rng.choices([0, 1, 2], [0.45, 0.4, 0.15])[0]
            tickets.append(
                Ticket(
                    ticket_id=f"T-{acct[4:]}-{i + 1:03d}",
                    created_at=(START + timedelta(days=rng.randrange(DAYS))).isoformat(),
                    priority=rng.choices(PRIORITIES, PRIORITY_WEIGHTS)[0],
                    status=status,
                    tags=tuple(sorted(rng.sample(TAGS, rng.choice([1, 1, 2])))),
                    assignee=assignee,
                    resolved_by=assignee if status in RESOLVED else None,
                    error_codes=tuple(sorted(rng.sample(ERROR_CODES, n_codes))),
                    first_response_minutes=rng.randint(5, 600),
                    subject=rng.choice(SUBJECTS),
                )
            )
        # Ticket ids are assigned in creation order, like a real ticketing system.
        tickets.sort(key=lambda t: (t.created_at, t.ticket_id))
        out[acct] = [
            Ticket(**{**asdict(t), "ticket_id": f"T-{acct[4:]}-{n + 1:03d}"}) for n, t in enumerate(tickets)
        ]
    return out


def _body(rng: random.Random, target_chars: int) -> str:
    parts: list[str] = []
    n = 0
    turn = 0
    while n < target_chars:
        line = ("Customer: " if turn % 2 == 0 else "Agent: ") + rng.choice(_CUST if turn % 2 == 0 else _AGENT)
        parts.append(line)
        n += len(line) + 1
        turn += 1
    return " ".join(parts)


@lru_cache(maxsize=1)
def structured_corpus() -> dict[str, list[Ticket]]:
    """The structured corpus, using the first seed at which every task's
    constraints hold (unique arg-max, needle present). Deterministic."""
    from support_audit.tasks import constraints_hold

    for seed in range(1000, 2000):
        corpus = _gen_structured(seed)
        if constraints_hold(corpus):
            return corpus
    raise RuntimeError("no seed satisfies task constraints")


@lru_cache(maxsize=64)
def export_payload(account_id: str, size: str) -> str:
    """The exact string the `export_tickets` tool returns. JSON Lines: a header
    line then one ticket per line, structured fields first and the transcript
    body last."""
    if account_id not in ACCOUNTS:
        return json.dumps({"error": f"unknown account_id {account_id!r}", "known": sorted(ACCOUNTS)})
    tickets = structured_corpus()[account_id]
    body_rng = random.Random(f"{account_id}:{size}")
    lines = [json.dumps({"account_id": account_id, "account_name": ACCOUNTS[account_id], "ticket_count": len(tickets),
                         "format": "one JSON ticket per line follows"})]
    for t in tickets:
        rec = asdict(t)
        rec["tags"] = list(t.tags)
        rec["error_codes"] = list(t.error_codes)
        rec["transcript"] = _body(body_rng, SIZES[size])
        lines.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(lines)


def accounts_payload() -> str:
    return json.dumps([{"account_id": k, "name": v} for k, v in ACCOUNTS.items()])
