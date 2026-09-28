"""Scan (and with --fix, redact) secrets in committed artifacts.

    uv run python scripts/scrub_secrets.py          # exit 1 if anything found
    uv run python scripts/scrub_secrets.py --fix    # redact in place

Run logs capture provider error bodies, which carry organisation ids, and any
env var a tool might echo. Looks for API-key shapes, OpenAI org ids, bearer
tokens, and the literal values of the credential env vars set in this shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_audit.runner import _SECRET_PATTERNS, scrub  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "artifacts", ROOT / "site" / "data"]


def main() -> int:
    fix = "--fix" in sys.argv
    hits = 0
    for base in TARGETS:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log", ".txt", ".md"}:
                continue
            text = path.read_text(errors="replace")
            clean = scrub(text)
            if clean != text:
                n = sum(len(p.findall(text)) for p, _ in _SECRET_PATTERNS) or 1
                hits += n
                print(f"{path.relative_to(ROOT)}: {n} secret-shaped string(s)")
                if fix:
                    path.write_text(clean)
    print("clean" if not hits else (f"redacted {hits}" if fix else f"{hits} found — run with --fix"))
    return 0 if (fix or not hits) else 1


if __name__ == "__main__":
    sys.exit(main())
