#!/usr/bin/env python3
"""Apply the narrow GKD B15 event-loop compatibility fix.

The pinned GKD recipe calls the async vLLM ``ServerAdapter.update_weights``
from a synchronous Ray worker.  Python 3.12 no longer creates an implicit
event loop for that thread, so ``get_event_loop()`` raises before the first
rollout.  This patcher is intentionally strict and idempotent: it only changes
the exact broken two-line call site produced by the server-side B14 fix.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


BROKEN = re.compile(
    r"^(?P<indent>[ \t]*)loop = "
    r"asyncio\.get_event_loop_policy\(\)\.get_event_loop\(\)\n"
    r"(?P=indent)loop\.run_until_complete\("
    r"(?P<coro>self\.rollout\.update_weights\([^\n]+\))\)\s*$",
    re.MULTILINE,
)

FIX_MARKER = "with asyncio.Runner() as runner:"


def patch_source(source: str) -> tuple[str, str]:
    """Return ``(new_source, status)`` for the single supported call site."""
    if FIX_MARKER in source and "runner.run(self.rollout.update_weights(" in source:
        return source, "already_patched"

    matches = list(BROKEN.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one B15 get_event_loop/update_weights call site; "
            f"found {len(matches)}. Refusing to patch an unknown backend revision."
        )

    match = matches[0]
    indent = match.group("indent")
    coro = match.group("coro")
    replacement = (
        f"{indent}with asyncio.Runner() as runner:\n"
        f"{indent}    runner.run({coro})"
    )
    patched = source[: match.start()] + replacement + source[match.end() :]
    ast.parse(patched)
    return patched, "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="GKD megatron_workers.py path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the fix is present without writing (default writes)",
    )
    args = parser.parse_args()

    source = args.target.read_text(encoding="utf-8")
    patched, status = patch_source(source)
    if args.check:
        if status != "already_patched":
            raise SystemExit("B15 compatibility fix is not applied")
        print(f"B15 compatibility fix verified: {args.target}")
        return 0

    if status == "patched":
        args.target.write_text(patched, encoding="utf-8")
        print(f"Applied B15 compatibility fix: {args.target}")
    else:
        print(f"B15 compatibility fix already applied: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
