#!/usr/bin/env python3
"""Patch B16: asyncio.get_running_loop() in vLLMAsyncRollout._init_zeromq().

Python 3.12 raises RuntimeError when get_running_loop() is called
outside an async coroutine.  _init_zeromq() runs inside __init__
(a sync context), so we need to create a new event loop when none
is running.

Idempotent — only patches the exact broken pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

# Exact two-line pattern to fix
BROKEN = re.compile(
    r"^(\s+)loop = asyncio\.get_running_loop\(\)\n"
    r"\1self\.zmq_loop_task = loop\.create_task\(self\._loop_forever\(\)\)",
    re.MULTILINE,
)

FIX = r"""\1try:
\1    loop = asyncio.get_running_loop()
\1except RuntimeError:
\1    loop = asyncio.new_event_loop()
\1    asyncio.set_event_loop(loop)
\1self.zmq_loop_task = loop.create_task(self._loop_forever())"""


def patch_source(source: str) -> tuple[str, str]:
    """Returns (patched_source, "ok"|"skip")."""
    if FIX in source:
        return source, "skip"

    new, count = BROKEN.subn(FIX, source)
    if count == 1:
        return new, "ok"
    if count > 1:
        return new, f"ok ({count} occurrences)"
    return source, "skip"


def patch_file(path: Path) -> str:
    original = path.read_text()
    patched, status = patch_source(original)
    if status != "skip":
        path.write_text(patched)
    return status


def main():
    import argparse
    p = argparse.ArgumentParser(description="Patch B16 event-loop bug")
    p.add_argument("target", type=Path, help="Path to vllm_rollout.py")
    args = p.parse_args()

    status = patch_file(args.target)
    print(f"B16 patch: {status} — {args.target}")


if __name__ == "__main__":
    main()
