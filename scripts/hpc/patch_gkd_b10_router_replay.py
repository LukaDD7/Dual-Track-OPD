#!/usr/bin/env python3
"""Patch B10: safe access to router_replay when actor config is plain dict.

GKD recipe's config provides actor as a plain dict (no router_replay),
but base MegatronWorker init expects it as a structured config.
Makes the access safe for both dict and OmegaConf struct.
Idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path

BROKEN = (
    r"self\.router_replay = self\.config\.actor\.router_replay\n"
    r"(?P<indent>\s+)"
    r"self\.enable_routing_replay = self\.router_replay\.mode != \"disabled\""
)

FIX = (
    r"self.router_replay = "
    r'self.config.actor.get("router_replay", None) '
    r"if isinstance(self.config.actor, dict) "
    r'else getattr(self.config.actor, "router_replay", None)\n'
    r"\g<indent>if self.router_replay is not None:\n"
    r'\g<indent>    self.enable_routing_replay = self.router_replay.mode != "disabled"'
)

IDEMPOTENT_MARKER = 'isinstance(self.config.actor, dict)'


def patch_source(source: str) -> tuple[str, str]:
    if IDEMPOTENT_MARKER in source:
        return source, "skip"

    new, count = re.subn(BROKEN, FIX, source)
    if count:
        return new, "ok"
    return source, "skip"


def patch_file(path: Path) -> str:
    original = path.read_text()
    patched, status = patch_source(original)
    if status != "skip":
        path.write_text(patched)
    return status


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path)
    args = p.parse_args()
    print(f"B10 patch: {patch_file(args.target)} - {args.target}")


if __name__ == "__main__":
    main()
