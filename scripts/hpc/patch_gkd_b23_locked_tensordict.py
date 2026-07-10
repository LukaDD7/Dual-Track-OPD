#!/usr/bin/env python3
"""Allow the GKD actor to cast attention_mask in a locked TensorDict.

Newer TensorDict versions reject item replacement while the container is
locked.  The GKD actor must replace the attention-mask tensor because its dtype
changes to bool, so ``set_`` (which preserves the original storage) is not an
appropriate substitute.  Unlock only for the replacement and restore the lock
on context exit.
"""

from __future__ import annotations

from pathlib import Path


OLD = '        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)'
NEW = (
    "        with data.batch.unlock_():\n"
    '            data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)'
)
MARKER = "with data.batch.unlock_():"


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        return source, "skip"
    if OLD not in source:
        return source, "no_match"
    return source.replace(OLD, NEW, 1), "ok"


def patch_file(path: Path) -> str:
    original = path.read_text(encoding="utf-8")
    patched, status = patch_source(original)
    if status == "ok":
        path.write_text(patched, encoding="utf-8")
    return status


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    status = patch_file(args.target)
    print(f"B23 TensorDict patch: {status} - {args.target}")
    return 0 if status in {"ok", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
