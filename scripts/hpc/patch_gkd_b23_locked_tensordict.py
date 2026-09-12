#!/usr/bin/env python3
"""Allow the GKD actor to construct writable TensorDict microbatches.

Newer TensorDict versions reject item replacement while the container is
locked.  The GKD actor replaces the attention-mask tensor and later adds KL and
teacher tensors to split microbatches.  Unlock the local batch at method entry
so all derived microbatches remain writable.  ``set_`` is not a substitute: it
cannot add the later keys and preserves storage when the mask dtype changes.
"""

from __future__ import annotations

from pathlib import Path


OLD = '        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)'
OLD_SCOPED_FIX = (
    "        with data.batch.unlock_():\n"
    '            data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)'
)
NEW = (
    "        data.batch.unlock_()\n"
    '        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)'
)
MARKER = "        data.batch.unlock_()\n"


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        return source, "skip"
    if OLD_SCOPED_FIX in source:
        return source.replace(OLD_SCOPED_FIX, NEW, 1), "ok"
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
