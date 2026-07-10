#!/usr/bin/env python3
"""Make the integrated GKD teacher vLLM memory fraction configurable.

The upstream recipe hard-codes ``gpu_memory_utilization=0.7``.  That is
unnecessarily large for the 0.6B smoke model and makes the teacher fail to
start whenever another process occupies roughly a third of an H200.
"""

from __future__ import annotations

from pathlib import Path


IMPORT_MARKER = "import os\n"
OLD_VALUE = "gpu_memory_utilization=0.7,"
NEW_VALUE = (
    'gpu_memory_utilization=float(os.environ.get('
    '"GKD_TEACHER_GPU_MEMORY_UTILIZATION", "0.7")),'
)


def patch_source(source: str) -> tuple[str, str]:
    if "GKD_TEACHER_GPU_MEMORY_UTILIZATION" in source:
        return source, "skip"
    if OLD_VALUE not in source:
        return source, "no_match"

    patched = source.replace("import argparse\n", "import argparse\nimport os\n", 1)
    patched = patched.replace(OLD_VALUE, NEW_VALUE, 1)
    return patched, "ok"


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
    print(f"Teacher memory patch: {status} - {args.target}")
    return 0 if status in {"ok", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
