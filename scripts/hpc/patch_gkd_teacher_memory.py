#!/usr/bin/env python3
"""Patch integrated GKD teacher compatibility with vLLM 0.11.

The upstream recipe hard-codes ``gpu_memory_utilization=0.7``.  That is
unnecessarily large for the 0.6B smoke model and makes the teacher fail to
start whenever another process occupies roughly a third of an H200.  It also
calls ``LLM.generate(prompt_token_ids=...)``, a keyword removed from the vLLM
0.11 signature; token IDs must be wrapped as token-prompt dictionaries.
"""

from __future__ import annotations

from pathlib import Path


IMPORT_MARKER = "import os\n"
OLD_VALUE = "gpu_memory_utilization=0.7,"
NEW_VALUE = (
    'gpu_memory_utilization=float(os.environ.get('
    '"GKD_TEACHER_GPU_MEMORY_UTILIZATION", "0.7")),'
)
OLD_GENERATE = "self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)"
NEW_GENERATE = (
    'self.llm.generate('
    '[{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids], '
    "sampling_params=sampling_params)"
)
GENERATE_MARKER = 'for token_ids in prompt_token_ids], sampling_params=sampling_params)'


def patch_source(source: str) -> tuple[str, str]:
    patched = source
    changed = False

    if "GKD_TEACHER_GPU_MEMORY_UTILIZATION" not in patched:
        if OLD_VALUE not in patched or "import argparse\n" not in patched:
            return source, "no_match"
        patched = patched.replace("import argparse\n", "import argparse\nimport os\n", 1)
        patched = patched.replace(OLD_VALUE, NEW_VALUE, 1)
        changed = True

    if GENERATE_MARKER not in patched:
        if OLD_GENERATE not in patched:
            return source, "no_match"
        patched = patched.replace(OLD_GENERATE, NEW_GENERATE, 1)
        changed = True

    return patched, "ok" if changed else "skip"


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
