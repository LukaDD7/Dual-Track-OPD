#!/usr/bin/env python3
"""Bridge verl's removed Transformers auto-model name under Transformers 5."""

from __future__ import annotations

from pathlib import Path


OLD_IMPORT = "    AutoModelForVision2Seq,\n"
ANCHOR = "from transformers.modeling_outputs import CausalLMOutputWithPast\n"
BRIDGE = (
    "try:\n"
    "    from transformers import AutoModelForVision2Seq\n"
    "except ImportError:  # Transformers 5 renamed the multimodal auto class.\n"
    "    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n\n"
)
MARKER = "AutoModelForImageTextToText as AutoModelForVision2Seq"


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        return source, "skip"
    if OLD_IMPORT not in source or ANCHOR not in source:
        return source, "no_match"
    patched = source.replace(OLD_IMPORT, "", 1)
    patched = patched.replace(ANCHOR, BRIDGE + ANCHOR, 1)
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
    print(f"Transformers 5 compatibility patch: {status} - {args.target}")
    return 0 if status in {"ok", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
