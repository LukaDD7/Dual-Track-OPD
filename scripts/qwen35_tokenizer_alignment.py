#!/usr/bin/env python3
"""Fail fast unless student and teacher assign the same meaning to every token ID.

The pinned verl OPD teacher consumes token IDs produced by the student tokenizer;
it does not tokenize text again.  Byte-for-byte tokenizer files may differ in
metadata, so this check compares the canonical ID-to-token mapping instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_to_token(model_dir: Path) -> tuple[dict[int, str], dict[str, Any]]:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json: {tokenizer_path}")
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = payload.get("model") or {}
    vocab = model.get("vocab")
    mapping: dict[int, str] = {}

    if isinstance(vocab, dict):
        entries = ((int(token_id), str(token)) for token, token_id in vocab.items())
    elif isinstance(vocab, list):
        entries = ((index, str(item[0] if isinstance(item, list) else item)) for index, item in enumerate(vocab))
    else:
        raise ValueError(f"unsupported tokenizer vocab in {tokenizer_path}")

    for token_id, token in entries:
        previous = mapping.setdefault(token_id, token)
        if previous != token:
            raise ValueError(f"conflicting base tokens for id {token_id} in {tokenizer_path}")

    for item in payload.get("added_tokens") or []:
        token_id = int(item["id"])
        token = str(item["content"])
        previous = mapping.setdefault(token_id, token)
        if previous != token:
            raise ValueError(f"conflicting added token for id {token_id} in {tokenizer_path}")

    canonical = json.dumps(sorted(mapping.items()), ensure_ascii=False, separators=(",", ":")).encode()
    summary = {
        "model_path": str(model_dir.resolve()),
        "tokenizer_json": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": _sha256(tokenizer_path),
        "tokenizer_type": model.get("type"),
        "id_count": len(mapping),
        "max_token_id": max(mapping, default=-1),
        "id_to_token_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return mapping, summary


def compare_tokenizers(student_dir: Path, teacher_dir: Path) -> dict[str, Any]:
    student_mapping, student = _id_to_token(student_dir)
    teacher_mapping, teacher = _id_to_token(teacher_dir)
    mismatched_ids = [
        token_id
        for token_id in sorted(student_mapping.keys() | teacher_mapping.keys())
        if student_mapping.get(token_id) != teacher_mapping.get(token_id)
    ]
    return {
        "schema_version": 1,
        "valid": not mismatched_ids,
        "reason": "exact ID-to-token mapping match" if not mismatched_ids else "token-ID semantics differ",
        "student": student,
        "teacher": teacher,
        "mismatch_count": len(mismatched_ids),
        "first_mismatched_ids": mismatched_ids[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_tokenizers(args.student_model, args.teacher_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["valid"]:
        print(
            "FATAL: student and teacher tokenizer-ID mappings differ; "
            "the pinned teacher consumes student token IDs directly"
        )
        print(f"Tokenizer audit: {args.output}")
        return 1
    print(f"Tokenizer alignment: PASS ({result['student']['id_count']} IDs; {args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
