#!/usr/bin/env python3
"""Run one real Geometry3K image through the online teacher scoring protocol."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import pandas as pd
from transformers import AutoTokenizer

from dual_track_opd.fc_opd.conditions import Condition, build_condition_inputs
from dual_track_opd.fc_opd.teacher_client import TeacherClient, score_teacher_conditions
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint


def _mapping(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return dict(value)
    raise TypeError(f"expected a mapping-compatible value, got {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--objective", choices=("gkd", "va_opd", "va_opd_jsd"), required=True)
    args = parser.parse_args()

    row = pd.read_parquet(args.data).iloc[0].to_dict()
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, local_files_only=True, trust_remote_code=True)
    response = "<think>Inspect the geometry diagram.</think>\n\\boxed{A}"
    response_ids = tuple(int(token) for token in tokenizer.encode(response, add_special_tokens=False))
    client = TeacherClient(
        args.teacher_url,
        expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
        timeout_seconds=1200,
    )
    conditions = (
        (Condition.FULL,)
        if args.objective == "gkd"
        else (Condition.FULL, Condition.DEGRADED)
    )
    scores = score_teacher_conditions(
        response_ids,
        str(row["question"]),
        build_condition_inputs({"condition_inputs": _mapping(row["condition_inputs"])}),
        conditions,
        client,
        response_text=response,
        request_prefix="geometry3k_warmup",
    )
    for condition in conditions:
        score = scores[condition]
        if score.token_ids.shape[1] != len(response_ids):
            raise RuntimeError(f"teacher response length mismatch for {condition.value}: {score.token_ids.shape}")
    print(f"Teacher warmup: PASS conditions={[condition.value for condition in conditions]}")


if __name__ == "__main__":
    main()
