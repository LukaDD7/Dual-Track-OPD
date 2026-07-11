#!/usr/bin/env python3
"""Run one real Geometry3K image through the online teacher scoring protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from dual_track_opd.fc_opd.conditions import Condition, build_condition_inputs
from dual_track_opd.fc_opd.teacher_client import TeacherClient, score_teacher_conditions
from dual_track_opd.fc_opd.teacher_protocol import TeacherScoreRequest, tokenizer_fingerprint
from dual_track_opd.fc_opd.prompt_contracts import geometry3k_training_prompt


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
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--max-first-eos-prob", type=float, default=0.20)
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
    prompt = geometry3k_training_prompt(str(row["question"]))
    scores = score_teacher_conditions(
        response_ids,
        str(row["question"]),
        build_condition_inputs({"condition_inputs": _mapping(row["condition_inputs"])}),
        conditions,
        client,
        response_text=response,
        prompt=prompt,
        request_prefix="geometry3k_warmup",
    )
    for condition in conditions:
        score = scores[condition]
        if score.token_ids.shape[1] != len(response_ids):
            raise RuntimeError(f"teacher response length mismatch for {condition.value}: {score.token_ids.shape}")
    if args.objective == "gkd":
        request = TeacherScoreRequest(
            request_id="geometry3k_alignment_probe",
            condition=Condition.FULL,
            question=str(row["question"]),
            condition_inputs=build_condition_inputs({"condition_inputs": _mapping(row["condition_inputs"])}),
            response_token_ids=(int(tokenizer.eos_token_id),),
            tokenizer_hash=client.metadata.tokenizer_hash,
            prompt=tuple(prompt),
        )
        diagnostic = client.diagnose_generation_alignment(request, max_new_tokens=4)
        if args.diagnostic_output is not None:
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_output.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
            print(f"Teacher alignment diagnostic: {args.diagnostic_output}")
        if not diagnostic.get("native_forced_topk_ids_match"):
            raise RuntimeError("teacher native generation and forced-forward top-k token IDs differ")
        max_diff = diagnostic.get("native_forced_max_logprob_diff")
        if max_diff is None or float(max_diff) > 5e-4:
            raise RuntimeError(f"teacher native/forced log-prob mismatch: {max_diff}")
        first_eos_prob = float(diagnostic["native_first_eos_probability"])
        if first_eos_prob > args.max_first_eos_prob:
            raise RuntimeError(
                f"teacher native generation assigns EOS probability {first_eos_prob:.4f} at first token; "
                "inspect the saved prompt/image diagnostic before training"
            )
    print(f"Teacher warmup: PASS conditions={[condition.value for condition in conditions]}")


if __name__ == "__main__":
    main()
