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
from dual_track_opd.fc_opd.teacher_client import TeacherClient, TeacherServiceError, score_teacher_conditions
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
    parser.add_argument(
        "--allow-high-teacher-eos",
        action="store_true",
        help="override the first-token EOS probability gate (known-risk: likely EOS collapse under pure GKD)",
    )
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
        question = str(row["question"])
        simple_prompt = ({"role": "user", "content": f"<image>\n{question}"},)
        variants = (
            ("exact_current", tuple(prompt), None),
            ("simple_image_question", simple_prompt, None),
            ("exact_enable_thinking", tuple(prompt), {"enable_thinking": True}),
            ("simple_enable_thinking", simple_prompt, {"enable_thinking": True}),
            (
                "assistant_think_prefill",
                tuple(prompt) + ({"role": "assistant", "content": "<think>\n"},),
                {"add_generation_prompt": False, "continue_final_message": True},
            ),
        )
        diagnostics = {}
        condition_inputs = build_condition_inputs({"condition_inputs": _mapping(row["condition_inputs"])})
        for name, variant_prompt, template_kwargs in variants:
            request = TeacherScoreRequest(
                request_id=f"geometry3k_alignment_probe:{name}",
                condition=Condition.FULL,
                question=question,
                condition_inputs=condition_inputs,
                response_token_ids=(int(tokenizer.eos_token_id),),
                tokenizer_hash=client.metadata.tokenizer_hash,
                prompt=variant_prompt,
                chat_template_kwargs=template_kwargs,
            )
            try:
                diagnostics[name] = client.diagnose_generation_alignment(request, max_new_tokens=4)
            except TeacherServiceError as exc:
                diagnostics[name] = {"error": str(exc)}
        baseline_failure = None
        for name, result in diagnostics.items():
            if "error" in result:
                if name == "exact_current":
                    baseline_failure = f"baseline teacher diagnostic failed: {result['error']}"
                print(f"Teacher prompt variant {name}: ERROR {result['error']}")
                continue

            top1_match = bool(result.get("native_forced_top1_match"))
            top10_overlap = float(result.get("native_forced_top10_overlap_ratio", 0.0))
            top1_logprob_diff = float(result.get("native_forced_top1_logprob_diff", 1.0))
            alignment_ok = top1_match and top10_overlap >= 0.90 and top1_logprob_diff <= 0.02
            result["alignment_gate_passed"] = alignment_ok
            result["alignment_gate_thresholds"] = {
                "require_top1_match": True,
                "minimum_top10_overlap_ratio": 0.90,
                "maximum_top1_logprob_diff": 0.02,
            }
            print(
                f"Teacher prompt variant {name}: "
                f"raw_eos={float(result['native_first_eos_probability']):.4f} "
                f"processed_eos={float(result['processed_first_eos_probability']):.4f} "
                f"generated={result.get('generated_text', '')!r} "
                f"top1_match={top1_match} top10_overlap={top10_overlap:.3f} "
                f"top1_logprob_diff={top1_logprob_diff:.4f} alignment_ok={alignment_ok}"
            )
            if name == "exact_current" and not alignment_ok:
                baseline_failure = (
                    "baseline teacher native/forced alignment failed: "
                    f"top1_match={top1_match}, top10_overlap={top10_overlap:.3f}, "
                    f"top1_logprob_diff={top1_logprob_diff:.4f}"
                )

        diagnostic = {
            "question": question,
            "max_first_eos_probability": args.max_first_eos_prob,
            "variants": diagnostics,
        }
        if args.diagnostic_output is not None:
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_output.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
            print(f"Teacher alignment diagnostic: {args.diagnostic_output}")
        if baseline_failure is not None:
            raise RuntimeError(baseline_failure)

        baseline = diagnostics["exact_current"]
        first_eos_prob = float(baseline["native_first_eos_probability"])
        if first_eos_prob > args.max_first_eos_prob:
            viable = [
                name
                for name, result in diagnostics.items()
                if result.get("alignment_gate_passed")
                and "native_first_eos_probability" in result
                and float(result["native_first_eos_probability"]) <= args.max_first_eos_prob
            ]
            msg = (
                f"teacher native generation assigns EOS probability {first_eos_prob:.4f} at first token "
                f"(threshold: {args.max_first_eos_prob}). "
                f"Under pure GKD this is a direct training target: forward-KL will drive the student to "
                f"output EOS immediately, causing response-length collapse. "
                f"Inspect {args.diagnostic_output} to check native/processed EOS, generated_token_ids, "
                f"and image_grid_thw before proceeding. Prompt variants below threshold: {viable}."
            )
            if args.allow_high_teacher_eos:
                print(f"WARNING: {msg}")
            else:
                raise RuntimeError(f"{msg}  Re-run with --allow-high-teacher-eos to override.")
    print(f"Teacher warmup: PASS conditions={[condition.value for condition in conditions]}")


if __name__ == "__main__":
    main()
