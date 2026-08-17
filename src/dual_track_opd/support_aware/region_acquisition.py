"""Region/path-level acquisition diagnostic (frozen plan follow-up).

The single-block micro-operator test has a native-q floor and cannot move the
whole path.  This diagnostic updates a *continuous semantic region* of the
confirmed usable teacher prefix (tokens before ``h*``) and compares:

- soft FKL (teacher Top-K + tail forward KL),
- full-vocabulary RKL,
- hard-token CE (SFT estimator),
- matched control (same-length region from another prompt, soft FKL).

Primary outcome: teacher-path log-likelihood / support dose-response
(``log P_student(suffix | prefix)`` and suffix ``KL(S||T)`` before vs after),
so a region update is judged by whether it actually raises the student's
probability of the teacher's continuation path.  Secondary: no-prefix native
rollout conversion.

No RL, no full training.  Each arm starts from the exact same frozen student
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .causal_image import build_image_conditions
from .causal_runtime import _prompt_inputs, load_runtime_models, response_chunk_logits
from .diagnostic import build_prompt, extract_image
from .micro_operator import (
    _block_token_logp,
    _micro_update,
    _native_pass_rate,
    _train_block_logits,
    _read_jsonl,
)
from .reasoning_blocks import blocks_with_token_spans


SCHEMA_VERSION = "support-aware-region-acquisition-v1"


def select_regions(
    *,
    block_table_path: str,
    rescue_uids: Sequence[str],
    region_mode: str,
    window_blocks: int | None,
    min_tokens: int,
    seed: int = 20260817,
) -> list[dict[str, Any]]:
    """Select contiguous pre-h* regions (whole prefix or last N blocks)."""

    rng = np.random.default_rng(seed)
    rescue_set = set(rescue_uids)
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(Path(block_table_path)):
        if row.get("prompt_id") in rescue_set and row.get("pre_hstar") is True:
            by_uid[row["prompt_id"]].append(row)
    regions: list[dict[str, Any]] = []
    for uid in sorted(by_uid):
        blocks = sorted(by_uid[uid], key=lambda row: int(row["block_index"]))
        pre = [row for row in blocks if int(row["end_token"]) > 0]
        if not pre:
            continue
        if region_mode == "full_prefix":
            selected = pre
        elif region_mode == "window_before_hstar":
            n = window_blocks if window_blocks and window_blocks > 0 else len(pre)
            selected = pre[-n:]
        else:
            raise ValueError(f"unsupported region_mode: {region_mode}")
        token_count = int(selected[-1]["end_token"]) - int(selected[0]["start_token"])
        if token_count < min_tokens:
            continue
        h_star = int(selected[-1].get("h_star") or 0)
        if h_star <= 0:
            continue
        regions.append(
            {
                "prompt_id": uid,
                "start_block": int(selected[0]["block_index"]),
                "end_block": int(selected[-1]["block_index"]),
                "start_token": int(selected[0]["start_token"]),
                "end_token": int(selected[-1]["end_token"]),
                "token_count": token_count,
                "h_star": h_star,
                "region_mode": region_mode,
            }
        )
    rng.shuffle(regions)
    return regions


def _suffix_token_logp(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    suffix_start: int,
    suffix_ids: torch.Tensor,
    device: str,
) -> float:
    """Mean log P_theta(teacher suffix | prefix) over the suffix span."""

    with torch.no_grad():
        logits = _train_block_logits(
            model,
            prompt_inputs,
            response_ids,
            start=suffix_start,
            end=len(response_ids),
            device=device,
        )
        logp = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, suffix_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        return float(logp.mean())


def run_region_acquisition(
    *,
    student_model_path: str,
    teacher_model_path: str,
    student_device: str,
    teacher_device: str,
    dtype: str,
    cohort_dir: str,
    cohort_parquet_path: str,
    proposal_dirs: Sequence[str],
    rescue_dirs: Sequence[str],
    block_table_path: str,
    output_dir: str,
    response_format: str,
    degraded_mode: str,
    region_mode: str,
    window_blocks: int | None,
    min_tokens: int,
    operators: Sequence[str],
    max_regions: int,
    steps: int,
    lr: float,
    fkl_top_k: int,
    rollout_k: int,
    max_continuation_tokens: int,
    seed: int,
) -> dict[str, Any]:
    import pandas as pd

    rescue_uids = []
    for source in rescue_dirs:
        for row in _read_jsonl(Path(source) / "minimal_rescue_prefixes.jsonl"):
            if row.get("meets_preregistered_rescue_rule") is True:
                rescue_uids.append(str(row["sample_uid"]))
    regions = select_regions(
        block_table_path=block_table_path,
        rescue_uids=rescue_uids,
        region_mode=region_mode,
        window_blocks=window_blocks,
        min_tokens=min_tokens,
        seed=seed,
    )
    if max_regions:
        regions = regions[:max_regions]
    print(f"selected {len(regions)} regions (mode={region_mode})", flush=True)
    if not regions:
        raise ValueError("no regions selected; check block table and rescue dirs")

    cohort_path = Path(cohort_parquet_path).expanduser().resolve()
    if not cohort_path.is_file():
        cohort_path = Path(cohort_dir).expanduser() / "cohort.parquet"
    frame = pd.read_parquet(cohort_path)
    frame.index = frame["sample_uid"].astype(str)

    retained_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in proposal_dirs:
        for row in _read_jsonl(Path(source) / "retained_proposals.jsonl"):
            retained_by_uid[str(row.get("sample_uid"))].append(row)
    selected_traces: dict[str, dict[str, Any]] = {}
    for uid, rows in retained_by_uid.items():
        candidates = [
            row
            for row in rows
            if row.get("correct") is True and row.get("retained_for_fkl") is True
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                int(row.get("reachability_rank") or 10**9),
                int(row.get("proposal_id") or 0),
            )
        )
        selected_traces[uid] = candidates[0]

    models = load_runtime_models(
        student_model_path=student_model_path,
        student_device=student_device,
        dtype=dtype,
        teacher_model_path=teacher_model_path,
        teacher_device=teacher_device,
    )
    teacher = models.teacher_model
    tokenizer = models.student_processor.tokenizer
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        uid = str(region["prompt_id"])
        trace = selected_traces.get(uid)
        if trace is None:
            continue
        cohort_row = frame.loc[uid].to_dict()
        cohort_row["sample_uid"] = uid
        response_ids = list(trace.get("response_token_ids") or [])
        region_start = int(region["start_token"])
        region_end = int(region["end_token"])
        suffix_start = int(region["h_star"])
        if not (0 <= region_start < region_end <= suffix_start <= len(response_ids)):
            continue
        question = str(cohort_row.get("question") or "").strip()
        gold_answer = str(cohort_row.get("answer") or "")
        prompt_text = build_prompt(question, response_format=response_format)
        image = extract_image(cohort_row).convert("RGB")
        images = build_image_conditions(image, degraded_mode=degraded_mode)
        prompt_inputs_s = _prompt_inputs(models.student_processor, images.full, prompt_text)
        prompt_inputs_t = _prompt_inputs(models.teacher_processor, images.full, prompt_text)
        suffix_ids = torch.tensor(
            [response_ids[suffix_start:]],
            dtype=torch.long,
            device=student_device,
        )
        teacher_region_logits, _ = response_chunk_logits(
            teacher,
            prompt_inputs_t,
            response_ids,
            start=region_start,
            end=region_end,
            device=teacher_device,
        )
        teacher_region_logits = teacher_region_logits.unsqueeze(0)

        frozen_path_logp = _suffix_token_logp(
            models.student_model,
            prompt_inputs_s,
            response_ids,
            suffix_start,
            suffix_ids,
            student_device,
        )
        baseline_q = _native_pass_rate(
            models.student_model,
            models.student_processor,
            image=images.full,
            prompt_text=prompt_text,
            gold_answer=gold_answer,
            k=rollout_k,
            max_continuation_tokens=max_continuation_tokens,
            seed=seed + index,
            device=student_device,
        )
        region_block_ids = torch.tensor(
            [response_ids[region_start:region_end]],
            dtype=torch.long,
            device=student_device,
        )
        region_len = region_end - region_start
        control_start = suffix_start
        control_end = min(suffix_start + region_len, len(response_ids))
        control_block_ids = torch.tensor(
            [response_ids[control_start:control_end]],
            dtype=torch.long,
            device=student_device,
        )
        teacher_control_logits, _ = response_chunk_logits(
            teacher,
            prompt_inputs_t,
            response_ids,
            start=control_start,
            end=control_end,
            device=teacher_device,
        )
        teacher_control_logits = teacher_control_logits.unsqueeze(0)
        for operator in operators:
            if operator == "control":
                op_start, op_end, op_ids, op_teacher_logits = (
                    control_start,
                    control_end,
                    control_block_ids,
                    teacher_control_logits,
                )
                update_operator = "FKL"
            else:
                op_start, op_end, op_ids, op_teacher_logits = (
                    region_start,
                    region_end,
                    region_block_ids,
                    teacher_region_logits,
                )
                update_operator = operator
            model_i = _fresh_copy(models.student_model, student_device)
            diag = _micro_update(
                student=model_i,
                prompt_inputs=prompt_inputs_s,
                response_ids=response_ids,
                start=op_start,
                end=op_end,
                block_ids=op_ids,
                teacher_logits=op_teacher_logits,
                operator=update_operator,
                steps=steps,
                lr=lr,
                device=student_device,
                fkl_top_k=fkl_top_k,
            )
            updated_path_logp = _suffix_token_logp(
                model_i,
                prompt_inputs_s,
                response_ids,
                suffix_start,
                suffix_ids,
                student_device,
            )
            q_i = _native_pass_rate(
                model_i,
                models.student_processor,
                image=images.full,
                prompt_text=prompt_text,
                gold_answer=gold_answer,
                k=rollout_k,
                max_continuation_tokens=max_continuation_tokens,
                seed=seed + index + 10**6,
                device=student_device,
            )
            del model_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(
                {
                    "prompt_id": uid,
                    "operator": operator,
                    "region_start_token": region_start,
                    "region_end_token": region_end,
                    "region_token_count": region_end - region_start,
                    "h_star": suffix_start,
                    "path_logp_before": frozen_path_logp,
                    "path_logp_after": updated_path_logp,
                    "path_logp_delta": updated_path_logp - frozen_path_logp,
                    "q_before": baseline_q,
                    "q_after": q_i,
                    "G_native": q_i - baseline_q,
                    "update_diag": diag,
                }
            )
            print(
                f"[{index + 1}/{len(regions)}] {uid} op={operator} "
                f"region={region_start}-{region_end} ({region_end - region_start} tok) "
                f"path_logp {frozen_path_logp:+.3f}->{updated_path_logp:+.3f} "
                f"(delta {updated_path_logp - frozen_path_logp:+.3f}) "
                f"q {baseline_q:.2f}->{q_i:.2f}",
                flush=True,
            )

    table_path = output / "region_acquisition_results.jsonl"
    with table_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_regions": len(results) // max(1, len(operators)),
        "n_results": len(results),
        "operators": list(operators),
        "region_mode": region_mode,
        "mean_path_logp_delta": {
            op: float(np.mean([r["path_logp_delta"] for r in results if r["operator"] == op]))
            for op in operators
        },
        "mean_G_native": {
            op: float(np.mean([r["G_native"] for r in results if r["operator"] == op]))
            for op in operators
        },
        "results_path": str(table_path),
    }
    report_path = output / "region_acquisition_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _fresh_copy(model: Any, device: str) -> Any:
    import copy

    model_i = copy.deepcopy(model)
    model_i.to(device)
    return model_i


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct")
    parser.add_argument("--teacher-model", default="${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct")
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--cohort-parquet-path")
    parser.add_argument("--proposal-dirs", nargs="+", required=True)
    parser.add_argument("--rescue-dirs", nargs="+", required=True)
    parser.add_argument("--block-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-format", default="legacy_answer")
    parser.add_argument("--degraded-mode", default="lowres_20_bilinear_nearest")
    parser.add_argument("--region-mode", default="full_prefix", choices=("full_prefix", "window_before_hstar"))
    parser.add_argument("--window-blocks", type=int)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--operators", nargs="+", default=("fkl", "rkl", "hardce", "control"))
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fkl-top-k", type=int, default=100)
    parser.add_argument("--rollout-k", type=int, default=8)
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operator_map = {"fkl": "FKL", "rkl": "RKL", "hardce": "HARD_CE", "control": "control"}
    operators = [operator_map.get(op, op) for op in args.operators]
    summary = run_region_acquisition(
        student_model_path=os.path.expandvars(args.student_model),
        teacher_model_path=os.path.expandvars(args.teacher_model),
        student_device=args.student_device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        cohort_dir=args.cohort_dir,
        cohort_parquet_path=args.cohort_parquet_path or "",
        proposal_dirs=args.proposal_dirs,
        rescue_dirs=args.rescue_dirs,
        block_table_path=args.block_table,
        output_dir=args.output_dir,
        response_format=args.response_format,
        degraded_mode=args.degraded_mode,
        region_mode=args.region_mode,
        window_blocks=args.window_blocks,
        min_tokens=args.min_tokens,
        operators=operators,
        max_regions=args.max_regions,
        steps=args.steps,
        lr=args.lr,
        fkl_top_k=args.fkl_top_k,
        rollout_k=args.rollout_k,
        max_continuation_tokens=args.max_continuation_tokens,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
