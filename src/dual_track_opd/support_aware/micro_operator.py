"""Micro-operator causal experiment (frozen plan B7).

From a frozen student checkpoint ``theta_0``, for each confirmed pre-h* candidate
reasoning block (high-F, high-R, low-F/low-R control) run a very short soft
Top-K+tail FKL or full-vocabulary RKL update on that block only (1-4 optimizer
steps), then measure the fresh
no-prefix native rollout improvement:

    theta_i^FKL = theta_0 - eta * grad L_FKL(B_i)
    q_i^FKL     = P_theta_i^FKL(R=1 | I, q)      # no teacher prefix
    G_i^FKL     = q_i^FKL - q_0

and the symmetric RKL quantity.  The report then tests:

    F_i up  => G_i^FKL - G_i^RKL up
    R_i up  => G_i^RKL - G_i^FKL up

This is diagnostic-only micro-training: no full FKL/RKL training, no RL, no
router.  Each candidate starts from the exact same frozen checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .causal_image import build_image_conditions
from .causal_runtime import (
    _append_response_prefix,
    _prompt_inputs,
    load_runtime_models,
    response_chunk_logits,
)
from .diagnostic import build_prompt, extract_image
from .prefix_intervention import generate_continuation
from .reasoning_blocks import blocks_with_token_spans
from .support_transition_loss import suffix_rkl_k1, topk_tail_fkl
from .verifier import verify_answer


SCHEMA_VERSION = "support-aware-micro-operator-v2"


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_candidate_blocks(
    *,
    block_table_path: str,
    rescue_uids: Sequence[str],
    per_prompt: int = 3,
    max_candidates: int = 60,
    min_tokens: int = 4,
    seed: int = 20260817,
) -> list[dict[str, Any]]:
    """Pick per-prompt high-F / high-R / low-control blocks (CPU)."""

    rng = np.random.default_rng(seed)
    rescue_set = set(rescue_uids)
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(Path(block_table_path)):
        if (
            row.get("prompt_id") in rescue_set
            and row.get("pre_hstar") is True
            and int(row.get("token_count") or 0) >= min_tokens
        ):
            by_uid[row["prompt_id"]].append(row)
    candidates: list[dict[str, Any]] = []
    uids = sorted(by_uid)
    rng.shuffle(uids)
    for uid in uids:
        rows = sorted(by_uid[uid], key=lambda row: float(row["block_index"]))
        high_f = max(rows, key=lambda row: float(row["F_i"]))
        high_r = max(rows, key=lambda row: float(row["R_i"]))
        control = min(rows, key=lambda row: abs(float(row["F_i"])) + abs(float(row["R_i"])))
        chosen: list[tuple[str, dict[str, Any]]] = []
        for kind, row in (("FKL", high_f), ("RKL", high_r), ("control", control)):
            if all(row["block_index"] != existing["block_index"] for _, existing in chosen):
                chosen.append((kind, row))
        for kind, row in chosen[:per_prompt]:
            candidates.append(
                {
                    "prompt_id": uid,
                    "block_index": row["block_index"],
                    "block_type": row.get("block_type"),
                    "start_token": row["start_token"],
                    "end_token": row["end_token"],
                    "operator": kind,
                    "F_i": row["F_i"],
                    "R_i": row["R_i"],
                    "C_i": row.get("C_i"),
                    "Q_i": row.get("Q_i"),
                    "pre_hstar": True,
                }
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _train_block_logits(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    start: int,
    end: int,
    device: str,
) -> torch.Tensor:
    """Train-mode forward over the block span (autograd kept)."""

    import inspect

    prefix = tuple(int(value) for value in response_ids[:end])
    model_inputs = _append_response_prefix(prompt_inputs, prefix, device=device)
    kwargs: dict[str, Any] = {"use_cache": False}
    used_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    if used_keep:
        kwargs["logits_to_keep"] = end - start + 1
    outputs = model(**model_inputs, **kwargs)
    logits = outputs.logits
    if used_keep and int(logits.shape[1]) == end - start + 1:
        selected = logits[:, :-1, :]
    else:
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        selected = logits[:, prompt_length - 1 + start : prompt_length - 1 + end, :]
    return selected


def _block_token_logp(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    start: int,
    end: int,
    block_ids: torch.Tensor,
    device: str,
) -> float:
    """Mean log P(block tokens) under the current model (dose sanity check)."""

    with torch.no_grad():
        logits = _train_block_logits(
            model, prompt_inputs, response_ids, start=start, end=end, device=device
        )
        logp = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, block_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        return float(logp.mean())


def _micro_update(
    *,
    student: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    start: int,
    end: int,
    block_ids: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    operator: str,
    steps: int,
    lr: float,
    device: str,
    fkl_top_k: int = 100,
) -> dict[str, Any]:
    """Run 1..steps optimizer updates; return loss/grad diagnostics."""

    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    mask = torch.ones((1, block_ids.numel()), dtype=torch.bool, device=device)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        student_logits = _train_block_logits(
            student, prompt_inputs, response_ids, start=start, end=end, device=device
        ).float()
        if operator == "FKL":
            if teacher_logits is None:
                raise ValueError("soft FKL candidate requires teacher logits")
            loss = topk_tail_fkl(
                student_logits,
                teacher_logits.to(student_logits.device).float(),
                mask,
                top_k=fkl_top_k,
            )
        else:
            if teacher_logits is None:
                raise ValueError("RKL candidate requires teacher logits")
            loss = suffix_rkl_k1(
                student_logits,
                teacher_logits.to(student_logits.device).float(),
                mask,
            )
        losses.append(float(loss.detach().float()))
        loss.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)))
        optimizer.step()
        del student_logits
    nan_params = sum(1 for parameter in parameters if bool(torch.isnan(parameter).any().item()))
    return {
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "nan_params": nan_params,
    }


def _native_pass_rate(
    model: Any,
    processor: Any,
    *,
    image,
    prompt_text: str,
    gold_answer: str,
    k: int,
    max_continuation_tokens: int,
    seed: int,
    device: str,
) -> float:
    correct = 0
    for index in range(k):
        generation = generate_continuation(
            model,
            processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=[],
            max_continuation_tokens=max_continuation_tokens,
            temperature=0.7,
            top_p=0.95,
            seed=seed + index,
            device=device,
        )
        record = generation["generation"]
        response_text = getattr(record, "response_text_display", None)
        verdict = verify_answer(str(response_text or ""), gold_answer)
        if verdict.get("correct"):
            correct += 1
    return correct / k


def run_micro_operator(
    *,
    student_model_path: str,
    teacher_model_path: str,
    student_device: str,
    teacher_device: str,
    dtype: str,
    cohort_dir: str,
    cohort_parquet_path: str,
    proposal_dirs: Sequence[str],
    candidate_blocks: Sequence[Mapping[str, Any]],
    output_dir: str,
    response_format: str,
    degraded_mode: str,
    max_continuation_tokens: int,
    rollout_k: int,
    steps: int,
    lr: float,
    fkl_top_k: int,
    seed: int,
) -> dict[str, Any]:
    import pandas as pd

    cohort_path = Path(cohort_parquet_path).expanduser().resolve()
    if not cohort_path.is_file():
        cohort_path = Path(cohort_dir).expanduser() / "cohort.parquet"
    frame = pd.read_parquet(cohort_path)
    frame.index = frame["sample_uid"].astype(str)

    retained_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in proposal_dirs:
        for row in _read_jsonl(Path(source) / "retained_proposals.jsonl"):
            retained_by_uid[str(row.get("sample_uid"))].append(row)
    selected: dict[str, dict[str, Any]] = {}
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
        selected[uid] = candidates[0]

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
    baseline_cache: dict[str, float] = {}
    for index, candidate in enumerate(candidate_blocks):
        uid = str(candidate["prompt_id"])
        trace = selected.get(uid)
        cohort_row = frame.loc[uid].to_dict()
        cohort_row["sample_uid"] = uid
        if trace is None:
            continue
        response_ids = list(trace.get("response_token_ids") or [])
        question = str(cohort_row.get("question") or "").strip()
        gold_answer = str(cohort_row.get("answer") or "")
        prompt_text = build_prompt(question, response_format=response_format)
        image = extract_image(cohort_row).convert("RGB")
        images = build_image_conditions(image, degraded_mode=degraded_mode)
        block_ids = torch.tensor(
            [response_ids[int(candidate["start_token"]) : int(candidate["end_token"])]],
            dtype=torch.long,
            device=student_device,
        )
        if block_ids.numel() == 0:
            continue

        prompt_inputs_s = _prompt_inputs(models.student_processor, images.full, prompt_text)
        prompt_inputs_t = _prompt_inputs(models.teacher_processor, images.full, prompt_text)
        teacher_logits, _ = response_chunk_logits(
            teacher,
            prompt_inputs_t,
            response_ids,
            start=int(candidate["start_token"]),
            end=int(candidate["end_token"]),
            device=teacher_device,
        )
        teacher_logits = teacher_logits.unsqueeze(0)
        frozen_block_logp = _block_token_logp(
            models.student_model,
            prompt_inputs_s,
            response_ids,
            start=int(candidate["start_token"]),
            end=int(candidate["end_token"]),
            block_ids=block_ids,
            device=student_device,
        )

        q0 = baseline_cache.get(uid)
        if q0 is None:
            q0 = _native_pass_rate(
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
            baseline_cache[uid] = q0

        gains: dict[str, float] = {}
        diags: dict[str, dict[str, Any]] = {}
        for operator in ("FKL", "RKL"):
            model_i = copy.deepcopy(models.student_model)
            model_i.to(student_device)
            update_diag = _micro_update(
                student=model_i,
                prompt_inputs=prompt_inputs_s,
                response_ids=response_ids,
                start=int(candidate["start_token"]),
                end=int(candidate["end_token"]),
                block_ids=block_ids,
                teacher_logits=teacher_logits,
                operator=operator,
                steps=steps,
                lr=lr,
                device=student_device,
                fkl_top_k=fkl_top_k,
            )
            q_i = _native_pass_rate(
                model_i,
                models.student_processor,
                image=images.full,
                prompt_text=prompt_text,
                gold_answer=gold_answer,
                k=rollout_k,
                max_continuation_tokens=max_continuation_tokens,
                seed=seed + index + (10**6 if operator == "RKL" else 0),
                device=student_device,
            )
            updated_block_logp = _block_token_logp(
                model_i,
                prompt_inputs_s,
                response_ids,
                start=int(candidate["start_token"]),
                end=int(candidate["end_token"]),
                block_ids=block_ids,
                device=student_device,
            )
            del model_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gains[operator] = q_i - q0
            diags[operator] = update_diag
            results.append(
                {
                    **dict(candidate),
                    "operator": operator,
                    "role": str(candidate["operator"]),
                    "q0": q0,
                    "q_i": q_i,
                    "G_i": q_i - q0,
                    "block_logp_delta": updated_block_logp - frozen_block_logp,
                    "update_diag": update_diag,
                }
            )
        print(
            f"[{index + 1}/{len(candidate_blocks)}] {uid} block "
            f"{candidate['block_index']} role={candidate['operator']} "
            f"F={float(candidate['F_i']):+.2f} R={float(candidate['R_i']):+.2f} "
            f"q0={q0:.2f} G_FKL={gains['FKL']:+.2f} G_RKL={gains['RKL']:+.2f} "
            f"loss_FKL={diags['FKL']['loss_last']:.2f}->{diags['FKL']['loss_first']:.2f} "
            f"loss_RKL={diags['RKL']['loss_last']:.2f}->{diags['RKL']['loss_first']:.2f} "
            f"gn={diags['RKL']['max_grad_norm']:.2f} nan={diags['RKL']['nan_params']}",
            flush=True,
        )

    table_path = output / "micro_operator_results.jsonl"
    with table_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = _summarize(results)
    report["schema_version"] = SCHEMA_VERSION
    report["operator_definitions"] = {
        "FKL": f"teacher_top_{fkl_top_k}_plus_tail_forward_kl",
        "RKL": "full_vocabulary_reverse_kl",
        "candidate_scope": "confirmed_pre_hstar_blocks_only",
    }
    report["results_path"] = str(table_path)
    report_path = output / "micro_operator_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_block: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_block[(str(row["prompt_id"]), int(row["block_index"]))][str(row["operator"])] = row
    blocks = list(by_block.values())
    pairs_fkl_adv: list[tuple[float, float]] = []
    pairs_rkl_adv: list[tuple[float, float]] = []
    for operators in blocks:
        if "FKL" not in operators or "RKL" not in operators:
            continue
        fkl = operators["FKL"]
        rkl = operators["RKL"]
        pairs_fkl_adv.append((float(fkl["F_i"]), float(fkl["G_i"]) - float(rkl["G_i"])))
        pairs_rkl_adv.append((float(rkl["R_i"]), float(rkl["G_i"]) - float(fkl["G_i"])))

    def _corr(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 4:
            return None
        x = np.asarray([a for a, _ in pairs], dtype=np.float64)
        y = np.asarray([b for _, b in pairs], dtype=np.float64)
        if float(x.std()) <= 1e-9 or float(y.std()) <= 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    fkl = [row for row in results if row["operator"] == "FKL"]
    rkl = [row for row in results if row["operator"] == "RKL"]
    return {
        "n_candidates": len(blocks),
        "n_results": len(results),
        "n_fkl": len(fkl),
        "n_rkl": len(rkl),
        "mean_G_fkl": float(np.mean([row["G_i"] for row in fkl])) if fkl else None,
        "mean_G_rkl": float(np.mean([row["G_i"] for row in rkl])) if rkl else None,
        "mean_G_fkl_by_role": {
            role: (
                float(np.mean([row["G_i"] for row in fkl if row.get("role") == role]))
                if any(row.get("role") == role for row in fkl)
                else None
            )
            for role in ("FKL", "RKL", "control")
        },
        "mean_block_logp_delta": {
            "FKL": float(np.mean([row.get("block_logp_delta", 0.0) for row in fkl])),
            "RKL": float(np.mean([row.get("block_logp_delta", 0.0) for row in rkl])),
        },
        "corr_F_vs_G_fkl_minus_rkl": _corr(pairs_fkl_adv),
        "corr_R_vs_G_rkl_minus_fkl": _corr(pairs_rkl_adv),
    }


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
    parser.add_argument("--block-table", required=True)
    parser.add_argument("--rescue-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-format", default="legacy_answer")
    parser.add_argument("--degraded-mode", default="lowres_20_bilinear_nearest")
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--rollout-k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fkl-top-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rescue_uids: list[str] = []
    for source in args.rescue_dirs:
        for row in _read_jsonl(Path(source) / "minimal_rescue_prefixes.jsonl"):
            if row.get("meets_preregistered_rescue_rule") is True:
                rescue_uids.append(str(row["sample_uid"]))
    candidates = select_candidate_blocks(
        block_table_path=args.block_table,
        rescue_uids=rescue_uids,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    print(f"selected {len(candidates)} candidate blocks")
    report = run_micro_operator(
        student_model_path=os.path.expandvars(args.student_model),
        teacher_model_path=os.path.expandvars(args.teacher_model),
        student_device=args.student_device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        cohort_dir=args.cohort_dir,
        cohort_parquet_path=args.cohort_parquet_path or "",
        proposal_dirs=args.proposal_dirs,
        candidate_blocks=candidates,
        output_dir=args.output_dir,
        response_format=args.response_format,
        degraded_mode=args.degraded_mode,
        max_continuation_tokens=args.max_continuation_tokens,
        rollout_k=args.rollout_k,
        steps=args.steps,
        lr=args.lr,
        fkl_top_k=args.fkl_top_k,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
