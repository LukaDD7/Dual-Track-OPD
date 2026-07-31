"""Step 1 — Frozen-Policy Support Diagnostic.

Generates student rollouts from a frozen pre-RL checkpoint, teacher/student
force-scores every response, classifies prompts by observed support state,
and tests whether ``teacher_gap`` can rank correct trajectories above wrong
ones.

Usage::

    python -m dual_track_opd.support_aware.diagnostic \\
        --config configs/experiment/support_aware_geometry3k_pilot.yaml \\
        --mode smoke
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .reporter import (
    DiagnosticRunMeta,
    write_prompt_support_summary_jsonl,
    write_resolved_config_yaml,
    write_rollouts_jsonl,
    write_run_manifest,
    write_selected_prompts_jsonl,
    write_summary_json,
)
from .scorer import (
    StudentScorer,
    StudentScorerConfig,
    TeacherScorer,
    TeacherScorerConfig,
)
from .support_state import (
    SupportState,
    classify_support_state,
)
from .verifier import verify_answer

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """{question}

Think step by step about this geometry problem. First analyze the diagram carefully, then reason through the solution, and finally give your answer as "Answer: <number>"."""


def build_prompt(question: str) -> str:
    return _PROMPT_TEMPLATE.format(question=question.strip())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticConfig:
    dataset_path: str = ""
    num_prompts: int = 128
    rollouts_per_prompt: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    seed: int = 42
    student_model_path: str = ""
    teacher_url: str = "http://127.0.0.1:18080"
    device: str = "cuda"
    dtype: str = "bfloat16"
    output_root: str = ""
    mode: str = "smoke"  # "smoke" or "full"

    @property
    def resolved_num_prompts(self) -> int:
        if self.mode == "smoke":
            return 8
        return self.num_prompts

    @property
    def resolved_rollouts_per_prompt(self) -> int:
        if self.mode == "smoke":
            return 2
        return self.rollouts_per_prompt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_select_prompts(
    dataset_path: str,
    num_prompts: int,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], Path]:
    """Load Geometry3K parquet and select prompts deterministically.

    Selection rule: sort by SHA256(sample_uid), take first N.
    """
    import pandas as pd

    path = Path(os.path.expandvars(dataset_path))
    df = pd.read_parquet(str(path))

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        # Validate required fields
        if not rec.get("sample_uid"):
            continue
        if not rec.get("question"):
            continue
        if not _has_valid_image(rec):
            continue
        records.append(rec)

    # Deterministic selection by SHA256(sample_uid)
    records.sort(key=lambda r: hashlib.sha256(str(r["sample_uid"]).encode()).hexdigest())

    selected = records[:num_prompts]

    # Record selection manifest
    manifest = {
        "dataset_path": str(path),
        "total_rows": len(df),
        "valid_rows": len(records),
        "selected_rows": len(selected),
        "selection_rule": "sort_by_sha256_sample_uid",
        "selection_sha256": hashlib.sha256(
            json.dumps([r["sample_uid"] for r in selected], sort_keys=True).encode()
        ).hexdigest(),
    }

    return selected, manifest


def _has_valid_image(record: Mapping[str, Any]) -> bool:
    """Check that the record contains valid image data."""
    images = record.get("images")
    if images is None:
        return False
    if isinstance(images, np.ndarray):
        return len(images) > 0
    if isinstance(images, list):
        return len(images) > 0 and images[0] is not None
    return False


def extract_image(record: Mapping[str, Any]) -> Image.Image:
    """Extract PIL Image from a Geometry3K parquet record."""
    images = record["images"]
    if isinstance(images, np.ndarray):
        item = images[0]
    elif isinstance(images, list):
        item = images[0]
    else:
        raise ValueError(f"unexpected images type: {type(images)}")

    if isinstance(item, dict):
        # Embedded bytes
        if "bytes" in item:
            return Image.open(io.BytesIO(item["bytes"]))
        # Path reference
        if "path" in item or "image_path" in item:
            img_path = item.get("path") or item.get("image_path")
            return Image.open(os.path.expandvars(str(img_path))).convert("RGB")
    if isinstance(item, bytes):
        return Image.open(io.BytesIO(item))
    if isinstance(item, str):
        return Image.open(os.path.expandvars(item)).convert("RGB")
    if isinstance(item, Image.Image):
        return item

    raise ValueError(f"cannot extract image from item type: {type(item)}")


def save_image_for_teacher(image: Image.Image, temp_dir: Path) -> str:
    """Save a PIL Image to a temp file and return its path."""
    path = temp_dir / f"{hashlib.sha256(image.tobytes()).hexdigest()[:16]}.png"
    if not path.exists():
        image.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _is_greedy(temperature: float) -> bool:
    return temperature <= 0.0


def rollout_seed(base_seed: int, source_index: int, rollout_id: int) -> int:
    return base_seed + source_index * 1000 + rollout_id


def generate_response(
    model,
    processor,
    question: str,
    image: Image.Image,
    prompt_text: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    device: str,
) -> tuple[str, tuple[int, ...]]:
    """Generate one response from the student model.

    Returns (response_text, response_token_ids).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt"
    ).to(device)

    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0, input_len:]
    response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    response_token_ids = tuple(int(t) for t in generated_ids.tolist())

    return response_text, response_token_ids


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_mean_logp(sampled_log_probs: Sequence[float]) -> float:
    if not sampled_log_probs:
        return float("nan")
    return float(sum(sampled_log_probs) / len(sampled_log_probs))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_diagnostic(config: DiagnosticConfig) -> dict[str, Any]:
    """Execute the full frozen-policy diagnostic pipeline.

    Returns the aggregate summary dict.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    start_time = time.time()

    # -- Git info ---------------------------------------------------------------
    git_commit = _get_git_commit()
    git_dirty = _get_git_dirty()

    # -- Resolve output directory -----------------------------------------------
    output_root = Path(os.path.expandvars(config.output_root))
    run_id = _make_run_id(config.mode)
    output_dir = output_root / "support_aware_opd" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Support-Aware Diagnostic: {run_id} ===")
    print(f"Output: {output_dir}")
    print(f"Mode: {config.mode}")

    # -- Load data --------------------------------------------------------------
    print("\n[1/6] Loading and selecting prompts...")
    prompts, selection_manifest = load_and_select_prompts(
        config.dataset_path,
        config.resolved_num_prompts,
    )
    print(f"  Selected {len(prompts)} prompts from {selection_manifest['valid_rows']} valid rows")

    # -- Init models ------------------------------------------------------------
    print("\n[2/6] Initializing models...")

    # Student model for generation + scoring
    student_path = os.path.expandvars(config.student_model_path.removeprefix("hf:"))
    print(f"  Loading student: {student_path}")
    processor = AutoProcessor.from_pretrained(student_path)
    torch_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        student_path, torch_dtype=torch_dtype, device_map=config.device
    )
    model.eval()

    # Student scorer (uses the same model for forced scoring)
    student_scorer = StudentScorer(
        StudentScorerConfig(
            model_path=config.student_model_path,
            device=config.device,
            dtype=config.dtype,
        )
    )

    # Teacher scorer
    teacher = TeacherScorer(TeacherScorerConfig(base_url=config.teacher_url))

    # -- Pre-flight checks ------------------------------------------------------
    print("\n[3/6] Pre-flight checks...")

    if not teacher.health():
        print(f"  ERROR: Teacher service at {config.teacher_url} is not healthy")
        print(f"  Start it first: bash scripts/hpc/start_fc_teacher.sh")
        return _fail_run(output_dir, config, start_time, git_commit, git_dirty)

    student_hash = student_scorer.tokenizer_hash()
    teacher_hash = teacher.tokenizer_hash
    print(f"  Student tokenizer: {student_hash[:16]}...")
    print(f"  Teacher tokenizer: {teacher_hash[:16]}...")
    if student_hash != teacher_hash:
        print(f"  FATAL: Tokenizer mismatch! Token-aligned RKL is not possible.")
        print(f"  Student: {student_hash}")
        print(f"  Teacher: {teacher_hash}")
        return _fail_run(output_dir, config, start_time, git_commit, git_dirty)
    print(f"  ✓ Tokenizers match — token-aligned scoring is viable")

    # -- Image extraction & temp storage ----------------------------------------
    temp_dir = Path(tempfile.mkdtemp(prefix="support_aware_images_"))
    print(f"\n  Image temp dir: {temp_dir}")

    # -- Generate and score ----------------------------------------------------
    K = config.resolved_rollouts_per_prompt
    all_rollouts: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    print(f"\n[4/6] Generating and scoring ({len(prompts)} prompts × {1+K} rollouts)...")

    missing_image = 0
    non_finite_count = 0
    malformed_count = 0
    total_rollouts = 0

    for pi, prompt in enumerate(prompts):
        sample_uid = str(prompt["sample_uid"])
        question = str(prompt.get("question", "")).strip()
        image = extract_image(prompt)
        image_path = save_image_for_teacher(image, temp_dir)
        prompt_text = build_prompt(question)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        gold_answer = prompt.get("answer")

        if image is None:
            missing_image += 1
            continue

        # Greedy response
        greedy_text, greedy_ids = generate_response(
            model, processor, question, image, prompt_text,
            temperature=0.0, top_p=1.0,
            max_new_tokens=config.max_new_tokens,
            seed=rollout_seed(config.seed, int(prompt.get("source_index", pi)), 0),
            device=config.device,
        )
        greedy_verdict = verify_answer(greedy_text, gold_answer)
        greedy_correct = greedy_verdict.get("correct")

        # Score greedy
        _score_and_record(
            all_rollouts, teacher, student_scorer,
            sample_uid=sample_uid, question=question, image=image,
            image_path=image_path, image_hash=image_hash,
            prompt_text=prompt_text, prompt_hash=prompt_hash,
            response_text=greedy_text, response_token_ids=greedy_ids,
            config=config, prompt_index=pi, rollout_id=0,
            is_greedy=True, gold_answer=gold_answer,
            generation_seed=rollout_seed(config.seed, int(prompt.get("source_index", pi)), 0),
            tokenizer_hash=student_hash,
            teacher_model_id=teacher.model_id,
            non_finite_count=non_finite_count,
            errors=errors,
        )
        total_rollouts += 1
        if greedy_verdict.get("malformed"):
            malformed_count += 1

        # Stochastic rollouts
        correct_count = 0
        prompt_rollout_hashes: list[str] = []
        for rollout_id in range(1, K + 1):
            gen_seed = rollout_seed(config.seed, int(prompt.get("source_index", pi)), rollout_id)
            resp_text, resp_ids = generate_response(
                model, processor, question, image, prompt_text,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                seed=gen_seed,
                device=config.device,
            )
            verdict = verify_answer(resp_text, gold_answer)
            if verdict.get("correct"):
                correct_count += 1
            if verdict.get("malformed"):
                malformed_count += 1

            _score_and_record(
                all_rollouts, teacher, student_scorer,
                sample_uid=sample_uid, question=question, image=image,
                image_path=image_path, image_hash=image_hash,
                prompt_text=prompt_text, prompt_hash=prompt_hash,
                response_text=resp_text, response_token_ids=resp_ids,
                config=config, prompt_index=pi, rollout_id=rollout_id,
                is_greedy=False, gold_answer=gold_answer,
                generation_seed=gen_seed,
                tokenizer_hash=student_hash,
                teacher_model_id=teacher.model_id,
                non_finite_count=non_finite_count,
                errors=errors,
            )
            total_rollouts += 1
            prompt_rollout_hashes.append(hashlib.sha256(resp_text.encode()).hexdigest())

        # Per-prompt support state
        unique_hashes = set(prompt_rollout_hashes)
        dup_rate = 0.0
        if len(prompt_rollout_hashes) > 0:
            dup_rate = 1.0 - len(unique_hashes) / len(prompt_rollout_hashes)

        state = classify_support_state(
            greedy_correct=greedy_correct,
            correct_count=correct_count,
            K=K,
        )

        # Compute ranking metrics from this prompt's rollouts
        prompt_rollouts = [r for r in all_rollouts if r["sample_uid"] == sample_uid and not r["is_greedy"]]
        rank_metrics = _compute_ranking_metrics(prompt_rollouts)

        prompt_summaries.append({
            "sample_uid": sample_uid,
            "K": K,
            "correct_count": correct_count,
            "greedy_correct": greedy_correct,
            "support_state": state.value,
            "unique_response_count": len(unique_hashes),
            "duplicate_rollout_rate": dup_rate,
            **rank_metrics,
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            print(f"  [{pi+1}/{len(prompts)}] {sample_uid}: state={state.value}, "
                  f"correct={correct_count}/{K}, greedy_correct={greedy_correct}")

    # -- Compute summary --------------------------------------------------------
    print(f"\n[5/6] Computing summary metrics...")

    # Support state counts
    state_counts = {
        s.value: sum(1 for p in prompt_summaries if p["support_state"] == s.value)
        for s in SupportState
    }

    # Correct-vs-wrong AUC for teacher_gap and teacher_mean_logp
    stochastic_rollouts = [r for r in all_rollouts if not r["is_greedy"]]
    auc_teacher_gap = _compute_auc(
        stochastic_rollouts, score_key="teacher_gap", correct_key="correct"
    )
    auc_teacher_mean = _compute_auc(
        stochastic_rollouts, score_key="teacher_mean_logp", correct_key="correct"
    )
    auc_student_mean = _compute_auc(
        stochastic_rollouts, score_key="student_mean_logp", correct_key="correct"
    )

    # Correct-tail rank metrics
    correct_tail_rank = _compute_correct_tail_ranks(prompt_summaries, stochastic_rollouts)

    # Response length stats
    lengths = [r.get("response_token_count", 0) for r in all_rollouts]
    length_percentiles = {
        "p10": int(np.percentile(lengths, 10)) if lengths else 0,
        "p50": int(np.percentile(lengths, 50)) if lengths else 0,
        "p90": int(np.percentile(lengths, 90)) if lengths else 0,
        "mean": float(np.mean(lengths)) if lengths else 0.0,
    }

    # Duplicate rate (overall)
    total_dups = sum(
        (p.get("K", 0) - p.get("unique_response_count", 0))
        for p in prompt_summaries
    )
    total_stochastic = len(stochastic_rollouts)
    overall_dup_rate = total_dups / max(total_stochastic, 1)

    # Greedy accuracy
    greedy_rollouts = [r for r in all_rollouts if r["is_greedy"]]
    greedy_correct_count = sum(1 for r in greedy_rollouts if r.get("correct") is True)
    greedy_accuracy = greedy_correct_count / max(len(greedy_rollouts), 1)

    # Empirical pass@1 and pass@K
    correct_tails = [p for p in prompt_summaries if p["support_state"] == "correct_tail"]
    exposed = [p for p in prompt_summaries if p["support_state"] == "exposed"]
    no_correct = [p for p in prompt_summaries if p["support_state"] == "no_correct_observed"]

    summary = {
        "run_id": run_id,
        "mode": config.mode,
        "num_prompts": len(prompts),
        "rollouts_per_prompt": K,
        "total_rollouts": total_rollouts,
        "greedy_accuracy": greedy_accuracy,
        "support_state_counts": state_counts,
        "correct_tail_count": len(correct_tails),
        "exposed_count": len(exposed),
        "no_correct_observed_count": len(no_correct),
        "auc_teacher_gap_correct_vs_wrong": auc_teacher_gap,
        "auc_teacher_mean_logp_correct_vs_wrong": auc_teacher_mean,
        "auc_student_mean_logp_correct_vs_wrong": auc_student_mean,
        "correct_tail_rank_metrics": correct_tail_rank,
        "response_length_percentiles": length_percentiles,
        "overall_duplicate_rollout_rate": overall_dup_rate,
        "malformed_response_count": malformed_count,
        "malformed_response_rate": malformed_count / max(total_rollouts, 1),
        "missing_image_count": missing_image,
        "non_finite_score_count": non_finite_count,
        "tokenizer_match": student_hash == teacher_hash,
        "student_tokenizer_hash": student_hash,
        "teacher_tokenizer_hash": teacher_hash,
        "teacher_model_id": teacher.model_id,
        "student_model_path": config.student_model_path,
        "selection_manifest": selection_manifest,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }

    # -- Acceptance gates ------------------------------------------------------
    print("\n[6/6] Acceptance gates...")
    gates = _check_gates(summary, config.mode)
    summary["acceptance_gates"] = gates
    for gate_name, passed, detail in gates:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {gate_name}: {detail}")

    # -- Write outputs ---------------------------------------------------------
    print(f"\nWriting outputs to {output_dir}...")

    meta = DiagnosticRunMeta(
        run_id=run_id,
        output_dir=output_dir,
        config={k: str(v) for k, v in config.__dict__.items()},
        git_commit=git_commit,
        git_dirty=git_dirty,
        num_prompts=len(prompts),
        rollouts_per_prompt=K,
        seed=config.seed,
        start_time=start_time,
        end_time=time.time(),
        exit_status="PASS" if all(p for _, p, _ in gates) else "GATE_FAIL",
    )

    write_rollouts_jsonl(output_dir, all_rollouts)
    write_prompt_support_summary_jsonl(output_dir, prompt_summaries)
    write_selected_prompts_jsonl(output_dir, [
        {"sample_uid": p["sample_uid"], "question": str(p.get("question", ""))[:200]}
        for p in prompts
    ])
    write_summary_json(output_dir, summary)
    write_run_manifest(output_dir, meta)
    write_resolved_config_yaml(output_dir, {k: str(v) for k, v in config.__dict__.items()})

    print(f"\nDone. Outputs: {output_dir}")
    print(f"  rollouts.jsonl                — {len(all_rollouts)} rows")
    print(f"  prompt_support_summary.jsonl  — {len(prompt_summaries)} rows")
    print(f"  summary.json")

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_run_id(mode: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"diag_{mode}_{ts}"


def _fail_run(
    output_dir: Path,
    config: DiagnosticConfig,
    start_time: float,
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    meta = DiagnosticRunMeta(
        run_id=output_dir.name,
        output_dir=output_dir,
        config={},
        git_commit=git_commit,
        git_dirty=git_dirty,
        num_prompts=0,
        rollouts_per_prompt=0,
        seed=config.seed,
        start_time=start_time,
        end_time=time.time(),
        exit_status="PREFLIGHT_FAIL",
    )
    write_run_manifest(output_dir, meta)
    return {"error": "preflight_failed", "run_id": output_dir.name}


def _score_and_record(
    all_rollouts: list[dict[str, Any]],
    teacher: TeacherScorer,
    student_scorer: StudentScorer,
    *,
    sample_uid: str,
    question: str,
    image: Image.Image,
    image_path: str,
    image_hash: str,
    prompt_text: str,
    prompt_hash: str,
    response_text: str,
    response_token_ids: tuple[int, ...],
    config: DiagnosticConfig,
    prompt_index: int,
    rollout_id: int,
    is_greedy: bool,
    gold_answer: Any,
    generation_seed: int,
    tokenizer_hash: str,
    teacher_model_id: str,
    non_finite_count: int,
    errors: list[str],
) -> None:
    """Score one rollout and append to all_rollouts."""
    rollout_uid = f"{sample_uid}:{'greedy' if is_greedy else f'rollout-{rollout_id}'}"
    verdict = verify_answer(response_text, gold_answer)

    # Teacher score
    t_result = teacher.score(
        request_id=rollout_uid,
        question=question,
        image_path=image_path,
        response_token_ids=response_token_ids,
        response_text=response_text,
    )
    teacher_mean_logp = t_result.mean_logp
    teacher_sampled = t_result.sampled_token_log_probs

    # Student score
    s_result = student_scorer.score(
        question=question,
        image=image,
        prompt_text=prompt_text,
        response_text=response_text,
        response_token_ids=response_token_ids,
    )
    student_mean_logp = s_result.mean_logp
    student_sampled = s_result.sampled_token_log_probs

    # Teacher gap
    import math
    teacher_gap = (
        teacher_mean_logp - student_mean_logp
        if (not math.isnan(teacher_mean_logp) and not math.isnan(student_mean_logp))
        else float("nan")
    )

    if math.isnan(teacher_mean_logp) or math.isnan(student_mean_logp):
        non_finite_count += 1

    if t_result.error:
        errors.append(f"{rollout_uid}: teacher_error={t_result.error}")
    if s_result.error:
        errors.append(f"{rollout_uid}: student_error={s_result.error}")

    all_rollouts.append({
        "run_id": _make_run_id(config.mode),
        "sample_uid": sample_uid,
        "source_index": prompt_index,
        "split": "train",
        "rollout_id": rollout_id,
        "is_greedy": is_greedy,
        "generation_seed": generation_seed,
        "temperature": 0.0 if is_greedy else config.temperature,
        "top_p": 1.0 if is_greedy else config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "response_text": response_text,
        "response_token_ids": list(response_token_ids),
        "response_token_count": len(response_token_ids),
        "response_hash": hashlib.sha256(response_text.encode()).hexdigest(),
        "answer_extracted": verdict.get("answer_extracted"),
        "gold_answer": verdict.get("gold_answer"),
        "format_valid": verdict.get("format_valid", True),
        "correct": verdict.get("correct"),
        "malformed": verdict.get("malformed", False),
        "student_sampled_token_log_probs": list(student_sampled),
        "teacher_sampled_token_log_probs": list(teacher_sampled),
        "student_mean_logp": student_mean_logp,
        "teacher_mean_logp": teacher_mean_logp,
        "teacher_gap": teacher_gap,
        "student_model_path": config.student_model_path,
        "teacher_model_id": teacher_model_id,
        "prompt_hash": prompt_hash,
        "image_hash": image_hash,
        "tokenizer_hash": tokenizer_hash,
        "errors": [e for e in (t_result.error, s_result.error) if e],
    })


def _compute_ranking_metrics(
    prompt_rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute ranking metrics for one prompt's stochastic rollouts."""
    if not prompt_rollouts:
        return {
            "teacher_rank_of_best_correct": None,
            "student_rank_of_best_correct": None,
            "teacher_gap_rank_of_best_correct": None,
            "teacher_top1_correct": None,
            "teacher_gap_top1_correct": None,
        }

    # Rank by teacher_gap (descending)
    by_gap = sorted(prompt_rollouts, key=lambda r: r.get("teacher_gap", float("-inf")), reverse=True)
    # Rank by teacher_mean_logp (descending)
    by_teacher = sorted(prompt_rollouts, key=lambda r: r.get("teacher_mean_logp", float("-inf")), reverse=True)
    # Rank by student_mean_logp (descending)
    by_student = sorted(prompt_rollouts, key=lambda r: r.get("student_mean_logp", float("-inf")), reverse=True)

    def best_correct_rank(ranked: list[dict[str, Any]]) -> int | None:
        for rank, r in enumerate(ranked):
            if r.get("correct") is True:
                return rank + 1  # 1-indexed
        return None

    return {
        "teacher_rank_of_best_correct": best_correct_rank(by_teacher),
        "student_rank_of_best_correct": best_correct_rank(by_student),
        "teacher_gap_rank_of_best_correct": best_correct_rank(by_gap),
        "teacher_top1_correct": by_teacher[0].get("correct") if by_teacher else None,
        "teacher_gap_top1_correct": by_gap[0].get("correct") if by_gap else None,
    }


def _compute_auc(
    rollouts: list[dict[str, Any]],
    score_key: str,
    correct_key: str = "correct",
) -> float | None:
    """Compute ROC AUC for a score predicting correctness."""
    import math
    scores = []
    labels = []
    for r in rollouts:
        val = r.get(score_key)
        lab = r.get(correct_key)
        if val is None or lab is None or not math.isfinite(float(val)):
            continue
        scores.append(float(val))
        labels.append(1.0 if lab else 0.0)

    if len(set(labels)) < 2:
        return None  # AUC undefined when all same class

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Manual AUC computation
        pairs = list(zip(scores, labels))
        pairs.sort(key=lambda x: x[0], reverse=True)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        auc = 0.0
        tp = 0
        for _, label in pairs:
            if label == 1:
                tp += 1
            else:
                auc += tp
        return auc / (n_pos * n_neg)


def _compute_correct_tail_ranks(
    prompt_summaries: list[dict[str, Any]],
    all_rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute rank metrics specifically for correct_tail prompts."""
    correct_tail_uids = {
        p["sample_uid"]
        for p in prompt_summaries
        if p["support_state"] == "correct_tail"
    }

    tail_rollouts = [r for r in all_rollouts if r["sample_uid"] in correct_tail_uids and not r["is_greedy"]]

    if not tail_rollouts:
        return {
            "rank1": None,
            "mrr": None,
            "num_prompts": 0,
            "num_rollouts": 0,
        }

    # Group by prompt
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for r in tail_rollouts:
        by_prompt.setdefault(r["sample_uid"], []).append(r)

    reciprocal_ranks = []
    top1_hits = 0
    for uid, rollouts in by_prompt.items():
        by_gap = sorted(rollouts, key=lambda r: r.get("teacher_gap", float("-inf")), reverse=True)
        for rank, r in enumerate(by_gap):
            if r.get("correct") is True:
                reciprocal_ranks.append(1.0 / (rank + 1))
                if rank == 0:
                    top1_hits += 1
                break

    return {
        "rank1": top1_hits / max(len(by_prompt), 1),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "num_prompts": len(by_prompt),
        "num_rollouts": len(tail_rollouts),
    }


def _check_gates(
    summary: dict[str, Any],
    mode: str,
) -> list[tuple[str, bool, str]]:
    """Check protocol and signal gates (runbook §4.7)."""
    gates: list[tuple[str, bool, str]] = []

    # Protocol gates
    tokenizer_match = summary.get("tokenizer_match", False)
    gates.append((
        "tokenizer_alignment",
        tokenizer_match,
        f"student and teacher tokenizers {'match' if tokenizer_match else 'DO NOT MATCH'}",
    ))

    missing = summary.get("missing_image_count", 0)
    gates.append((
        "zero_missing_images",
        missing == 0,
        f"{missing} missing images",
    ))

    non_finite = summary.get("non_finite_score_count", 0)
    gates.append((
        "finite_scores",
        non_finite == 0,
        f"{non_finite} non-finite scores",
    ))

    malformed_rate = summary.get("malformed_response_rate", 0.0)
    gates.append((
        "malformed_rate_ok",
        malformed_rate <= 0.10,
        f"malformed rate = {malformed_rate:.3f} (threshold: 0.10)",
    ))

    dup_rate = summary.get("overall_duplicate_rollout_rate", 0.0)
    gates.append((
        "duplicate_rate_ok",
        dup_rate <= 0.25,
        f"duplicate rate = {dup_rate:.3f} (threshold: 0.25)",
    ))

    # Signal gates (only for full mode)
    if mode == "full":
        correct_tail_count = summary.get("correct_tail_count", 0)
        gates.append((
            "sufficient_correct_tails",
            correct_tail_count >= 20,
            f"{correct_tail_count} correct_tail prompts (need ≥ 20; try 256 if < 20)",
        ))

        auc = summary.get("auc_teacher_gap_correct_vs_wrong")
        if auc is not None:
            gates.append((
                "teacher_gap_auc",
                auc >= 0.60,
                f"teacher_gap AUC = {auc:.3f} (threshold: 0.60)",
            ))
        else:
            gates.append((
                "teacher_gap_auc",
                False,
                "AUC undefined (possibly all correct or all wrong)",
            ))

    return gates


def _get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _get_git_dirty() -> bool:
    import subprocess
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return len(result) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str) -> dict[str, Any]:
    import yaml as _yaml
    with open(os.path.expandvars(path)) as fh:
        raw = _yaml.safe_load(fh) or {}
    return _resolve_env_vars(raw)


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR} patterns in strings."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _nested_get(cfg: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Get a value from nested dict keys."""
    for key in keys:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(key, {})
    return cfg if not isinstance(cfg, dict) or cfg else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1 — Frozen-Policy Support Diagnostic"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--mode", type=str, default="smoke",
                        choices=["smoke", "full"],
                        help="smoke=8 prompts × 2 rollouts, full=128 prompts × 8 rollouts")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path")
    parser.add_argument("--student-model-path", type=str, default=None,
                        help="Override student model path")
    parser.add_argument("--teacher-url", type=str, default=None,
                        help="Override teacher URL")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override output root")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Override number of prompts (for full mode)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    yaml_cfg = _load_yaml_config(args.config)

    # Merge CLI overrides (CLI > YAML nested key > default)
    config = DiagnosticConfig(
        dataset_path=args.dataset or _nested_get(yaml_cfg, "data", "dataset_path", default=""),
        num_prompts=args.num_prompts or _nested_get(yaml_cfg, "data", "num_prompts", default=128),
        rollouts_per_prompt=_nested_get(yaml_cfg, "diagnostic", "rollouts_per_prompt", default=8),
        temperature=_nested_get(yaml_cfg, "diagnostic", "temperature", default=0.7),
        top_p=_nested_get(yaml_cfg, "diagnostic", "top_p", default=0.95),
        max_new_tokens=_nested_get(yaml_cfg, "diagnostic", "max_new_tokens", default=512),
        seed=args.seed if args.seed is not None else _nested_get(yaml_cfg, "diagnostic", "seed", default=42),
        student_model_path=args.student_model_path or _nested_get(yaml_cfg, "models", "student", default=""),
        teacher_url=args.teacher_url or _nested_get(yaml_cfg, "teacher", "url", default="http://127.0.0.1:18080"),
        device=args.device or _nested_get(yaml_cfg, "hardware", "student_gpu", default="cuda"),
        dtype=args.dtype or _nested_get(yaml_cfg, "hardware", "dtype", default="bfloat16"),
        output_root=args.output_root or _nested_get(yaml_cfg, "output", "root", default=os.path.expandvars("$DTOPD_OUTPUT_ROOT")),
        mode=args.mode,
    )

    if not config.dataset_path:
        print("ERROR: dataset_path is required", file=sys.stderr)
        return 1
    if not config.student_model_path:
        print("ERROR: student_model_path is required", file=sys.stderr)
        return 1

    summary = run_diagnostic(config)

    # Return code based on gates
    gates = summary.get("acceptance_gates", [])
    if gates and all(passed for _, passed, _ in gates):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
