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
import math
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
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    output_root: str = ""
    mode: str = "smoke"  # "smoke" or "full"
    resume_run_id: str | None = None
    run_id_override: str | None = None
    prompt_start: int = 0
    prompt_end: int | None = None

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


def slice_prompts(
    prompts: list[dict[str, Any]],
    prompt_start: int,
    prompt_end: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Slice the deterministic prompt selection for sharded parallel runs.

    Shards are processed by separate processes/instances (each writing its own
    output dir), then combined with :func:`merge_shard_runs`.  Slicing is
    applied after the SHA256-sorted selection, so every shard shares the same
    ordering and resume-within-a-shard stays deterministic.
    """
    start = max(prompt_start, 0)
    end = len(prompts) if prompt_end is None else min(prompt_end, len(prompts))
    sliced = prompts[start:end]
    manifest = {
        "prompt_start": start,
        "prompt_end": end,
        "selection_rule": "sort_by_sha256_sample_uid (sharded slice)",
    }
    return sliced, manifest


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
    )
    # Move to whichever GPU the model is on (avoids CUDA_VISIBLE_DEVICES issues)
    if torch.cuda.is_available():
        inputs = inputs.to(f"cuda:{torch.cuda.current_device()}")

    do_sample = temperature > 0
    # transformers ≥ 5.x ignores temperature/top_p as generate() kwargs;
    # set them on the model's generation_config before calling generate().
    model.generation_config.do_sample = do_sample
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    if do_sample:
        model.generation_config.temperature = temperature
        model.generation_config.top_p = top_p
    else:
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    with torch.no_grad():
        outputs = model.generate(**inputs)

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


@dataclass
class _ResumeState:
    """Incremental state loaded from a partial run's JSONL files."""

    all_rollouts: list[dict[str, Any]]
    prompt_summaries: list[dict[str, Any]]
    completed_uids: set[str]
    errors: list[str]
    malformed_count: int
    non_finite_count: int
    total_rollouts: int


def _is_nonfinite(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _load_resume_state(output_dir: Path) -> _ResumeState:
    """Load rollouts/summaries written by a previous partial run.

    Only prompts that have a completed summary line are treated as done.
    Rollouts left behind by a prompt that was killed mid-write (no summary yet)
    are discarded from the in-memory state; the final writer overwrites the
    JSONL files with the full in-memory state when the run completes.
    """
    summaries: list[dict[str, Any]] = []
    summaries_path = output_dir / "prompt_support_summary.jsonl"
    if summaries_path.exists():
        with summaries_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))

    completed_uids = {
        str(s.get("sample_uid")) for s in summaries if s.get("sample_uid")
    }

    rollouts: list[dict[str, Any]] = []
    rollouts_path = output_dir / "rollouts.jsonl"
    if rollouts_path.exists():
        with rollouts_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if str(rec.get("sample_uid")) in completed_uids:
                    rollouts.append(rec)

    errors: list[str] = []
    malformed_count = 0
    non_finite_count = 0
    for rec in rollouts:
        errors.extend(rec.get("errors") or [])
        if rec.get("malformed"):
            malformed_count += 1
        if _is_nonfinite(rec.get("teacher_mean_logp")) or _is_nonfinite(
            rec.get("student_mean_logp")
        ):
            non_finite_count += 1

    return _ResumeState(
        all_rollouts=rollouts,
        prompt_summaries=summaries,
        completed_uids=completed_uids,
        errors=errors,
        malformed_count=malformed_count,
        non_finite_count=non_finite_count,
        total_rollouts=len(rollouts),
    )


def _build_summary(
    *,
    run_id: str,
    mode: str,
    num_prompts: int,
    K: int,
    all_rollouts: list[dict[str, Any]],
    prompt_summaries: list[dict[str, Any]],
    malformed_count: int,
    missing_image: int,
    non_finite_count: int,
    student_hash: str,
    teacher_hash: str,
    teacher_model_id: str,
    student_model_path: str,
    selection_manifest: Mapping[str, Any],
    git_commit: str,
    git_dirty: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    """Aggregate rollouts/prompt summaries into the final summary dict + gates.

    Shared by the full pipeline and the shard merge path so both produce
    identical metrics.
    """
    if verbose:
        print("\n[5/6] Computing summary metrics...")

    total_rollouts = len(all_rollouts)

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
        "mode": mode,
        "num_prompts": num_prompts,
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
        "teacher_model_id": teacher_model_id,
        "student_model_path": student_model_path,
        "selection_manifest": dict(selection_manifest),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }

    # -- Acceptance gates ------------------------------------------------------
    gates = _check_gates(summary, mode)
    summary["acceptance_gates"] = gates
    if verbose:
        print("\n[6/6] Acceptance gates...")
        for gate_name, passed, detail in gates:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {gate_name}: {detail}")
    return summary


def merge_shard_runs(
    shard_dirs: Sequence[str],
    output_dir: str,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Combine rollouts/prompt summaries from sharded runs into one summary.

    Each shard writes its own run directory (prompts are disjoint after
    ``slice_prompts``).  This reloads the JSONL files, dedupes defensively, and
    recomputes the aggregate summary + acceptance gates exactly like the single
    full run would.
    """
    rollouts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_rollout: set[tuple[str, bool, int]] = set()
    seen_uids: set[str] = set()
    manifests: list[dict[str, Any]] = []

    for d in shard_dirs:
        p = Path(d)
        rollouts_path = p / "rollouts.jsonl"
        if rollouts_path.exists():
            with rollouts_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (
                        str(rec.get("sample_uid")),
                        bool(rec.get("is_greedy")),
                        int(rec.get("rollout_id", -1)),
                    )
                    if key in seen_rollout:
                        continue
                    seen_rollout.add(key)
                    rollouts.append(rec)
        summaries_path = p / "prompt_support_summary.jsonl"
        if summaries_path.exists():
            with summaries_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    uid = str(rec.get("sample_uid"))
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    summaries.append(rec)
        summary_path = p / "summary.json"
        if summary_path.exists():
            try:
                with summary_path.open(encoding="utf-8") as fh:
                    manifests.append(json.load(fh))
            except (OSError, ValueError):
                pass

    if not rollouts:
        raise ValueError("no rollouts found in shard dirs")
    if not summaries:
        raise ValueError("no prompt summaries found in shard dirs")

    hashes = {str(r.get("tokenizer_hash")) for r in rollouts}
    if len(hashes) > 1:
        raise ValueError(f"tokenizer hash mismatch across shards: {hashes}")
    ks = {p.get("K") for p in summaries if p.get("K") is not None}
    if len(ks) > 1:
        raise ValueError(f"rollouts_per_prompt mismatch across shards: {ks}")

    tokenizer_hash = next(iter(hashes))
    malformed_count = sum(1 for r in rollouts if r.get("malformed"))
    non_finite_count = sum(
        1
        for r in rollouts
        if _is_nonfinite(r.get("teacher_mean_logp"))
        or _is_nonfinite(r.get("student_mean_logp"))
    )
    missing_image = sum(int(m.get("missing_image_count") or 0) for m in manifests)

    first_manifest = (
        (manifests[0].get("selection_manifest") or {})
        if manifests
        else {}
    )
    selection_manifest = {
        "dataset_path": first_manifest.get("dataset_path"),
        "valid_rows": first_manifest.get("valid_rows"),
        "merged_from": [str(Path(d).name) for d in shard_dirs],
        "selected_rows": len(summaries),
        "selection_rule": "sort_by_sha256_sample_uid (sharded merge)",
        "selection_sha256": hashlib.sha256(
            json.dumps([p["sample_uid"] for p in summaries], sort_keys=True).encode()
        ).hexdigest(),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(
        run_id=out.name or "merged",
        mode=mode,
        num_prompts=len(summaries),
        K=next(iter(ks)) if ks else 8,
        all_rollouts=rollouts,
        prompt_summaries=summaries,
        malformed_count=malformed_count,
        missing_image=missing_image,
        non_finite_count=non_finite_count,
        student_hash=tokenizer_hash,
        teacher_hash=tokenizer_hash,
        teacher_model_id=rollouts[0].get("teacher_model_id", ""),
        student_model_path=rollouts[0].get("student_model_path", ""),
        selection_manifest=selection_manifest,
        git_commit="merged",
        git_dirty=True,
        verbose=False,
    )
    write_rollouts_jsonl(out, rollouts)
    write_prompt_support_summary_jsonl(out, summaries)
    write_summary_json(out, summary)
    return summary


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
    if config.resume_run_id:
        run_id = config.resume_run_id
    elif config.run_id_override:
        run_id = config.run_id_override
    else:
        run_id = _make_run_id(config.mode)
    if config.resume_run_id:
        output_dir = output_root / "support_aware_opd" / run_id
        if not (output_dir / "prompt_support_summary.jsonl").exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"FATAL: cannot resume — no prompt_support_summary.jsonl in {output_dir}",
                file=sys.stderr,
            )
            return _fail_run(output_dir, config, start_time, git_commit, git_dirty)
    else:
        output_dir = output_root / "support_aware_opd" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Open incremental output files so we don't lose data if the run is killed
    rollouts_path = output_dir / "rollouts.jsonl"
    summaries_path = output_dir / "prompt_support_summary.jsonl"
    rollouts_fh = rollouts_path.open("a", encoding="utf-8")  # append for resume
    summaries_fh = summaries_path.open("a", encoding="utf-8")

    # Reload incremental state when resuming a partial run.
    resume_state = _load_resume_state(output_dir) if config.resume_run_id else None

    print(f"=== Support-Aware Diagnostic: {run_id} ===")
    print(f"Output: {output_dir}")
    print(f"Mode: {config.mode}")
    if resume_state is not None:
        print(
            f"  RESUME: {len(resume_state.prompt_summaries)} prompts / "
            f"{len(resume_state.all_rollouts)} rollouts already on disk; "
            f"skipping completed prompts"
        )

    # -- Load data --------------------------------------------------------------
    print("\n[1/6] Loading and selecting prompts...")
    prompts, selection_manifest = load_and_select_prompts(
        config.dataset_path,
        config.resolved_num_prompts,
    )
    prompts, slice_manifest = slice_prompts(
        prompts, config.prompt_start, config.prompt_end
    )
    selection_manifest = {
        **selection_manifest,
        **slice_manifest,
        "selected_rows": len(prompts),
    }
    print(f"  Selected {len(prompts)} prompts from {selection_manifest['valid_rows']} valid rows")

    # -- Init models ------------------------------------------------------------
    print("\n[2/6] Initializing models...")

    # Student model for generation + scoring
    student_path = os.path.expandvars(config.student_model_path.removeprefix("hf:"))
    print(f"  Loading student: {student_path}")
    processor = AutoProcessor.from_pretrained(student_path)
    torch_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        student_path, torch_dtype=torch_dtype, device_map="auto"
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
    all_rollouts: list[dict[str, Any]] = (
        list(resume_state.all_rollouts) if resume_state is not None else []
    )
    prompt_summaries: list[dict[str, Any]] = (
        list(resume_state.prompt_summaries) if resume_state is not None else []
    )
    errors: list[str] = list(resume_state.errors) if resume_state is not None else []

    print(f"\n[4/6] Generating and scoring ({len(prompts)} prompts × {1+K} rollouts)...")

    missing_image = 0
    nf_container = [
        resume_state.non_finite_count if resume_state is not None else 0
    ]  # mutable so _score_and_record can increment
    malformed_count = resume_state.malformed_count if resume_state is not None else 0
    total_rollouts = resume_state.total_rollouts if resume_state is not None else 0

    for pi, prompt in enumerate(prompts):
        sample_uid = str(prompt["sample_uid"])
        if resume_state is not None and sample_uid in resume_state.completed_uids:
            continue
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

        # -- Phase A: Generate all responses first ---------------------------------
        # Collect metadata for all 9 rollouts (1 greedy + K stochastic) before
        # scoring, so we can batch-score teacher and student in single calls.
        batch_meta: list[dict[str, Any]] = []

        # Greedy
        greedy_seed = rollout_seed(config.seed, int(prompt.get("source_index", pi)), 0)
        greedy_text, greedy_ids = generate_response(
            model, processor, question, image, prompt_text,
            temperature=0.0, top_p=1.0,
            max_new_tokens=config.max_new_tokens,
            seed=greedy_seed,
            device=config.device,
        )
        greedy_verdict = verify_answer(greedy_text, gold_answer)
        greedy_correct = greedy_verdict.get("correct")
        batch_meta.append(dict(
            rollout_id=0, is_greedy=True, generation_seed=greedy_seed,
            response_text=greedy_text, response_token_ids=greedy_ids,
            verdict=greedy_verdict,
        ))

        # Stochastic
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
            batch_meta.append(dict(
                rollout_id=rollout_id, is_greedy=False, generation_seed=gen_seed,
                response_text=resp_text, response_token_ids=resp_ids,
                verdict=verdict,
            ))
            prompt_rollout_hashes.append(hashlib.sha256(resp_text.encode()).hexdigest())

        # -- Phase B: Batch-score all responses ------------------------------------
        # Teacher batch: one HTTP call for all 9 rollouts
        batch_size = len(batch_meta)
        batch_request_ids = [
            f"{sample_uid}:greedy" if m["is_greedy"] else f"{sample_uid}:rollout-{m['rollout_id']}"
            for m in batch_meta
        ]
        t_results = teacher.score_batch(
            request_ids=batch_request_ids,
            questions=[question] * batch_size,
            image_paths=[image_path] * batch_size,
            prompt_texts=[prompt_text] * batch_size,
            response_token_ids_list=[m["response_token_ids"] for m in batch_meta],
            response_texts=[m["response_text"] for m in batch_meta],
        )

        # Student batch: one forward pass for all 9 rollouts
        s_results = student_scorer.score_batch(
            questions=[question] * batch_size,
            images=[image] * batch_size,
            prompt_texts=[prompt_text] * batch_size,
            response_texts=[m["response_text"] for m in batch_meta],
            response_token_ids_list=[m["response_token_ids"] for m in batch_meta],
        )

        # -- Phase C: Build and write records --------------------------------------
        for i, meta in enumerate(batch_meta):
            t_result = t_results[i] if i < len(t_results) else None
            s_result = s_results[i] if i < len(s_results) else None
            # Fallback for batch failures
            teacher_sampled = t_result.sampled_token_log_probs if t_result else ()
            teacher_mean_logp = t_result.mean_logp if t_result else float("nan")
            student_sampled = s_result.sampled_token_log_probs if s_result else ()
            student_mean_logp = s_result.mean_logp if s_result else float("nan")

            import math
            teacher_gap = (
                teacher_mean_logp - student_mean_logp
                if (not math.isnan(teacher_mean_logp) and not math.isnan(student_mean_logp))
                else float("nan")
            )
            if math.isnan(teacher_mean_logp) or math.isnan(student_mean_logp):
                nf_container[0] += 1
            if t_result and t_result.error:
                errors.append(f"{sample_uid}:rollout-{meta['rollout_id']}: teacher_error={t_result.error}")
            if s_result and s_result.error:
                errors.append(f"{sample_uid}:rollout-{meta['rollout_id']}: student_error={s_result.error}")

            rollout_uid = f"{sample_uid}:greedy" if meta["is_greedy"] else f"{sample_uid}:rollout-{meta['rollout_id']}"
            all_rollouts.append({
                "run_id": run_id,
                "sample_uid": sample_uid,
                "source_index": pi,
                "split": "train",
                "rollout_id": meta["rollout_id"],
                "is_greedy": meta["is_greedy"],
                "generation_seed": meta["generation_seed"],
                "temperature": 0.0 if meta["is_greedy"] else config.temperature,
                "top_p": 1.0 if meta["is_greedy"] else config.top_p,
                "max_new_tokens": config.max_new_tokens,
                "prompt_hash": prompt_hash,
                "image_hash": image_hash,
                "gold_answer": gold_answer,
                "response_text": meta["response_text"],
                "response_token_ids": meta["response_token_ids"],
                "response_token_count": len(meta["response_token_ids"]),
                "response_hash": hashlib.sha256(meta["response_text"].encode()).hexdigest(),
                "correct": meta["verdict"].get("correct"),
                "answer_extracted": meta["verdict"].get("extracted"),
                "format_valid": meta["verdict"].get("format_valid", True),
                "malformed": meta["verdict"].get("malformed", False),
                "teacher_mean_logp": teacher_mean_logp,
                "teacher_sampled_token_log_probs": teacher_sampled,
                "student_mean_logp": student_mean_logp,
                "student_sampled_token_log_probs": student_sampled,
                "teacher_gap": teacher_gap,
                "teacher_model_id": teacher.model_id,
                "student_model_path": config.student_model_path,
                "tokenizer_hash": student_hash,
                "errors": [e for e in errors if e.startswith(f"{sample_uid}:")],
            })
            _write_rollout_line(rollouts_fh, all_rollouts[-1])
            total_rollouts += 1

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

        prompt_summary = {
            "sample_uid": sample_uid,
            "K": K,
            "correct_count": correct_count,
            "greedy_correct": greedy_correct,
            "support_state": state.value,
            "unique_response_count": len(unique_hashes),
            "duplicate_rollout_rate": dup_rate,
            **rank_metrics,
        }
        prompt_summaries.append(prompt_summary)
        # Incremental write: flush prompt summary immediately
        _write_json_line(summaries_fh, prompt_summary)

        # Write partial summary every prompt (cheap — just a few KB)
        prompts_done = len(prompt_summaries)
        _write_partial_summary(output_dir, {
            "run_id": run_id, "mode": config.mode,
            "prompts_done": prompts_done, "prompts_total": len(prompts),
            "rollouts_done": total_rollouts,
            "elapsed_s": time.time() - start_time,
            "missing_image": missing_image,
            "malformed_count": malformed_count,
            "non_finite_count": nf_container[0],
            "errors_tail": errors[-10:],
        })

        if prompts_done % 10 == 0 or prompts_done == 1:
            print(f"  [{prompts_done}/{len(prompts)}] {sample_uid}: state={state.value}, "
                  f"correct={correct_count}/{K}, greedy_correct={greedy_correct}")

    # -- Compute summary --------------------------------------------------------
    summary = _build_summary(
        run_id=run_id,
        mode=config.mode,
        num_prompts=len(prompts),
        K=K,
        all_rollouts=all_rollouts,
        prompt_summaries=prompt_summaries,
        malformed_count=malformed_count,
        missing_image=missing_image,
        non_finite_count=nf_container[0],
        student_hash=student_hash,
        teacher_hash=teacher_hash,
        teacher_model_id=teacher.model_id,
        student_model_path=config.student_model_path,
        selection_manifest=selection_manifest,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    gates = summary["acceptance_gates"]

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

    # Close incremental file handles before overwriting with final versions
    rollouts_fh.close()
    summaries_fh.close()

    write_rollouts_jsonl(output_dir, all_rollouts)
    write_prompt_support_summary_jsonl(output_dir, prompt_summaries)
    write_selected_prompts_jsonl(output_dir, [
        {"sample_uid": p["sample_uid"], "question": str(p.get("question", ""))[:200]}
        for p in prompts
    ])
    write_summary_json(output_dir, summary)
    write_run_manifest(output_dir, meta)
    write_resolved_config_yaml(output_dir, {k: str(v) for k, v in config.__dict__.items()})

    # Remove partial summary — run is complete
    partial_path = output_dir / "_partial_summary.json"
    if partial_path.exists():
        partial_path.unlink()

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


def _write_json_line(fh, record: dict[str, Any]) -> None:
    """Write a single JSON line to a file handle and flush."""
    fh.write(json.dumps(record, ensure_ascii=False, default=_default))
    fh.write("\n")
    fh.flush()


def _write_rollout_line(fh, record: dict[str, Any]) -> None:
    """Write a single rollout record to the rollouts JSONL file and flush."""
    _write_json_line(fh, record)


def _write_partial_summary(output_dir: Path, info: dict[str, Any]) -> None:
    """Write a lightweight partial summary so progress is visible on disk."""
    path = output_dir / "_partial_summary.json"
    # Atomic write: write to temp file then rename
    tmp = output_dir / "_partial_summary.json.tmp"
    tmp.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.rename(path)


def _default(obj: Any) -> Any:
    """JSON default serializer — reused from reporter module."""
    from pathlib import Path as _Path
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


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
    run_id: str,
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
    non_finite_count: list[int],  # mutable container so caller sees increments
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
        prompt_text=prompt_text,
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
        non_finite_count[0] += 1

    if t_result.error:
        errors.append(f"{rollout_uid}: teacher_error={t_result.error}")
    if s_result.error:
        errors.append(f"{rollout_uid}: student_error={s_result.error}")

    all_rollouts.append({
        "run_id": run_id,
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
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an existing partial run by run_id (e.g. diag_full_20260801_123045)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run directory name (avoids timestamp collisions for "
        "parallel pipelines started in the same second)",
    )
    parser.add_argument(
        "--prompt-start",
        type=int,
        default=None,
        help="First prompt index in the deterministic selection (sharded runs)",
    )
    parser.add_argument(
        "--prompt-end",
        type=int,
        default=None,
        help="Exclusive end prompt index in the deterministic selection",
    )
    parser.add_argument(
        "--merge-shards",
        nargs="+",
        default=None,
        help="Merge completed shard run directories into one summary",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default=None,
        help="Output directory for a --merge-shards result",
    )
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="full",
        choices=["smoke", "full"],
        help="Mode to use when scoring the merged summary gates",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # -- Shard merge mode -----------------------------------------------------
    if args.merge_shards:
        if not args.merge_output:
            print("ERROR: --merge-output is required with --merge-shards", file=sys.stderr)
            return 1
        try:
            summary = merge_shard_runs(
                args.merge_shards,
                args.merge_output,
                mode=args.merge_mode,
            )
        except Exception as exc:
            print(f"ERROR: merge failed: {exc}", file=sys.stderr)
            return 1
        print(f"=== Merged summary ({len(summary.get('acceptance_gates', []))} gates) ===")
        for gate_name, passed, detail in summary["acceptance_gates"]:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {gate_name}: {detail}")
        print(f"Outputs: {args.merge_output}")
        return 0

    if not args.config:
        print("ERROR: --config is required", file=sys.stderr)
        return 1

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
        resume_run_id=args.resume,
        run_id_override=args.run_id,
        prompt_start=args.prompt_start if args.prompt_start is not None else 0,
        prompt_end=args.prompt_end,
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
