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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.student_rollout_signal_audit import build_rollout_prompt
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

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


def build_prompt(question: str, response_format: str = "legacy_answer") -> str:
    if response_format == "legacy_answer":
        return _PROMPT_TEMPLATE.format(question=question.strip())
    return build_rollout_prompt(question, response_format=response_format).text


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
    response_format: str = "legacy_answer"
    malformed_rate_max: float = 0.10
    duplicate_rate_max: float = 0.25
    min_correct_tails: int = 20
    teacher_gap_auc_min: float = 0.60
    correct_tail_rank1_above_random: bool = True
    truncation_rate_max: float = 0.15
    bootstrap_seed: int = 42
    bootstrap_resamples: int = 10_000

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

    @property
    def gate_config(self) -> dict[str, Any]:
        return {
            "malformed_rate_max": self.malformed_rate_max,
            "duplicate_rate_max": self.duplicate_rate_max,
            "min_correct_tails": self.min_correct_tails,
            "teacher_gap_auc_min": self.teacher_gap_auc_min,
            "correct_tail_rank1_above_random": self.correct_tail_rank1_above_random,
            "truncation_rate_max": self.truncation_rate_max,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_select_prompts(
    dataset_path: str,
    num_prompts: int,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        "dataset_sha256": _sha256_file(path),
        "total_rows": len(df),
        "valid_rows": len(records),
        "selected_rows": len(selected),
        "selection_rule": "sort_by_sha256_sample_uid",
        "selection_sha256": hashlib.sha256(
            json.dumps([r["sample_uid"] for r in selected], sort_keys=True).encode()
        ).hexdigest(),
    }

    return selected, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass(frozen=True)
class GenerationRecord:
    """Lossless generation output plus display-only text and stop metadata."""

    response_token_ids_raw: tuple[int, ...]
    response_text_display: str
    response_text_raw: str
    response_token_hash: str
    prompt_token_hash: str
    finish_reason: Literal["stop", "length", "unknown"]
    terminal_token_id: int | None
    content_mask: tuple[bool, ...]


def _generation_stop_token_ids(model, tokenizer) -> set[int]:
    raw_ids: list[Any] = [getattr(tokenizer, "eos_token_id", None)]
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        configured = getattr(generation_config, "eos_token_id", None)
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            raw_ids.extend(configured)
        else:
            raw_ids.append(configured)
    return {int(token_id) for token_id in raw_ids if token_id is not None}


def generation_record_from_token_ids(
    *,
    model,
    tokenizer,
    response_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    response_text_display: str | None = None,
) -> GenerationRecord:
    """Build auditable generation metadata without changing the action IDs.

    This helper is shared by live generation and append-only migration of old
    runs.  In particular, display decoding is never used to reconstruct the
    response token sequence.
    """

    response_ids = tuple(int(token_id) for token_id in response_token_ids)
    prompt_ids = tuple(int(token_id) for token_id in prompt_token_ids)
    if response_text_display is None:
        response_text_display = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    response_text_raw = tokenizer.decode(
        response_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    stop_token_ids = _generation_stop_token_ids(model, tokenizer)
    terminal_token_id = (
        response_ids[-1]
        if response_ids and response_ids[-1] in stop_token_ids
        else None
    )
    if terminal_token_id is not None:
        finish_reason: Literal["stop", "length", "unknown"] = "stop"
    elif len(response_ids) >= max_new_tokens:
        finish_reason = "length"
    else:
        finish_reason = "unknown"
    content_mask = tuple(
        not (terminal_token_id is not None and index == len(response_ids) - 1)
        for index in range(len(response_ids))
    )
    return GenerationRecord(
        response_token_ids_raw=response_ids,
        response_text_display=response_text_display,
        response_text_raw=response_text_raw,
        response_token_hash=hash_token_ids(response_ids),
        prompt_token_hash=hash_token_ids(prompt_ids),
        finish_reason=finish_reason,
        terminal_token_id=terminal_token_id,
        content_mask=content_mask,
    )


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
) -> GenerationRecord:
    """Generate one response from the student model.

    Display text is decoded separately; exact scoring must use the returned raw
    token IDs and never re-tokenize the text.
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
    response_token_ids = tuple(int(t) for t in generated_ids.tolist())
    prompt_ids = tuple(int(token_id) for token_id in inputs["input_ids"][0].tolist())
    return generation_record_from_token_ids(
        model=model,
        tokenizer=processor.tokenizer,
        response_token_ids=response_token_ids,
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_mean_logp(sampled_log_probs: Sequence[float]) -> float:
    if not sampled_log_probs:
        return float("nan")
    return float(sum(sampled_log_probs) / len(sampled_log_probs))


def compute_masked_logp(
    sampled_log_probs: Sequence[float],
    mask: Sequence[bool],
) -> tuple[float, float, int]:
    """Return sum, mean, and count under an explicit token mask."""

    if len(sampled_log_probs) != len(mask):
        raise ValueError("log-probabilities and token mask must have equal length")
    selected = [float(value) for value, keep in zip(sampled_log_probs, mask, strict=True) if keep]
    if not selected:
        return 0.0, float("nan"), 0
    return float(sum(selected)), float(sum(selected) / len(selected)), len(selected)


def assert_exact_score_alignment(
    *,
    expected_token_ids: Sequence[int],
    teacher_result: TeacherScorer.ScoreResult,
    student_result: StudentScorer.ScoreResult,
) -> None:
    """Fail closed unless both scorers evaluated the generated action sequence."""

    expected = tuple(int(token_id) for token_id in expected_token_ids)
    expected_mask = (True,) * len(expected)
    if teacher_result.scored_token_ids != expected:
        raise RuntimeError("teacher scored token IDs differ from generated response IDs")
    if student_result.scored_token_ids != expected:
        raise RuntimeError("student scored token IDs differ from generated response IDs")
    if teacher_result.response_mask != expected_mask:
        raise RuntimeError("teacher response mask differs from generated response mask")
    if student_result.response_mask != expected_mask:
        raise RuntimeError("student response mask differs from generated response mask")
    expected_hash = hash_token_ids(expected)
    if teacher_result.scored_token_hash != expected_hash:
        raise RuntimeError("teacher scored token hash differs from generated response hash")
    if student_result.scored_token_hash != expected_hash:
        raise RuntimeError("student scored token hash differs from generated response hash")
    if len(teacher_result.sampled_token_log_probs) != len(expected):
        raise RuntimeError("teacher log-probability count differs from generated token count")
    if len(student_result.sampled_token_log_probs) != len(expected):
        raise RuntimeError("student log-probability count differs from generated token count")


def build_exact_scored_record(
    *,
    base_record: Mapping[str, Any],
    generation: GenerationRecord,
    verdict: Mapping[str, Any],
    teacher_result: TeacherScorer.ScoreResult,
    student_result: StudentScorer.ScoreResult,
) -> dict[str, Any]:
    """Combine one immutable rollout with exact-ID scorer/verifier evidence."""

    record_errors = [
        message
        for message in (
            f"teacher_error={teacher_result.error}" if teacher_result.error else None,
            f"student_error={student_result.error}" if student_result.error else None,
        )
        if message is not None
    ]
    exact_token_alignment = False
    if not record_errors:
        assert_exact_score_alignment(
            expected_token_ids=generation.response_token_ids_raw,
            teacher_result=teacher_result,
            student_result=student_result,
        )
        exact_token_alignment = True

    teacher_sampled = teacher_result.sampled_token_log_probs
    student_sampled = student_result.sampled_token_log_probs
    if exact_token_alignment:
        teacher_sum_all, teacher_mean_all, teacher_count_all = compute_masked_logp(
            teacher_sampled, (True,) * len(teacher_sampled)
        )
        student_sum_all, student_mean_all, student_count_all = compute_masked_logp(
            student_sampled, (True,) * len(student_sampled)
        )
        teacher_sum_content, teacher_mean_content, teacher_count_content = compute_masked_logp(
            teacher_sampled, generation.content_mask
        )
        student_sum_content, student_mean_content, student_count_content = compute_masked_logp(
            student_sampled, generation.content_mask
        )
    else:
        teacher_sum_all = student_sum_all = float("nan")
        teacher_mean_all = student_mean_all = float("nan")
        teacher_sum_content = student_sum_content = float("nan")
        teacher_mean_content = student_mean_content = float("nan")
        teacher_count_all = student_count_all = 0
        teacher_count_content = student_count_content = 0

    teacher_gap = (
        teacher_mean_content - student_mean_content
        if math.isfinite(teacher_mean_content) and math.isfinite(student_mean_content)
        else float("nan")
    )
    teacher_gap_all = (
        teacher_mean_all - student_mean_all
        if math.isfinite(teacher_mean_all) and math.isfinite(student_mean_all)
        else float("nan")
    )
    teacher_terminal_logp = (
        float(teacher_sampled[-1])
        if generation.terminal_token_id is not None and teacher_sampled
        else None
    )
    student_terminal_logp = (
        float(student_sampled[-1])
        if generation.terminal_token_id is not None and student_sampled
        else None
    )

    record = {
        **dict(base_record),
        "response_text": generation.response_text_display,
        "response_text_display": generation.response_text_display,
        "response_text_raw": generation.response_text_raw,
        "response_token_ids": generation.response_token_ids_raw,
        "response_token_hash": generation.response_token_hash,
        "response_token_count": len(generation.response_token_ids_raw),
        "content_token_count": sum(generation.content_mask),
        "content_mask": generation.content_mask,
        "finish_reason": generation.finish_reason,
        "terminal_token_id": generation.terminal_token_id,
        "response_hash": hashlib.sha256(
            generation.response_text_display.encode()
        ).hexdigest(),
        "correct": verdict.get("correct"),
        "answer_extracted": verdict.get("answer_extracted"),
        "gold_answer": verdict.get("gold_answer", base_record.get("gold_answer")),
        "format_valid": verdict.get("format_valid", False),
        "malformed": verdict.get("malformed", False),
        "teacher_sum_logp": teacher_sum_content,
        "teacher_mean_logp": teacher_mean_content,
        "teacher_sum_logp_all": teacher_sum_all,
        "teacher_mean_logp_all": teacher_mean_all,
        "teacher_terminal_logp": teacher_terminal_logp,
        "teacher_scored_token_count": teacher_count_all,
        "teacher_content_token_count": teacher_count_content,
        "teacher_scored_token_hash": teacher_result.scored_token_hash,
        "teacher_response_mask": teacher_result.response_mask,
        "teacher_sampled_token_log_probs": teacher_sampled,
        "student_sum_logp": student_sum_content,
        "student_mean_logp": student_mean_content,
        "student_sum_logp_all": student_sum_all,
        "student_mean_logp_all": student_mean_all,
        "student_terminal_logp": student_terminal_logp,
        "student_scored_token_count": student_count_all,
        "student_content_token_count": student_count_content,
        "student_scored_token_hash": student_result.scored_token_hash,
        "student_response_mask": student_result.response_mask,
        "student_sampled_token_log_probs": student_sampled,
        "teacher_gap": teacher_gap,
        "teacher_gap_all": teacher_gap_all,
        "exact_token_alignment": exact_token_alignment,
        "errors": record_errors,
    }
    return record


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


def _validate_resume_state_for_config(
    state: _ResumeState,
    config: DiagnosticConfig,
) -> None:
    """Prevent mixing historical text-retokenized rows with exact-ID rows."""

    for record in state.all_rollouts:
        uid = str(record.get("sample_uid") or "unknown")
        rollout_id = record.get("rollout_id")
        response_ids = tuple(
            int(token_id) for token_id in record.get("response_token_ids") or ()
        )
        expected_hash = hash_token_ids(response_ids) if response_ids else ""
        expected_mask = [True] * len(response_ids)
        if (
            not response_ids
            or record.get("response_token_hash") != expected_hash
            or record.get("exact_token_alignment") is not True
            or not record.get("prompt_token_hash")
            or record.get("prompt_version") != config.response_format
            or record.get("teacher_scored_token_hash") != expected_hash
            or record.get("student_scored_token_hash") != expected_hash
            or list(record.get("teacher_response_mask") or ()) != expected_mask
            or list(record.get("student_response_mask") or ()) != expected_mask
        ):
            raise ValueError(
                f"{uid}:rollout-{rollout_id} predates or violates the exact-token "
                "scoring contract; do not mix it with new rows. Recover/finish the "
                "cohort separately and use support_aware.rescore into a new output."
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
    gate_config: Mapping[str, Any] | None = None,
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 10_000,
    shard_completeness_rate: float = 1.0,
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
    correct_tail_rank = _compute_correct_tail_ranks(
        prompt_summaries,
        stochastic_rollouts,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    prompt_signal_all = _compute_prompt_signal_metrics(
        stochastic_rollouts,
        seed=bootstrap_seed + 10,
        resamples=bootstrap_resamples,
    )
    prompt_signal_by_finish_reason = {
        finish_reason: _compute_prompt_signal_metrics(
            [
                rollout
                for rollout in stochastic_rollouts
                if finish_reason == "all" or rollout.get("finish_reason") == finish_reason
            ],
            seed=bootstrap_seed + 20 + index * 10,
            resamples=bootstrap_resamples,
        )
        for index, finish_reason in enumerate(("all", "stop", "length"))
    }
    length_bins = {
        "le_512": lambda length: length <= 512,
        "513_2048": lambda length: 512 < length <= 2048,
        "gt_2048": lambda length: length > 2048,
    }
    prompt_signal_by_length_bin = {
        label: _compute_prompt_signal_metrics(
            [
                rollout
                for rollout in stochastic_rollouts
                if predicate(int(rollout.get("response_token_count") or 0))
            ],
            seed=bootstrap_seed + 100 + index * 10,
            resamples=bootstrap_resamples,
        )
        for index, (label, predicate) in enumerate(length_bins.items())
    }

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

    finish_reason_counts = {
        reason: sum(1 for rollout in all_rollouts if rollout.get("finish_reason") == reason)
        for reason in ("stop", "length", "unknown")
    }
    truncation_rate = finish_reason_counts["length"] / max(total_rollouts, 1)
    exact_alignment_count = sum(
        1 for rollout in all_rollouts if rollout.get("exact_token_alignment") is True
    )
    exact_alignment_rate = exact_alignment_count / max(total_rollouts, 1)
    prompt_token_hash_count = sum(
        1 for rollout in all_rollouts if bool(rollout.get("prompt_token_hash"))
    )
    prompt_token_hash_rate = prompt_token_hash_count / max(total_rollouts, 1)
    prompt_version_counts: dict[str, int] = {}
    for rollout in all_rollouts:
        version = str(rollout.get("prompt_version") or "unavailable")
        prompt_version_counts[version] = prompt_version_counts.get(version, 0) + 1

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
        "prompt_signal_metrics": prompt_signal_all,
        "prompt_signal_by_finish_reason": prompt_signal_by_finish_reason,
        "prompt_signal_by_length_bin": prompt_signal_by_length_bin,
        "response_length_percentiles": length_percentiles,
        "finish_reason_counts": finish_reason_counts,
        "truncation_rate": truncation_rate,
        "exact_token_alignment_count": exact_alignment_count,
        "exact_token_alignment_rate": exact_alignment_rate,
        "prompt_token_hash_count": prompt_token_hash_count,
        "prompt_token_hash_rate": prompt_token_hash_rate,
        "prompt_version_counts": prompt_version_counts,
        "shard_completeness_rate": float(shard_completeness_rate),
        "scoring_policy": "exact_raw_ids_content_primary_terminal_separate_v1",
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
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "gate_config": dict(gate_config or {}),
    }

    # -- Acceptance gates ------------------------------------------------------
    gates = _check_gates(summary, mode, gate_config=gate_config)
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
    """Strictly validate and combine a complete set of disjoint shard runs."""
    import yaml

    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    rollouts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    selected_prompts: list[dict[str, Any]] = []
    seen_rollout: set[tuple[str, bool, int]] = set()
    seen_uids: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    resolved_configs: list[dict[str, Any]] = []
    shard_intervals: list[tuple[int, int, list[str]]] = []

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
        return rows

    def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
        ignored = {
            "prompt_start",
            "prompt_end",
            "run_id_override",
            "resume_run_id",
            # Per-shard append-only migration provenance; validated through
            # each run_manifest/source artifact hash rather than equality.
            "source_run",
            "output_dir",
        }
        return {str(key): value for key, value in config.items() if key not in ignored}

    for d in shard_dirs:
        p = Path(d)
        required = (
            "rollouts.jsonl",
            "prompt_support_summary.jsonl",
            "selected_prompts.jsonl",
            "summary.json",
            "run_manifest.json",
            "resolved_config.yaml",
        )
        missing = [name for name in required if not (p / name).is_file()]
        if missing:
            raise ValueError(f"{p}: incomplete shard; missing {missing}")

        shard_rollouts = read_jsonl(p / "rollouts.jsonl")
        shard_prompt_summaries = read_jsonl(p / "prompt_support_summary.jsonl")
        shard_selected = read_jsonl(p / "selected_prompts.jsonl")
        shard_summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        run_manifest = json.loads((p / "run_manifest.json").read_text(encoding="utf-8"))
        resolved_config = yaml.safe_load((p / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        if not isinstance(resolved_config, dict):
            raise ValueError(f"{p}: resolved_config.yaml must contain a mapping")
        if run_manifest.get("exit_status") not in {"PASS", "GATE_FAIL"}:
            raise ValueError(f"{p}: shard is not complete (exit_status={run_manifest.get('exit_status')})")

        shard_uids = [str(row.get("sample_uid", "")) for row in shard_selected]
        if any(not uid for uid in shard_uids) or len(shard_uids) != len(set(shard_uids)):
            raise ValueError(f"{p}: selected prompt UIDs are empty or duplicated")
        summary_uids = {str(row.get("sample_uid", "")) for row in shard_prompt_summaries}
        if (
            len(shard_prompt_summaries) != len(shard_uids)
            or summary_uids != set(shard_uids)
        ):
            raise ValueError(f"{p}: selected prompts and prompt summaries do not match")
        if seen_uids.intersection(shard_uids):
            overlap = sorted(seen_uids.intersection(shard_uids))[:10]
            raise ValueError(f"{p}: shard UID overlap detected: {overlap}")

        shard_k_values = {int(row.get("K", -1)) for row in shard_prompt_summaries}
        if len(shard_k_values) != 1:
            raise ValueError(f"{p}: inconsistent K within shard: {shard_k_values}")
        shard_k = next(iter(shard_k_values))
        expected_keys = {
            (uid, is_greedy, rollout_id)
            for uid in shard_uids
            for is_greedy, rollout_id in (
                [(True, 0)] + [(False, rollout_id) for rollout_id in range(1, shard_k + 1)]
            )
        }
        actual_keys: set[tuple[str, bool, int]] = set()
        for record in shard_rollouts:
            key = (
                str(record.get("sample_uid", "")),
                bool(record.get("is_greedy")),
                int(record.get("rollout_id", -1)),
            )
            if key in actual_keys or key in seen_rollout:
                raise ValueError(f"{p}: duplicate rollout key {key}")
            actual_keys.add(key)
        if actual_keys != expected_keys:
            missing_keys = sorted(expected_keys - actual_keys)[:10]
            extra_keys = sorted(actual_keys - expected_keys)[:10]
            raise ValueError(
                f"{p}: incomplete rollout key set; missing={missing_keys}, extra={extra_keys}"
            )

        for record in shard_rollouts:
            response_ids = tuple(int(token_id) for token_id in record.get("response_token_ids") or ())
            response_hash = hash_token_ids(response_ids) if response_ids else ""
            if not response_ids or record.get("response_token_hash") != response_hash:
                raise ValueError(f"{p}: missing/invalid raw response token hash")
            if record.get("exact_token_alignment") is not True:
                raise ValueError(f"{p}: rollout lacks proven exact-token alignment")
            if not record.get("prompt_token_hash"):
                raise ValueError(f"{p}: rollout lacks prompt token hash")
            if record.get("teacher_scored_token_hash") != response_hash:
                raise ValueError(f"{p}: teacher scored-token hash differs from raw response")
            if record.get("student_scored_token_hash") != response_hash:
                raise ValueError(f"{p}: student scored-token hash differs from raw response")
            expected_mask = [True] * len(response_ids)
            if list(record.get("teacher_response_mask") or ()) != expected_mask:
                raise ValueError(f"{p}: teacher response mask differs from exact response mask")
            if list(record.get("student_response_mask") or ()) != expected_mask:
                raise ValueError(f"{p}: student response mask differs from exact response mask")

        start = int(resolved_config.get("prompt_start") or 0)
        raw_end = resolved_config.get("prompt_end")
        end = start + len(shard_uids) if raw_end in (None, "None", "") else int(raw_end)
        if end - start != len(shard_uids):
            raise ValueError(
                f"{p}: configured prompt interval [{start}, {end}) does not match "
                f"{len(shard_uids)} selected prompts"
            )
        shard_intervals.append((start, end, shard_uids))

        seen_uids.update(shard_uids)
        seen_rollout.update(actual_keys)
        rollouts.extend(shard_rollouts)
        summaries.extend(shard_prompt_summaries)
        selected_prompts.extend(shard_selected)
        shard_summaries.append(shard_summary)
        run_manifests.append(run_manifest)
        resolved_configs.append(resolved_config)

    reference_config = normalized_config(resolved_configs[0])
    for index, resolved_config in enumerate(resolved_configs[1:], start=1):
        if normalized_config(resolved_config) != reference_config:
            raise ValueError(f"shard {index} resolved config differs from shard 0")

    selection_manifests = [
        summary.get("selection_manifest") or {} for summary in shard_summaries
    ]
    selection_total_counts = {
        int(manifest["selection_total_rows"])
        for manifest in selection_manifests
        if manifest.get("selection_total_rows") is not None
    }
    if len(selection_total_counts) > 1:
        raise ValueError(
            f"selection total mismatch across shards: {selection_total_counts}"
        )
    expected_prompt_count = (
        next(iter(selection_total_counts))
        if selection_total_counts
        else int(reference_config.get("num_prompts", len(seen_uids)))
    )
    if len(seen_uids) != expected_prompt_count:
        raise ValueError(
            f"incomplete shard coverage: got {len(seen_uids)} prompts, expected {expected_prompt_count}"
        )
    cursor = 0
    for start, end, _ in sorted(shard_intervals):
        if start != cursor:
            raise ValueError(
                f"incomplete/overlapping shard intervals: expected start {cursor}, got {start}"
            )
        cursor = end
    if cursor != expected_prompt_count:
        raise ValueError(
            f"incomplete shard intervals: covered [0, {cursor}), expected [0, {expected_prompt_count})"
        )

    ks = {p.get("K") for p in summaries if p.get("K") is not None}
    if len(ks) != 1:
        raise ValueError(f"rollouts_per_prompt mismatch across shards: {ks}")
    K = int(next(iter(ks)))
    student_hashes = {
        str(record.get("student_tokenizer_hash") or record.get("tokenizer_hash") or "")
        for record in rollouts
    }
    teacher_hashes = {
        str(record.get("teacher_tokenizer_hash") or record.get("tokenizer_hash") or "")
        for record in rollouts
    }
    if len(student_hashes) != 1 or "" in student_hashes:
        raise ValueError(f"student tokenizer hash mismatch across shards: {student_hashes}")
    if len(teacher_hashes) != 1 or "" in teacher_hashes:
        raise ValueError(f"teacher tokenizer hash mismatch across shards: {teacher_hashes}")
    student_hash = next(iter(student_hashes))
    teacher_hash = next(iter(teacher_hashes))
    if student_hash != teacher_hash:
        raise ValueError("student and teacher tokenizer hashes differ")

    teacher_model_ids = {str(record.get("teacher_model_id") or "") for record in rollouts}
    student_model_paths = {str(record.get("student_model_path") or "") for record in rollouts}
    if (
        len(teacher_model_ids) != 1
        or "" in teacher_model_ids
        or len(student_model_paths) != 1
        or "" in student_model_paths
    ):
        raise ValueError("model identity mismatch across shards")
    git_commits = {str(manifest.get("git_commit") or "") for manifest in run_manifests}
    if len(git_commits) != 1 or "" in git_commits:
        raise ValueError(f"git commit mismatch across shards: {git_commits}")
    git_dirty_states = {bool(manifest.get("git_dirty")) for manifest in run_manifests}
    if len(git_dirty_states) != 1:
        raise ValueError(f"git dirty state mismatch across shards: {git_dirty_states}")

    dataset_hashes = {str(manifest.get("dataset_sha256") or "") for manifest in selection_manifests}
    selection_hashes = {str(manifest.get("selection_sha256") or "") for manifest in selection_manifests}
    if len(dataset_hashes) != 1 or "" in dataset_hashes:
        raise ValueError(f"dataset content hash mismatch/missing across shards: {dataset_hashes}")
    if len(selection_hashes) != 1 or "" in selection_hashes:
        raise ValueError(f"selection manifest hash mismatch/missing across shards: {selection_hashes}")

    uid_order = sorted(
        seen_uids,
        key=lambda uid: hashlib.sha256(uid.encode()).hexdigest(),
    )
    for start, end, shard_uids in shard_intervals:
        if shard_uids != uid_order[start:end]:
            raise ValueError(
                f"shard UID order/content does not match deterministic selection slice "
                f"[{start}, {end})"
            )
    union_selection_hash = hashlib.sha256(
        json.dumps(uid_order, sort_keys=True).encode()
    ).hexdigest()
    source_selection_hash = next(iter(selection_hashes))
    if union_selection_hash != source_selection_hash:
        raise ValueError(
            "merged UID union does not match the source selection manifest hash"
        )

    prompt_versions = {str(record.get("prompt_version") or "") for record in rollouts}
    expected_prompt_version = str(reference_config.get("response_format") or "")
    if (
        len(prompt_versions) != 1
        or "" in prompt_versions
        or (expected_prompt_version and next(iter(prompt_versions)) != expected_prompt_version)
    ):
        raise ValueError(
            f"prompt version mismatch across shards/config: {prompt_versions}, "
            f"config={expected_prompt_version!r}"
        )
    scoring_policies = {
        str(summary.get("scoring_policy") or "") for summary in shard_summaries
    }
    if scoring_policies != {"exact_raw_ids_content_primary_terminal_separate_v1"}:
        raise ValueError(f"scoring policy mismatch/missing across shards: {scoring_policies}")

    malformed_count = sum(1 for r in rollouts if r.get("malformed"))
    non_finite_count = sum(
        1
        for r in rollouts
        if _is_nonfinite(r.get("teacher_mean_logp"))
        or _is_nonfinite(r.get("student_mean_logp"))
    )
    missing_image = sum(int(summary.get("missing_image_count") or 0) for summary in shard_summaries)
    first_manifest = selection_manifests[0]
    selection_manifest = {
        "dataset_path": first_manifest.get("dataset_path"),
        "dataset_sha256": next(iter(dataset_hashes)),
        "valid_rows": first_manifest.get("valid_rows"),
        "merged_from": [str(Path(d).name) for d in shard_dirs],
        "selection_total_rows": expected_prompt_count,
        "selected_rows": len(summaries),
        "selection_rule": "sort_by_sha256_sample_uid (sharded merge)",
        "selection_sha256": source_selection_hash,
    }

    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"merge output already exists: {out}")
    gate_config = shard_summaries[0].get("gate_config") or {}
    bootstrap_seed = int(shard_summaries[0].get("bootstrap_seed") or 42)
    bootstrap_resamples = int(shard_summaries[0].get("bootstrap_resamples") or 10_000)
    git_commit = next(iter(git_commits))
    git_dirty = next(iter(git_dirty_states))
    summary = _build_summary(
        run_id=out.name or "merged",
        mode=mode,
        num_prompts=len(summaries),
        K=K,
        all_rollouts=rollouts,
        prompt_summaries=summaries,
        malformed_count=malformed_count,
        missing_image=missing_image,
        non_finite_count=non_finite_count,
        student_hash=student_hash,
        teacher_hash=teacher_hash,
        teacher_model_id=next(iter(teacher_model_ids)),
        student_model_path=next(iter(student_model_paths)),
        selection_manifest=selection_manifest,
        git_commit=git_commit,
        git_dirty=git_dirty,
        gate_config=gate_config,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        shard_completeness_rate=1.0,
        verbose=False,
    )
    uid_rank = {uid: index for index, uid in enumerate(uid_order)}
    rollouts.sort(
        key=lambda row: (
            uid_rank[str(row["sample_uid"])],
            0 if row.get("is_greedy") else 1,
            int(row.get("rollout_id") or 0),
        )
    )
    summaries.sort(key=lambda row: uid_rank[str(row["sample_uid"])])
    selected_prompts.sort(key=lambda row: uid_rank[str(row["sample_uid"])])
    merged_config = dict(resolved_configs[0])
    merged_config["prompt_start"] = "0"
    merged_config["prompt_end"] = str(expected_prompt_count)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        write_rollouts_jsonl(temp_out, rollouts)
        write_prompt_support_summary_jsonl(temp_out, summaries)
        write_selected_prompts_jsonl(temp_out, selected_prompts)
        write_summary_json(temp_out, summary)
        write_resolved_config_yaml(temp_out, merged_config)
        write_run_manifest(temp_out, DiagnosticRunMeta(
            run_id=out.name,
            output_dir=out,
            config=merged_config,
            git_commit=git_commit,
            git_dirty=git_dirty,
            num_prompts=len(summaries),
            rollouts_per_prompt=K,
            seed=int(reference_config.get("seed", 42)),
            start_time=time.time(),
            end_time=time.time(),
            exit_status=(
                "PASS"
                if all(passed for _, passed, _ in summary["acceptance_gates"])
                else "GATE_FAIL"
            ),
        ))
        temp_out.rename(out)
    except Exception:
        import shutil

        shutil.rmtree(temp_out, ignore_errors=True)
        raise
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
    if resume_state is not None:
        try:
            _validate_resume_state_for_config(resume_state, config)
        except ValueError as exc:
            rollouts_fh.close()
            summaries_fh.close()
            print(f"FATAL: incompatible resume state: {exc}", file=sys.stderr)
            return {"error": "incompatible_resume", "detail": str(exc), "run_id": run_id}

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
    selection_total_rows = int(selection_manifest["selected_rows"])
    prompts, slice_manifest = slice_prompts(
        prompts, config.prompt_start, config.prompt_end
    )
    selection_manifest = {
        **selection_manifest,
        **slice_manifest,
        "selection_total_rows": selection_total_rows,
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

    student_hash = tokenizer_fingerprint(processor.tokenizer)

    # Student scorer reuses the live rollout model.  Loading a second 4B copy
    # wastes memory and risks processor/version drift.
    student_scorer = StudentScorer(
        StudentScorerConfig(
            model_path=config.student_model_path,
            device=config.device,
            dtype=config.dtype,
        ),
        model=model,
        processor=processor,
    )

    # Fail during construction if the teacher does not share the student's
    # exact response-token action space.
    teacher = TeacherScorer(TeacherScorerConfig(
        base_url=config.teacher_url,
        expected_tokenizer_hash=student_hash,
    ))

    # -- Pre-flight checks ------------------------------------------------------
    print("\n[3/6] Pre-flight checks...")

    if not teacher.health():
        print(f"  ERROR: Teacher service at {config.teacher_url} is not healthy")
        print("  Start it first: bash scripts/hpc/start_fc_teacher.sh")
        return _fail_run(output_dir, config, start_time, git_commit, git_dirty)

    scorer_student_hash = student_scorer.tokenizer_hash()
    teacher_hash = teacher.tokenizer_hash
    print(f"  Student tokenizer: {student_hash[:16]}...")
    print(f"  Teacher tokenizer: {teacher_hash[:16]}...")
    if student_hash != scorer_student_hash or student_hash != teacher_hash:
        print("  FATAL: Tokenizer mismatch! Token-aligned RKL is not possible.")
        print(f"  Student: {student_hash}")
        print(f"  Student scorer: {scorer_student_hash}")
        print(f"  Teacher: {teacher_hash}")
        return _fail_run(output_dir, config, start_time, git_commit, git_dirty)
    print("  ✓ Tokenizers match — token-aligned scoring is viable")

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
        prompt_text = build_prompt(question, config.response_format)
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
        greedy_generation = generate_response(
            model, processor, question, image, prompt_text,
            temperature=0.0, top_p=1.0,
            max_new_tokens=config.max_new_tokens,
            seed=greedy_seed,
            device=config.device,
        )
        greedy_verdict = verify_answer(
            greedy_generation.response_text_display,
            gold_answer,
        )
        greedy_correct = greedy_verdict.get("correct")
        if greedy_verdict.get("malformed"):
            malformed_count += 1
        batch_meta.append(dict(
            rollout_id=0, is_greedy=True, generation_seed=greedy_seed,
            generation=greedy_generation,
            response_text=greedy_generation.response_text_display,
            response_token_ids=greedy_generation.response_token_ids_raw,
            verdict=greedy_verdict,
        ))

        # Stochastic
        correct_count = 0
        prompt_rollout_hashes: list[str] = []
        for rollout_id in range(1, K + 1):
            gen_seed = rollout_seed(config.seed, int(prompt.get("source_index", pi)), rollout_id)
            generation = generate_response(
                model, processor, question, image, prompt_text,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                seed=gen_seed,
                device=config.device,
            )
            verdict = verify_answer(generation.response_text_display, gold_answer)
            if verdict.get("correct"):
                correct_count += 1
            if verdict.get("malformed"):
                malformed_count += 1
            batch_meta.append(dict(
                rollout_id=rollout_id, is_greedy=False, generation_seed=gen_seed,
                generation=generation,
                response_text=generation.response_text_display,
                response_token_ids=generation.response_token_ids_raw,
                verdict=verdict,
            ))
            prompt_rollout_hashes.append(generation.response_token_hash)

        # -- Phase B: Batch-score all responses ------------------------------------
        # Teacher transport batch: one HTTP request for the 1 greedy + K
        # stochastic rollouts.  The Transformers teacher backend intentionally
        # scores requests sequentially inside that transport batch to preserve
        # exact multimodal token alignment; this is not one B=1+K tensor
        # forward.  See the experiment contract before changing this path.
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
            generation: GenerationRecord = meta["generation"]
            if t_result is None:
                t_result = TeacherScorer.ScoreResult(
                    request_id=batch_request_ids[i],
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    error="missing teacher batch result",
                )
            if s_result is None:
                s_result = StudentScorer.ScoreResult(
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    token_count=0,
                    error="missing student batch result",
                )

            record = build_exact_scored_record(
                base_record={
                    "run_id": run_id,
                    "sample_uid": sample_uid,
                    "source_index": int(prompt.get("source_index", pi)),
                    "split": "train",
                    "rollout_id": meta["rollout_id"],
                    "is_greedy": meta["is_greedy"],
                    "generation_seed": meta["generation_seed"],
                    "temperature": 0.0 if meta["is_greedy"] else config.temperature,
                    "top_p": 1.0 if meta["is_greedy"] else config.top_p,
                    "max_new_tokens": config.max_new_tokens,
                    "prompt_hash": prompt_hash,
                    "prompt_token_hash": generation.prompt_token_hash,
                    "prompt_version": config.response_format,
                    "image_hash": image_hash,
                    "gold_answer": gold_answer,
                    "teacher_model_id": teacher.model_id,
                    "student_model_path": config.student_model_path,
                    "tokenizer_hash": student_hash,
                    "student_tokenizer_hash": student_hash,
                    "teacher_tokenizer_hash": teacher_hash,
                },
                generation=generation,
                verdict=meta["verdict"],
                teacher_result=t_result,
                student_result=s_result,
            )
            if not math.isfinite(float(record["teacher_mean_logp"])) or not math.isfinite(
                float(record["student_mean_logp"])
            ):
                nf_container[0] += 1

            rollout_uid = f"{sample_uid}:greedy" if meta["is_greedy"] else f"{sample_uid}:rollout-{meta['rollout_id']}"
            for error in record["errors"]:
                errors.append(f"{rollout_uid}: {error}")
            all_rollouts.append(record)
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
        gate_config=config.gate_config,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
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
    print("  summary.json")

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


def _random_first_correct_mrr(K: int, correct_count: int) -> float:
    """Exact E[1/R] for the first correct rank in a random permutation."""

    if K <= 0 or correct_count <= 0 or correct_count > K:
        return float("nan")
    denominator = math.comb(K, correct_count)
    return float(sum(
        (math.comb(K - rank, correct_count - 1) / denominator) / rank
        for rank in range(1, K - correct_count + 2)
    ))


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if finite.size == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "count": 0,
            "bootstrap_seed": seed,
            "bootstrap_resamples": resamples,
        }
    if finite.size == 1 or resamples <= 0:
        mean = float(finite.mean())
        return {
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "count": int(finite.size),
            "bootstrap_seed": seed,
            "bootstrap_resamples": max(int(resamples), 0),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(int(resamples), int(finite.size)), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "count": int(finite.size),
        "bootstrap_seed": seed,
        "bootstrap_resamples": int(resamples),
    }


def _compute_prompt_signal_metrics(
    rollouts: Sequence[dict[str, Any]],
    *,
    allowed_uids: set[str] | None = None,
    seed: int = 42,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Macro, within-prompt teacher-gap discrimination and rank lift."""

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        if rollout.get("is_greedy"):
            continue
        uid = str(rollout.get("sample_uid", ""))
        if allowed_uids is not None and uid not in allowed_uids:
            continue
        gap = rollout.get("teacher_gap")
        correct = rollout.get("correct")
        if correct is None or gap is None or not math.isfinite(float(gap)):
            continue
        by_prompt.setdefault(uid, []).append(rollout)

    auc_values: list[float] = []
    rank1_values: list[float] = []
    random_rank1_values: list[float] = []
    rank1_lifts: list[float] = []
    mrr_values: list[float] = []
    random_mrr_values: list[float] = []
    mrr_lifts: list[float] = []
    prompt_rows: list[dict[str, Any]] = []

    for uid, prompt_rollouts in sorted(by_prompt.items()):
        correct_rows = [row for row in prompt_rollouts if row.get("correct") is True]
        wrong_rows = [row for row in prompt_rollouts if row.get("correct") is False]
        if not correct_rows or not wrong_rows:
            continue
        pair_credits = [
            1.0 if float(correct["teacher_gap"]) > float(wrong["teacher_gap"])
            else 0.5 if float(correct["teacher_gap"]) == float(wrong["teacher_gap"])
            else 0.0
            for correct in correct_rows
            for wrong in wrong_rows
        ]
        prompt_auc = float(np.mean(pair_credits))
        ranked = sorted(
            prompt_rollouts,
            key=lambda row: float(row["teacher_gap"]),
            reverse=True,
        )
        max_gap = float(ranked[0]["teacher_gap"])
        top_ties = [row for row in ranked if float(row["teacher_gap"]) == max_gap]
        rank1_credit = sum(row.get("correct") is True for row in top_ties) / len(top_ties)
        first_correct_rank = next(
            rank
            for rank, row in enumerate(ranked, start=1)
            if row.get("correct") is True
        )
        mrr = 1.0 / first_correct_rank
        K = len(prompt_rollouts)
        correct_count = len(correct_rows)
        random_rank1 = correct_count / K
        random_mrr = _random_first_correct_mrr(K, correct_count)

        auc_values.append(prompt_auc)
        rank1_values.append(rank1_credit)
        random_rank1_values.append(random_rank1)
        rank1_lifts.append(rank1_credit - random_rank1)
        mrr_values.append(mrr)
        random_mrr_values.append(random_mrr)
        mrr_lifts.append(mrr - random_mrr)
        prompt_rows.append({
            "sample_uid": uid,
            "K": K,
            "correct_count": correct_count,
            "within_prompt_auc": prompt_auc,
            "rank1_credit": rank1_credit,
            "random_rank1": random_rank1,
            "rank1_lift": rank1_credit - random_rank1,
            "mrr": mrr,
            "random_mrr": random_mrr,
            "mrr_lift": mrr - random_mrr,
        })

    return {
        "eligible_prompt_count": len(prompt_rows),
        "within_prompt_auc": _bootstrap_mean_ci(
            auc_values, seed=seed, resamples=resamples
        ),
        "rank1": _bootstrap_mean_ci(
            rank1_values, seed=seed + 1, resamples=resamples
        ),
        "random_rank1": float(np.mean(random_rank1_values)) if random_rank1_values else None,
        "rank1_lift": _bootstrap_mean_ci(
            rank1_lifts, seed=seed + 2, resamples=resamples
        ),
        "mrr": _bootstrap_mean_ci(
            mrr_values, seed=seed + 3, resamples=resamples
        ),
        "random_mrr": float(np.mean(random_mrr_values)) if random_mrr_values else None,
        "mrr_lift": _bootstrap_mean_ci(
            mrr_lifts, seed=seed + 4, resamples=resamples
        ),
        "per_prompt": prompt_rows,
    }


def _compute_correct_tail_ranks(
    prompt_summaries: list[dict[str, Any]],
    all_rollouts: list[dict[str, Any]],
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Compute corrected rank metrics specifically for correct-tail prompts."""
    correct_tail_uids = {
        p["sample_uid"]
        for p in prompt_summaries
        if p["support_state"] == "correct_tail"
    }
    metrics = _compute_prompt_signal_metrics(
        all_rollouts,
        allowed_uids=correct_tail_uids,
        seed=seed,
        resamples=resamples,
    )
    # Preserve the old scalar keys for downstream readers while adding the
    # corrected baselines and confidence intervals.
    metrics["num_prompts"] = metrics["eligible_prompt_count"]
    metrics["num_rollouts"] = sum(
        1
        for rollout in all_rollouts
        if rollout.get("sample_uid") in correct_tail_uids and not rollout.get("is_greedy")
    )
    metrics["rank1_mean"] = metrics["rank1"]["mean"]
    metrics["mrr_mean"] = metrics["mrr"]["mean"]
    return metrics


def _check_gates(
    summary: dict[str, Any],
    mode: str,
    *,
    gate_config: Mapping[str, Any] | None = None,
) -> list[tuple[str, bool, str]]:
    """Check protocol and signal gates (runbook §4.7)."""
    gates: list[tuple[str, bool, str]] = []
    configured = {
        "malformed_rate_max": 0.10,
        "duplicate_rate_max": 0.25,
        "min_correct_tails": 20,
        "teacher_gap_auc_min": 0.60,
        "correct_tail_rank1_above_random": True,
        "truncation_rate_max": 0.15,
        **dict(gate_config or {}),
    }

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
    malformed_max = float(configured["malformed_rate_max"])
    gates.append((
        "malformed_rate_ok",
        malformed_rate <= malformed_max,
        f"malformed rate = {malformed_rate:.3f} (threshold: {malformed_max:.3f})",
    ))

    dup_rate = summary.get("overall_duplicate_rollout_rate", 0.0)
    duplicate_max = float(configured["duplicate_rate_max"])
    gates.append((
        "duplicate_rate_ok",
        dup_rate <= duplicate_max,
        f"duplicate rate = {dup_rate:.3f} (threshold: {duplicate_max:.3f})",
    ))

    exact_alignment_rate = float(summary.get("exact_token_alignment_rate", 0.0))
    gates.append((
        "exact_token_identity",
        exact_alignment_rate == 1.0,
        f"exact generated/student/teacher token identity rate = {exact_alignment_rate:.3f}",
    ))

    prompt_hash_rate = float(summary.get("prompt_token_hash_rate", 0.0))
    gates.append((
        "prompt_token_hash_available",
        prompt_hash_rate == 1.0,
        f"prompt token hash availability rate = {prompt_hash_rate:.3f}",
    ))

    completeness_rate = float(summary.get("shard_completeness_rate", 0.0))
    gates.append((
        "complete_rollout_coverage",
        completeness_rate == 1.0,
        f"validated rollout/shard completeness rate = {completeness_rate:.3f}",
    ))

    truncation_rate = float(summary.get("truncation_rate", 1.0))
    truncation_max = float(configured["truncation_rate_max"])
    gates.append((
        "truncation_rate_ok",
        truncation_rate <= truncation_max,
        f"length-truncation rate = {truncation_rate:.3f} (threshold: {truncation_max:.3f})",
    ))

    # Signal gates (only for full mode)
    if mode == "full":
        correct_tail_count = summary.get("correct_tail_count", 0)
        min_correct_tails = int(configured["min_correct_tails"])
        gates.append((
            "sufficient_correct_tails",
            correct_tail_count >= min_correct_tails,
            f"{correct_tail_count} correct_tail prompts (need ≥ {min_correct_tails})",
        ))

        signal = summary.get("correct_tail_rank_metrics") or {}
        auc_info = signal.get("within_prompt_auc") or {}
        auc = auc_info.get("mean")
        auc_low = auc_info.get("ci95_low")
        auc_min = float(configured["teacher_gap_auc_min"])
        if auc is not None and auc_low is not None:
            auc_passed = float(auc) >= auc_min and float(auc_low) > 0.50
            gates.append((
                "teacher_gap_within_prompt_auc",
                auc_passed,
                f"macro AUC = {float(auc):.3f}, CI95 low = {float(auc_low):.3f} "
                f"(point threshold: {auc_min:.3f}; lower bound must exceed 0.50)",
            ))
        else:
            gates.append((
                "teacher_gap_within_prompt_auc",
                False,
                "within-prompt AUC/CI undefined",
            ))

        if bool(configured["correct_tail_rank1_above_random"]):
            rank_lift = signal.get("rank1_lift") or {}
            rank_lift_mean = rank_lift.get("mean")
            rank_lift_low = rank_lift.get("ci95_low")
            passed = rank_lift_low is not None and float(rank_lift_low) > 0.0
            gates.append((
                "correct_tail_rank1_above_random",
                passed,
                "rank@1 lift = "
                f"{rank_lift_mean if rank_lift_mean is not None else 'undefined'}, "
                f"CI95 low = {rank_lift_low if rank_lift_low is not None else 'undefined'}",
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
        response_format=_nested_get(
            yaml_cfg,
            "diagnostic",
            "response_format",
            default="legacy_answer",
        ),
        malformed_rate_max=_nested_get(
            yaml_cfg, "gates", "malformed_rate_max", default=0.10
        ),
        duplicate_rate_max=_nested_get(
            yaml_cfg, "gates", "duplicate_rate_max", default=0.25
        ),
        min_correct_tails=_nested_get(
            yaml_cfg, "gates", "min_correct_tails", default=20
        ),
        teacher_gap_auc_min=_nested_get(
            yaml_cfg, "gates", "teacher_gap_auc_min", default=0.60
        ),
        correct_tail_rank1_above_random=_nested_get(
            yaml_cfg,
            "gates",
            "correct_tail_rank1_above_random",
            default=True,
        ),
        truncation_rate_max=_nested_get(
            yaml_cfg, "gates", "truncation_rate_max", default=0.15
        ),
        bootstrap_seed=_nested_get(
            yaml_cfg, "diagnostic", "bootstrap_seed", default=42
        ),
        bootstrap_resamples=_nested_get(
            yaml_cfg, "diagnostic", "bootstrap_resamples", default=10_000
        ),
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
