"""Frozen-policy causal prefix intervention over verified teacher proposals.

This experiment asks which answer-free teacher prefix is sufficient to rescue
the frozen student's continuation distribution.  It is not training and does
not claim that a prefix has been internalized.  Fresh unaided continuations and
same-prompt wrong-student prefixes are mandatory controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .diagnostic import build_prompt, extract_image, generation_record_from_token_ids
from .verifier import verify_answer


SCHEMA_VERSION = "support-aware-prefix-intervention-v1"
DEFAULT_HORIZONS = (64, 128, 256, 512)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:\\boxed\s*\{|(?:^|\n)\s*(?:final\s+)?answer\s*:)",
    flags=re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _selection_key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


def _safe_uid(uid: str) -> str:
    prefix = "".join(character if character.isalnum() else "_" for character in uid)
    return f"{prefix[:80]}_{hashlib.sha256(uid.encode()).hexdigest()[:12]}"


def expected_group_utility(correct_count: int, K: int, group_size: int = 8) -> float:
    """Jeffreys-posterior E[1-p^G-(1-p)^G] in closed form."""

    if not 0 <= correct_count <= K or K <= 0 or group_size <= 0:
        raise ValueError("invalid correct_count, K, or group_size")
    alpha = correct_count + 0.5
    beta = K - correct_count + 0.5

    def beta_moment(left: float, right: float) -> float:
        value = 1.0
        for index in range(group_size):
            value *= (left + index) / (left + right + index)
        return value

    return float(1.0 - beta_moment(alpha, beta) - beta_moment(beta, alpha))


def posterior_lift_probability(
    treatment_correct: int,
    control_correct: int,
    K: int,
    *,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    """Return posterior mean pass-rate lift and P(p_treatment > p_control)."""

    rng = np.random.default_rng(seed)
    treatment = rng.beta(treatment_correct + 0.5, K - treatment_correct + 0.5, draws)
    control = rng.beta(control_correct + 0.5, K - control_correct + 0.5, draws)
    return float(np.mean(treatment - control)), float(np.mean(treatment > control))


def prefix_leakage_reason(prefix_text: str, gold_answer: Any) -> str | None:
    """Reject prefixes that expose a final-answer channel before continuation."""

    if FINAL_ANSWER_PATTERN.search(prefix_text):
        return "final_answer_marker"
    verdict = verify_answer(prefix_text, gold_answer)
    if verdict.get("answer_extracted") is not None or verdict.get("correct") is True:
        return "verifier_extractable_answer"
    return None


@dataclass(frozen=True)
class InterventionConfig:
    proposal_dir: str
    k32_run_dir: str
    cohort_dir: str
    output_dir: str
    student_model_path: str
    cohort_parquet_path: str | None = None
    prompt_manifest: str | None = None
    wrong_source_run_dir: str | None = None
    stage1_k: int = 4
    stage2_k: int = 8
    adaptive_confirm: bool = False
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    continuations_per_arm: int = 8
    max_continuation_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 20260806
    response_format: str = "legacy_answer"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    shard_index: int = 0
    num_shards: int = 1
    max_prompts: int | None = None
    posterior_draws: int = 20_000
    rescue_min_mean_lift: float = 0.20
    rescue_min_probability: float = 0.90

    def validate(self) -> None:
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if self.continuations_per_arm <= 0 or self.max_continuation_tokens <= 0:
            raise ValueError("continuation counts and limits must be positive")
        if self.stage1_k <= 0 or self.stage2_k <= 0:
            raise ValueError("stage continuation counts must be positive")
        if self.adaptive_confirm and self.stage2_k <= self.stage1_k:
            raise ValueError("adaptive confirmation requires stage2_k > stage1_k")
        if not 0 <= self.shard_index < self.num_shards or self.num_shards <= 0:
            raise ValueError("invalid shard assignment")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")
        if not 0 <= self.rescue_min_probability <= 1:
            raise ValueError("rescue probability threshold must lie in [0, 1]")


def load_config(path: str | Path, args: argparse.Namespace) -> InterventionConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    horizons = tuple(args.horizons or raw["intervention"]["horizons"])
    config = InterventionConfig(
        proposal_dir=str(args.proposal_dir or raw["data"]["proposal_dir"]),
        k32_run_dir=str(args.k32_run_dir or raw["data"]["k32_run_dir"]),
        cohort_dir=str(args.cohort_dir or raw["data"]["cohort_dir"]),
        output_dir=str(args.output_dir or raw["output"]["dir"]),
        student_model_path=str(raw["model"]["student"]),
        cohort_parquet_path=(
            None
            if (args.cohort_parquet_path or raw["data"].get("cohort_parquet_path")) in (None, "")
            else str(args.cohort_parquet_path or raw["data"]["cohort_parquet_path"])
        ),
        prompt_manifest=(
            None
            if (args.prompt_manifest or raw["data"].get("prompt_manifest")) in (None, "")
            else str(args.prompt_manifest or raw["data"]["prompt_manifest"])
        ),
        wrong_source_run_dir=(
            None
            if (args.wrong_source_run_dir or raw["data"].get("wrong_source_run_dir")) in (None, "")
            else str(args.wrong_source_run_dir or raw["data"]["wrong_source_run_dir"])
        ),
        stage1_k=int(args.stage1_k or raw["intervention"].get("stage1_k", 4)),
        stage2_k=int(args.stage2_k or raw["intervention"].get("stage2_k", 8)),
        adaptive_confirm=bool(
            args.adaptive_confirm
            if args.adaptive_confirm is not None
            else raw["intervention"].get("adaptive_confirm", False)
        ),
        horizons=tuple(int(value) for value in horizons),
        continuations_per_arm=int(
            args.continuations_per_arm or raw["generation"]["continuations_per_arm"]
        ),
        max_continuation_tokens=int(
            args.max_continuation_tokens or raw["generation"]["max_continuation_tokens"]
        ),
        temperature=float(raw["generation"].get("temperature", 0.7)),
        top_p=float(raw["generation"].get("top_p", 0.95)),
        seed=int(raw["generation"].get("seed", 20260806)),
        response_format=str(raw["generation"].get("response_format", "legacy_answer")),
        dtype=str(raw["hardware"].get("dtype", "bfloat16")),
        device=str(raw["hardware"].get("device", "cuda:0")),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        max_prompts=args.max_prompts,
        posterior_draws=int(raw["analysis"].get("posterior_draws", 20_000)),
        rescue_min_mean_lift=float(raw["analysis"].get("rescue_min_mean_lift", 0.20)),
        rescue_min_probability=float(raw["analysis"].get("rescue_min_probability", 0.90)),
    )
    config = InterventionConfig(**{
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in asdict(config).items()
    })
    config.validate()
    return config


def select_intervention_records(config: InterventionConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join top-1 retained proposal, cohort row, and wrong K32 controls."""

    proposal_dir = Path(config.proposal_dir).expanduser().resolve()
    k32_dir = Path(config.k32_run_dir).expanduser().resolve()
    cohort_dir = Path(config.cohort_dir).expanduser().resolve()
    cohort_path = (
        Path(config.cohort_parquet_path).expanduser().resolve()
        if config.cohort_parquet_path
        else cohort_dir / "cohort.parquet"
    )
    wrong_source_dir = (
        Path(config.wrong_source_run_dir).expanduser().resolve()
        if config.wrong_source_run_dir
        else k32_dir
    )
    required = (
        proposal_dir / "run_manifest.json",
        proposal_dir / "summary.json",
        proposal_dir / "retained_proposals.jsonl",
        wrong_source_dir / "rollouts.jsonl",
        cohort_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prefix intervention inputs: {missing}")
    proposal_summary = json.loads((proposal_dir / "summary.json").read_text(encoding="utf-8"))
    if proposal_summary.get("complete") is not True:
        raise ValueError("proposal cache is incomplete")
    if config.wrong_source_run_dir is None:
        validation = json.loads((k32_dir / "k32_validation.json").read_text(encoding="utf-8"))
        if validation.get("valid") is not True or int(validation.get("K") or 0) != 32:
            raise ValueError("prefix intervention requires a protocol-valid K=32 run")

    retained = _read_jsonl(proposal_dir / "retained_proposals.jsonl")
    proposal_by_uid: dict[str, dict[str, Any]] = {}
    for row in retained:
        uid = str(row.get("sample_uid") or "")
        if row.get("correct") is not True or not row.get("retained_for_fkl"):
            raise ValueError(f"invalid retained proposal for {uid!r}")
        current = proposal_by_uid.get(uid)
        if current is None or int(row.get("reachability_rank") or 10**9) < int(
            current.get("reachability_rank") or 10**9
        ):
            proposal_by_uid[uid] = row

    wrong_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(wrong_source_dir / "rollouts.jsonl"):
        if (
            not row.get("is_greedy")
            and row.get("correct") is False
            and row.get("exact_token_alignment") is True
            and row.get("response_token_ids")
        ):
            wrong_by_uid[str(row["sample_uid"])].append(row)
    for rows in wrong_by_uid.values():
        # Use the most student-probable wrong path as the strongest plausible
        # history-length control, not an arbitrary low-quality failure.
        rows.sort(key=lambda row: (
            -float(row["student_mean_logp"])
            if row.get("student_mean_logp") is not None
            and math.isfinite(float(row["student_mean_logp"]))
            else math.inf,
            int(row.get("rollout_id") or 0),
            row.get("response_token_hash", ""),
        ))

    import pandas as pd

    frame = pd.read_parquet(cohort_path).copy()
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort contains duplicate sample_uid")
    available_uids = sorted(
        set(proposal_by_uid).intersection(frame.index).intersection(wrong_by_uid),
        key=_selection_key,
    )
    if config.prompt_manifest:
        manifest_path_value = Path(config.prompt_manifest).expanduser().resolve()
        if not manifest_path_value.is_file():
            raise FileNotFoundError(f"missing prompt manifest: {manifest_path_value}")
        manifest_uids = [str(row.get("sample_uid") or "") for row in _read_jsonl(manifest_path_value)]
        if any(not uid for uid in manifest_uids) or len(set(manifest_uids)) != len(manifest_uids):
            raise ValueError("prompt manifest must contain unique non-empty sample_uids")
        unavailable_report: dict[str, str] = {}
        for uid in manifest_uids:
            if uid not in proposal_by_uid:
                unavailable_report[uid] = "teacher_trace_unavailable"
            elif uid not in frame.index:
                unavailable_report[uid] = "cohort_missing"
            elif uid not in wrong_by_uid:
                unavailable_report[uid] = "wrong_control_unavailable"
        available_uids = [uid for uid in manifest_uids if uid not in unavailable_report]
    start = len(available_uids) * config.shard_index // config.num_shards
    end = len(available_uids) * (config.shard_index + 1) // config.num_shards
    shard_uids = available_uids[start:end]
    if config.max_prompts is not None:
        shard_uids = shard_uids[: config.max_prompts]
    records = []
    for uid in shard_uids:
        cohort_row = frame.loc[uid].to_dict()
        cohort_row["sample_uid"] = uid
        records.append({
            "sample_uid": uid,
            "cohort": cohort_row,
            "teacher_proposal": proposal_by_uid[uid],
            "wrong_rollouts": wrong_by_uid[uid],
        })
    provenance = {
        "proposal_manifest_sha256": _sha256_file(proposal_dir / "run_manifest.json"),
        "proposal_summary_sha256": _sha256_file(proposal_dir / "summary.json"),
        "retained_proposals_sha256": _sha256_file(proposal_dir / "retained_proposals.jsonl"),
        "wrong_source_rollouts_sha256": _sha256_file(wrong_source_dir / "rollouts.jsonl"),
        "k32_validation_sha256": (
            _sha256_file(k32_dir / "k32_validation.json") if config.wrong_source_run_dir is None else None
        ),
        "cohort_sha256": _sha256_file(cohort_path),
        "prompt_manifest_sha256": (
            _sha256_file(Path(config.prompt_manifest)) if config.prompt_manifest else None
        ),
        "all_selected_uids": available_uids,
        "expected_uids": shard_uids,
        "shard_start": start,
        "shard_end": end,
        "selection_mode": "manifest" if config.prompt_manifest else "k32_universe",
        "wrong_source": str(wrong_source_dir),
        "manifest_unavailable_uids": (
            unavailable_report if config.prompt_manifest else None
        ),
    }
    return records, provenance


def _load_student(config: InterventionConfig):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(config.student_model_path)
    torch_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        config.student_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=Path(config.student_model_path).exists(),
    ).to(config.device)
    model.eval()
    return model, processor, tokenizer_fingerprint(processor.tokenizer)


def _extend_prompt_inputs(inputs: Mapping[str, Any], prefix_ids: Sequence[int], device: str):
    import torch

    encoded = {key: value.to(device) for key, value in inputs.items()}
    if not prefix_ids:
        return encoded
    batch_size = int(encoded["input_ids"].shape[0])
    if batch_size != 1:
        raise ValueError("prefix generation expects batch size one")
    prefix = torch.tensor([list(prefix_ids)], dtype=encoded["input_ids"].dtype, device=device)
    encoded["input_ids"] = torch.cat([encoded["input_ids"], prefix], dim=1)
    attention_extension = torch.ones(
        (1, len(prefix_ids)), dtype=encoded["attention_mask"].dtype, device=device
    )
    encoded["attention_mask"] = torch.cat([encoded["attention_mask"], attention_extension], dim=1)
    encoded.pop("position_ids", None)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in encoded:
            extension = torch.zeros(
                (1, len(prefix_ids)), dtype=encoded[key].dtype, device=device
            )
            encoded[key] = torch.cat([encoded[key], extension], dim=-1)
    return encoded


def generate_continuation(
    model,
    processor,
    *,
    image,
    prompt_text: str,
    prefix_ids: Sequence[int],
    max_continuation_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Generate after an exact response prefix and return auditable raw IDs."""

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[chat], images=[image], return_tensors="pt")
    inputs = _extend_prompt_inputs(prompt_inputs, prefix_ids, device)
    input_width = int(inputs["input_ids"].shape[1])
    model.generation_config.do_sample = temperature > 0
    model.generation_config.max_new_tokens = max_continuation_tokens
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    model.generation_config.temperature = temperature if temperature > 0 else None
    model.generation_config.top_p = top_p if temperature > 0 else None
    with torch.no_grad():
        outputs = model.generate(**inputs)
    continuation_ids = tuple(int(value) for value in outputs[0, input_width:].tolist())
    composite_ids = tuple(int(value) for value in prefix_ids) + continuation_ids
    prompt_ids = tuple(int(value) for value in prompt_inputs["input_ids"][0].tolist())
    generation = generation_record_from_token_ids(
        model=model,
        tokenizer=processor.tokenizer,
        response_token_ids=composite_ids,
        prompt_token_ids=prompt_ids,
        max_new_tokens=len(prefix_ids) + max_continuation_tokens,
    )
    return {
        "generation": generation,
        "continuation_token_ids": continuation_ids,
        "continuation_token_hash": hash_token_ids(continuation_ids),
    }


def _candidate_prefix(
    token_ids: Sequence[int],
    *,
    horizon: int,
    tokenizer,
    gold_answer: Any,
) -> tuple[tuple[int, ...] | None, str | None, str]:
    if len(token_ids) < horizon:
        return None, "source_shorter_than_horizon", ""
    prefix = tuple(int(value) for value in token_ids[:horizon])
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in prefix:
        return None, "terminal_token_in_prefix", ""
    text = tokenizer.decode(
        prefix,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    leakage = prefix_leakage_reason(text, gold_answer)
    return (None, leakage, text) if leakage else (prefix, None, text)


def _choose_wrong_prefix(
    wrong_rollouts: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    tokenizer,
    gold_answer: Any,
) -> tuple[tuple[int, ...] | None, str | None, str, Mapping[str, Any] | None]:
    last_reason = "no_wrong_source_long_enough"
    for row in wrong_rollouts:
        prefix, reason, text = _candidate_prefix(
            row.get("response_token_ids") or (),
            horizon=horizon,
            tokenizer=tokenizer,
            gold_answer=gold_answer,
        )
        if prefix is not None:
            return prefix, None, text, row
        if reason:
            last_reason = reason
    return None, last_reason, "", None


def _rescue_decision(
    uid: str,
    horizon: int,
    teacher: Mapping[str, Any],
    wrong: Mapping[str, Any],
    baseline: Mapping[str, Any],
    config: InterventionConfig,
) -> dict[str, Any]:
    """Preregistered rescue rule for one (prompt, horizon) with per-unit K."""

    K = int(teacher.get("K") or config.continuations_per_arm)
    if int(wrong.get("K") or config.continuations_per_arm) != K or int(
        baseline.get("K") or config.continuations_per_arm
    ) != K:
        raise ValueError(f"{uid}:{horizon}: rescue arms disagree on continuation K")
    teacher_wrong_lift, teacher_wrong_probability = posterior_lift_probability(
        int(teacher["correct_count"]),
        int(wrong["correct_count"]),
        K,
        seed=config.seed + int(_selection_key(uid)[:8], 16) + horizon,
        draws=config.posterior_draws,
    )
    teacher_unaided_lift, teacher_unaided_probability = posterior_lift_probability(
        int(teacher["correct_count"]),
        int(baseline["correct_count"]),
        K,
        seed=config.seed + int(_selection_key(uid)[8:16], 16) + horizon,
        draws=config.posterior_draws,
    )
    rescued = (
        teacher_wrong_lift >= config.rescue_min_mean_lift
        and teacher_unaided_lift >= config.rescue_min_mean_lift
        and teacher_wrong_probability >= config.rescue_min_probability
        and teacher_unaided_probability >= config.rescue_min_probability
    )
    return {
        "sample_uid": uid,
        "horizon": horizon,
        "teacher_correct_count": teacher["correct_count"],
        "wrong_correct_count": wrong["correct_count"],
        "unaided_correct_count": baseline["correct_count"],
        "teacher_minus_wrong_posterior_mean": teacher_wrong_lift,
        "teacher_gt_wrong_probability": teacher_wrong_probability,
        "teacher_minus_unaided_posterior_mean": teacher_unaided_lift,
        "teacher_gt_unaided_probability": teacher_unaided_probability,
        "rescue_K": K,
        "rescue_stage": (
            "stage2"
            if config.adaptive_confirm and K == config.stage2_k
            else "stage1"
        ),
        "meets_preregistered_rescue_rule": rescued,
    }


def _aggregate(output_dir: Path, expected_uids: Sequence[str], config: InterventionConfig) -> dict[str, Any]:
    prompt_results = []
    result_dir = output_dir / "prompt_results"
    for path in sorted(result_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if str(row.get("sample_uid")) in expected_uids:
            prompt_results.append(row)
    by_uid = {str(row["sample_uid"]): row for row in prompt_results}
    if len(by_uid) != len(prompt_results):
        raise ValueError("duplicate prompt results in prefix-intervention output")
    completed = [uid for uid in expected_uids if uid in by_uid]
    rollouts = [item for uid in completed for item in by_uid[uid]["rollouts"]]
    units = [item for uid in completed for item in by_uid[uid]["intervention_units"]]

    unit_by_key = {
        (str(row["sample_uid"]), str(row["arm"]), int(row["horizon"])): row
        for row in units if not row.get("skipped_reason")
    }
    rescue_rows = []
    minimal_rescue = []
    for uid in completed:
        baseline = unit_by_key.get((uid, "unaided", 0))
        if baseline is None:
            continue
        first_rescue = None
        for horizon in config.horizons:
            teacher = unit_by_key.get((uid, "teacher_prefix", horizon))
            wrong = unit_by_key.get((uid, "wrong_student_prefix", horizon))
            if teacher is None or wrong is None:
                continue
            rescue = _rescue_decision(uid, horizon, teacher, wrong, baseline, config)
            rescue_rows.append(rescue)
            if rescue["meets_preregistered_rescue_rule"] and first_rescue is None:
                first_rescue = rescue
        if first_rescue is not None:
            minimal_rescue.append(first_rescue)

    def arm_summary(arm: str) -> dict[str, Any]:
        selected = [row for row in units if row.get("arm") == arm]
        valid = [row for row in selected if not row.get("skipped_reason")]
        return {
            "unit_count": len(selected),
            "valid_unit_count": len(valid),
            "skipped_unit_count": len(selected) - len(valid),
            "continuation_count": sum(int(row.get("K") or 0) for row in valid),
            "correct_count": sum(int(row.get("correct_count") or 0) for row in valid),
            "mean_pass_rate": float(np.mean([row["pass_rate"] for row in valid])) if valid else None,
            "mean_expected_U8": float(np.mean([row["expected_U8"] for row in valid])) if valid else None,
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_prompt_count": len(expected_uids),
        "completed_prompt_count": len(completed),
        "complete": completed == list(expected_uids),
        "rollout_count": len(rollouts),
        "intervention_unit_count": len(units),
        "adaptive_confirm": bool(config.adaptive_confirm),
        "stage1_unit_count": sum(1 for row in units if int(row.get("K") or 0) == config.stage1_k),
        "stage2_unit_count": sum(1 for row in units if int(row.get("K") or 0) == config.stage2_k),
        "arm_summary": {
            arm: arm_summary(arm)
            for arm in ("unaided", "teacher_prefix", "wrong_student_prefix")
        },
        "rescue_comparison_count": len(rescue_rows),
        "prompt_with_minimal_rescue_count": len(minimal_rescue),
        "completed_uids": completed,
        "missing_uids": [uid for uid in expected_uids if uid not in by_uid],
    }
    _write_jsonl(output_dir / "continuation_rollouts.jsonl", rollouts)
    _write_jsonl(output_dir / "intervention_units.jsonl", units)
    _write_jsonl(output_dir / "rescue_comparisons.jsonl", rescue_rows)
    _write_jsonl(output_dir / "minimal_rescue_prefixes.jsonl", minimal_rescue)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run(config: InterventionConfig) -> dict[str, Any]:
    import torch
    import transformers

    config.validate()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "prompt_results"
    result_dir.mkdir(exist_ok=True)
    records, provenance = select_intervention_records(config)
    expected_uids = list(provenance["expected_uids"])
    git_commit, git_dirty = _git_state()
    json_config = json.loads(json.dumps(asdict(config)))
    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config": json_config,
        "provenance": provenance,
        "started_at_unix": time.time(),
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config") != json_config or existing.get("provenance") != provenance:
            raise ValueError("existing manifest differs from requested config/provenance")
        if existing.get("git_commit") != git_commit or bool(existing.get("git_dirty")) != git_dirty:
            raise ValueError("resume requires the identical git commit and dirty state")
        manifest = existing
        manifest["status"] = "running"
    _write_json(manifest_path, manifest)
    completed = {
        str(json.loads(path.read_text(encoding="utf-8"))["sample_uid"])
        for path in result_dir.glob("*.json")
    }
    pending = [record for record in records if record["sample_uid"] not in completed]
    if not pending:
        summary = _aggregate(output, expected_uids, config)
        manifest["status"] = "completed" if summary["complete"] else "partial"
        manifest["summary"] = summary
        _write_json(manifest_path, manifest)
        return summary

    model, processor, tokenizer_hash = _load_student(config)
    proposal_manifest = json.loads(
        (Path(config.proposal_dir).expanduser().resolve() / "run_manifest.json").read_text(encoding="utf-8")
    )
    proposal_student = str(proposal_manifest.get("config", {}).get("student_model_path") or "")
    if proposal_student and Path(proposal_student).expanduser().resolve() != Path(
        config.student_model_path
    ).expanduser().resolve():
        raise ValueError("proposal cache and intervention use different student model paths")
    manifest["student_tokenizer_hash"] = tokenizer_hash
    manifest["runtime"].update({
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device_count": int(torch.cuda.device_count()),
    })
    _write_json(manifest_path, manifest)

    for prompt_index, record in enumerate(pending, start=1):
        uid = str(record["sample_uid"])
        cohort = record["cohort"]
        question = str(cohort.get("question") or "").strip()
        gold_answer = cohort.get("answer")
        if not question:
            raise ValueError(f"{uid}: missing question")
        image = extract_image(cohort).convert("RGB")
        prompt_text = build_prompt(question, config.response_format)
        teacher_ids = tuple(int(value) for value in record["teacher_proposal"]["response_token_ids"])
        intervention_units: list[dict[str, Any]] = []
        rollout_rows: list[dict[str, Any]] = []

        def _arm_specs_for(build_horizons: Sequence[int]) -> list[tuple[str, int, tuple[int, ...] | None, str | None, str, Mapping[str, Any] | None]]:
            specs: list[tuple[str, int, tuple[int, ...] | None, str | None, str, Mapping[str, Any] | None]] = [
                ("unaided", 0, (), None, "", None)
            ]
            for horizon in build_horizons:
                teacher_prefix, teacher_reason, teacher_text = _candidate_prefix(
                    teacher_ids,
                    horizon=horizon,
                    tokenizer=processor.tokenizer,
                    gold_answer=gold_answer,
                )
                specs.append((
                    "teacher_prefix", horizon, teacher_prefix, teacher_reason, teacher_text,
                    record["teacher_proposal"],
                ))
                wrong_prefix, wrong_reason, wrong_text, wrong_source = _choose_wrong_prefix(
                    record["wrong_rollouts"],
                    horizon=horizon,
                    tokenizer=processor.tokenizer,
                    gold_answer=gold_answer,
                )
                specs.append((
                    "wrong_student_prefix", horizon, wrong_prefix, wrong_reason, wrong_text, wrong_source,
                ))
            return specs

        def _execute_arms(
            arm_specs: Sequence[tuple[str, int, tuple[int, ...] | None, str | None, str, Mapping[str, Any] | None]],
            K: int,
        ) -> list[dict[str, Any]]:
            units: list[dict[str, Any]] = []
            for arm_index, (
                arm,
                horizon,
                prefix_ids,
                skipped_reason,
                prefix_text,
                source,
            ) in enumerate(arm_specs):
                unit_rollouts: list[dict[str, Any]] = []
                if skipped_reason is None and prefix_ids is not None:
                    for continuation_id in range(1, K + 1):
                        generation_seed = (
                            config.seed
                            + int(_selection_key(uid)[:8], 16)
                            + arm_index * 100_000
                            + continuation_id
                        )
                        generated = generate_continuation(
                            model,
                            processor,
                            image=image,
                            prompt_text=prompt_text,
                            prefix_ids=prefix_ids,
                            max_continuation_tokens=config.max_continuation_tokens,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            seed=generation_seed,
                            device=config.device,
                        )
                        generation = generated["generation"]
                        verdict = verify_answer(generation.response_text_display, gold_answer)
                        rollout = {
                            "schema_version": SCHEMA_VERSION,
                            "sample_uid": uid,
                            "observed_stratum": record["teacher_proposal"].get("observed_stratum"),
                            "arm": arm,
                            "horizon": horizon,
                            "continuation_id": continuation_id,
                            "generation_seed": generation_seed,
                            "prefix_token_ids": list(prefix_ids),
                            "prefix_token_hash": hash_token_ids(prefix_ids),
                            "prefix_text_display": prefix_text,
                            "source_response_token_hash": source.get("response_token_hash") if source else None,
                            "continuation_token_ids": list(generated["continuation_token_ids"]),
                            "continuation_token_hash": generated["continuation_token_hash"],
                            "response_token_ids": list(generation.response_token_ids_raw),
                            "response_token_hash": generation.response_token_hash,
                            "response_text_display": generation.response_text_display,
                            "response_text_raw": generation.response_text_raw,
                            "finish_reason": generation.finish_reason,
                            "terminal_token_id": generation.terminal_token_id,
                            "correct": verdict.get("correct"),
                            "malformed": bool(verdict.get("malformed")),
                            "answer_extracted": verdict.get("answer_extracted"),
                            "gold_answer": verdict.get("gold_answer"),
                        }
                        unit_rollouts.append(rollout)
                        rollout_rows.append(rollout)
                correct_count = sum(row.get("correct") is True for row in unit_rollouts)
                K_actual = len(unit_rollouts)
                unit = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_uid": uid,
                    "observed_stratum": record["teacher_proposal"].get("observed_stratum"),
                    "arm": arm,
                    "horizon": horizon,
                    "prefix_token_hash": hash_token_ids(prefix_ids or ()),
                    "source_response_token_hash": source.get("response_token_hash") if source else None,
                    "skipped_reason": skipped_reason,
                    "K": K_actual,
                    "correct_count": correct_count,
                    "pass_rate": correct_count / K_actual if K_actual else None,
                    "expected_U8": expected_group_utility(correct_count, K_actual) if K_actual else None,
                    "malformed_count": sum(bool(row.get("malformed")) for row in unit_rollouts),
                    "truncated_count": sum(row.get("finish_reason") == "length" for row in unit_rollouts),
                    "unique_continuation_count": len({row["continuation_token_hash"] for row in unit_rollouts}),
                }
                units.append(unit)
            return units

        intervention_units = _execute_arms(_arm_specs_for(config.horizons), config.stage1_k)
        if config.adaptive_confirm:
            baseline_unit = next(
                unit for unit in intervention_units if unit["arm"] == "unaided" and unit["horizon"] == 0
            )
            candidate = None
            for horizon in config.horizons:
                teacher_unit = next(
                    (
                        unit
                        for unit in intervention_units
                        if unit["arm"] == "teacher_prefix" and unit["horizon"] == horizon
                    ),
                    None,
                )
                wrong_unit = next(
                    (
                        unit
                        for unit in intervention_units
                        if unit["arm"] == "wrong_student_prefix" and unit["horizon"] == horizon
                    ),
                    None,
                )
                if teacher_unit is None or wrong_unit is None:
                    continue
                decision = _rescue_decision(uid, horizon, teacher_unit, wrong_unit, baseline_unit, config)
                if decision["meets_preregistered_rescue_rule"]:
                    candidate = horizon
                    break
            if candidate is not None:
                intervention_units.extend(
                    _execute_arms(_arm_specs_for([candidate]), config.stage2_k)
                )

        prompt_result = {
            "schema_version": SCHEMA_VERSION,
            "sample_uid": uid,
            "observed_stratum": record["teacher_proposal"].get("observed_stratum"),
            "teacher_proposal_token_hash": record["teacher_proposal"].get("response_token_hash"),
            "student_tokenizer_hash": tokenizer_hash,
            "intervention_units": intervention_units,
            "rollouts": rollout_rows,
        }
        target = result_dir / f"{_safe_uid(uid)}.json"
        temporary = target.with_suffix(".json.tmp")
        _write_json(temporary, prompt_result)
        temporary.replace(target)
        image.close()
        _aggregate(output, expected_uids, config)
        valid_units = sum(not unit.get("skipped_reason") for unit in intervention_units)
        print(
            f"[{prompt_index}/{len(pending)}] {uid}: "
            f"valid_units={valid_units}, rollouts={len(rollout_rows)}",
            flush=True,
        )

    summary = _aggregate(output, expected_uids, config)
    manifest["status"] = "completed" if summary["complete"] else "partial"
    manifest["finished_at_unix"] = time.time()
    manifest["summary"] = summary
    _write_json(manifest_path, manifest)
    return summary


def merge_shards(shard_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard is required")
    shards = [Path(path).expanduser().resolve() for path in shard_dirs]
    manifests = [json.loads((path / "run_manifest.json").read_text(encoding="utf-8")) for path in shards]
    if any(manifest.get("status") != "completed" for manifest in manifests):
        raise ValueError("all prefix-intervention shards must be completed")
    commits = {str(manifest.get("git_commit") or "") for manifest in manifests}
    if len(commits) != 1 or "" in commits:
        raise ValueError(f"shard commits differ: {sorted(commits)}")
    allow_dirty = os.environ.get("DTOPD_ALLOW_DIRTY_MERGE") == "1"
    if {bool(manifest.get("git_dirty")) for manifest in manifests} != {False} and not allow_dirty:
        raise ValueError("all shards must use clean worktrees")
    full_uid_sets = {tuple(manifest["provenance"]["all_selected_uids"]) for manifest in manifests}
    if len(full_uid_sets) != 1:
        raise ValueError("shards disagree on selected UID universe")
    expected_indices = set(range(int(manifests[0]["config"]["num_shards"])))
    actual_indices = {int(manifest["config"]["shard_index"]) for manifest in manifests}
    if actual_indices != expected_indices:
        raise ValueError(
            f"shard index coverage differs: {sorted(actual_indices)} "
            f"vs {sorted(expected_indices)}"
        )
    normalized = []
    seen: set[str] = set()
    expected_uids: list[str] = []
    for manifest in sorted(manifests, key=lambda item: int(item["config"]["shard_index"])):
        config = dict(manifest["config"])
        config.pop("output_dir", None)
        config.pop("shard_index", None)
        normalized.append(config)
        shard_uids = list(manifest["provenance"]["expected_uids"])
        overlap = seen.intersection(shard_uids)
        if overlap:
            raise ValueError(f"shards overlap: {sorted(overlap)[:5]}")
        seen.update(shard_uids)
        expected_uids.extend(shard_uids)
    if any(config != normalized[0] for config in normalized[1:]):
        raise ValueError("shard configs differ")
    if seen != set(next(iter(full_uid_sets))):
        raise ValueError("shards do not cover selected UID universe")

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        result_dir = temporary / "prompt_results"
        result_dir.mkdir()
        for shard in shards:
            for source in (shard / "prompt_results").glob("*.json"):
                target = result_dir / source.name
                if target.exists():
                    raise ValueError(f"duplicate prompt artifact: {source.name}")
                target.write_bytes(source.read_bytes())
        merged_config = dict(manifests[0]["config"])
        merged_config["output_dir"] = str(output)
        merged_config["shard_index"] = 0
        merged_config["num_shards"] = 1
        summary = _aggregate(temporary, expected_uids, InterventionConfig(**merged_config))
        if not summary["complete"]:
            raise ValueError("merged prefix intervention is incomplete")
        _write_json(temporary / "run_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "git_commit": next(iter(commits)),
            "git_dirty": False,
            "merged_from": [str(path) for path in shards],
            "config": merged_config,
            "provenance": {
                "expected_uids": expected_uids,
                "source_manifest_sha256": [_sha256_file(path / "run_manifest.json") for path in shards],
            },
            "summary": summary,
        })
        temporary.replace(output)
        return summary
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--proposal-dir")
    run_parser.add_argument("--k32-run-dir")
    run_parser.add_argument("--cohort-dir")
    run_parser.add_argument("--cohort-parquet-path")
    run_parser.add_argument("--prompt-manifest")
    run_parser.add_argument("--wrong-source-run-dir")
    run_parser.add_argument("--stage1-k", type=int)
    run_parser.add_argument("--stage2-k", type=int)
    run_parser.add_argument("--adaptive-confirm", action="store_true", default=None)
    run_parser.add_argument("--no-adaptive-confirm", action="store_false", dest="adaptive_confirm", default=None)
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--horizons", nargs="+", type=int)
    run_parser.add_argument("--continuations-per-arm", type=int)
    run_parser.add_argument("--max-continuation-tokens", type=int)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--max-prompts", type=int)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--shard-dirs", nargs="+", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            run(load_config(args.config, args))
            if args.command == "run"
            else merge_shards(args.shard_dirs, args.output_dir)
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
