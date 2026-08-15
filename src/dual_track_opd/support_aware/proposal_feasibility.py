"""Verified teacher-proposal feasibility for support-confirmed prompts.

This is the first executable step after the frozen-policy K=32 diagnostic.  It
does not update the student.  Instead, it asks whether a stronger teacher can
produce verifier-passing trajectories for prompts where the student has zero
or rare observed support, and whether those trajectories are close enough to
the frozen student to serve as a TREK-like forward-KL bridge dataset.

The module deliberately preserves teacher-generated raw token IDs.  Student
reachability is computed by teacher-forcing those exact IDs under the frozen
student; display text is used only by the verifier and is never re-tokenized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .diagnostic import build_prompt, extract_image, generate_response
from .k32_cohort import observed_stratum
from .scorer import StudentScorer, StudentScorerConfig
from .verifier import verify_answer


SCHEMA_VERSION = "support-aware-proposal-feasibility-v1"
DEFAULT_STATES = ("no_correct_observed", "rare_success", "mixed_support")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _selection_key(sample_uid: str) -> str:
    return hashlib.sha256(sample_uid.encode()).hexdigest()


def _safe_uid(sample_uid: str) -> str:
    prefix = "".join(character if character.isalnum() else "_" for character in sample_uid)
    return f"{prefix[:80]}_{hashlib.sha256(sample_uid.encode()).hexdigest()[:12]}"


def _model_identity(model_path: str) -> dict[str, Any]:
    resolved = Path(model_path).expanduser().resolve()
    config_path = resolved / "config.json"
    return {
        "path": str(resolved),
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else None,
    }


def _quantile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    position = probability * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def trimmed_length_normalized_nll(
    sampled_token_log_probs: Sequence[float],
    *,
    content_mask: Sequence[bool] | None = None,
    low_trim_fraction: float = 0.10,
    high_trim_fraction: float = 0.02,
) -> dict[str, Any]:
    """Compute deterministic two-sided trimmed mean student NLL.

    The lowest-loss tokens are trimmed to reduce boilerplate/format dominance;
    the highest-loss tokens are trimmed to reduce isolated rare-token outliers.
    Fractions are converted to floor counts so at least one token is retained.
    """

    if not 0 <= low_trim_fraction < 1 or not 0 <= high_trim_fraction < 1:
        raise ValueError("trim fractions must lie in [0, 1)")
    if low_trim_fraction + high_trim_fraction >= 1:
        raise ValueError("trim fractions must sum to less than one")
    if content_mask is None:
        content_mask = (True,) * len(sampled_token_log_probs)
    if len(content_mask) != len(sampled_token_log_probs):
        raise ValueError("content mask and log probabilities must have equal length")

    losses = sorted(
        -float(logp)
        for logp, keep in zip(sampled_token_log_probs, content_mask, strict=True)
        if keep and math.isfinite(float(logp))
    )
    if not losses:
        raise ValueError("no finite content-token log probabilities")
    low_count = math.floor(len(losses) * low_trim_fraction)
    high_count = math.floor(len(losses) * high_trim_fraction)
    if low_count + high_count >= len(losses):
        raise ValueError("trim configuration removes every token")
    stop = len(losses) - high_count if high_count else len(losses)
    retained = losses[low_count:stop]
    return {
        "trimmed_length_normalized_nll": float(sum(retained) / len(retained)),
        "content_token_count": len(losses),
        "retained_token_count": len(retained),
        "low_trim_count": low_count,
        "high_trim_count": high_count,
        "low_trim_fraction": float(low_trim_fraction),
        "high_trim_fraction": float(high_trim_fraction),
    }


def mark_reachable_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    retain_count: int,
) -> list[dict[str, Any]]:
    """Mark the lowest-NLL verifier-passing proposals as retained."""

    if retain_count <= 0:
        raise ValueError("retain_count must be positive")
    rows = [dict(row) for row in proposals]
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("correct") is True
            and math.isfinite(float(row.get("trimmed_length_normalized_nll", float("nan"))))
        ),
        key=lambda row: (
            float(row["trimmed_length_normalized_nll"]),
            int(row.get("proposal_id", 0)),
        ),
    )
    retained_ids = {int(row["proposal_id"]) for row in eligible[:retain_count]}
    for row in rows:
        row["retained_for_fkl"] = int(row.get("proposal_id", -1)) in retained_ids
        row["reachability_rank"] = next(
            (
                rank
                for rank, candidate in enumerate(eligible, start=1)
                if int(candidate["proposal_id"]) == int(row.get("proposal_id", -1))
            ),
            None,
        )
    return rows


@dataclass(frozen=True)
class ProposalConfig:
    k32_run_dir: str
    cohort_dir: str
    output_dir: str
    student_model_path: str
    teacher_model_path: str
    cohort_parquet_path: str | None = None
    pool256_run_dir: str | None = None
    prompt_manifest: str | None = None
    states: tuple[str, ...] = DEFAULT_STATES
    proposals_per_prompt: int = 4
    retain_count: int = 2
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 4096
    seed: int = 20260805
    response_format: str = "legacy_answer"
    low_trim_fraction: float = 0.10
    high_trim_fraction: float = 0.02
    dtype: str = "bfloat16"
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"
    shard_index: int = 0
    num_shards: int = 1
    max_prompts: int | None = None

    def validate(self) -> None:
        if not self.states or any(state not in DEFAULT_STATES for state in self.states):
            raise ValueError(f"states must be a subset of {DEFAULT_STATES}")
        if self.proposals_per_prompt <= 0 or self.retain_count <= 0:
            raise ValueError("proposal and retention counts must be positive")
        if self.retain_count > self.proposals_per_prompt:
            raise ValueError("retain_count cannot exceed proposals_per_prompt")
        if self.max_new_tokens <= 0 or self.num_shards <= 0:
            raise ValueError("max_new_tokens and num_shards must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must lie in [0, num_shards)")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive when provided")


def load_config(path: str | Path, overrides: argparse.Namespace) -> ProposalConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    override = (
        lambda name, default=None: getattr(overrides, name, None)
        if getattr(overrides, name, None) is not None
        else default
    )
    config = ProposalConfig(
        k32_run_dir=str(overrides.k32_run_dir or raw["data"]["k32_run_dir"]),
        cohort_dir=str(overrides.cohort_dir or raw["data"]["cohort_dir"]),
        output_dir=str(overrides.output_dir or raw["output"]["dir"]),
        student_model_path=str(raw["models"]["student"]),
        teacher_model_path=str(raw["models"]["teacher"]),
        cohort_parquet_path=(
            None
            if override("cohort_parquet_path", raw.get("data", {}).get("cohort_parquet_path"))
            in (None, "")
            else str(
                override("cohort_parquet_path", raw.get("data", {}).get("cohort_parquet_path"))
            )
        ),
        pool256_run_dir=(
            None
            if override("pool256_run_dir", raw.get("data", {}).get("pool256_run_dir")) in (None, "")
            else str(override("pool256_run_dir", raw.get("data", {}).get("pool256_run_dir")))
        ),
        prompt_manifest=(
            None
            if override("prompt_manifest", raw.get("data", {}).get("prompt_manifest")) in (None, "")
            else str(override("prompt_manifest", raw.get("data", {}).get("prompt_manifest")))
        ),
        states=tuple(raw["selection"].get("states", DEFAULT_STATES)),
        proposals_per_prompt=int(
            overrides.proposals_per_prompt or raw["generation"]["proposals_per_prompt"]
        ),
        retain_count=int(raw["selection"].get("retain_count", 2)),
        temperature=float(raw["generation"].get("temperature", 0.7)),
        top_p=float(raw["generation"].get("top_p", 0.95)),
        max_new_tokens=int(overrides.max_new_tokens or raw["generation"]["max_new_tokens"]),
        seed=int(raw["generation"].get("seed", 20260805)),
        response_format=str(raw["generation"].get("response_format", "legacy_answer")),
        low_trim_fraction=float(raw["reachability"].get("low_trim_fraction", 0.10)),
        high_trim_fraction=float(raw["reachability"].get("high_trim_fraction", 0.02)),
        dtype=str(raw["hardware"].get("dtype", "bfloat16")),
        teacher_device=str(raw["hardware"].get("teacher_device", "cuda:0")),
        student_device=str(raw["hardware"].get("student_device", "cuda:1")),
        shard_index=int(overrides.shard_index),
        num_shards=int(overrides.num_shards),
        max_prompts=overrides.max_prompts,
    )
    config = ProposalConfig(
        **{
            key: os.path.expandvars(value) if isinstance(value, str) else value
            for key, value in asdict(config).items()
        }
    )
    config.validate()
    return config


def select_prompt_records(config: ProposalConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    k32_run = Path(config.k32_run_dir).expanduser().resolve()
    cohort_dir = Path(config.cohort_dir).expanduser().resolve()
    cohort_path = (
        Path(config.cohort_parquet_path).expanduser().resolve()
        if config.cohort_parquet_path
        else cohort_dir / "cohort.parquet"
    )
    pool256_mode = config.pool256_run_dir is not None
    if pool256_mode:
        pool256_root = Path(config.pool256_run_dir).expanduser().resolve()
        summary_path = pool256_root / "frontier_analysis" / "frontier_prompts.jsonl"
    else:
        summary_path = k32_run / "prompt_support_summary.jsonl"
    required = (summary_path, cohort_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing proposal inputs: {missing}")
    validation_path = k32_run / "k32_validation.json"
    if not pool256_mode:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("valid") is not True or int(validation.get("K") or 0) != 32:
            raise ValueError("proposal selection requires a protocol-valid K=32 run")

    summaries = _read_jsonl(summary_path)
    summary_by_uid: dict[str, dict[str, Any]] = {}
    for row in summaries:
        uid = str(row.get("sample_uid") or "")
        K = int(row.get("K") or 0)
        if not uid or uid in summary_by_uid:
            raise ValueError(f"invalid prompt summary row: {uid!r}")
        if pool256_mode:
            stratum = str(row.get("observed_support_stratum") or "")
            if K != 8 or stratum not in config.states:
                continue
        else:
            if K != 32:
                raise ValueError(f"invalid K=32 prompt summary row: {uid!r}, K={K}")
            stratum = observed_stratum(int(row.get("correct_count") or 0), K)
            if stratum not in config.states:
                continue
        summary_by_uid[uid] = {**row, "observed_stratum": stratum, "K": K}

    frame = pd.read_parquet(cohort_path)
    if "sample_uid" not in frame.columns:
        raise ValueError("cohort parquet has no sample_uid column")
    frame = frame.copy()
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort parquet contains duplicate sample_uid values")
    missing_uids = sorted(set(summary_by_uid) - set(frame.index))
    if missing_uids:
        raise ValueError(f"selected K=32 UIDs missing from cohort: {missing_uids[:5]}")

    ordered_uids = sorted(summary_by_uid, key=_selection_key)
    manifest_rows: list[dict[str, Any]] = []
    if config.prompt_manifest:
        manifest_path_value = Path(config.prompt_manifest).expanduser().resolve()
        if not manifest_path_value.is_file():
            raise FileNotFoundError(f"missing prompt manifest: {manifest_path_value}")
        manifest_rows = _read_jsonl(manifest_path_value)
        manifest_uids = [str(row.get("sample_uid") or "") for row in manifest_rows]
        if any(not uid for uid in manifest_uids) or len(set(manifest_uids)) != len(manifest_uids):
            raise ValueError("prompt manifest must contain unique non-empty sample_uids")
        missing_from_pool = [uid for uid in manifest_uids if uid not in summary_by_uid]
        if missing_from_pool:
            raise ValueError(
                f"manifest uids missing from selected pool strata: {missing_from_pool[:10]}"
            )
        ordered_uids = manifest_uids
    start = len(ordered_uids) * config.shard_index // config.num_shards
    end = len(ordered_uids) * (config.shard_index + 1) // config.num_shards
    shard_uids = ordered_uids[start:end]
    if config.max_prompts is not None:
        shard_uids = shard_uids[: config.max_prompts]
    records: list[dict[str, Any]] = []
    for uid in shard_uids:
        record = frame.loc[uid].to_dict()
        record["sample_uid"] = uid
        record["k32_summary"] = summary_by_uid[uid]
        records.append(record)
    provenance = {
        "k32_validation_sha256": None if pool256_mode else _sha256_file(validation_path),
        "k32_prompt_summary_sha256": None if pool256_mode else _sha256_file(summary_path),
        "pool256_frontier_sha256": _sha256_file(summary_path) if pool256_mode else None,
        "cohort_parquet_sha256": _sha256_file(cohort_path),
        "prompt_manifest_sha256": (
            _sha256_file(Path(config.prompt_manifest)) if config.prompt_manifest else None
        ),
        "selection_mode": "pool256_manifest" if pool256_mode else "k32_states",
        "all_selected_uids": ordered_uids,
        "shard_start": start,
        "shard_end": end,
        "expected_uids": shard_uids,
    }
    return records, provenance


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _load_models(config: ProposalConfig):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from dual_track_opd.fc_opd.teacher_transformers import TransformersTeacherScorer

    teacher = TransformersTeacherScorer(
        model_id=config.teacher_model_path,
        top_k=32,
        dtype=config.dtype,
        device=config.teacher_device,
    )
    student_processor = AutoProcessor.from_pretrained(config.student_model_path)
    torch_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
    student_model = AutoModelForImageTextToText.from_pretrained(
        config.student_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=Path(config.student_model_path).exists(),
    ).to(config.student_device)
    student = StudentScorer(
        StudentScorerConfig(
            model_path=config.student_model_path,
            device=config.student_device,
            dtype=config.dtype,
        ),
        model=student_model,
        processor=student_processor,
    )
    student_hash = tokenizer_fingerprint(student_processor.tokenizer)
    if teacher.metadata.tokenizer_hash != student_hash:
        raise ValueError(
            "teacher/student tokenizer mismatch; exact teacher proposal IDs cannot be "
            "forced under the student"
        )
    return teacher, student, student_hash


def _aggregate(output_dir: Path, expected_uids: Sequence[str]) -> dict[str, Any]:
    result_dir = output_dir / "prompt_results"
    prompt_results = []
    for path in sorted(result_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_uid") in expected_uids:
            prompt_results.append(value)
    by_uid = {str(row["sample_uid"]): row for row in prompt_results}
    completed_uids = [uid for uid in expected_uids if uid in by_uid]
    proposal_rows = [proposal for uid in completed_uids for proposal in by_uid[uid]["proposals"]]
    retained_rows = [row for row in proposal_rows if row.get("retained_for_fkl")]

    def proposal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        retained = [row for row in rows if row.get("retained_for_fkl")]
        nll_values = [
            float(row["trimmed_length_normalized_nll"])
            for row in retained
            if row.get("trimmed_length_normalized_nll") is not None
        ]
        return {
            "proposal_count": len(rows),
            "correct_proposal_count": sum(row.get("correct") is True for row in rows),
            "retained_proposal_count": len(retained),
            "malformed_proposal_count": sum(bool(row.get("malformed")) for row in rows),
            "truncated_proposal_count": sum(row.get("finish_reason") == "length" for row in rows),
            "generated_token_count": sum(int(row.get("response_token_count") or 0) for row in rows),
            "retained_nll_p10": _quantile(nll_values, 0.10),
            "retained_nll_median": _quantile(nll_values, 0.50),
            "retained_nll_p90": _quantile(nll_values, 0.90),
        }

    state_summary: dict[str, dict[str, Any]] = {}
    for state in DEFAULT_STATES:
        state_prompts = [by_uid[uid] for uid in completed_uids if by_uid[uid]["observed_stratum"] == state]
        if not state_prompts:
            continue
        available = [row for row in state_prompts if int(row["correct_proposal_count"]) > 0]
        state_proposals = [proposal for row in state_prompts for proposal in row["proposals"]]
        state_summary[state] = {
            "prompt_count": len(state_prompts),
            "proposal_available_count": len(available),
            "proposal_available_rate": len(available) / len(state_prompts),
            **proposal_metrics(state_proposals),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_prompt_count": len(expected_uids),
        "completed_prompt_count": len(completed_uids),
        "complete": completed_uids == list(expected_uids),
        **proposal_metrics(proposal_rows),
        "state_summary": state_summary,
        "completed_uids": completed_uids,
        "missing_uids": [uid for uid in expected_uids if uid not in by_uid],
    }
    _write_jsonl(output_dir / "proposals.jsonl", proposal_rows)
    _write_jsonl(output_dir / "retained_proposals.jsonl", retained_rows)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run(config: ProposalConfig) -> dict[str, Any]:
    import torch

    config.validate()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "prompt_results"
    result_dir.mkdir(exist_ok=True)
    records, provenance = select_prompt_records(config)
    expected_uids = list(provenance["expected_uids"])
    git_commit, git_dirty = _git_state()
    manifest_path = output_dir / "run_manifest.json"
    json_config = json.loads(json.dumps(asdict(config)))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config": json_config,
        "provenance": provenance,
        "output_dir": str(output_dir),
        "model_identity": {
            "teacher": _model_identity(config.teacher_model_path),
            "student": _model_identity(config.student_model_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "started_at_unix": time.time(),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config") != manifest["config"] or existing.get("provenance") != provenance:
            raise ValueError("existing output manifest does not match requested config/provenance")
        manifest = existing
        manifest["status"] = "running"
    _write_json(manifest_path, manifest)

    already_complete = {
        str(json.loads(path.read_text(encoding="utf-8"))["sample_uid"])
        for path in result_dir.glob("*.json")
    }
    pending = [record for record in records if str(record["sample_uid"]) not in already_complete]
    if not pending:
        summary = _aggregate(output_dir, expected_uids)
        manifest["status"] = "completed" if summary["complete"] else "partial"
        manifest["finished_at_unix"] = time.time()
        _write_json(manifest_path, manifest)
        return summary

    teacher, student, tokenizer_hash = _load_models(config)
    import transformers

    manifest["runtime"].update({
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device_count": int(torch.cuda.device_count()),
    })
    manifest["tokenizer_hash"] = tokenizer_hash
    _write_json(manifest_path, manifest)
    teacher_cuda_index = _cuda_index(config.teacher_device)
    temp_dir = Path(tempfile.mkdtemp(prefix="proposal_feasibility_images_"))
    try:
        for prompt_index, record in enumerate(pending, start=1):
            prompt_started = time.time()
            uid = str(record["sample_uid"])
            question = str(record.get("question") or "").strip()
            if not question:
                raise ValueError(f"{uid}: missing question")
            gold_answer = record.get("answer")
            image = extract_image(record).convert("RGB")
            prompt_text = build_prompt(question, config.response_format)
            proposals: list[dict[str, Any]] = []
            generations = []
            for proposal_id in range(1, config.proposals_per_prompt + 1):
                if teacher_cuda_index is not None:
                    torch.cuda.set_device(teacher_cuda_index)
                generation_seed = config.seed + int(_selection_key(uid)[:8], 16) + proposal_id
                generation = generate_response(
                    teacher.model,
                    teacher.processor,
                    question,
                    image,
                    prompt_text,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_new_tokens=config.max_new_tokens,
                    seed=generation_seed,
                    device=config.teacher_device,
                )
                verdict = verify_answer(generation.response_text_display, gold_answer)
                generations.append((proposal_id, generation_seed, generation, verdict))

            correct_generations = [item for item in generations if item[3].get("correct") is True]
            student_scores = student.score_batch(
                questions=[question] * len(correct_generations),
                images=[image] * len(correct_generations),
                prompt_texts=[prompt_text] * len(correct_generations),
                response_texts=[item[2].response_text_display for item in correct_generations],
                response_token_ids_list=[item[2].response_token_ids_raw for item in correct_generations],
            ) if correct_generations else []
            score_by_proposal = {
                proposal_id: score
                for (proposal_id, _, _, _), score in zip(
                    correct_generations, student_scores, strict=True
                )
            }
            for proposal_id, generation_seed, generation, verdict in generations:
                proposal: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_uid": uid,
                    "observed_stratum": record["k32_summary"]["observed_stratum"],
                    "k32_correct_count": int(record["k32_summary"]["correct_count"]),
                    "k32_K": int(record["k32_summary"]["K"]),
                    "proposal_id": proposal_id,
                    "generation_seed": generation_seed,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_new_tokens": config.max_new_tokens,
                    "response_text_display": generation.response_text_display,
                    "response_text_raw": generation.response_text_raw,
                    "response_token_ids": list(generation.response_token_ids_raw),
                    "response_token_hash": generation.response_token_hash,
                    "response_token_count": len(generation.response_token_ids_raw),
                    "content_mask": list(generation.content_mask),
                    "finish_reason": generation.finish_reason,
                    "terminal_token_id": generation.terminal_token_id,
                    "prompt_token_hash": generation.prompt_token_hash,
                    "correct": verdict.get("correct"),
                    "malformed": bool(verdict.get("malformed")),
                    "answer_extracted": verdict.get("answer_extracted"),
                    "gold_answer": verdict.get("gold_answer"),
                    "student_scored_token_hash": None,
                    "student_sampled_token_log_probs": None,
                    "trimmed_length_normalized_nll": None,
                }
                score = score_by_proposal.get(proposal_id)
                if score is not None:
                    if score.error:
                        raise RuntimeError(f"{uid}:proposal-{proposal_id}: {score.error}")
                    expected_hash = hash_token_ids(generation.response_token_ids_raw)
                    if score.scored_token_hash != expected_hash:
                        raise RuntimeError(f"{uid}:proposal-{proposal_id}: student token hash mismatch")
                    reachability = trimmed_length_normalized_nll(
                        score.sampled_token_log_probs,
                        content_mask=generation.content_mask,
                        low_trim_fraction=config.low_trim_fraction,
                        high_trim_fraction=config.high_trim_fraction,
                    )
                    proposal.update(reachability)
                    proposal["student_scored_token_hash"] = score.scored_token_hash
                    proposal["student_sampled_token_log_probs"] = list(
                        score.sampled_token_log_probs
                    )
                proposals.append(proposal)
            proposals = mark_reachable_proposals(proposals, retain_count=config.retain_count)
            prompt_result = {
                "schema_version": SCHEMA_VERSION,
                "sample_uid": uid,
                "observed_stratum": record["k32_summary"]["observed_stratum"],
                "k32_correct_count": int(record["k32_summary"]["correct_count"]),
                "k32_K": int(record["k32_summary"]["K"]),
                "teacher_model_id": teacher.metadata.model_id,
                "student_model_path": config.student_model_path,
                "tokenizer_hash": tokenizer_hash,
                "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "proposal_count": len(proposals),
                "correct_proposal_count": sum(row.get("correct") is True for row in proposals),
                "retained_proposal_count": sum(bool(row["retained_for_fkl"]) for row in proposals),
                "wall_seconds": time.time() - prompt_started,
                "proposals": proposals,
            }
            target = result_dir / f"{_safe_uid(uid)}.json"
            temporary = target.with_suffix(".json.tmp")
            _write_json(temporary, prompt_result)
            temporary.replace(target)
            summary = _aggregate(output_dir, expected_uids)
            print(
                f"[{prompt_index}/{len(pending)}] {uid}: "
                f"correct={prompt_result['correct_proposal_count']}/{config.proposals_per_prompt}, "
                f"retained={prompt_result['retained_proposal_count']}",
                flush=True,
            )
            image.close()
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    summary = _aggregate(output_dir, expected_uids)
    manifest["status"] = "completed" if summary["complete"] else "partial"
    manifest["finished_at_unix"] = time.time()
    manifest["summary"] = summary
    _write_json(manifest_path, manifest)
    return summary


def merge_shards(shard_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    shards = [Path(path).expanduser().resolve() for path in shard_dirs]
    manifests = [json.loads((path / "run_manifest.json").read_text(encoding="utf-8")) for path in shards]
    for path, manifest in zip(shards, manifests, strict=True):
        if manifest.get("status") != "completed":
            raise ValueError(f"incomplete proposal shard: {path}")
    normalized_configs = []
    expected_uids: list[str] = []
    seen: set[str] = set()
    all_selected_sets = {
        tuple(manifest["provenance"]["all_selected_uids"])
        for manifest in manifests
    }
    if len(all_selected_sets) != 1:
        raise ValueError("proposal shards disagree on the full selected UID set")
    git_commits = {str(manifest.get("git_commit") or "") for manifest in manifests}
    if len(git_commits) != 1 or "" in git_commits:
        raise ValueError(f"proposal shard git commits differ: {sorted(git_commits)}")
    dirty_states = {bool(manifest.get("git_dirty")) for manifest in manifests}
    allow_dirty = os.environ.get("DTOPD_ALLOW_DIRTY_MERGE") == "1"
    if dirty_states != {False} and not allow_dirty:
        raise ValueError("proposal shards must be produced from clean worktrees")
    expected_shard_indices = set(range(int(manifests[0]["config"]["num_shards"])))
    actual_shard_indices = {int(manifest["config"]["shard_index"]) for manifest in manifests}
    if actual_shard_indices != expected_shard_indices:
        raise ValueError(
            f"proposal shard index coverage differs: {sorted(actual_shard_indices)} "
            f"vs {sorted(expected_shard_indices)}"
        )
    for manifest in manifests:
        config = dict(manifest["config"])
        for key in ("output_dir", "shard_index"):
            config.pop(key, None)
        normalized_configs.append(config)
        shard_uids = list(manifest["provenance"]["expected_uids"])
        overlap = seen.intersection(shard_uids)
        if overlap:
            raise ValueError(f"proposal shards overlap: {sorted(overlap)[:5]}")
        seen.update(shard_uids)
        expected_uids.extend(shard_uids)
    if any(config != normalized_configs[0] for config in normalized_configs[1:]):
        raise ValueError("proposal shard configs differ")
    full_selected_uids = next(iter(all_selected_sets))
    if seen != set(full_selected_uids):
        raise ValueError("proposal shard UID union does not cover the selected cohort")

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
                    raise ValueError(f"duplicate proposal prompt artifact: {source.name}")
                target.write_bytes(source.read_bytes())
        merged_config = dict(manifests[0]["config"])
        merged_config["output_dir"] = str(output)
        merged_config["shard_index"] = 0
        merged_config["num_shards"] = 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "merged_from": [str(path) for path in shards],
            "git_commits": sorted(git_commits),
            "config": merged_config,
            "provenance": {
                "expected_uids": expected_uids,
                "source_shard_manifest_sha256": [
                    _sha256_file(path / "run_manifest.json") for path in shards
                ],
            },
        }
        summary = _aggregate(temporary, expected_uids)
        if not summary["complete"]:
            raise ValueError(f"merged proposal output is incomplete: {summary['missing_uids'][:5]}")
        manifest["summary"] = summary
        _write_json(temporary / "run_manifest.json", manifest)
        temporary.replace(output)
        return summary
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one resumable proposal shard")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--k32-run-dir")
    run_parser.add_argument("--cohort-dir")
    run_parser.add_argument("--cohort-parquet-path")
    run_parser.add_argument("--pool256-run-dir")
    run_parser.add_argument("--prompt-manifest")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--max-prompts", type=int)
    run_parser.add_argument("--proposals-per-prompt", type=int)
    run_parser.add_argument("--max-new-tokens", type=int)
    merge_parser = subparsers.add_parser("merge", help="strictly merge completed shards")
    merge_parser.add_argument("--shard-dirs", nargs="+", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run(load_config(args.config, args))
        else:
            result = merge_shards(args.shard_dirs, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
