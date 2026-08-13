"""End-to-end Phase-I causal state localization for Qwen3-VL Geometry3K.

The command consumes the repository's immutable K=32 diagnostic and verified
teacher-proposal cache.  It does not update model weights.  Expensive probes
are independently feature-gated and every completed trajectory is committed
atomically for safe resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids

from .causal_dataset import ProbeInput, select_probe_inputs, sha256_file
from .causal_image import build_image_conditions, image_sha256
from .causal_probes import ActionabilityThresholds, classify_actionability, detect_reachability_barriers
from .causal_report import write_reports
from .causal_runtime import (
    condition_continuations,
    direct_answer_estimate,
    fixed_trajectory_visual_statistics,
    load_runtime_models,
    model_identity,
    teacher_path_support_statistics,
    teacher_relay_estimate,
    transport_estimate,
)
from .causal_schema import CandidateWindow, CausalStateRecord, StudentSupport
from .diagnostic import build_prompt, extract_image
from .prefix_intervention import prefix_leakage_reason
from .visual_js import CandidateSelectionConfig, multiscale_curves, select_visual_candidates


RUN_SCHEMA_VERSION = "causal-state-probe-run-v1"


@dataclass(frozen=True)
class FeatureFlags:
    visual_js: bool = False
    visual_continuation: bool = False
    teacher_relay: bool = False
    teacher_transport: bool = False
    answer_leakage: bool = False
    student_teacher_reference: bool = False
    teacher_path_support: bool = False

    @property
    def needs_teacher(self) -> bool:
        return self.teacher_relay or self.student_teacher_reference or self.teacher_path_support


@dataclass(frozen=True)
class ProbeConfig:
    k32_run_dir: str
    cohort_dir: str
    proposal_dir: str | None
    output_dir: str
    student_model_path: str
    teacher_model_path: str | None
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    dtype: str = "bfloat16"
    response_format: str = "legacy_answer"
    trajectory_outcomes: tuple[str, ...] = ("correct", "wrong")
    max_trajectories_per_outcome: int = 1
    degraded_mode: str = "blur_sigma_2"
    null_value: int = 255
    js_chunk_size: int = 64
    candidate_windows: tuple[int, ...] = (8, 16, 32, 64)
    candidate_topk_high: int = 1
    candidate_topk_drop: int = 2
    candidate_fixed_positions: tuple[float, ...] = (0.2, 0.5, 0.8)
    candidate_min: int = 3
    candidate_max: int = 6
    candidate_nms_radius: int = 16
    candidate_snap_radius: int = 12
    continuation_k: int = 8
    relay_k: int = 8
    transport_k: int = 8
    answer_k: int = 8
    max_continuation_tokens: int = 2048
    max_answer_tokens: int = 64
    relay_lengths: tuple[int, ...] = (32, 64, 128)
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 20260808
    posterior_draws: int = 20_000
    min_mean_gain: float = 0.20
    min_posterior_probability: float = 0.90
    max_low_gain: float = 0.10
    teacher_path_window: int = 32
    teacher_path_top_k: int = 3
    features: FeatureFlags = FeatureFlags()
    max_prompts: int | None = None
    shard_index: int = 0
    num_shards: int = 1
    work_slice_index: int = 0
    work_slice_total: int = 1

    def validate(self) -> None:
        if not self.k32_run_dir or not self.cohort_dir or not self.output_dir:
            raise ValueError("K32, cohort, and output paths are required")
        if not self.student_model_path:
            raise ValueError("student model path is required")
        if self.features.needs_teacher and not self.teacher_model_path:
            raise ValueError("enabled teacher probes require teacher_model_path")
        if (self.features.teacher_transport or self.features.answer_leakage or self.features.teacher_path_support) and not self.proposal_dir:
            raise ValueError("transport/leakage/teacher-path probes require proposal_dir")
        if self.features.visual_continuation and not self.features.visual_js:
            raise ValueError("visual continuation requires visual_js candidate localization")
        if self.features.teacher_relay and not self.features.visual_continuation:
            raise ValueError("teacher relay requires the same-prefix student continuation baseline")
        if self.features.answer_leakage and not self.features.teacher_transport:
            raise ValueError("answer leakage is defined on accepted teacher-transport prefixes")
        if self.js_chunk_size <= 0 or self.max_continuation_tokens <= 0 or self.max_answer_tokens <= 0:
            raise ValueError("token limits must be positive")
        if any(value <= 0 for value in (self.continuation_k, self.relay_k, self.transport_k, self.answer_k)):
            raise ValueError("probe K values must be positive")
        if not self.relay_lengths or any(value <= 0 for value in self.relay_lengths):
            raise ValueError("relay lengths must be positive")
        if not 0 <= self.shard_index < self.num_shards or self.num_shards <= 0:
            raise ValueError("invalid shard assignment")
        if self.work_slice_total <= 0 or not 0 <= self.work_slice_index < self.work_slice_total:
            raise ValueError("invalid work-slice assignment")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")
        self.candidate_config().validate()
        self.actionability_thresholds().validate()

    def candidate_config(self) -> CandidateSelectionConfig:
        return CandidateSelectionConfig(
            windows=self.candidate_windows,
            topk_high=self.candidate_topk_high,
            topk_drop=self.candidate_topk_drop,
            fixed_positions=self.candidate_fixed_positions,
            min_candidates=self.candidate_min,
            max_candidates=self.candidate_max,
            nms_radius=self.candidate_nms_radius,
            snap_radius=self.candidate_snap_radius,
        )

    def actionability_thresholds(self) -> ActionabilityThresholds:
        return ActionabilityThresholds(
            min_mean_gain=self.min_mean_gain,
            min_posterior_probability=self.min_posterior_probability,
            max_low_gain=self.max_low_gain,
        )


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _nested(raw: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = raw
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def load_config(path: str | Path, args: argparse.Namespace | None = None) -> ProbeConfig:
    import yaml

    raw = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    override = lambda name, default=None: getattr(args, name, None) if args is not None and getattr(args, name, None) is not None else default
    feature_raw = _nested(raw, "features", default={}) or {}
    features = FeatureFlags(**{
        key: bool(feature_raw.get(key, False))
        for key in FeatureFlags.__dataclass_fields__
    })
    config = ProbeConfig(
        k32_run_dir=str(override("k32_run_dir", _nested(raw, "data", "k32_run_dir", default=""))),
        cohort_dir=str(override("cohort_dir", _nested(raw, "data", "cohort_dir", default=""))),
        proposal_dir=(
            None if override("proposal_dir", _nested(raw, "data", "proposal_dir")) in (None, "")
            else str(override("proposal_dir", _nested(raw, "data", "proposal_dir")))
        ),
        output_dir=str(override("output_dir", _nested(raw, "output", "dir", default=""))),
        student_model_path=str(_nested(raw, "models", "student", default="")),
        teacher_model_path=_nested(raw, "models", "teacher"),
        student_device=str(override("student_device", _nested(raw, "hardware", "student_device", default="cuda:0"))),
        teacher_device=str(override("teacher_device", _nested(raw, "hardware", "teacher_device", default="cuda:1"))),
        dtype=str(_nested(raw, "hardware", "dtype", default="bfloat16")),
        response_format=str(_nested(raw, "data", "response_format", default="legacy_answer")),
        trajectory_outcomes=tuple(_nested(raw, "selection", "trajectory_outcomes", default=("correct", "wrong"))),
        max_trajectories_per_outcome=int(_nested(raw, "selection", "max_trajectories_per_outcome", default=1)),
        degraded_mode=str(_nested(raw, "visual", "degraded_mode", default="blur_sigma_2")),
        null_value=int(_nested(raw, "visual", "null_value", default=255)),
        js_chunk_size=int(_nested(raw, "visual", "js_chunk_size", default=64)),
        candidate_windows=tuple(int(value) for value in _nested(raw, "candidates", "windows", default=(8, 16, 32, 64))),
        candidate_topk_high=int(_nested(raw, "candidates", "topk_high", default=1)),
        candidate_topk_drop=int(_nested(raw, "candidates", "topk_drop", default=2)),
        candidate_fixed_positions=tuple(float(value) for value in _nested(raw, "candidates", "fixed_positions", default=(0.2, 0.5, 0.8))),
        candidate_min=int(_nested(raw, "candidates", "min_candidates", default=3)),
        candidate_max=int(_nested(raw, "candidates", "max_candidates", default=6)),
        candidate_nms_radius=int(_nested(raw, "candidates", "nms_radius", default=16)),
        candidate_snap_radius=int(_nested(raw, "candidates", "snap_radius", default=12)),
        continuation_k=int(override("continuation_k", _nested(raw, "continuation", "k", default=8))),
        relay_k=int(override("relay_k", _nested(raw, "relay", "k", default=8))),
        transport_k=int(override("transport_k", _nested(raw, "transport", "k", default=8))),
        answer_k=int(override("answer_k", _nested(raw, "leakage", "k", default=8))),
        max_continuation_tokens=int(override("max_continuation_tokens", _nested(raw, "continuation", "max_new_tokens", default=2048))),
        max_answer_tokens=int(override("max_answer_tokens", _nested(raw, "leakage", "max_new_tokens", default=64))),
        relay_lengths=tuple(int(value) for value in (override("relay_lengths", None) or _nested(raw, "relay", "lengths", default=(32, 64, 128)))),
        temperature=float(_nested(raw, "generation", "temperature", default=0.7)),
        top_p=float(_nested(raw, "generation", "top_p", default=0.95)),
        seed=int(_nested(raw, "generation", "seed", default=20260808)),
        posterior_draws=int(_nested(raw, "analysis", "posterior_draws", default=20_000)),
        min_mean_gain=float(_nested(raw, "analysis", "min_mean_gain", default=0.20)),
        min_posterior_probability=float(_nested(raw, "analysis", "min_posterior_probability", default=0.90)),
        max_low_gain=float(_nested(raw, "analysis", "max_low_gain", default=0.10)),
        teacher_path_window=int(_nested(raw, "teacher_path", "window", default=32)),
        teacher_path_top_k=int(_nested(raw, "teacher_path", "top_k", default=3)),
        features=features,
        max_prompts=override("max_prompts"),
        shard_index=int(override("shard_index", 0)),
        num_shards=int(override("num_shards", 1)),
        work_slice_index=int(override("work_slice_index", 0)),
        work_slice_total=int(override("work_slice_total", 1)),
    )
    config.validate()
    return config


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    # Unique temp name: concurrent runners share the same manifest/result
    # directory, so a fixed ".tmp" suffix would race (one process replaces the
    # other's temp file before it is renamed).  mkstemp + os.replace keeps the
    # write atomic while allowing safe concurrent writers.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _safe_name(value: str) -> str:
    prefix = "".join(character if character.isalnum() else "_" for character in value)
    return f"{prefix[:80]}_{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def _work_id(item: ProbeInput) -> str:
    rollout = item.student_rollout
    return f"{item.sample_uid}:rollout-{int(rollout.get('rollout_id') or 0)}:{rollout.get('response_token_hash')}"


def _work_slice_items(items: Sequence[ProbeInput], index: int, total: int) -> list[ProbeInput]:
    """Deterministically split pending work units across concurrent runners.

    `index`/`total` must already be validated (0 <= index < total, total >= 1).
    With `total == 1` the full list is returned, so resume behavior is unchanged.
    """
    if total <= 1:
        return list(items)
    return [item for i, item in enumerate(items) if i % total == index]


def _seed(config: ProbeConfig, work_id: str, candidate_anchor: int, salt: str) -> int:
    digest = hashlib.sha256(f"{work_id}:{candidate_anchor}:{salt}".encode()).hexdigest()
    return config.seed + int(digest[:8], 16) % 1_000_000_000


def _mean_signal(rows: Sequence[Mapping[str, Any]], key: str, start: int, end: int) -> float:
    return float(fmean(float(rows[index][key]) for index in range(start, end)))


def _model_preflight(path_value: str | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    expanded = Path(os.path.expandvars(path_value)).expanduser()
    result = model_identity(str(expanded))
    result["exists"] = expanded.exists()
    result["config_exists"] = (expanded / "config.json").is_file() if expanded.is_dir() else None
    return result


def preflight(config: ProbeConfig, *, load_tokenizers: bool = False) -> dict[str, Any]:
    config.validate()
    checks = {
        "k32_validation": (Path(config.k32_run_dir).expanduser() / "k32_validation.json").is_file(),
        "k32_rollouts": (Path(config.k32_run_dir).expanduser() / "rollouts.jsonl").is_file(),
        "k32_support": (Path(config.k32_run_dir).expanduser() / "prompt_support_summary.jsonl").is_file(),
        "cohort": (Path(config.cohort_dir).expanduser() / "cohort.parquet").is_file(),
        "proposals": True if config.proposal_dir is None else (Path(config.proposal_dir).expanduser() / "retained_proposals.jsonl").is_file(),
    }
    student_model = _model_preflight(config.student_model_path)
    teacher_model = _model_preflight(config.teacher_model_path)
    models_valid = bool(student_model and student_model.get("exists") and student_model.get("config_exists"))
    if config.features.needs_teacher:
        models_valid = bool(models_valid and teacher_model and teacher_model.get("exists") and teacher_model.get("config_exists"))
    result: dict[str, Any] = {
        "valid": all(checks.values()) and models_valid,
        "checks": checks,
        "student_model": student_model,
        "teacher_model": teacher_model,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "features": asdict(config.features),
    }
    try:
        import torch
        import transformers

        result["torch"] = str(torch.__version__)
        result["transformers"] = str(transformers.__version__)
        result["cuda_runtime"] = str(torch.version.cuda)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
    except ImportError as exc:
        result["runtime_error"] = str(exc)
        result["valid"] = False
    if load_tokenizers and result["valid"]:
        from transformers import AutoProcessor
        from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

        student = AutoProcessor.from_pretrained(
            config.student_model_path,
            local_files_only=Path(config.student_model_path).exists(),
            trust_remote_code=True,
        )
        result["student_tokenizer_hash"] = tokenizer_fingerprint(student.tokenizer)
        if config.teacher_model_path:
            teacher = AutoProcessor.from_pretrained(
                config.teacher_model_path,
                local_files_only=Path(config.teacher_model_path).exists(),
                trust_remote_code=True,
            )
            result["teacher_tokenizer_hash"] = tokenizer_fingerprint(teacher.tokenizer)
            result["tokenizer_aligned"] = result["student_tokenizer_hash"] == result["teacher_tokenizer_hash"]
            result["valid"] = bool(result["valid"] and result["tokenizer_aligned"])
    return result


def _teacher_rows(item: ProbeInput) -> list[dict[str, Any]]:
    rows = []
    for proposal in item.teacher_proposals:
        rows.append({
            "proposal_id": proposal.get("proposal_id"),
            "correct": proposal.get("correct"),
            "reachability_rank": proposal.get("reachability_rank"),
            "response_token_ids": list(proposal.get("response_token_ids") or ()),
            "response_token_hash": proposal.get("response_token_hash"),
            "response_text": proposal.get("response_text_display"),
            "student_path_support": None,
            "reachability_barriers": [],
        })
    return rows


def _build_record(item: ProbeInput, config: ProbeConfig, models: Any) -> CausalStateRecord:
    rollout = item.student_rollout
    response_ids = tuple(int(value) for value in rollout["response_token_ids"])
    response_hash = str(rollout.get("response_token_hash") or hash_token_ids(response_ids))
    question = str(item.cohort.get("question") or "").strip()
    gold_answer = item.cohort.get("answer")
    prompt_text = build_prompt(question, config.response_format)
    image = extract_image(item.cohort).convert("RGB")
    images = build_image_conditions(
        image,
        degraded_mode=config.degraded_mode,
        null_value=config.null_value,
    )
    work_id = _work_id(item)
    teacher_rows = _teacher_rows(item)

    fixed = fixed_trajectory_visual_statistics(
        models.student_model,
        models.student_processor,
        prompt_text=prompt_text,
        images=images,
        response_ids=response_ids,
        device=config.student_device,
        chunk_size=config.js_chunk_size,
    )
    token_rows = [dict(row) for row in fixed.token_signals]
    selection_signal = [
        max(float(row["js_full_degraded"]), float(row["js_full_null"]))
        for row in token_rows
    ]
    selected = select_visual_candidates(
        selection_signal,
        decoded_tokens=fixed.decoded_tokens,
        config=config.candidate_config(),
    )
    curves_degraded = multiscale_curves(
        [float(row["js_full_degraded"]) for row in token_rows], config.candidate_windows
    )
    curves_null = multiscale_curves(
        [float(row["js_full_null"]) for row in token_rows], config.candidate_windows
    )

    cross_model_student_path = None
    if config.features.student_teacher_reference:
        cross_model_student_path = teacher_path_support_statistics(
            models=models,
            image=images.full,
            prompt_text=prompt_text,
            teacher_token_ids=response_ids,
            student_device=config.student_device,
            teacher_device=config.teacher_device,
            chunk_size=config.js_chunk_size,
        )
        for token_row, reference in zip(token_rows, cross_model_student_path, strict=True):
            token_row["student_teacher_js"] = reference["student_teacher_js"]
            token_row["student_teacher_overlap"] = reference["topk_overlap"]

    if config.features.teacher_path_support and teacher_rows:
        teacher_token_ids = tuple(int(value) for value in teacher_rows[0]["response_token_ids"])
        path_rows = teacher_path_support_statistics(
            models=models,
            image=images.full,
            prompt_text=prompt_text,
            teacher_token_ids=teacher_token_ids,
            student_device=config.student_device,
            teacher_device=config.teacher_device,
            chunk_size=config.js_chunk_size,
        )
        barriers = detect_reachability_barriers(
            [float(row["student_nll"]) for row in path_rows],
            window=config.teacher_path_window,
            top_k=config.teacher_path_top_k,
        )
        teacher_rows[0]["student_path_support"] = path_rows
        teacher_rows[0]["reachability_barriers"] = [asdict(value) for value in barriers]

    unaided_full = None
    if config.features.teacher_transport:
        unaided = condition_continuations(
            models.student_model,
            models.student_processor,
            images=images,
            prompt_text=prompt_text,
            prefix_ids=(),
            gold_answer=gold_answer,
            k=config.transport_k,
            max_continuation_tokens=config.max_continuation_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=_seed(config, work_id, 0, "unaided"),
            device=config.student_device,
        )
        unaided_full = next(value for value in unaided if value.condition == "full")

    candidates: list[CandidateWindow] = []
    for candidate_index, value in enumerate(selected):
        prefix_ids = response_ids[: value.anchor + 1]
        notes: list[str] = []
        visual_estimates = ()
        visual_fine = visual_all = None
        baseline_full = None
        if config.features.visual_continuation:
            visual_estimates = condition_continuations(
                models.student_model,
                models.student_processor,
                images=images,
                prompt_text=prompt_text,
                prefix_ids=prefix_ids,
                gold_answer=gold_answer,
                k=config.continuation_k,
                max_continuation_tokens=config.max_continuation_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=_seed(config, work_id, value.anchor, "visual"),
                device=config.student_device,
            )
            visual_by_name = {estimate.condition: estimate for estimate in visual_estimates}
            baseline_full = visual_by_name["full"]
            visual_fine = baseline_full.pass_rate - visual_by_name["degraded"].pass_rate
            visual_all = baseline_full.pass_rate - visual_by_name["null"].pass_rate

        relay_gains: dict[str, float] = {}
        relay_probabilities: dict[str, float] = {}
        relay_estimates: dict[str, Any] = {}
        if config.features.teacher_relay and baseline_full is not None:
            from .causal_probes import compare_estimates

            for relay_length in config.relay_lengths:
                estimate = teacher_relay_estimate(
                    models=models,
                    image=images.full,
                    prompt_text=prompt_text,
                    student_prefix_ids=prefix_ids,
                    gold_answer=gold_answer,
                    relay_length=relay_length,
                    k=config.relay_k,
                    max_student_tokens=config.max_continuation_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=_seed(config, work_id, value.anchor, f"relay-{relay_length}"),
                    student_device=config.student_device,
                    teacher_device=config.teacher_device,
                )
                relay_estimates[str(relay_length)] = estimate
                if estimate.n_malformed:
                    notes.append(
                        f"relay_l{relay_length}_answer_leakage_or_malformed:{estimate.n_malformed}"
                    )
                if estimate.n == baseline_full.n:
                    lift = compare_estimates(
                        estimate,
                        baseline_full,
                        seed=_seed(config, work_id, value.anchor, f"relay-posterior-{relay_length}"),
                        draws=config.posterior_draws,
                    )
                    relay_gains[str(relay_length)] = lift.mean_lift
                    relay_probabilities[str(relay_length)] = lift.posterior_probability
                else:
                    relay_gains[str(relay_length)] = estimate.pass_rate - baseline_full.pass_rate

        transported = wrong_prefix_estimate = answer_estimate = None
        transport_gain = transport_probability = answer_leakage = None
        transport_vs_wrong_gain = transport_vs_wrong_probability = wrong_prefix_gain = None
        if config.features.teacher_transport and teacher_rows and unaided_full is not None:
            from .causal_probes import compare_estimates

            teacher_ids = tuple(int(token_id) for token_id in teacher_rows[0]["response_token_ids"])
            prefix_length = min(
                len(teacher_ids),
                max(1, int(round(value.relative_position * max(len(teacher_ids) - 1, 1))) + 1),
            )
            teacher_prefix = teacher_ids[:prefix_length]
            prefix_text = models.student_processor.tokenizer.decode(
                list(teacher_prefix), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            leakage_reason = prefix_leakage_reason(prefix_text, gold_answer)
            if leakage_reason is not None:
                notes.append(f"transport_skipped:{leakage_reason}")
            else:
                transported = transport_estimate(
                    models=models,
                    image=images.full,
                    prompt_text=prompt_text,
                    teacher_prefix_ids=teacher_prefix,
                    gold_answer=gold_answer,
                    k=config.transport_k,
                    max_continuation_tokens=config.max_continuation_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=_seed(config, work_id, value.anchor, "transport"),
                    student_device=config.student_device,
                )
                lift = compare_estimates(
                    transported,
                    unaided_full,
                    seed=_seed(config, work_id, value.anchor, "transport-posterior"),
                    draws=config.posterior_draws,
                )
                transport_gain = lift.mean_lift
                transport_probability = lift.posterior_probability
                wrong_row = item.wrong_control_rollout
                wrong_ids = tuple(int(token_id) for token_id in (
                    () if wrong_row is None else wrong_row.get("response_token_ids") or ()
                ))
                if len(wrong_ids) < prefix_length:
                    notes.append("wrong_prefix_skipped:source_shorter_than_teacher_prefix")
                else:
                    wrong_prefix = wrong_ids[:prefix_length]
                    wrong_text = models.student_processor.tokenizer.decode(
                        list(wrong_prefix), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    wrong_leakage = prefix_leakage_reason(wrong_text, gold_answer)
                    if wrong_leakage is not None:
                        notes.append(f"wrong_prefix_skipped:{wrong_leakage}")
                    else:
                        wrong_prefix_estimate = transport_estimate(
                            models=models,
                            image=images.full,
                            prompt_text=prompt_text,
                            teacher_prefix_ids=wrong_prefix,
                            gold_answer=gold_answer,
                            k=config.transport_k,
                            max_continuation_tokens=config.max_continuation_tokens,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            seed=_seed(config, work_id, value.anchor, "wrong-prefix"),
                            student_device=config.student_device,
                            condition="wrong_prefix_control",
                        )
                        wrong_lift = compare_estimates(
                            wrong_prefix_estimate,
                            unaided_full,
                            seed=_seed(config, work_id, value.anchor, "wrong-prefix-posterior"),
                            draws=config.posterior_draws,
                        )
                        teacher_vs_wrong = compare_estimates(
                            transported,
                            wrong_prefix_estimate,
                            seed=_seed(config, work_id, value.anchor, "transport-vs-wrong-posterior"),
                            draws=config.posterior_draws,
                        )
                        wrong_prefix_gain = wrong_lift.mean_lift
                        transport_vs_wrong_gain = teacher_vs_wrong.mean_lift
                        transport_vs_wrong_probability = teacher_vs_wrong.posterior_probability
                if config.features.answer_leakage:
                    answer_estimate = direct_answer_estimate(
                        models.student_model,
                        models.student_processor,
                        image=images.full,
                        prompt_text=prompt_text,
                        prefix_ids=teacher_prefix,
                        gold_answer=gold_answer,
                        k=config.answer_k,
                        max_answer_tokens=config.max_answer_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        seed=_seed(config, work_id, value.anchor, "leakage"),
                        device=config.student_device,
                    )
                    answer_leakage = answer_estimate.pass_rate

        relay_gain = max(relay_gains.values()) if relay_gains else None
        best_relay_length = (
            max(relay_gains, key=lambda key: relay_gains[key]) if relay_gains else None
        )
        relay_probability = (
            relay_probabilities.get(best_relay_length) if best_relay_length is not None else None
        )
        actionable_transport_gain = (
            None if transport_vs_wrong_gain is None or transport_gain is None
            else min(transport_gain, transport_vs_wrong_gain)
        )
        actionable_transport_probability = (
            None if transport_vs_wrong_probability is None or transport_probability is None
            else min(transport_probability, transport_vs_wrong_probability)
        )
        state_class = classify_actionability(
            relay_gain=relay_gain,
            transport_gain=actionable_transport_gain,
            relay_probability=relay_probability,
            transport_probability=actionable_transport_probability,
            thresholds=config.actionability_thresholds(),
        )
        candidates.append(CandidateWindow(
            candidate_id=f"{work_id}:candidate-{candidate_index}",
            start=value.start,
            end=value.end,
            anchor=value.anchor,
            relative_position=value.relative_position,
            sources=value.sources,
            snapped=value.snapped,
            student_js_full_degraded=_mean_signal(token_rows, "js_full_degraded", value.start, value.end),
            student_js_full_null=_mean_signal(token_rows, "js_full_null", value.start, value.end),
            js_drop_full_degraded=float(curves_degraded["drop"][value.anchor]),
            js_drop_full_null=float(curves_null["drop"][value.anchor]),
            entropy_full=_mean_signal(token_rows, "entropy_full", value.start, value.end),
            entropy_degraded=_mean_signal(token_rows, "entropy_degraded", value.start, value.end),
            entropy_null=_mean_signal(token_rows, "entropy_null", value.start, value.end),
            visual_continuations=tuple(visual_estimates),
            visual_fine_gain=visual_fine,
            visual_all_gain=visual_all,
            relay_continuations=relay_estimates,
            relay_gain_by_length=relay_gains,
            relay_probability_by_length=relay_probabilities,
            unaided_continuation=unaided_full,
            transport_continuation=transported,
            wrong_prefix_continuation=wrong_prefix_estimate,
            transport_gain=transport_gain,
            transport_probability=transport_probability,
            transport_vs_wrong_gain=transport_vs_wrong_gain,
            transport_vs_wrong_probability=transport_vs_wrong_probability,
            wrong_prefix_gain=wrong_prefix_gain,
            answer_leakage_continuation=answer_estimate,
            answer_leakage=answer_leakage,
            student_teacher_overlap=(
                None if cross_model_student_path is None
                else _mean_signal(token_rows, "student_teacher_overlap", value.start, value.end)
            ),
            state_class=state_class,
            notes=tuple(notes),
        ))

    support = StudentSupport.from_counts(
        int(item.support_summary.get("correct_count") or 0),
        int(item.support_summary.get("K") or 32),
    )
    record = CausalStateRecord(
        prompt_id=item.sample_uid,
        dataset=str(item.cohort.get("data_source") or item.cohort.get("dataset") or "geometry3k"),
        question=question,
        image_path=None,
        image_sha256=image_sha256(images.full),
        ground_truth=gold_answer,
        student_support=support,
        trajectory_id=work_id,
        trajectory_correct=rollout.get("correct"),
        trajectory_token_ids=response_ids,
        trajectory_token_hash=response_hash,
        trajectory_text=str(rollout.get("response_text_display") or rollout.get("response_text") or ""),
        teacher_trajectories=tuple(teacher_rows),
        token_signals=tuple(token_rows),
        candidate_windows=tuple(candidates),
        metadata={
            "student_rollout_id": rollout.get("rollout_id"),
            "student_finish_reason": rollout.get("finish_reason"),
            "prompt_token_hash": fixed.prompt_token_hash,
            "tokenizer_hash": models.tokenizer_hash,
            "layout": fixed.layout_metadata,
            "features": asdict(config.features),
            # Review P1-4: per-trajectory implementation identity so old
            # (chunked logits_to_keep) and new (single full-prefix forward)
            # units can be stratified instead of silently mixed.
            "implementation_version": "single_forward_v2",
        },
    )
    record.validate()
    return record


def run(config: ProbeConfig) -> dict[str, Any]:
    config.validate()
    if not config.features.visual_js:
        raise ValueError("causal-state run requires features.visual_js=true")
    output = Path(config.output_dir).expanduser().resolve()
    result_dir = output / "trajectory_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    inputs, provenance = select_probe_inputs(
        k32_run_dir=config.k32_run_dir,
        cohort_dir=config.cohort_dir,
        proposal_dir=config.proposal_dir,
        trajectory_outcomes=config.trajectory_outcomes,
        max_trajectories_per_outcome=config.max_trajectories_per_outcome,
        max_prompts=config.max_prompts,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    provenance["expected_work_ids"] = [_work_id(item) for item in inputs]
    commit, dirty = _git_state()
    config_dict = {
        key: value
        for key, value in json.loads(json.dumps(asdict(config))).items()
        if key not in ("work_slice_index", "work_slice_total")
    }
    manifest_path = output / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config_dict,
        "provenance": provenance,
        "output_dir": str(output),
        "models": {
            "student": model_identity(config.student_model_path),
            "teacher": None if config.teacher_model_path is None else model_identity(config.teacher_model_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "started_at_unix": time.time(),
    }
    import yaml

    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_dict, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config") != config_dict or existing.get("provenance") != provenance:
            raise ValueError("existing causal-probe manifest differs from requested run")
        manifest = existing
        manifest["status"] = "running"
    _write_json_atomic(manifest_path, manifest)

    completed = {
        str(json.loads(path.read_text(encoding="utf-8"))["trajectory_id"])
        for path in result_dir.glob("*.json")
    }
    pending = [item for item in inputs if _work_id(item) not in completed]
    if config.work_slice_total > 1:
        pending = _work_slice_items(pending, config.work_slice_index, config.work_slice_total)
    models = None
    if pending:
        models = load_runtime_models(
            student_model_path=config.student_model_path,
            student_device=config.student_device,
            dtype=config.dtype,
            teacher_model_path=config.teacher_model_path if config.features.needs_teacher else None,
            teacher_device=config.teacher_device,
        )
        expected_tokenizer_hash = str(provenance.get("k32_tokenizer_hash") or "")
        if not expected_tokenizer_hash or models.tokenizer_hash != expected_tokenizer_hash:
            raise ValueError(
                "current model tokenizer differs from immutable K=32/proposal token-ID evidence"
            )
        import torch
        import transformers

        manifest["runtime"].update({
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device_count": int(torch.cuda.device_count()),
            "tokenizer_hash": models.tokenizer_hash,
            "backend": f"transformers-{transformers.__version__}",
        })
        _write_json_atomic(manifest_path, manifest)
        for index, item in enumerate(pending, start=1):
            record = _build_record(item, config, models)
            result_path = result_dir / f"{_safe_name(record.trajectory_id)}.json"
            _write_json_atomic(result_path, record.to_dict())
            print(f"[{index}/{len(pending)}] completed {record.trajectory_id}", flush=True)

    if config.work_slice_total > 1 and config.work_slice_index != config.work_slice_total - 1:
        # Non-final slices only contribute trajectory results; the final
        # slice owns the shared summary/manifest finalization.
        print(f"[slice {config.work_slice_index}/{config.work_slice_total}] done, summary owned by slice {config.work_slice_total - 1}", flush=True)
        return {"slice_index": config.work_slice_index, "slice_total": config.work_slice_total}

    if config.work_slice_total > 1:
        # The final slice must wait for every other slice's trajectory JSON
        # before writing the shared summary; otherwise it can finalize a
        # partial run while peers are still computing (2026-08-12 s1 incident).
        wait_seconds = int(os.environ.get("CAUSAL_PROBE_FINAL_WAIT_SECONDS", "21600"))
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            present = {p.name for p in result_dir.glob("*.json")}
            if all(
                f"{_safe_name(_work_id(item))}.json" in present
                for item in inputs
            ):
                break
            missing = [
                _work_id(item)
                for item in inputs
                if f"{_safe_name(_work_id(item))}.json" not in present
            ]
            print(
                f"[final slice] waiting for {len(missing)} peer result(s): {missing[:3]}",
                flush=True,
            )
            time.sleep(60)

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("*.json"))
    ]
    expected_ids = {_work_id(item) for item in inputs}
    records = [record for record in records if str(record.get("trajectory_id")) in expected_ids]
    report_paths = write_reports(output, records)
    completed_ids = {str(record["trajectory_id"]) for record in records}
    complete = completed_ids == expected_ids
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update({
        "complete": complete,
        "expected_work_units": len(expected_ids),
        "completed_work_units": len(completed_ids),
        "missing_work_ids": sorted(expected_ids - completed_ids),
        "report_paths": report_paths,
    })
    _write_json_atomic(output / "summary.json", summary)
    manifest["status"] = "completed" if complete else "partial"
    manifest["finished_at_unix"] = time.time()
    manifest["summary_sha256"] = sha256_file(output / "summary.json")
    _write_json_atomic(manifest_path, manifest)
    return summary


def summarize(input_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    dirty_shards: list[str] = []
    for directory in input_dirs:
        root = Path(directory).expanduser().resolve()
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_SCHEMA_VERSION or manifest.get("status") != "completed":
            raise ValueError(f"incomplete or incompatible causal-probe shard: {root}")
        if manifest.get("git_dirty") is not False:
            # Review P1-2: surface dirty-worktree provenance instead of refusing
            # to merge.  The 2026-08-08 probe ran every shard from the same
            # dirty worktree (ab7d054 + uncommitted local fixes), so all shards
            # share the same code identity and the merge remains interpretable.
            dirty_shards.append(str(root))
        manifests.append(manifest)
        for path in (root / "trajectory_results").glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            trajectory_id = str(record.get("trajectory_id") or "")
            if not trajectory_id or trajectory_id in records:
                raise ValueError(f"duplicate or missing trajectory_id while merging: {trajectory_id!r}")
            records[trajectory_id] = record
    if not manifests:
        raise ValueError("at least one completed shard is required")
    commits = {str(value.get("git_commit")) for value in manifests}
    if len(commits) != 1:
        raise ValueError("causal-probe shards use different repo commits")

    def normalized_config(value: Mapping[str, Any]) -> dict[str, Any]:
        config = dict(value.get("config") or {})
        config.pop("output_dir", None)
        config.pop("shard_index", None)
        return config

    reference_config = normalized_config(manifests[0])
    if any(normalized_config(value) != reference_config for value in manifests[1:]):
        raise ValueError("causal-probe shard configs differ")
    num_shards = int(reference_config.get("num_shards") or 1)
    shard_indices = sorted(int((value.get("config") or {}).get("shard_index") or 0) for value in manifests)
    if shard_indices != list(range(num_shards)):
        raise ValueError(f"expected shard indices 0..{num_shards - 1}, got {shard_indices}")
    provenance_hash_keys = (
        "k32_validation_sha256",
        "k32_rollouts_sha256",
        "k32_support_summary_sha256",
        "cohort_sha256",
        "retained_proposals_sha256",
    )
    reference_hashes = {
        key: (manifests[0].get("provenance") or {}).get(key)
        for key in provenance_hash_keys
    }
    for manifest in manifests[1:]:
        current = {key: (manifest.get("provenance") or {}).get(key) for key in provenance_hash_keys}
        if current != reference_hashes:
            raise ValueError("causal-probe shards use different immutable inputs")
    expected_work_ids = {
        str(work_id)
        for manifest in manifests
        for work_id in (manifest.get("provenance") or {}).get("expected_work_ids", ())
    }
    if expected_work_ids and set(records) != expected_work_ids:
        raise ValueError("merged records do not exactly cover shard work IDs")
    paths = write_reports(output_dir, [records[key] for key in sorted(records)])
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    summary["merged_input_dirs"] = [str(Path(value).expanduser().resolve()) for value in input_dirs]
    summary["report_paths"] = paths
    _write_json_atomic(Path(paths["summary_json"]), summary)
    merged_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed",
        "git_commit": next(iter(commits)),
        "git_dirty": bool(dirty_shards),
        "dirty_shards": sorted(dirty_shards),
        "config": reference_config,
        "input_hashes": reference_hashes,
        "input_shards": [str(Path(value).expanduser().resolve()) for value in input_dirs],
        "record_count": len(records),
        "summary_sha256": sha256_file(Path(paths["summary_json"])),
    }
    _write_json_atomic(Path(output_dir).expanduser().resolve() / "run_manifest.json", merged_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--k32-run-dir")
        sub.add_argument("--cohort-dir")
        sub.add_argument("--proposal-dir")
        sub.add_argument("--output-dir")
        sub.add_argument("--student-device")
        sub.add_argument("--teacher-device")
        sub.add_argument("--max-prompts", type=int)
        sub.add_argument("--continuation-k", type=int)
        sub.add_argument("--relay-k", type=int)
        sub.add_argument("--transport-k", type=int)
        sub.add_argument("--answer-k", type=int)
        sub.add_argument("--max-continuation-tokens", type=int)
        sub.add_argument("--max-answer-tokens", type=int)
        sub.add_argument("--relay-lengths", nargs="+", type=int)
        sub.add_argument("--shard-index", type=int, default=0)
        sub.add_argument("--num-shards", type=int, default=1)
        sub.add_argument("--work-slice-index", type=int, default=0)
        sub.add_argument("--work-slice-total", type=int, default=1)
        if command == "preflight":
            sub.add_argument("--load-tokenizers", action="store_true")
    merge = subparsers.add_parser("summarize")
    merge.add_argument("--input-dir", action="append", required=True)
    merge.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summarize":
        print(json.dumps(summarize(args.input_dir, args.output_dir), indent=2, sort_keys=True))
        return 0
    if (
        getattr(args, "work_slice_index", 0) != 0
        or getattr(args, "work_slice_total", 1) != 1
    ) and args.command not in ("run", "preflight"):
        parser.error("--work-slice-index/--work-slice-total only apply to run/preflight")
    config = load_config(args.config, args)
    if args.command == "preflight":
        result = preflight(config, load_tokenizers=args.load_tokenizers)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if "slice_index" in result:
        return 0
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
