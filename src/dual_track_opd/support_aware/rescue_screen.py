"""Adaptive answer-free prefix rescue screen (frozen policy, 32B teacher).

Two-stage screen from the STP-OPD plan (handoff 2026-08-06, section 4.2):
1) generate M verified teacher proposals and keep the first correct one;
2) generate unaided student rollouts to source a long wrong-student prefix;
3) Stage-1 probe h=128/256 at K=4 with provisional promote (mean lift >= 0.15
and P >= 0.80 against both controls); 4) if h128 promotes also probe h64, and
if neither promotes but the teacher had a success, probe h512; 5) Stage-2
strict K=8 at the candidate horizon (lift >= 0.20, P >= 0.90).

Nothing trains.  The adaptive state machine is a pure function so it can be
unit-tested without GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .diagnostic import build_prompt, extract_image, generate_response
from .prefix_intervention import (
    _candidate_prefix,
    _load_student,
    generate_continuation,
    posterior_lift_probability,
)
from .verifier import verify_answer


SCHEMA_VERSION = "support-aware-rescue-screen-v1"
STAGE1_HORIZONS = (128, 256)
ALL_HORIZONS = (64, 128, 256, 512)
PROVISIONAL_MIN_LIFT = 0.15
PROVISIONAL_MIN_PROBABILITY = 0.80
STRICT_MIN_LIFT = 0.20
STRICT_MIN_PROBABILITY = 0.90


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


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


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    return int(device.split(":", 1)[1]) if ":" in device else 0


@dataclass(frozen=True)
class RescueScreenConfig:
    cohort_path: str
    teacher_model_path: str
    student_model_path: str
    output_dir: str
    response_format: str = "legacy_answer"
    proposals_per_prompt: int = 4
    proposal_temperature: float = 0.7
    proposal_top_p: float = 0.95
    proposal_max_new_tokens: int = 4096
    stage1_k: int = 4
    stage2_k: int = 8
    wrong_source_k: int = 16
    max_continuation_tokens: int = 2048
    continuation_temperature: float = 0.7
    continuation_top_p: float = 0.95
    seed: int = 20260806
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"
    dtype: str = "bfloat16"
    max_prompts: int | None = None
    num_shards: int = 1
    shard_index: int = 0
    posterior_draws: int = 20_000

    def validate(self) -> None:
        if self.proposals_per_prompt <= 0 or self.stage1_k <= 0 or self.stage2_k <= 0:
            raise ValueError("generation counts must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("invalid shard assignment")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")


def load_config(path: str | Path, args: argparse.Namespace) -> RescueScreenConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = raw.get("data", {})
    model = raw.get("model", {})
    screen = raw.get("screen", {})
    config = RescueScreenConfig(
        cohort_path=str(args.cohort_path or data.get("cohort_path") or ""),
        teacher_model_path=str(args.teacher_model_path or model.get("teacher") or ""),
        student_model_path=str(args.student_model_path or model.get("student") or ""),
        output_dir=str(args.output_dir or raw.get("output", {}).get("dir") or ""),
        response_format=str(args.response_format or screen.get("response_format", "legacy_answer")),
        proposals_per_prompt=int(args.proposals_per_prompt or screen.get("proposals_per_prompt", 4)),
        proposal_temperature=float(screen.get("proposal_temperature", 0.7)),
        proposal_top_p=float(screen.get("proposal_top_p", 0.95)),
        proposal_max_new_tokens=int(screen.get("proposal_max_new_tokens", 4096)),
        stage1_k=int(args.stage1_k or screen.get("stage1_k", 4)),
        stage2_k=int(args.stage2_k or screen.get("stage2_k", 8)),
        wrong_source_k=int(args.wrong_source_k or screen.get("wrong_source_k", 16)),
        max_continuation_tokens=int(
            args.max_continuation_tokens or screen.get("max_continuation_tokens", 2048)
        ),
        continuation_temperature=float(screen.get("continuation_temperature", 0.7)),
        continuation_top_p=float(screen.get("continuation_top_p", 0.95)),
        seed=int(args.seed or screen.get("seed", 20260806)),
        teacher_device=str(args.teacher_device or screen.get("teacher_device", "cuda:0")),
        student_device=str(args.student_device or screen.get("student_device", "cuda:1")),
        dtype=str(screen.get("dtype", "bfloat16")),
        max_prompts=args.max_prompts,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        posterior_draws=int(screen.get("posterior_draws", 20_000)),
    )
    config = RescueScreenConfig(**{
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in asdict(config).items()
    })
    config.validate()
    return config


def select_cohort_records(config: RescueScreenConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(config.cohort_path)
    required = {"sample_uid", "question", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cohort missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort contains duplicate sample_uid")
    available = sorted(frame.index, key=_selection_key)
    start = len(available) * config.shard_index // config.num_shards
    end = len(available) * (config.shard_index + 1) // config.num_shards
    uids = available[start:end]
    if config.max_prompts is not None:
        uids = uids[: config.max_prompts]
    records = []
    for uid in uids:
        row = frame.loc[uid].to_dict()
        row["sample_uid"] = uid
        records.append(row)
    return records, {
        "all_selected_uids": available,
        "expected_uids": uids,
        "shard_start": start,
        "shard_end": end,
        "cohort_path": config.cohort_path,
    }


def decide_stage1(
    posteriors: Mapping[int, Mapping[str, dict[str, float]]],
    teacher_success: Mapping[int, bool],
) -> tuple[str, int | None, list[str]]:
    """Pure adaptive-horizon state machine (testable without GPUs).

    posteriors[h][arm] has keys mean_lift / probability for arm in
    {"vs_wrong", "vs_unaided"} (teacher minus control).
    """

    def promote(h: int) -> bool:
        row = posteriors.get(h)
        if row is None:
            return False
        for key in ("vs_wrong", "vs_unaided"):
            value = row.get(key)
            if (
                value is None
                or value.get("mean_lift") is None
                or value.get("probability") is None
            ):
                return False
            if value["mean_lift"] < PROVISIONAL_MIN_LIFT:
                return False
            if value["probability"] < PROVISIONAL_MIN_PROBABILITY:
                return False
        return True

    notes: list[str] = []
    if promote(128):
        notes.append("stage1 h128 promoted")
        if promote(64):
            notes.append("h64 promoted -> candidate h64")
            return "candidate", 64, notes
        notes.append("h64 not promoted -> candidate h128")
        return "candidate", 128, notes
    if promote(256):
        notes.append("stage1 h256 promoted -> candidate h256")
        return "candidate", 256, notes
    if teacher_success.get(128) or teacher_success.get(256):
        if promote(512):
            notes.append("h512 promoted -> candidate h512")
            return "candidate", 512, notes
        notes.append("no stage1 promote; h512 not promoted")
    else:
        notes.append("no stage1 promote and no teacher success")
    return "rescue_negative", None, notes


def decide_stage2(
    posterior: Mapping[str, dict[str, float]],
) -> tuple[str, list[str]]:
    """Strict K=8 decision at the candidate horizon."""

    notes: list[str] = []
    for key in ("vs_wrong", "vs_unaided"):
        value = posterior.get(key)
        if (
            value is None
            or value.get("mean_lift") is None
            or value.get("probability") is None
        ):
            return "stage2_failed", notes + [f"{key} posterior missing"]
        if value["mean_lift"] < STRICT_MIN_LIFT:
            return "stage2_failed", notes + [
                f"{key} mean lift {value['mean_lift']:.3f} < {STRICT_MIN_LIFT}"
            ]
        if value["probability"] < STRICT_MIN_PROBABILITY:
            return "stage2_failed", notes + [
                f"{key} P {value['probability']:.3f} < {STRICT_MIN_PROBABILITY}"
            ]
    return "rescue_positive", notes + ["strict K=8 rescue rule satisfied"]


def _posterior_row(
    teacher: dict[str, Any],
    control: dict[str, Any],
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    mean_lift, probability = posterior_lift_probability(
        teacher["correct_count"],
        control["correct_count"],
        teacher["K"],
        seed=seed,
        draws=draws,
    )
    return {"mean_lift": mean_lift, "probability": probability}


def _run_arm(
    model,
    processor,
    *,
    image,
    prompt_text: str,
    gold_answer: Any,
    prefix_ids: tuple[int, ...] | None,
    seeds: Sequence[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> dict[str, Any]:
    correct = 0
    rollouts: list[dict[str, Any]] = []
    for seed in seeds:
        generation = generate_continuation(
            model,
            processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=list(prefix_ids) if prefix_ids else [],
            max_continuation_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=int(seed),
            device=device,
        )
        verdict = verify_answer(generation["generation"].response_text_display, gold_answer)
        is_correct = bool(verdict.get("correct") is True)
        correct += int(is_correct)
        rollouts.append({
            "seed": int(seed),
            "prefix_token_hash": hash_token_ids(prefix_ids) if prefix_ids else "none",
            "continuation_token_hash": generation["continuation_token_hash"],
            "correct": is_correct,
            "malformed": bool(verdict.get("malformed")),
            "response_token_count": len(generation["generation"].response_token_ids_raw),
            "finish_reason": generation["generation"].finish_reason,
        })
    return {"correct_count": correct, "K": len(seeds), "rollouts": rollouts}


def _screen_wrong_source(
    model,
    processor,
    *,
    image,
    prompt_text: str,
    gold_answer: Any,
    seeds: Sequence[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[tuple[int, ...] | None, dict[str, Any]]:
    """Generate unaided rollouts and return (wrong_ids, summary) for the
    wrong-prefix control.  The wrong source is the first verifier-wrong
    rollout long enough to cover the largest horizon; longer is preferred."""

    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        generation = generate_continuation(
            model,
            processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=[],
            max_continuation_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=int(seed),
            device=device,
        )
        record = generation["generation"]
        verdict = verify_answer(record.response_text_display, gold_answer)
        candidates.append({
            "seed": int(seed),
            "correct": bool(verdict.get("correct") is True),
            "malformed": bool(verdict.get("malformed")),
            "response_token_count": len(record.response_token_ids_raw),
            "response_token_ids_raw": tuple(int(value) for value in record.response_token_ids_raw),
        })
    summary = {
        "generated_count": len(candidates),
        "correct_count": sum(1 for item in candidates if item["correct"]),
    }
    eligible = [
        item for item in candidates
        if not item["correct"] and item["response_token_count"] >= max(ALL_HORIZONS)
    ]
    if not eligible:
        return None, summary
    eligible.sort(key=lambda item: -item["response_token_count"])
    return eligible[0]["response_token_ids_raw"], summary


def screen_prompt(
    record: Mapping[str, Any],
    *,
    teacher,
    teacher_processor,
    student,
    student_processor,
    config: RescueScreenConfig,
) -> dict[str, Any]:
    import torch

    uid = str(record["sample_uid"])
    question = str(record.get("question") or "").strip()
    gold_answer = record.get("answer")
    image = extract_image(record).convert("RGB")
    prompt_text = build_prompt(question, config.response_format)
    teacher_cuda_index = _cuda_index(config.teacher_device)
    student_cuda_index = _cuda_index(config.student_device)

    result: dict[str, Any] = {
        "sample_uid": uid,
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "minimal_horizon": None,
        "proposal_count": 0,
        "correct_proposal_count": 0,
        "proposal_token_count": 0,
        "continuation_token_count": 0,
        "notes": [],
        "posteriors": {},
    }

    # 1. Verified teacher proposals.
    proposals: list[tuple[int, int, Any, dict[str, Any]]] = []
    for proposal_id in range(1, config.proposals_per_prompt + 1):
        if teacher_cuda_index is not None:
            torch.cuda.set_device(teacher_cuda_index)
        generation_seed = config.seed + int(_selection_key(uid)[:8], 16) + proposal_id
        generation = generate_response(
            teacher,
            teacher_processor,
            question,
            image,
            prompt_text,
            temperature=config.proposal_temperature,
            top_p=config.proposal_top_p,
            max_new_tokens=config.proposal_max_new_tokens,
            seed=generation_seed,
            device=config.teacher_device,
        )
        verdict = verify_answer(generation.response_text_display, gold_answer)
        proposals.append((proposal_id, generation_seed, generation, verdict))
        result["proposal_token_count"] += len(generation.response_token_ids_raw)
    result["proposal_count"] = len(proposals)
    correct_proposals = [item for item in proposals if item[3].get("correct") is True]
    result["correct_proposal_count"] = len(correct_proposals)
    if not correct_proposals:
        result["status"] = "proposal_unavailable"
        result["notes"].append("no verified correct teacher proposal")
        return result
    teacher_ids = tuple(int(value) for value in correct_proposals[0][2].response_token_ids_raw)

    # 2. Wrong-prefix source from unaided student rollouts.
    if student_cuda_index is not None:
        torch.cuda.set_device(student_cuda_index)
    wrong_ids, wrong_source_summary = _screen_wrong_source(
        student,
        student_processor,
        image=image,
        prompt_text=prompt_text,
        gold_answer=gold_answer,
        seeds=range(config.wrong_source_k),
        max_tokens=config.max_continuation_tokens,
        temperature=config.continuation_temperature,
        top_p=config.continuation_top_p,
        device=config.student_device,
    )
    result["wrong_source"] = wrong_source_summary
    if wrong_ids is None:
        result["status"] = "wrong_prefix_unavailable"
        result["notes"].append("no long verifier-wrong unaided rollout for the wrong-prefix control")
        return result

    # 3. Adaptive two-stage probe.
    def probe(horizon: int, seeds: Sequence[int]) -> dict[str, Any]:
        teacher_prefix, teacher_reason, _ = _candidate_prefix(
            teacher_ids,
            horizon=horizon,
            tokenizer=student_processor.tokenizer,
            gold_answer=gold_answer,
        )
        if teacher_prefix is None:
            return {"skipped": teacher_reason or "teacher_prefix_unavailable"}
        wrong_prefix, wrong_reason, _ = _candidate_prefix(
            wrong_ids,
            horizon=horizon,
            tokenizer=student_processor.tokenizer,
            gold_answer=gold_answer,
        )
        arms: dict[str, Any] = {}
        for arm_name, prefix_ids in (
            ("teacher_prefix", teacher_prefix),
            ("wrong_prefix", wrong_prefix),
            ("unaided", None),
        ):
            if prefix_ids is None and arm_name != "unaided":
                arms[arm_name] = {"skipped": wrong_reason or "wrong_prefix_unavailable"}
                continue
            arms[arm_name] = _run_arm(
                student,
                student_processor,
                image=image,
                prompt_text=prompt_text,
                gold_answer=gold_answer,
                prefix_ids=prefix_ids,
                seeds=seeds,
                max_tokens=config.max_continuation_tokens,
                temperature=config.continuation_temperature,
                top_p=config.continuation_top_p,
                device=config.student_device,
            )
            result["continuation_token_count"] += int(
                sum(rollout["response_token_count"] for rollout in arms[arm_name]["rollouts"])
            )
        return arms

    def posterior_for(arms: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "vs_wrong": _posterior_row(
                arms["teacher_prefix"],
                arms["wrong_prefix"],
                seed=config.seed,
                draws=config.posterior_draws,
            ),
            "vs_unaided": _posterior_row(
                arms["teacher_prefix"],
                arms["unaided"],
                seed=config.seed,
                draws=config.posterior_draws,
            ),
        }

    stage1_seeds = list(range(config.stage1_k))
    posteriors: dict[int, dict[str, Any]] = {}
    teacher_success: dict[int, bool] = {}
    for horizon in STAGE1_HORIZONS:
        arms = probe(horizon, stage1_seeds)
        if "teacher_prefix" not in arms:
            posteriors[horizon] = {"skipped": arms.get("skipped")}
            teacher_success[horizon] = False
            continue
        posteriors[horizon] = posterior_for(arms)
        teacher_success[horizon] = arms["teacher_prefix"]["correct_count"] > 0

    decision, candidate, notes = decide_stage1(posteriors, teacher_success)
    result["notes"].extend(notes)
    for horizon in STAGE1_HORIZONS:
        result["posteriors"][str(horizon)] = posteriors.get(horizon)

    if decision != "candidate" or candidate is None:
        result["status"] = decision
        return result

    if candidate in (64, 512):
        arms = probe(candidate, stage1_seeds)
        if "teacher_prefix" in arms:
            result["posteriors"][str(candidate)] = posterior_for(arms)

    arms = probe(candidate, list(range(config.stage2_k)))
    if "teacher_prefix" not in arms:
        result["status"] = "stage2_failed"
        result["notes"].append("stage2 candidate prefix unavailable")
        return result
    stage2_posterior = posterior_for(arms)
    result["posteriors"][f"stage2_h{candidate}"] = stage2_posterior
    stage2_decision, stage2_notes = decide_stage2(stage2_posterior)
    result["notes"].extend(stage2_notes)
    if stage2_decision == "rescue_positive":
        result["status"] = "rescue_positive"
        result["minimal_horizon"] = candidate
    else:
        result["status"] = "stage2_failed"
    return result


class _StudentNamespace:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def run(config: RescueScreenConfig) -> dict[str, Any]:
    config.validate()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "prompt_results"
    result_dir.mkdir(exist_ok=True)
    records, provenance = select_cohort_records(config)
    expected_uids = list(provenance["expected_uids"])
    git_commit, git_dirty = _git_state()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config": json.loads(json.dumps(asdict(config))),
        "provenance": provenance,
        "started_at_unix": time.time(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    already_done = {
        str(_read_json(path)["sample_uid"])
        for path in result_dir.glob("*.json")
    }
    pending = [record for record in records if str(record["sample_uid"]) not in already_done]
    if pending:
        from transformers import AutoProcessor

        from dual_track_opd.fc_opd.teacher_transformers import TransformersTeacherScorer

        teacher = TransformersTeacherScorer(
            model_id=config.teacher_model_path,
            top_k=32,
            dtype=config.dtype,
            device=config.teacher_device,
        )
        student_processor = AutoProcessor.from_pretrained(config.student_model_path)
        if teacher.metadata.tokenizer_hash != tokenizer_fingerprint(student_processor.tokenizer):
            raise ValueError("teacher/student tokenizer mismatch; exact teacher token IDs "
                             "cannot be used as student prefixes")
        student, student_processor, _ = _load_student(
            _StudentNamespace(
                student_model_path=config.student_model_path,
                dtype=config.dtype,
                device=config.student_device,
            )
        )
        for record in pending:
            outcome = screen_prompt(
                record,
                teacher=teacher.model,
                teacher_processor=teacher.processor,
                student=student,
                student_processor=student_processor,
                config=config,
            )
            _write_json(result_dir / f"{outcome['sample_uid']}.json", outcome)
    return aggregate(output_dir, expected_uids, manifest)


def aggregate(output_dir: Path, expected_uids: Sequence[str], manifest: dict[str, Any]) -> dict[str, Any]:
    result_dir = output_dir / "prompt_results"
    by_uid: dict[str, dict[str, Any]] = {}
    for path in result_dir.glob("*.json"):
        value = _read_json(path)
        by_uid[str(value["sample_uid"])] = value
    completed = [uid for uid in expected_uids if uid in by_uid]
    statuses = [by_uid[uid].get("status") for uid in completed]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_prompt_count": len(expected_uids),
        "completed_prompt_count": len(completed),
        "complete": completed == list(expected_uids),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "rescue_positive_count": statuses.count("rescue_positive"),
        "minimal_horizons": {
            uid: by_uid[uid].get("minimal_horizon")
            for uid in completed
            if by_uid[uid].get("minimal_horizon") is not None
        },
        "completed_uids": completed,
        "missing_uids": [uid for uid in expected_uids if uid not in by_uid],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "screen_results.jsonl", [by_uid[uid] for uid in completed])
    manifest["status"] = "completed" if summary["complete"] else "partial"
    manifest["finished_at_unix"] = time.time()
    _write_json(output_dir / "run_manifest.json", manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the adaptive rescue screen")
    run_parser.add_argument("--config", default=None)
    run_parser.add_argument("--cohort-path", default=None)
    run_parser.add_argument("--teacher-model-path", default=None)
    run_parser.add_argument("--student-model-path", default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--response-format", default=None)
    run_parser.add_argument("--proposals-per-prompt", type=int, default=None)
    run_parser.add_argument("--stage1-k", type=int, default=None)
    run_parser.add_argument("--stage2-k", type=int, default=None)
    run_parser.add_argument("--wrong-source-k", type=int, default=None)
    run_parser.add_argument("--max-continuation-tokens", type=int, default=None)
    run_parser.add_argument("--teacher-device", default=None)
    run_parser.add_argument("--student-device", default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--max-prompts", type=int, default=None)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        return 2
    config = load_config(args.config, args)
    summary = run(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
