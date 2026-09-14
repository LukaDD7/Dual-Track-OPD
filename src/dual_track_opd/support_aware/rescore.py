"""Append-only exact-token migration for historical support-aware runs.

The source run is treated as immutable evidence.  This module reuses its raw
student rollout IDs and cohort, but recomputes student/teacher forced scores,
verification, terminal-token policy, prompt-level statistics, and gates into a
new atomically-created output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .diagnostic import (
    _build_summary,
    _compute_ranking_metrics,
    _get_git_commit,
    _get_git_dirty,
    build_exact_scored_record,
    build_prompt,
    extract_image,
    generation_record_from_token_ids,
    load_and_select_prompts,
    save_image_for_teacher,
)
from .reporter import (
    DiagnosticRunMeta,
    write_prompt_support_summary_jsonl,
    write_resolved_config_yaml,
    write_rollouts_jsonl,
    write_run_manifest,
    write_selected_prompts_jsonl,
    write_summary_json,
)
from .scorer import StudentScorer, StudentScorerConfig, TeacherScorer, TeacherScorerConfig
from .support_state import classify_support_state
from .verifier import verify_answer


@dataclass(frozen=True)
class RescoreConfig:
    source_run: str
    output_dir: str
    student_model: str
    teacher_url: str = "http://127.0.0.1:18080"
    device: str = "cuda"
    dtype: str = "bfloat16"
    bootstrap_seed: int = 42
    bootstrap_resamples: int = 10_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _source_snapshot(source: Path) -> dict[str, str]:
    required = (
        "rollouts.jsonl",
        "prompt_support_summary.jsonl",
        "selected_prompts.jsonl",
        "summary.json",
        "run_manifest.json",
        "resolved_config.yaml",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"{source}: incomplete source run; missing {missing}")
    return {name: _sha256_file(source / name) for name in required}


def _int_config(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value in (None, "None", ""):
        return default
    return int(value)


def _validate_source_cohort(
    rollouts: Sequence[Mapping[str, Any]],
    prompt_summaries: Sequence[Mapping[str, Any]],
    selected_prompts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], int, bool]:
    selected_uids = [str(row.get("sample_uid") or "") for row in selected_prompts]
    if any(not uid for uid in selected_uids) or len(selected_uids) != len(set(selected_uids)):
        raise ValueError("source selected prompts contain empty or duplicate UIDs")
    summary_uids = [str(row.get("sample_uid") or "") for row in prompt_summaries]
    if len(summary_uids) != len(selected_uids) or set(summary_uids) != set(selected_uids):
        raise ValueError("source prompt summaries do not match selected prompts")
    k_values = {int(row.get("K", -1)) for row in prompt_summaries}
    if len(k_values) != 1 or next(iter(k_values)) <= 0:
        raise ValueError(f"source has invalid/inconsistent K: {k_values}")
    K = next(iter(k_values))
    expected = {
        (uid, is_greedy, rollout_id)
        for uid in selected_uids
        for is_greedy, rollout_id in (
            [(True, 0)] + [(False, rollout_id) for rollout_id in range(1, K + 1)]
        )
    }
    actual: set[tuple[str, bool, int]] = set()
    hashes_original = True
    for row in rollouts:
        key = (
            str(row.get("sample_uid") or ""),
            bool(row.get("is_greedy")),
            int(row.get("rollout_id", -1)),
        )
        if key in actual:
            raise ValueError(f"source contains duplicate rollout key {key}")
        actual.add(key)
        response_ids = tuple(int(token_id) for token_id in row.get("response_token_ids") or ())
        if not response_ids:
            raise ValueError(f"source rollout {key} has no raw response token IDs")
        expected_hash = hash_token_ids(response_ids)
        stored_hash = row.get("response_token_hash")
        if stored_hash is None:
            hashes_original = False
        elif stored_hash != expected_hash:
            raise ValueError(f"source rollout {key} has a corrupt response token hash")
    if actual != expected:
        raise ValueError(
            "source rollout cohort is incomplete; "
            f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
        )
    return selected_uids, K, hashes_original


def _prompt_input_ids(processor, *, image, prompt_text: str) -> tuple[int, ...]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = processor(
        text=[chat_text], images=[image], return_tensors="pt"
    )
    return tuple(int(token_id) for token_id in encoded["input_ids"][0].tolist())


def _resolved_gate_config(source_summary: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "malformed_rate_max": 0.10,
        "duplicate_rate_max": 0.25,
        "min_correct_tails": 20,
        "teacher_gap_auc_min": 0.60,
        "correct_tail_rank1_above_random": True,
        "truncation_rate_max": 0.15,
    }
    return {**defaults, **dict(source_summary.get("gate_config") or {})}


def rescore_existing(
    config: RescoreConfig,
    *,
    model=None,
    processor=None,
    teacher: TeacherScorer | None = None,
) -> dict[str, Any]:
    """Re-score one historical run into a new directory without mutating it."""

    source = Path(os.path.expandvars(config.source_run)).resolve()
    output = Path(os.path.expandvars(config.output_dir)).resolve()
    if source == output or source in output.parents:
        raise ValueError("rescore output must not be the source directory or inside it")
    if output.exists():
        raise FileExistsError(f"rescore output already exists: {output}")
    source_hashes_before = _source_snapshot(source)
    source_rollouts = _read_jsonl(source / "rollouts.jsonl")
    source_prompt_summaries = _read_jsonl(source / "prompt_support_summary.jsonl")
    source_selected = _read_jsonl(source / "selected_prompts.jsonl")
    source_summary = _read_json(source / "summary.json")
    source_manifest = _read_json(source / "run_manifest.json")
    source_config = _load_yaml(source / "resolved_config.yaml")
    selected_uids, K, response_hashes_original = _validate_source_cohort(
        source_rollouts, source_prompt_summaries, source_selected
    )

    dataset_path = str(
        (source_summary.get("selection_manifest") or {}).get("dataset_path") or ""
    )
    if not dataset_path:
        raise ValueError("source summary does not record the dataset path")
    full_selection_count = _int_config(source_config, "num_prompts", len(selected_uids))
    dataset_rows, selection_manifest = load_and_select_prompts(
        dataset_path, full_selection_count
    )
    dataset_by_uid = {str(row["sample_uid"]): row for row in dataset_rows}
    missing_uids = sorted(set(selected_uids) - set(dataset_by_uid))
    if missing_uids:
        raise ValueError(f"source UIDs are absent from the recorded dataset: {missing_uids[:10]}")
    source_dataset_hash = str(
        (source_summary.get("selection_manifest") or {}).get("dataset_sha256") or ""
    )
    if source_dataset_hash and source_dataset_hash != selection_manifest["dataset_sha256"]:
        raise ValueError("recorded dataset content hash no longer matches the source dataset")

    if (model is None) != (processor is None):
        raise ValueError("model and processor must be provided together")
    student_path = os.path.expandvars(config.student_model.removeprefix("hf:"))
    if model is None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(student_path)
        torch_dtype = (
            getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
        )
        model = AutoModelForImageTextToText.from_pretrained(
            student_path, torch_dtype=torch_dtype, device_map="auto"
        )
    model.eval()
    student_hash = tokenizer_fingerprint(processor.tokenizer)
    student_scorer = StudentScorer(
        StudentScorerConfig(
            model_path=config.student_model,
            device=config.device,
            dtype=config.dtype,
        ),
        model=model,
        processor=processor,
    )
    teacher = teacher or TeacherScorer(TeacherScorerConfig(
        base_url=config.teacher_url,
        expected_tokenizer_hash=student_hash,
    ))
    if not teacher.health():
        raise RuntimeError(f"teacher service at {config.teacher_url} is not healthy")
    if teacher.tokenizer_hash != student_hash:
        raise ValueError("student and teacher tokenizer hashes do not match")

    source_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in source_rollouts:
        source_by_uid.setdefault(str(row["sample_uid"]), []).append(row)
    for rows in source_by_uid.values():
        rows.sort(key=lambda row: (0 if row.get("is_greedy") else 1, int(row["rollout_id"])))

    start_time = time.time()
    run_id = output.name
    temp_images = Path(tempfile.mkdtemp(prefix="support_aware_rescore_images_"))
    rescored_rollouts: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []
    original_prompt_hash_count = 0
    malformed_count = 0
    non_finite_count = 0
    try:
        for prompt_index, uid in enumerate(selected_uids):
            dataset_row = dataset_by_uid[uid]
            image = extract_image(dataset_row).convert("RGB")
            try:
                image_path = save_image_for_teacher(image, temp_images)
                question = str(dataset_row.get("question") or "").strip()
                gold_answer = dataset_row.get("answer")
                source_rows = source_by_uid[uid]
                versions = {
                    str(row.get("prompt_version") or source_config.get("response_format") or "legacy_answer")
                    for row in source_rows
                }
                if len(versions) != 1:
                    raise ValueError(f"{uid}: inconsistent source prompt versions {versions}")
                prompt_version = next(iter(versions))
                prompt_text = build_prompt(question, prompt_version)
                prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
                prompt_ids = _prompt_input_ids(
                    processor, image=image, prompt_text=prompt_text
                )
                prompt_token_hash = hash_token_ids(prompt_ids)
                for row in source_rows:
                    old_prompt_hash = row.get("prompt_token_hash")
                    if old_prompt_hash:
                        original_prompt_hash_count += 1
                        if old_prompt_hash != prompt_token_hash:
                            raise ValueError(
                                f"{uid}: reconstructed prompt token hash differs from source"
                            )

                response_ids_list = [
                    tuple(int(token_id) for token_id in row["response_token_ids"])
                    for row in source_rows
                ]
                display_texts = [
                    str(row.get("response_text_display") or row.get("response_text") or "")
                    for row in source_rows
                ]
                teacher_results = teacher.score_batch(
                    request_ids=[
                        f"{uid}:greedy" if row.get("is_greedy")
                        else f"{uid}:rollout-{row['rollout_id']}"
                        for row in source_rows
                    ],
                    questions=[question] * len(source_rows),
                    image_paths=[image_path] * len(source_rows),
                    prompt_texts=[prompt_text] * len(source_rows),
                    response_token_ids_list=response_ids_list,
                    response_texts=display_texts,
                )
                student_results = student_scorer.score_batch(
                    questions=[question] * len(source_rows),
                    images=[image] * len(source_rows),
                    prompt_texts=[prompt_text] * len(source_rows),
                    response_texts=display_texts,
                    response_token_ids_list=response_ids_list,
                )
                if len(teacher_results) != len(source_rows) or len(student_results) != len(source_rows):
                    raise RuntimeError(f"{uid}: scorer result count mismatch")

                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                prompt_records: list[dict[str, Any]] = []
                for source_row, response_ids, display_text, teacher_result, student_result in zip(
                    source_rows,
                    response_ids_list,
                    display_texts,
                    teacher_results,
                    student_results,
                    strict=True,
                ):
                    max_new_tokens = int(source_row.get("max_new_tokens") or 4096)
                    generation = generation_record_from_token_ids(
                        model=model,
                        tokenizer=processor.tokenizer,
                        response_token_ids=response_ids,
                        prompt_token_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        response_text_display=display_text,
                    )
                    verdict = verify_answer(display_text, gold_answer)
                    base_record = {
                        "run_id": run_id,
                        "source_run_id": source_row.get("run_id") or source_manifest.get("run_id"),
                        "sample_uid": uid,
                        "source_index": source_row.get("source_index", prompt_index),
                        "split": source_row.get("split", "train"),
                        "rollout_id": int(source_row["rollout_id"]),
                        "is_greedy": bool(source_row.get("is_greedy")),
                        "generation_seed": source_row.get("generation_seed"),
                        "temperature": source_row.get("temperature"),
                        "top_p": source_row.get("top_p"),
                        "max_new_tokens": max_new_tokens,
                        "prompt_hash": prompt_hash,
                        "prompt_token_hash": prompt_token_hash,
                        "prompt_version": prompt_version,
                        "image_hash": image_hash,
                        "gold_answer": gold_answer,
                        "teacher_model_id": teacher.model_id,
                        "student_model_path": config.student_model,
                        "tokenizer_hash": student_hash,
                        "student_tokenizer_hash": student_hash,
                        "teacher_tokenizer_hash": teacher.tokenizer_hash,
                        "migration_source_response_token_hash": generation.response_token_hash,
                        "migration_status": "rescored_exact_raw_ids_v1",
                    }
                    record = build_exact_scored_record(
                        base_record=base_record,
                        generation=generation,
                        verdict=verdict,
                        teacher_result=teacher_result,
                        student_result=student_result,
                    )
                    malformed_count += int(bool(record.get("malformed")))
                    if not math.isfinite(float(record["teacher_mean_logp"])) or not math.isfinite(
                        float(record["student_mean_logp"])
                    ):
                        non_finite_count += 1
                    prompt_records.append(record)
                    rescored_rollouts.append(record)

                stochastic = [row for row in prompt_records if not row["is_greedy"]]
                correct_count = sum(row.get("correct") is True for row in stochastic)
                greedy = next(row for row in prompt_records if row["is_greedy"])
                response_hashes = [str(row["response_token_hash"]) for row in stochastic]
                unique_count = len(set(response_hashes))
                support_state = classify_support_state(
                    greedy_correct=greedy.get("correct"),
                    correct_count=correct_count,
                    K=K,
                )
                prompt_summaries.append({
                    "sample_uid": uid,
                    "K": K,
                    "correct_count": correct_count,
                    "greedy_correct": greedy.get("correct"),
                    "support_state": support_state.value,
                    "unique_response_count": unique_count,
                    "duplicate_rollout_rate": 1.0 - unique_count / K,
                    **_compute_ranking_metrics(stochastic),
                })
            finally:
                image.close()
    finally:
        shutil.rmtree(temp_images, ignore_errors=True)

    source_hashes_after = _source_snapshot(source)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source run changed during rescoring; refusing to publish output")

    selection_manifest = {
        **selection_manifest,
        **dict(source_summary.get("selection_manifest") or {}),
        "dataset_sha256": selection_manifest["dataset_sha256"],
        "selection_total_rows": full_selection_count,
        "selected_rows": len(selected_uids),
        "source_run_path": str(source),
    }
    gate_config = _resolved_gate_config(source_summary)
    git_commit = _get_git_commit()
    git_dirty = _get_git_dirty()
    mode = str(source_summary.get("mode") or "full")
    summary = _build_summary(
        run_id=run_id,
        mode=mode,
        num_prompts=len(selected_uids),
        K=K,
        all_rollouts=rescored_rollouts,
        prompt_summaries=prompt_summaries,
        malformed_count=malformed_count,
        missing_image=0,
        non_finite_count=non_finite_count,
        student_hash=student_hash,
        teacher_hash=teacher.tokenizer_hash,
        teacher_model_id=teacher.model_id,
        student_model_path=config.student_model,
        selection_manifest=selection_manifest,
        git_commit=git_commit,
        git_dirty=git_dirty,
        gate_config=gate_config,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
        shard_completeness_rate=1.0,
        verbose=True,
    )

    teacher_metadata = teacher.metadata
    model_config = getattr(model, "config", None)
    student_revision = str(getattr(model_config, "_commit_hash", None) or "unavailable")
    chat_template = str(getattr(processor.tokenizer, "chat_template", "") or "")
    source_prompt_hash_total = len(source_rollouts)
    prompt_hash_status = (
        "original_validated"
        if original_prompt_hash_count == source_prompt_hash_total
        else "backfilled"
    )
    resolved_config = {
        **source_config,
        "source_run": str(source),
        "output_dir": str(output),
        "student_model": config.student_model,
        "teacher_url": config.teacher_url,
        "bootstrap_seed": config.bootstrap_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "migration_version": "rescored_exact_raw_ids_v1",
    }
    extra_manifest = {
        "migration_version": "rescored_exact_raw_ids_v1",
        "source_run_path": str(source),
        "source_rollouts_sha256": source_hashes_before["rollouts.jsonl"],
        "source_run_manifest_sha256": source_hashes_before["run_manifest.json"],
        "source_artifact_sha256": source_hashes_before,
        "source_git_commit": source_manifest.get("git_commit", "unavailable"),
        "source_git_dirty": source_manifest.get("git_dirty", "unavailable"),
        "rescore_git_commit": git_commit,
        "rescore_git_dirty": git_dirty,
        "student_model_id": str(getattr(model_config, "_name_or_path", None) or config.student_model),
        "student_model_revision": student_revision,
        "student_tokenizer_hash": student_hash,
        "student_chat_template_hash": hashlib.sha256(chat_template.encode()).hexdigest(),
        "teacher_model_id": teacher_metadata.model_id,
        "teacher_model_revision": teacher_metadata.git_revision,
        "teacher_tokenizer_hash": teacher_metadata.tokenizer_hash,
        "teacher_protocol_version": teacher_metadata.protocol_version,
        "prompt_versions": sorted({str(row["prompt_version"]) for row in rescored_rollouts}),
        "prompt_token_hash_status": prompt_hash_status,
        "response_token_hash_status": (
            "original_validated" if response_hashes_original else "backfilled"
        ),
        "verifier_version": "canonical_explicit_final_answer_v1",
        "terminal_token_policy": "exclude_only_observed_final_configured_stop_from_content",
        "fields_reused": [
            "selected_prompt_cohort",
            "response_token_ids",
            "display_text",
            "generation_seed",
            "sampling_parameters",
        ],
        "fields_recomputed": [
            "prompt_token_hash",
            "student_forced_scores",
            "teacher_forced_scores",
            "canonical_verdict",
            "terminal_metadata",
            "support_state",
            "prompt_statistics",
            "acceptance_gates",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_rollouts_jsonl(temp_output, rescored_rollouts)
        write_prompt_support_summary_jsonl(temp_output, prompt_summaries)
        write_selected_prompts_jsonl(temp_output, source_selected)
        write_summary_json(temp_output, summary)
        write_resolved_config_yaml(temp_output, resolved_config)
        write_run_manifest(temp_output, DiagnosticRunMeta(
            run_id=run_id,
            output_dir=output,
            config=resolved_config,
            git_commit=git_commit,
            git_dirty=git_dirty,
            num_prompts=len(selected_uids),
            rollouts_per_prompt=K,
            seed=_int_config(source_config, "seed", 42),
            start_time=start_time,
            end_time=time.time(),
            exit_status=(
                "PASS"
                if all(passed for _, passed, _ in summary["acceptance_gates"])
                else "GATE_FAIL"
            ),
            extra=extra_manifest,
        ))
        if _source_snapshot(source) != source_hashes_before:
            raise RuntimeError("source run changed before publish; refusing output")
        temp_output.rename(output)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append-only exact-token rescoring of a support-aware run"
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:18080")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = rescore_existing(RescoreConfig(
            source_run=args.source_run,
            output_dir=args.output_dir,
            student_model=args.student_model,
            teacher_url=args.teacher_url,
            device=args.device,
            dtype=args.dtype,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        ))
    except Exception as exc:
        print(f"ERROR: exact-token rescoring failed: {exc}")
        return 1
    print(f"Rescored output: {args.output_dir}")
    return 0 if all(passed for _, passed, _ in summary["acceptance_gates"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
