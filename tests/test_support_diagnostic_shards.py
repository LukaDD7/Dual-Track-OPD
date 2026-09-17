"""Strict validation tests for sharded diagnostic merging."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.support_aware.diagnostic import merge_shard_runs, slice_prompts
from dual_track_opd.support_aware.reporter import (
    DiagnosticRunMeta,
    write_prompt_support_summary_jsonl,
    write_resolved_config_yaml,
    write_rollouts_jsonl,
    write_run_manifest,
    write_selected_prompts_jsonl,
    write_summary_json,
)


def _fake_prompt(uid: str) -> dict[str, object]:
    return {
        "sample_uid": uid,
        "question": f"Question {uid}",
        "answer": "42",
        "images": {"array": [[0]]},
    }


def test_slice_prompts_bounds_and_manifest() -> None:
    prompts = [_fake_prompt(f"p{i}") for i in range(10)]

    sliced, manifest = slice_prompts(prompts, 2, 6)
    assert [p["sample_uid"] for p in sliced] == ["p2", "p3", "p4", "p5"]
    assert manifest["prompt_start"] == 2
    assert manifest["prompt_end"] == 6

    sliced_all, manifest_all = slice_prompts(prompts, 0, None)
    assert len(sliced_all) == 10
    assert manifest_all["prompt_end"] == 10


def _rollout(
    uid: str,
    *,
    is_greedy: bool,
    rollout_id: int,
    correct: bool,
    gap: float,
) -> dict[str, object]:
    response_ids = (3, 4)
    token_hash = hash_token_ids(response_ids)
    return {
        "run_id": "shard",
        "sample_uid": uid,
        "is_greedy": is_greedy,
        "rollout_id": rollout_id,
        "correct": correct,
        "teacher_mean_logp": -0.3 if correct else -0.6,
        "student_mean_logp": -0.4 if correct else -0.5,
        "teacher_gap": gap,
        "response_token_ids": response_ids,
        "response_token_hash": token_hash,
        "response_token_count": 2,
        "content_token_count": 1,
        "content_mask": (True, False),
        "finish_reason": "stop",
        "malformed": False,
        "exact_token_alignment": True,
        "prompt_token_hash": "prompt-hash",
        "prompt_version": "legacy_answer",
        "student_tokenizer_hash": "same-hash",
        "teacher_tokenizer_hash": "same-hash",
        "teacher_scored_token_hash": token_hash,
        "student_scored_token_hash": token_hash,
        "teacher_response_mask": (True, True),
        "student_response_mask": (True, True),
        "teacher_model_id": "teacher",
        "student_model_path": "student",
    }


def _prompt_rollouts(uid: str, n_correct: int, greedy_correct: bool):
    rows = [_rollout(uid, is_greedy=True, rollout_id=0, correct=greedy_correct, gap=0.0)]
    for rollout_id in range(1, 5):
        correct = rollout_id <= n_correct
        rows.append(_rollout(
            uid,
            is_greedy=False,
            rollout_id=rollout_id,
            correct=correct,
            gap=1.0 if correct else -1.0,
        ))
    return rows


def _summary_line(uid: str, state: str, n_correct: int) -> dict[str, object]:
    return {
        "sample_uid": uid,
        "K": 4,
        "correct_count": n_correct,
        "greedy_correct": bool(state == "exposed" and n_correct >= 2),
        "support_state": state,
        "unique_response_count": 4,
        "duplicate_rollout_rate": 0.0,
        "teacher_rank_of_best_correct": 1,
        "student_rank_of_best_correct": 1,
        "teacher_gap_rank_of_best_correct": 1,
        "teacher_top1_correct": True,
        "teacher_gap_top1_correct": True,
    }


def _selection_hash(uids: list[str]) -> str:
    order = sorted(uids, key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())
    return hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()


def _write_shard(
    path: Path,
    *,
    uids: list[str],
    states: list[tuple[str, int, bool]],
    start: int,
    end: int,
    full_selection_hash: str,
    num_prompts: int = 4,
) -> None:
    path.mkdir()
    rollouts = []
    prompt_summaries = []
    for uid, (state, correct_count, greedy_correct) in zip(uids, states, strict=True):
        rollouts.extend(_prompt_rollouts(uid, correct_count, greedy_correct))
        prompt_summaries.append(_summary_line(uid, state, correct_count))
    write_rollouts_jsonl(path, rollouts)
    write_prompt_support_summary_jsonl(path, prompt_summaries)
    write_selected_prompts_jsonl(path, [{"sample_uid": uid} for uid in uids])
    selection_manifest = {
        "dataset_path": "/dataset.parquet",
        "dataset_sha256": "dataset-hash",
        "valid_rows": 10,
        "selection_sha256": full_selection_hash,
    }
    write_summary_json(path, {
        "selection_manifest": selection_manifest,
        "scoring_policy": "exact_raw_ids_content_primary_terminal_separate_v1",
        "missing_image_count": 0,
        "gate_config": {},
        "bootstrap_seed": 42,
        "bootstrap_resamples": 100,
    })
    resolved = {
        "num_prompts": str(num_prompts),
        "rollouts_per_prompt": "4",
        "seed": "42",
        "response_format": "legacy_answer",
        "student_model_path": "student",
        "prompt_start": str(start),
        "prompt_end": str(end),
    }
    write_resolved_config_yaml(path, resolved)
    now = time.time()
    write_run_manifest(path, DiagnosticRunMeta(
        run_id=path.name,
        output_dir=path,
        config=resolved,
        git_commit="abc123",
        git_dirty=False,
        num_prompts=len(uids),
        rollouts_per_prompt=4,
        seed=42,
        start_time=now,
        end_time=now,
        exit_status="GATE_FAIL",
    ))


def _valid_shards(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    all_uids = sorted(
        ["p1", "p2", "p3", "p4"],
        key=lambda uid: hashlib.sha256(uid.encode()).hexdigest(),
    )
    selection_hash = _selection_hash(all_uids)
    _write_shard(
        shard_a,
        uids=all_uids[:2],
        states=[("exposed", 3, True), ("correct_tail", 1, False)],
        start=0,
        end=2,
        full_selection_hash=selection_hash,
    )
    _write_shard(
        shard_b,
        uids=all_uids[2:],
        states=[("no_correct_observed", 0, False), ("correct_tail", 1, False)],
        start=2,
        end=4,
        full_selection_hash=selection_hash,
    )
    return shard_a, shard_b


def test_merge_shard_runs_recomputes_summary_and_writes_atomically(tmp_path) -> None:
    shard_a, shard_b = _valid_shards(tmp_path)
    merged = tmp_path / "merged"
    summary = merge_shard_runs([str(shard_a), str(shard_b)], str(merged))

    assert summary["num_prompts"] == 4
    assert summary["total_rollouts"] == 20
    assert summary["greedy_accuracy"] == 0.25
    assert summary["correct_tail_count"] == 2
    assert summary["auc_teacher_gap_correct_vs_wrong"] == 1.0
    assert summary["tokenizer_match"] is True
    assert summary["shard_completeness_rate"] == 1.0
    assert summary["selection_manifest"]["merged_from"] == ["shard_a", "shard_b"]
    assert len(_read_lines(merged / "rollouts.jsonl")) == 20
    assert (merged / "run_manifest.json").exists()


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_merge_rejects_duplicate_rollout_without_creating_output(tmp_path) -> None:
    shard_a, shard_b = _valid_shards(tmp_path)
    rows = _read_lines(shard_b / "rollouts.jsonl")
    rows.append(rows[0])
    write_rollouts_jsonl(shard_b, rows)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="duplicate rollout"):
        merge_shard_runs([str(shard_a), str(shard_b)], str(output))
    assert not output.exists()


def test_merge_rejects_missing_rollout_and_config_mismatch(tmp_path) -> None:
    shard_a, shard_b = _valid_shards(tmp_path)
    rows = _read_lines(shard_b / "rollouts.jsonl")[:-1]
    write_rollouts_jsonl(shard_b, rows)
    with pytest.raises(ValueError, match="incomplete rollout key set"):
        merge_shard_runs([str(shard_a), str(shard_b)], str(tmp_path / "missing"))

    # Restore the cohort, then change a non-shard config field.
    _, shard_b = _valid_shards(tmp_path / "second")
    shard_a = tmp_path / "second" / "shard_a"
    config = _load_yaml(shard_b / "resolved_config.yaml")
    config["student_model_path"] = "different-student"
    write_resolved_config_yaml(shard_b, config)
    with pytest.raises(ValueError, match="resolved config differs"):
        merge_shard_runs([str(shard_a), str(shard_b)], str(tmp_path / "mismatch"))


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def test_merge_rejects_overlapping_intervals_and_uids(tmp_path) -> None:
    selection_hash = _selection_hash(["p1", "p2", "p3", "p4"])
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    _write_shard(
        shard_a,
        uids=["p1", "p2"],
        states=[("exposed", 3, True), ("correct_tail", 1, False)],
        start=0,
        end=2,
        full_selection_hash=selection_hash,
    )
    _write_shard(
        shard_b,
        uids=["p2", "p4"],
        states=[("correct_tail", 1, False), ("correct_tail", 1, False)],
        start=2,
        end=4,
        full_selection_hash=selection_hash,
    )
    with pytest.raises(ValueError, match="UID overlap"):
        merge_shard_runs([str(shard_a), str(shard_b)], str(tmp_path / "overlap"))
