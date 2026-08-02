"""Tests for sharded parallel runs (--prompt-start/--prompt-end) and merging."""

from __future__ import annotations

import json

from dual_track_opd.support_aware.diagnostic import (
    merge_shard_runs,
    slice_prompts,
)
from dual_track_opd.support_aware.reporter import (
    write_prompt_support_summary_jsonl,
    write_rollouts_jsonl,
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

    sliced_clamped, _ = slice_prompts(prompts, 8, 100)
    assert [p["sample_uid"] for p in sliced_clamped] == ["p8", "p9"]


def _rollout(
    uid: str,
    *,
    is_greedy: bool,
    rollout_id: int,
    correct: bool,
    gap: float,
) -> dict[str, object]:
    return {
        "run_id": "shard",
        "sample_uid": uid,
        "is_greedy": is_greedy,
        "rollout_id": rollout_id,
        "correct": correct,
        "teacher_mean_logp": -0.3 if correct else -0.6,
        "student_mean_logp": -0.2 if correct else -0.5,
        "teacher_gap": gap,
        "response_token_count": 500,
        "malformed": False,
        "tokenizer_hash": "same-hash",
        "teacher_model_id": "teacher",
        "student_model_path": "student",
    }


def _prompt_rollouts(uid: str, n_correct: int, greedy_correct: bool):
    rollouts = [_rollout(uid, is_greedy=True, rollout_id=0, correct=greedy_correct, gap=0.0)]
    for rid in range(1, 5):
        correct = rid <= n_correct
        rollouts.append(
            _rollout(uid, is_greedy=False, rollout_id=rid, correct=correct, gap=1.0 if correct else -1.0)
        )
    return rollouts


def _summary_line(uid: str, state: str, n_correct: int) -> dict[str, object]:
    return {
        "sample_uid": uid,
        "K": 4,
        "correct_count": n_correct,
        "greedy_correct": True if state == "exposed" and n_correct >= 2 else False,
        "support_state": state,
        "unique_response_count": 4,
        "duplicate_rollout_rate": 0.0,
        "teacher_rank_of_best_correct": 1,
        "student_rank_of_best_correct": 1,
        "teacher_gap_rank_of_best_correct": 1,
        "teacher_top1_correct": True,
        "teacher_gap_top1_correct": True,
    }


def test_merge_shard_runs_recomputes_summary(tmp_path) -> None:
    # Shard A: p1 exposed, p2 correct_tail
    shard_a = tmp_path / "shard_a"
    shard_a.mkdir()
    write_rollouts_jsonl(
        shard_a,
        _prompt_rollouts("p1", n_correct=3, greedy_correct=True)
        + _prompt_rollouts("p2", n_correct=1, greedy_correct=False),
    )
    write_prompt_support_summary_jsonl(
        shard_a,
        [_summary_line("p1", "exposed", 3), _summary_line("p2", "correct_tail", 1)],
    )

    # Shard B: p3 no_correct_observed, p4 correct_tail (with one duplicate line)
    shard_b = tmp_path / "shard_b"
    shard_b.mkdir()
    shard_b_rollouts = (
        _prompt_rollouts("p3", n_correct=0, greedy_correct=False)
        + _prompt_rollouts("p4", n_correct=1, greedy_correct=False)
    )
    shard_b_rollouts.append(shard_b_rollouts[0])  # defensive dedupe check
    write_rollouts_jsonl(shard_b, shard_b_rollouts)
    write_prompt_support_summary_jsonl(
        shard_b,
        [
            _summary_line("p3", "no_correct_observed", 0),
            _summary_line("p4", "correct_tail", 1),
        ],
    )

    merged = tmp_path / "merged"
    summary = merge_shard_runs([str(shard_a), str(shard_b)], str(merged))

    assert summary["num_prompts"] == 4
    assert summary["total_rollouts"] == 20  # 5 per prompt, duplicate removed
    assert summary["greedy_accuracy"] == 0.25
    assert summary["correct_tail_count"] == 2
    assert summary["exposed_count"] == 1
    assert summary["no_correct_observed_count"] == 1
    assert summary["auc_teacher_gap_correct_vs_wrong"] == 1.0
    assert summary["auc_teacher_mean_logp_correct_vs_wrong"] == 1.0
    assert summary["tokenizer_match"] is True
    assert summary["selection_manifest"]["merged_from"] == ["shard_a", "shard_b"]

    gate_names = [g[0] for g in summary["acceptance_gates"]]
    assert len(gate_names) == 7
    assert "teacher_gap_auc" in gate_names

    # Merged files written
    merged_rollouts = [json.loads(l) for l in (merged / "rollouts.jsonl").read_text().splitlines() if l.strip()]
    assert len(merged_rollouts) == 20
    assert (merged / "summary.json").exists()
