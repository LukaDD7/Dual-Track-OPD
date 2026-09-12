"""Tests for the Step 1 diagnostic's partial-run resume support."""

from __future__ import annotations

import json

import pytest

from dual_track_opd.support_aware.diagnostic import (
    DiagnosticConfig,
    _is_nonfinite,
    _load_resume_state,
    _validate_resume_state_for_config,
)


def _rollout(uid: str, rollout_id: int, *, correct: bool, teacher=1.0,
             student=1.0, malformed=False, errors=None) -> dict:
    return {
        "run_id": "diag_full_20260801_000000",
        "sample_uid": uid,
        "rollout_id": rollout_id,
        "is_greedy": rollout_id == 0,
        "correct": correct,
        "malformed": malformed,
        "teacher_mean_logp": teacher,
        "student_mean_logp": student,
        "errors": errors or [],
    }


def _summary(uid: str, state: str = "correct_tail") -> dict:
    return {"sample_uid": uid, "support_state": state, "K": 8}


def test_load_resume_state_keeps_completed_prompts_only(tmp_path):
    uid1, uid2, uid3 = "p1", "p2", "p3"
    with (tmp_path / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
        for uid in (uid1, uid2):
            for rid in range(9):
                fh.write(json.dumps(_rollout(uid, rid, correct=(rid == 1))) + "\n")
        # Prompt killed mid-write: rollouts exist but no summary line yet.
        for rid in range(3):
            fh.write(json.dumps(_rollout(uid3, rid, correct=False)) + "\n")
    with (tmp_path / "prompt_support_summary.jsonl").open("w", encoding="utf-8") as fh:
        for uid in (uid1, uid2):
            fh.write(json.dumps(_summary(uid)) + "\n")

    state = _load_resume_state(tmp_path)

    assert state.completed_uids == {uid1, uid2}
    assert len(state.prompt_summaries) == 2
    assert len(state.all_rollouts) == 18
    assert state.total_rollouts == 18
    assert all(r["sample_uid"] != uid3 for r in state.all_rollouts)
    assert state.malformed_count == 0
    assert state.non_finite_count == 0
    assert state.errors == []

    with pytest.raises(ValueError, match="do not mix it with new rows"):
        _validate_resume_state_for_config(
            state,
            DiagnosticConfig(dataset_path="dataset", student_model_path="student"),
        )


def test_load_resume_state_counts_errors_malformed_and_nonfinite(tmp_path):
    with (tmp_path / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_rollout(
            "p1", 1, correct=True, teacher=float("nan"), errors=["boom"]
        )) + "\n")
        fh.write(json.dumps(_rollout("p1", 2, correct=False, malformed=True)) + "\n")
        fh.write(json.dumps(_rollout("p1", 3, correct=False)) + "\n")
    with (tmp_path / "prompt_support_summary.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_summary("p1")) + "\n")

    state = _load_resume_state(tmp_path)

    assert state.non_finite_count == 1
    assert state.malformed_count == 1
    assert state.errors == ["boom"]
    assert state.total_rollouts == 3


def test_load_resume_state_missing_files(tmp_path):
    state = _load_resume_state(tmp_path)

    assert state.completed_uids == set()
    assert state.all_rollouts == []
    assert state.prompt_summaries == []
    assert state.total_rollouts == 0
    assert state.errors == []


def test_is_nonfinite():
    assert _is_nonfinite(float("nan")) is True
    assert _is_nonfinite(float("inf")) is True
    assert _is_nonfinite(1.0) is False
    assert _is_nonfinite(None) is True
