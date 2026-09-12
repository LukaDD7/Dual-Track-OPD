import torch

from dual_track_opd.fc_opd.alignment import (
    apply_alignment_weights,
    compute_alignment_weights,
    compute_rollout_group_stats,
    default_outcome_metadata,
)


def test_noop_alignment_does_not_change_loss():
    loss = torch.tensor([1.0, 3.0, 5.0])
    weights = compute_alignment_weights({"response_token_ids": [1, 2, 3]}, strategy="none")

    assert weights.available
    assert torch.isclose(apply_alignment_weights(loss, weights), loss.mean())


def test_token_weights_modify_weighted_loss():
    loss = torch.tensor([1.0, 3.0])
    weighted = apply_alignment_weights(loss, [1.0, 3.0])

    assert torch.isclose(weighted, torch.tensor(2.5))


def test_missing_outcome_metadata_returns_noop_warning():
    weights = compute_alignment_weights(
        {"response_token_ids": [1, 2]},
        strategy="correctness_contrastive",
    )

    assert not weights.available
    assert "outcome_metadata_unavailable" in weights.warnings


def test_default_outcome_metadata_marks_ground_truth_posthoc_only():
    outcome = default_outcome_metadata({"answer": "B"})

    assert outcome["ground_truth_available"] is True
    assert outcome["correctness_used_for_prompt"] is False
    assert outcome["correctness_used_for_evidence_generation"] is False


def test_success_failure_grouping_detects_contrast():
    stats = compute_rollout_group_stats(
        [
            {"rollout_group_uid": "g", "rollout_id": 0, "outcome_metadata": {"is_correct": True}},
            {"rollout_group_uid": "g", "rollout_id": 1, "outcome_metadata": {"is_correct": False}},
        ]
    )

    assert stats["g"]["num_success"] == 1
    assert stats["g"]["num_failure"] == 1
    assert stats["g"]["has_success_failure_contrast"] is True
