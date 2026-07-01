import math

import pytest
import torch

from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK
from dual_track_opd.fc_opd.visual_grounding_gap_loss import (
    build_verifier_learning_value_gate,
    compute_grouped_sparse_kl_loss,
    compute_rollout_va_weights,
    compute_va_raw,
    compute_visual_grounding_gap_token_weights,
    split_high_low_va_groups,
)


def _teacher_topk(probabilities, *, ids=None):
    ids = ids or [list(range(len(probabilities[0]))) for _ in probabilities]
    return TeacherTopK(
        token_ids=torch.tensor([ids], dtype=torch.long),
        log_probs=torch.log(torch.tensor([probabilities], dtype=torch.float32)),
    )


def _chunk_masks(length):
    masks = {
        "visible_evidence": torch.zeros((1, length), dtype=torch.bool),
        "diagram_inference": torch.zeros((1, length), dtype=torch.bool),
        "reasoning": torch.zeros((1, length), dtype=torch.bool),
        "answer": torch.zeros((1, length), dtype=torch.bool),
    }
    masks["visible_evidence"][0, :length] = True
    return masks


def test_compute_va_raw_preserves_signed_values():
    full = _teacher_topk([[0.8, 0.2], [0.1, 0.9]])
    degraded = _teacher_topk([[0.4, 0.6], [0.8, 0.2]])
    sampled = torch.tensor([[0, 0]])

    va = compute_va_raw(full, degraded, sampled)
    assert va[0, 0].item() > 0
    assert va[0, 1].item() < 0


def test_rollout_weights_sum_to_k_within_prompt_group():
    va = torch.tensor([[3.0, 0.0], [1.0, 0.0], [9.0, 0.0]])
    weights = compute_rollout_va_weights(va, prompt_ids=["a", "a", "b"], top_q=0.5)
    assert weights[:2].sum().item() == pytest.approx(2.0)
    assert weights[2].item() == pytest.approx(1.0)
    assert weights[0].item() > weights[1].item()


def test_high_va_minority_is_split_from_low_va_tokens():
    va = torch.tensor([[0.0, 10.0, 0.1, 0.0, 0.0]])
    chunk = torch.ones_like(va, dtype=torch.bool)
    high, low = split_high_low_va_groups(va, chunk, top_q=0.2, min_high_tokens=1)
    assert high.sum().item() == 1
    assert high[0, 1]
    assert low.sum().item() == 4


def test_verifier_gate_zeroes_correct_answer_chunk_by_default():
    gate = build_verifier_learning_value_gate({"correct": True, "format_valid": True})
    assert gate["chunk_gates"]["answer"] == 0.0
    wrong = build_verifier_learning_value_gate({"correct": False, "format_valid": True})
    assert wrong["chunk_gates"]["visible_evidence"] > gate["chunk_gates"]["visible_evidence"]


def test_token_weights_are_nonnegative_without_student_logits():
    va_raw = torch.tensor([[0.0, 2.0, -1.0]])
    chunks = _chunk_masks(3)
    weights = compute_visual_grounding_gap_token_weights(
        va_raw,
        teacher_vfs=torch.tensor([[0.0, 1.0, 0.5]]),
        student_vfs=torch.tensor([[0.0, 0.0, 0.5]]),
        chunk_masks=chunks,
        verifier_outcomes=[{"correct": False, "format_valid": True}],
        inputs_are_ranked=True,
    )
    assert torch.all(weights.token_weights >= 0)
    assert weights.high_va_mask[0, 1]
    assert weights.token_weights[0, 1] > weights.token_weights[0, 0]


def test_grouped_sparse_kl_keeps_high_va_minority_from_dilution():
    teacher_full = _teacher_topk(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    teacher_degraded = _teacher_topk(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    sampled = torch.tensor([[0, 0, 0, 0, 0]])
    student_logits = torch.log(
        torch.tensor(
            [
                [
                    [0.9, 0.1],
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
    )
    chunks = _chunk_masks(5)
    teacher_vfs_rank = torch.tensor([[0.0, 1.0, 0.2, 0.3, 0.4]])
    student_vfs_rank = torch.tensor([[0.0, 0.0, 0.2, 0.3, 0.4]])

    result = compute_grouped_sparse_kl_loss(
        student_logits,
        teacher_full,
        teacher_degraded,
        sampled,
        teacher_vfs_rank,
        student_vfs_rank,
        chunks,
        verifier_outcomes=[{"correct": False, "format_valid": True}],
        inputs_are_ranked=True,
    )

    assert result.loss.item() > 0
    assert result.token_mean_loss.item() > result.per_token_kl.mean().item()
    assert torch.all(result.va_pos >= 0)
    assert result.va_raw[0, 1].item() > 0
    assert result.chunk_gap_gates.gate["visible_evidence"][0].item() > 0.5


def test_grouped_sparse_kl_correct_answer_gate_is_nonnegative_zero_loss():
    teacher_full = _teacher_topk([[0.8, 0.2]], ids=[[0, 1]])
    teacher_degraded = _teacher_topk([[0.8, 0.2]], ids=[[0, 1]])
    sampled = torch.tensor([[0]])
    student_logits = torch.log(torch.tensor([[[0.2, 0.8]]], dtype=torch.float32))
    chunks = {
        "visible_evidence": torch.zeros((1, 1), dtype=torch.bool),
        "diagram_inference": torch.zeros((1, 1), dtype=torch.bool),
        "reasoning": torch.zeros((1, 1), dtype=torch.bool),
        "answer": torch.ones((1, 1), dtype=torch.bool),
    }

    result = compute_grouped_sparse_kl_loss(
        student_logits,
        teacher_full,
        teacher_degraded,
        sampled,
        torch.tensor([[1.0]]),
        torch.tensor([[0.0]]),
        chunks,
        verifier_outcomes=[{"correct": True, "format_valid": True}],
        inputs_are_ranked=True,
    )
    assert torch.isfinite(result.loss)
    assert result.loss.item() == pytest.approx(0.0)
