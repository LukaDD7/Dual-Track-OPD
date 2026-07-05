import math

import pytest
import torch

from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK
from dual_track_opd.fc_opd.va_opd_loss import (
    compute_rollout_va_weights,
    compute_va,
    compute_va_opd_loss,
)


def _teacher(token_ids: torch.Tensor, probs: torch.Tensor) -> TeacherTopK:
    alt = (token_ids + 1).remainder(8)
    topk_ids = torch.stack((token_ids, alt), dim=-1)
    topk_probs = torch.stack((probs, 1.0 - probs), dim=-1)
    return TeacherTopK(
        token_ids=topk_ids,
        log_probs=topk_probs.log(),
        tail_log_prob=torch.full(token_ids.shape, math.log(1e-6)),
        sampled_log_probs=probs.log(),
    )


def test_rollout_weights_sum_to_one_within_each_prompt_group():
    va = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 0.0],
            [8.0, 2.0],
            [2.0, 1.0],
        ]
    )

    weights = compute_rollout_va_weights(va, prompt_ids=["a", "a", "b", "b"])

    assert weights[:2].sum().item() == pytest.approx(1.0)
    assert weights[2:].sum().item() == pytest.approx(1.0)
    assert weights[0].item() > weights[1].item()
    assert weights[2].item() > weights[3].item()


def test_va_uses_exact_sampled_log_probs_when_available():
    sampled = torch.tensor([[1, 2]])
    full = _teacher(sampled, torch.tensor([[0.8, 0.7]]))
    degraded = _teacher(sampled, torch.tensor([[0.4, 0.9]]))

    va = compute_va(full, degraded, sampled)

    assert va[0, 0].item() == pytest.approx(math.log(0.8) - math.log(0.4))
    assert va[0, 1].item() == pytest.approx(math.log(0.7) - math.log(0.9))


def test_va_opd_requires_prompt_ids_for_multi_rollout_batches():
    sampled = torch.tensor([[1, 2], [1, 2]])
    full = _teacher(sampled, torch.full_like(sampled, 0.8, dtype=torch.float32))
    degraded = _teacher(sampled, torch.full_like(sampled, 0.4, dtype=torch.float32))
    student_logits = torch.zeros((2, 2, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="prompt_ids"):
        compute_va_opd_loss(student_logits, full, degraded, sampled)


def test_va_opd_loss_is_grouped_by_prompt_and_finite():
    sampled = torch.tensor([[1, 2], [1, 2]])
    full = _teacher(sampled, torch.tensor([[0.8, 0.8], [0.6, 0.6]]))
    degraded = _teacher(sampled, torch.tensor([[0.4, 0.4], [0.5, 0.5]]))
    student_logits = torch.zeros((2, 2, 8), dtype=torch.float32)

    result = compute_va_opd_loss(
        student_logits,
        full,
        degraded,
        sampled,
        prompt_ids=["same", "same"],
    )

    assert torch.isfinite(result.loss)
    assert result.rollout_weights.sum().item() == pytest.approx(1.0)
    assert result.rollout_weights[0].item() > result.rollout_weights[1].item()
    assert result.va_pos.shape == sampled.shape
