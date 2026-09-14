import math

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.loss import FCOPDLossConfig, compute_fc_opd_loss, sparse_forward_kl
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK


def _teacher(probabilities, ids=None, tail=None):
    ids = list(range(len(probabilities))) if ids is None else ids
    return TeacherTopK(
        token_ids=torch.tensor([[ids]], dtype=torch.int64),
        log_probs=torch.log(torch.tensor([[probabilities]], dtype=torch.float32)),
        tail_log_prob=None if tail is None else torch.tensor([[math.log(tail)]], dtype=torch.float32),
    )


def test_sparse_kl_is_zero_for_identical_full_distribution():
    probabilities = torch.tensor([0.6, 0.3, 0.1])
    logits = probabilities.log().reshape(1, 1, 3)
    loss = sparse_forward_kl(logits, _teacher(probabilities.tolist()))
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_sparse_kl_increases_under_controlled_perturbation():
    teacher = _teacher([0.8, 0.2])
    matching = torch.log(torch.tensor([[[0.8, 0.2]]]))
    perturbed = torch.log(torch.tensor([[[0.2, 0.8]]]))
    matching_loss = sparse_forward_kl(matching, teacher)
    perturbed_loss = sparse_forward_kl(perturbed, teacher)
    assert matching_loss.item() < 1e-6
    assert perturbed_loss.item() > matching_loss.item()


def test_sparse_kl_backpropagates_to_student_logits_only():
    student = torch.zeros((1, 1, 2), requires_grad=True)
    teacher = _teacher([0.9, 0.1])
    loss = sparse_forward_kl(student, teacher).mean()
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.log_probs.grad is None


def test_tail_bucket_matches_coarse_student_distribution():
    teacher = _teacher([0.6, 0.2], ids=[0, 1], tail=0.2)
    student = torch.log(torch.tensor([[[0.6, 0.2, 0.1, 0.1]]]))
    loss = sparse_forward_kl(student, teacher)
    assert loss.item() < 1e-6


def test_duplicate_teacher_ids_are_supported():
    teacher = _teacher([0.3, 0.3, 0.4], ids=[0, 0, 1])
    student = torch.log(torch.tensor([[[0.6, 0.4]]]))
    loss = sparse_forward_kl(student, teacher)
    assert loss.item() < 1e-6


def test_fc_opd_loss_respects_router_weights_and_response_mask():
    student = torch.log(
        torch.tensor(
            [
                [
                    [0.8, 0.2],
                    [0.5, 0.5],
                    [0.7, 0.3],
                ]
            ]
        )
    )
    full = TeacherTopK(
        token_ids=torch.tensor([[[0, 1], [0, 1], [0, 1]]]),
        log_probs=torch.log(torch.tensor([[[0.8, 0.2], [0.9, 0.1], [0.7, 0.3]]])),
    )
    task = TeacherTopK(
        token_ids=torch.tensor([[[0, 1], [0, 1], [0, 1]]]),
        log_probs=torch.log(torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.2, 0.8]]])),
    )
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    weights = {
        Condition.FULL: torch.tensor([[1.0, 0.0, 0.0]]),
        Condition.TASK: torch.tensor([[0.0, 1.0, 0.0]]),
    }
    loss, metrics = compute_fc_opd_loss(
        student,
        {Condition.FULL: full, Condition.TASK: task},
        {
            "visual_evidence": torch.tensor([[1, 0, 0]], dtype=torch.bool),
            "reasoning": torch.tensor([[0, 1, 0]], dtype=torch.bool),
            "answer": torch.tensor([[0, 0, 0]], dtype=torch.bool),
        },
        weights,
        response_mask,
    )
    assert loss.item() < 1e-6
    assert metrics["selection/full"].item() == pytest.approx(0.5)
    assert metrics["selection/task"].item() == pytest.approx(0.5)


def test_empty_response_mask_returns_finite_zero():
    student = torch.zeros((1, 1, 2))
    teacher = _teacher([0.5, 0.5])
    loss, _ = compute_fc_opd_loss(
        student,
        {Condition.FULL: teacher},
        {
            "visual_evidence": torch.zeros((1, 1), dtype=torch.bool),
            "reasoning": torch.zeros((1, 1), dtype=torch.bool),
            "answer": torch.zeros((1, 1), dtype=torch.bool),
        },
        {Condition.FULL: torch.zeros((1, 1))},
        torch.zeros((1, 1), dtype=torch.bool),
        FCOPDLossConfig(),
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
