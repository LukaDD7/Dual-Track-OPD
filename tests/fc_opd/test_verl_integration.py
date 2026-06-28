import math

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.online_batch import OnlineFCOPDBatchOutput, OnlineFCOPDSampleOutput
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK
from dual_track_opd.fc_opd.verl_integration import online_batch_output_to_verl_tensors
from dual_track_opd.fc_opd.verl_sparse_kd import compute_verl_sparse_topk_kd


def _topk(probabilities, token_ids=None, tail=None):
    ids = token_ids if token_ids is not None else list(range(len(probabilities)))
    return TeacherTopK(
        token_ids=torch.tensor([[[ids[0], ids[1]], [ids[0], ids[1]]]], dtype=torch.long),
        log_probs=torch.log(torch.tensor([[[probabilities[0], probabilities[1]], [probabilities[0], probabilities[1]]]])),
        tail_log_prob=None if tail is None else torch.full((1, 2), math.log(tail)),
    )


def _sample_output():
    teacher = {
        Condition.FULL: _topk([0.8, 0.2]),
        Condition.DEGRADED: _topk([0.6, 0.4], tail=1e-6),
    }
    weights = {
        Condition.FULL: torch.tensor([[1.0, 0.0]]),
        Condition.DEGRADED: torch.tensor([[0.0, 0.5]]),
    }
    return OnlineFCOPDSampleOutput(
        sample_uid="sample-1",
        response_token_ids=torch.tensor([[0, 1]], dtype=torch.long),
        teacher_scores=teacher,
        student_scores={},
        capability_scores={},
        verifier={"correct": False, "format_valid": True},
        verifier_learning_value_gate={"outcome_class": "wrong_but_format_valid"},
        condition_weights=weights,
        chunk_masks={},
        grouped_loss_tensors={},
        loss=torch.tensor(0.0),
        metrics={},
    )


def test_online_outputs_stack_to_verl_tensor_schema():
    batch = OnlineFCOPDBatchOutput(samples=(_sample_output(),), loss=torch.tensor(0.0), metrics={})

    tensors = online_batch_output_to_verl_tensors(
        batch,
        condition_order=(Condition.FULL, Condition.DEGRADED),
    )

    assert tensors.teacher_topk_indices.shape == (1, 2, 2, 2)
    assert tensors.teacher_topk_log_probs.shape == (1, 2, 2, 2)
    assert tensors.condition_weights.shape == (1, 2, 2)
    assert tensors.teacher_tail_log_prob is not None
    assert tensors.teacher_tail_log_prob.shape == (1, 2, 2)
    assert tensors.as_batch_dict()["fc_condition_ids"].tolist() == [0, 1]


def test_verl_sparse_kd_is_zero_for_matching_topk_distribution():
    student = torch.log(torch.tensor([[[0.8, 0.2], [0.6, 0.4]]], dtype=torch.float32))
    teacher_ids = torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.long)
    teacher_log_probs = torch.log(torch.tensor([[[[0.8, 0.2], [0.6, 0.4]]]], dtype=torch.float32))
    weights = torch.ones((1, 1, 2), dtype=torch.float32)
    response_mask = torch.ones((1, 2), dtype=torch.bool)

    output = compute_verl_sparse_topk_kd(student, teacher_ids, teacher_log_probs, weights, response_mask)

    assert torch.allclose(output.per_token_loss, torch.zeros_like(output.per_token_loss), atol=1e-6)
    assert output.active_weight.tolist() == [[1.0, 1.0]]


def test_verl_sparse_kd_backpropagates_to_current_logits():
    student = torch.zeros((1, 2, 2), requires_grad=True)
    teacher_ids = torch.tensor([[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]], dtype=torch.long)
    teacher_log_probs = torch.log(
        torch.tensor([[[[0.9, 0.1], [0.9, 0.1]], [[0.2, 0.8], [0.2, 0.8]]]], dtype=torch.float32)
    )
    weights = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    response_mask = torch.ones((1, 2), dtype=torch.bool)

    output = compute_verl_sparse_topk_kd(student, teacher_ids, teacher_log_probs, weights, response_mask)
    loss = output.per_token_loss.sum() / output.active_weight.sum().clamp_min(1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum().item() > 0


def test_verl_sparse_kd_validates_condition_weight_shape():
    student = torch.zeros((1, 2, 2))
    teacher_ids = torch.zeros((1, 1, 2, 2), dtype=torch.long)
    teacher_log_probs = torch.log(torch.full((1, 1, 2, 2), 0.5))
    with pytest.raises(ValueError, match="condition_weights"):
        compute_verl_sparse_topk_kd(
            student,
            teacher_ids,
            teacher_log_probs,
            torch.ones((1, 2)),
            torch.ones((1, 2), dtype=torch.bool),
        )
