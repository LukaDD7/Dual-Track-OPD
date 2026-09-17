import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.four_condition_offline_builder import (
    build_verifier_learning_value_gate,
    compute_student_deficit_capability_scores,
)
from dual_track_opd.fc_opd.router import RouterConfig, route_condition_weights


def _masks():
    return {
        "visible_evidence": torch.tensor([[1, 0, 0, 0]], dtype=torch.bool),
        "diagram_inference": torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
        "reasoning": torch.tensor([[0, 0, 1, 0]], dtype=torch.bool),
        "answer": torch.tensor([[0, 0, 0, 1]], dtype=torch.bool),
    }


def test_student_deficit_router_maps_capabilities_to_positive_conditions():
    response_mask = torch.ones((1, 4), dtype=torch.bool)
    capability_scores = {
        "visual_detail": {"positive": "full", "valid": True, "final_token_weight": [1.0, 0.0, 0.0, 0.0]},
        "evidence_selection": {"positive": "task_visible", "valid": True, "final_token_weight": [1.0, 0.0, 0.0, 0.0]},
        "visual_text_inference": {"positive": "task_infer", "valid": True, "final_token_weight": [0.0, 2.0, 0.0, 0.0]},
        "solving": {"positive": "task_solve", "valid": True, "final_token_weight": [0.0, 0.0, 1.0, 3.0]},
    }

    weights = route_condition_weights(
        {"capability_scores": capability_scores},
        _masks(),
        RouterConfig(mode="student_deficit_chunk_gated"),
        response_mask=response_mask,
        available_conditions=[
            Condition.FULL,
            Condition.TASK_VISIBLE,
            Condition.TASK_INFER,
            Condition.TASK_SOLVE,
        ],
    )

    assert weights[Condition.FULL][0, 0] > 0
    assert weights[Condition.TASK_VISIBLE][0, 0] > 0
    assert weights[Condition.TASK_INFER][0, 1] == 1
    assert weights[Condition.TASK_SOLVE][0, 2] == 1
    assert weights[Condition.TASK_SOLVE][0, 3] == 1
    assert torch.allclose(sum(weights.values()), torch.ones_like(response_mask, dtype=torch.float32))


def test_student_deficit_router_ignores_invalid_solve_like_infer_capability():
    response_mask = torch.ones((1, 4), dtype=torch.bool)
    capability_scores = {
        "visual_text_inference": {
            "positive": "task_infer",
            "valid": False,
            "invalid_reason": "task_infer_solve_like",
            "final_token_weight": [0.0, 9.0, 0.0, 0.0],
        },
        "solving": {"positive": "task_solve", "negative": "task_visible", "valid": True, "final_token_weight": [0.0, 0.0, 1.0, 1.0]},
    }
    weights = route_condition_weights(
        {"capability_scores": capability_scores},
        _masks(),
        RouterConfig(mode="student_deficit_chunk_gated"),
        response_mask=response_mask,
        available_conditions=[Condition.FULL, Condition.TASK_INFER, Condition.TASK_SOLVE],
    )

    assert torch.all(weights[Condition.TASK_INFER] == 0)
    assert weights[Condition.TASK_SOLVE][0, 2] == 1
    assert weights[Condition.TASK_SOLVE][0, 3] == 1


def _block(values):
    return {"actual_token_log_probs": values, "token_ids": [[0] for _ in values], "log_probs": [[0.0] for _ in values]}


def test_student_deficit_router_uses_verifier_learning_value_gate():
    teacher = {
        "full": _block([0.0, 0.0, 0.0, 0.0]),
        "degraded": _block([0.0, 0.0, 0.0, 0.0]),
        "free": _block([0.0, 0.0, 0.0, 0.0]),
        "task_visible": _block([0.0, 0.0, 0.0, 0.0]),
        "task_infer": _block([0.0, 1.0, 0.0, 0.0]),
        "task_solve": _block([0.0, 0.0, 1.0, 1.0]),
    }
    student = {condition: _block([0.0, 0.0, 0.0, 0.0]) for condition in teacher}
    chunks = {
        "visible_evidence": [[0, 1]],
        "diagram_inference": [[1, 2]],
        "reasoning": [[2, 3]],
        "answer": [[3, 4]],
    }
    correct_scores = compute_student_deficit_capability_scores(
        teacher_condition_scores=teacher,
        student_condition_scores=student,
        chunk_spans=chunks,
        verifier_learning_value_gate=build_verifier_learning_value_gate(
            {"correct": True, "format_valid": True, "malformed": False, "reward": 1.0}
        ),
        max_capabilities_per_token=4,
    )
    wrong_scores = compute_student_deficit_capability_scores(
        teacher_condition_scores=teacher,
        student_condition_scores=student,
        chunk_spans=chunks,
        verifier_learning_value_gate=build_verifier_learning_value_gate(
            {"correct": False, "format_valid": True, "malformed": False, "reward": 0.25}
        ),
        max_capabilities_per_token=4,
    )

    assert wrong_scores["visual_text_inference"]["final_token_weight"][1] > correct_scores["visual_text_inference"]["final_token_weight"][1]
    assert correct_scores["solving"]["final_token_weight"][3] > 0.0
    assert wrong_scores["solving"]["final_token_weight"][3] > correct_scores["solving"]["final_token_weight"][3]

    weights = route_condition_weights(
        {"capability_scores": wrong_scores},
        _masks(),
        RouterConfig(mode="student_deficit_chunk_gated"),
        response_mask=torch.ones((1, 4), dtype=torch.bool),
        available_conditions=[Condition.FULL, Condition.TASK_INFER, Condition.TASK_SOLVE],
    )
    assert weights[Condition.TASK_INFER][0, 1] == 1
    assert weights[Condition.TASK_SOLVE][0, 3] == 1
