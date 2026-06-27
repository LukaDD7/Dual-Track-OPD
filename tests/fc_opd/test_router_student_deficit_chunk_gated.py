import torch

from dual_track_opd.fc_opd.conditions import Condition
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
