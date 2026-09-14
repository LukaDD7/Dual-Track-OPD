import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.router import RouterConfig, route_condition_weights


def _masks():
    return {
        "visual_evidence": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        "reasoning": torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool),
        "answer": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool),
    }


def test_router_zero_selects_one_condition_for_all_tokens():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    weights = route_condition_weights(
        {},
        _masks(),
        RouterConfig(mode="single", single_condition=Condition.FULL),
        response_mask=response_mask,
        available_conditions=[Condition.FULL],
    )
    assert torch.all(weights[Condition.FULL] == 1)
    assert not weights[Condition.FULL].requires_grad


def test_router_one_uses_deterministic_chunk_mapping():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    weights = route_condition_weights(
        {},
        _masks(),
        RouterConfig(),
        response_mask=response_mask,
        available_conditions=[Condition.FULL, Condition.TASK],
    )
    assert torch.equal(weights[Condition.TASK], torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32))
    assert torch.equal(weights[Condition.FULL], torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32))
    assert torch.equal(sum(weights.values()), response_mask.float())


def test_invalid_format_routes_all_tokens_to_fallback_condition():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    weights = route_condition_weights(
        {},
        _masks(),
        RouterConfig(),
        response_mask=response_mask,
        available_conditions=[Condition.FULL, Condition.TASK],
        format_valid=torch.tensor([False]),
    )
    assert torch.all(weights[Condition.FULL] == 1)
    assert torch.all(weights[Condition.TASK] == 0)


def test_overlapping_chunk_masks_are_rejected():
    masks = _masks()
    masks["answer"][0, 0] = True
    with pytest.raises(ValueError, match="must not overlap"):
        route_condition_weights(
            {},
            masks,
            RouterConfig(),
            response_mask=torch.ones((1, 5), dtype=torch.bool),
        )


def test_chunk_gated_routing_is_sparse_and_chunk_aware():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    masks = {
        "visible_evidence": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        "diagram_inference": torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.bool),
        "reasoning": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool),
        "answer": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool),
    }
    weights = route_condition_weights(
        {
            "visual_detail_delta": torch.ones((1, 5)),
            "diagram_infer_delta": torch.ones((1, 5)),
            "solve_delta": torch.ones((1, 5)),
        },
        masks,
        RouterConfig(mode="chunk_gated", max_conditions_per_token=2),
        response_mask=response_mask,
        available_conditions=[
            Condition.FULL,
            Condition.FREE,
            Condition.TASK_VISIBLE,
            Condition.TASK_INFER,
            Condition.TASK_SOLVE,
        ],
    )

    assert torch.all(sum(weights.values()) == 1.0)
    assert torch.equal(weights[Condition.TASK_SOLVE], torch.tensor([[0, 0, 0.0, 0.5, 1.0]]))
    assert weights[Condition.TASK_VISIBLE][0, 0] > 0
    assert weights[Condition.TASK_INFER][0, 2] > 0


def test_chunk_gated_contrastive_keeps_visual_condition_pairs():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    masks = {
        "visible_evidence": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        "diagram_inference": torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.bool),
        "reasoning": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool),
        "answer": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool),
    }
    weights = route_condition_weights(
        {},
        masks,
        RouterConfig(mode="chunk_gated_contrastive", max_conditions_per_token=4),
        response_mask=response_mask,
        available_conditions=[
            Condition.FULL,
            Condition.DEGRADED,
            Condition.FREE,
            Condition.TASK_VISIBLE,
            Condition.TASK_INFER,
            Condition.TASK_SOLVE,
        ],
    )

    assert weights[Condition.FULL][0, 0] > 0
    assert weights[Condition.DEGRADED][0, 0] > 0
    assert weights[Condition.TASK_VISIBLE][0, 0] > 0
    assert weights[Condition.FREE][0, 0] > 0
    assert torch.allclose(sum(weights.values()), torch.ones_like(response_mask, dtype=torch.float32))


def test_chunk_router_maps_legacy_task_to_task_visible_when_needed():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    weights = route_condition_weights(
        {},
        _masks(),
        RouterConfig(
            mode="chunk",
            chunk_condition={
                "visual_evidence": Condition.TASK,
                "reasoning": Condition.FULL,
                "answer": Condition.FULL,
            },
        ),
        response_mask=response_mask,
        available_conditions=[Condition.FULL, Condition.TASK_VISIBLE],
    )

    assert torch.equal(weights[Condition.TASK_VISIBLE], torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32))


def test_uniform_all_conditions_remains_available_for_ablation():
    response_mask = torch.ones((1, 5), dtype=torch.bool)
    weights = route_condition_weights(
        {},
        _masks(),
        RouterConfig(mode="uniform_all_conditions"),
        response_mask=response_mask,
        available_conditions=[Condition.FULL, Condition.TASK],
    )
    assert torch.all(weights[Condition.FULL] == 0.5)
    assert torch.all(weights[Condition.TASK] == 0.5)
