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
