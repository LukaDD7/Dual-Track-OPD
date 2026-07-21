import pytest
import torch

from dual_track_opd.va_opd.objective import build_va_token_weights, compute_rollout_weights


def test_rollout_weights_reproduce_paper_figure_population_std():
    rollout_means = torch.tensor([[0.045], [0.030], [0.020], [0.010]])
    mask = torch.ones_like(rollout_means, dtype=torch.bool)

    weights, summaries, _ = compute_rollout_weights(
        rollout_means,
        response_mask=mask,
        prompt_ids=["geometry-1"] * 4,
        expected_rollouts=4,
    )

    assert summaries.tolist() == pytest.approx([0.045, 0.030, 0.020, 0.010])
    assert weights.tolist() == pytest.approx([0.656, 0.206, 0.095, 0.044], abs=1.5e-3)
    assert weights.sum().item() == pytest.approx(1.0)


def test_rollout_weights_fail_when_sibling_group_is_incomplete():
    va = torch.ones((3, 2))
    with pytest.raises(ValueError, match="exactly 4 sibling"):
        compute_rollout_weights(
            va,
            response_mask=torch.ones_like(va, dtype=torch.bool),
            prompt_ids=["same"] * 3,
            expected_rollouts=4,
        )


def test_native_token_weights_encode_grouped_equation_for_seq_sum_aggregation():
    full = torch.tensor(
        [
            [-0.10, -0.20, -0.30, -0.40, -0.50],
            [-0.10, -0.20, -0.30, -0.40, -0.50],
            [-0.10, -0.20, -0.30, -0.40, -0.50],
            [-0.10, -0.20, -0.30, -0.40, -0.50],
        ]
    )
    degraded = full - torch.tensor([[0.5], [0.3], [0.2], [0.1]])
    mask = torch.ones_like(full, dtype=torch.bool)

    result = build_va_token_weights(
        full,
        degraded,
        response_mask=mask,
        prompt_ids=["same"] * 4,
        expected_rollouts=4,
    )

    # One high token and four low tokens per sequence.  Each rollout's token
    # multipliers sum to K*w, and the whole prompt group sums to K.
    assert result.high_mask.sum(dim=1).tolist() == [1, 1, 1, 1]
    assert result.low_mask.sum(dim=1).tolist() == [4, 4, 4, 4]
    assert result.token_weights.sum(dim=1).tolist() == pytest.approx(
        (4.0 * result.rollout_weights).tolist()
    )
    assert result.token_weights.sum().item() == pytest.approx(4.0)


def test_one_token_response_preserves_rollout_mass():
    full = torch.tensor([[-0.1], [-0.1], [-0.1], [-0.1]])
    degraded = full - torch.tensor([[0.4], [0.3], [0.2], [0.1]])
    result = build_va_token_weights(
        full,
        degraded,
        response_mask=torch.ones_like(full, dtype=torch.bool),
        prompt_ids=["same"] * 4,
    )

    assert result.degenerate_sequence_count == 4
    assert result.token_weights.sum().item() == pytest.approx(4.0)
