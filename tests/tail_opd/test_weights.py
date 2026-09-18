import math

import pytest
import torch

from dual_track_opd.tail_opd import (
    compute_diagnostics,
    compute_nll_tail_rollout_weights,
    validate_fixed_group_size,
)


def test_uniform_recovery_and_no_gradient():
    old_log_probs = torch.log(torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]))
    old_log_probs.requires_grad_(True)
    response_mask = torch.ones(4, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a"] * 4)

    assert torch.allclose(result.weights, torch.full((4,), 0.25))
    assert torch.allclose(result.scales, torch.ones(4))
    assert not result.weights.requires_grad
    assert not result.scales.requires_grad

    distillation = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    scaled_distillation = distillation * result.scales[:, None]
    torch.testing.assert_close(scaled_distillation, distillation)


def test_higher_nll_receives_more_weight():
    old_log_probs = torch.tensor(
        [
            [-0.1, -0.1],
            [-0.2, -0.2],
            [-0.3, -0.3],
            [-0.4, -0.4],
        ]
    )
    response_mask = torch.ones(4, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a"] * 4)

    assert torch.allclose(result.weights.sum(), torch.tensor(1.0))
    assert torch.allclose(result.scales.mean(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(result.weights[1:] > result.weights[:-1])


def test_group_isolation_and_noncontiguous_siblings():
    old_log_probs = torch.tensor(
        [
            [-0.1, -0.1],
            [-0.2, -0.2],
            [-0.4, -0.4],
            [-0.8, -0.8],
        ]
    )
    response_mask = torch.ones(4, 2)
    ids = ["a", "b", "a", "b"]
    before = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ids)

    modified = old_log_probs.clone()
    modified[1] = torch.tensor([-0.001, -0.001])
    modified[3] = torch.tensor([-100.0, -100.0])
    after = compute_nll_tail_rollout_weights(modified, response_mask, ids)

    a_indices = torch.tensor([0, 2])
    torch.testing.assert_close(before.weights[a_indices], after.weights[a_indices])
    torch.testing.assert_close(before.scales[a_indices], after.scales[a_indices])
    torch.testing.assert_close(after.scales.mean(), torch.tensor(1.0), atol=1e-6, rtol=0.0)


def test_padding_is_ignored():
    old_log_probs = torch.tensor([[-0.1, -0.2, -99.0], [-0.4, -0.8, -99.0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a", "a"])
    expected_scores = torch.tensor([0.15, 0.6])
    assert torch.allclose(result.scores, expected_scores, atol=1e-6)
    assert result.weights[1] > result.weights[0]


def test_single_rollout_group_is_uniform():
    old_log_probs = torch.tensor([[-0.1, -2.0]])
    response_mask = torch.ones(1, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a"])
    assert torch.allclose(result.weights, torch.ones(1))
    assert torch.allclose(result.scales, torch.ones(1))


def test_core_weights_support_arbitrary_group_sizes_but_contract_rejects_incomplete_groups():
    old_log_probs = torch.tensor([[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]])
    response_mask = torch.ones_like(old_log_probs)
    ids = ["a", "b", "a", "b", "b"]

    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ids)
    torch.testing.assert_close(result.weights[[0, 2]].sum(), torch.tensor(1.0))
    torch.testing.assert_close(result.weights[[1, 3, 4]].sum(), torch.tensor(1.0))

    with pytest.raises(ValueError, match=r"'b'=3"):
        validate_fixed_group_size(ids, expected_group_size=2)


def test_fixed_group_size_contract_accepts_noncontiguous_complete_groups():
    validate_fixed_group_size(["a", "b", "a", "b"], expected_group_size=2)


def test_zero_valid_response_tokens_fail():
    with pytest.raises(ValueError, match="zero valid tokens"):
        compute_nll_tail_rollout_weights(
            torch.tensor([[-0.1, -0.2]]),
            torch.zeros(1, 2),
            ["a"],
        )


@pytest.mark.parametrize("invalid_log_prob", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_nll_scores_fail(invalid_log_prob):
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_nll_tail_rollout_weights(
            torch.tensor([[invalid_log_prob]]),
            torch.ones(1, 1),
            ["a"],
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_nonpositive_temperature_fails(temperature):
    with pytest.raises(ValueError, match="temperature must be positive"):
        compute_nll_tail_rollout_weights(
            torch.tensor([[-0.1]]),
            torch.ones(1, 1),
            ["a"],
            temperature=temperature,
        )


def test_negative_eps_fails():
    with pytest.raises(ValueError, match="eps must be non-negative"):
        compute_nll_tail_rollout_weights(
            torch.tensor([[-0.1]]),
            torch.ones(1, 1),
            ["a"],
            eps=-1e-6,
        )


def test_diagnostics_reports_effective_rollouts():
    old_log_probs = torch.tensor([[-0.1, -0.1], [-0.5, -0.5]])
    response_mask = torch.ones(2, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a", "a"])
    diagnostics = compute_diagnostics(result, response_mask)

    expected_effective_rollouts = (1.0 / result.weights.pow(2).sum()).item()
    assert diagnostics["tail_opd/effective_rollouts"] == pytest.approx(expected_effective_rollouts)
    assert diagnostics["tail_opd/scale_mean"] == pytest.approx(1.0)


def _uniform_group_entropy(group_count: int, group_size: int = 4) -> float:
    batch_size = group_count * group_size
    old_log_probs = torch.full((batch_size, 2), -0.25)
    response_mask = torch.ones_like(old_log_probs)
    ids = [f"group-{index // group_size}" for index in range(batch_size)]
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ids)
    return compute_diagnostics(result, response_mask)["tail_opd/weight_entropy"]


def test_weight_entropy_is_per_group_mean_and_batch_size_invariant():
    entropy_two_groups = _uniform_group_entropy(2)
    entropy_eight_groups = _uniform_group_entropy(8)

    assert entropy_two_groups == pytest.approx(math.log(4))
    assert entropy_eight_groups == pytest.approx(math.log(4))
    assert entropy_eight_groups == pytest.approx(entropy_two_groups)
