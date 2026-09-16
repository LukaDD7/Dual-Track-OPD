import pytest
import torch

from dual_track_opd.tail_opd import compute_diagnostics, compute_nll_tail_rollout_weights


def test_uniform_recovery_and_no_gradient():
    old_log_probs = torch.log(torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]))
    old_log_probs.requires_grad_(True)
    response_mask = torch.ones(4, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a"] * 4)

    assert torch.allclose(result.weights, torch.full((4,), 0.25))
    assert torch.allclose(result.scales, torch.ones(4))
    assert not result.weights.requires_grad
    assert not result.scales.requires_grad


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
            [-1.0, -1.0],
            [-0.1, -0.1],
            [-1.0, -1.0],
        ]
    )
    response_mask = torch.ones(4, 2)
    ids = ["a", "a", "b", "b"]
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ids)

    assert torch.allclose(result.weights[:2], result.weights[2:])
    assert torch.allclose(result.scales[:2], result.scales[2:])
    assert torch.allclose(result.scales.mean(), torch.tensor(1.0), atol=1e-6)


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


def test_diagnostics_reports_effective_rollouts():
    old_log_probs = torch.tensor([[-0.1, -0.1], [-0.5, -0.5]])
    response_mask = torch.ones(2, 2)
    result = compute_nll_tail_rollout_weights(old_log_probs, response_mask, ["a", "a"])
    diagnostics = compute_diagnostics(result, response_mask)

    expected_effective_rollouts = (1.0 / result.weights.pow(2).sum()).item()
    assert diagnostics["tail_opd/effective_rollouts"] == pytest.approx(expected_effective_rollouts)
    assert diagnostics["tail_opd/scale_mean"] == pytest.approx(1.0)
