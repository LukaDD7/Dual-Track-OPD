import torch

from dual_track_opd.losses.token_weighting import combine_weights, normalize_token_weights


def test_normalize_token_weights_mean_one_under_mask():
    weights = torch.tensor([[1.0, 3.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]])
    normalized = normalize_token_weights(weights, mask=mask)
    assert torch.allclose(normalized, torch.tensor([[0.5, 1.5, 0.0]]))


def test_combine_weights_sum_and_product():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(combine_weights(a, b, mode="sum"), torch.tensor([[4.0, 6.0]]))
    assert torch.allclose(combine_weights(a, b, mode="product"), torch.tensor([[3.0, 8.0]]))

