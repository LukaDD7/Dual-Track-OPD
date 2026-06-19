import torch

from dual_track_opd.losses.visual_anchor import build_visual_anchor_weights


def test_visual_anchor_topk_behavior():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    weights = build_visual_anchor_weights(scores, topk_ratio=0.5, high_weight=3.0, low_weight=1.0)
    assert torch.allclose(weights, torch.tensor([[1.0, 3.0, 1.0, 3.0]]))


def test_visual_anchor_respects_mask():
    scores = torch.tensor([[0.1, 0.9, 0.8]])
    mask = torch.tensor([[1, 0, 1]])
    weights = build_visual_anchor_weights(scores, mask=mask, topk_ratio=0.5)
    assert weights[0, 1].item() == 0.0
    assert weights[0, 2].item() == 2.0

