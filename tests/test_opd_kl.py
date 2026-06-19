import torch

from dual_track_opd.losses.opd_kl import kl_divergence_from_logits, masked_weighted_kl


def test_kl_shape_none_reduction():
    student = torch.randn(2, 3, 5)
    teacher = torch.randn(2, 3, 5)
    kl = kl_divergence_from_logits(student, teacher)
    assert kl.shape == (2, 3)
    assert torch.all(kl >= -1e-6)


def test_masked_weighted_kl_uses_mask():
    student = torch.zeros(1, 2, 2)
    teacher = torch.tensor([[[4.0, -4.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    loss = masked_weighted_kl(student, teacher, mask=mask)
    expected = kl_divergence_from_logits(student, teacher)[0, 0]
    assert torch.allclose(loss, expected)


def test_logits_shape_validation():
    student = torch.randn(2, 3, 5)
    teacher = torch.randn(2, 3, 4)
    try:
        kl_divergence_from_logits(student, teacher)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")

