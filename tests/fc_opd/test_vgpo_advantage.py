import pytest
import torch
from dual_track_opd.fc_opd.vgpo_advantage import compute_vgpo_advantage


def test_vgpo_modulated_advantage_same_sign():
    """Modulated advantage preserves sign of original."""
    adv = torch.tensor([[1.0, -0.5, 2.0, -1.0, 0.0]])
    vfs = torch.tensor([[0.9, 0.1, 0.8, 0.3, 0.5]])
    mask = torch.ones_like(adv, dtype=torch.bool)

    result = compute_vgpo_advantage(adv, vfs, mask)
    mod = result.modulated
    assert mod[0, 0] > 0   # positive stays positive
    assert mod[0, 1] < 0   # negative stays negative
    assert abs(mod[0, 4]) < 1e-4  # zero stays near zero


def test_vgpo_preserves_zero_advantage():
    """Zero advantage stays zero."""
    adv = torch.zeros((2, 4))
    vfs = torch.rand((2, 4))
    mask = torch.ones((2, 4), dtype=torch.bool)
    result = compute_vgpo_advantage(adv, vfs, mask)
    assert result.modulated.abs().max() < 1e-4


def test_vgpo_psi_zero_centered():
    """Intra-trajectory weights ψ sum to zero within each rollout."""
    vfs = torch.rand((3, 10))
    adv = torch.ones((3, 10))
    mask = torch.ones((3, 10), dtype=torch.bool)
    result = compute_vgpo_advantage(adv, vfs, mask)
    for i in range(3):
        assert abs(result.psi[i].sum().item()) < 1e-5


def test_vgpo_phi_zero_centered_across_siblings():
    """Inter-trajectory weights φ sum to zero within each prompt group."""
    B = 6
    vfs = torch.rand((B, 8))
    adv = torch.ones((B, 8))
    mask = torch.ones((B, 8), dtype=torch.bool)
    prompt_ids = torch.tensor([0, 0, 0, 0, 1, 1])  # group of 4 + group of 2
    result = compute_vgpo_advantage(adv, vfs, mask, prompt_ids=prompt_ids)
    assert abs(result.phi[:4].sum().item()) < 1e-5  # group 0 sums to 0
    assert abs(result.phi[4:].sum().item()) < 1e-5  # group 1 sums to 0


def test_vgpo_higher_vfs_gives_higher_amplification():
    """Tokens with higher visual focus get larger advantage amplification."""
    adv = torch.tensor([[1.0, 1.0, 1.0]])
    vfs_high = torch.tensor([[0.9, 0.5, 0.1]])
    mask = torch.ones_like(adv, dtype=torch.bool)
    result = compute_vgpo_advantage(adv, vfs_high, mask)
    # Token 0 (VFS=0.9) should have higher advantage than token 2 (VFS=0.1)
    assert result.modulated[0, 0] > result.modulated[0, 2]
