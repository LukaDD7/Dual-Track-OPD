"""VGPO: Visually-Guided Policy Optimization advantage modulation.

Faithful implementation of ACL 2026 (arXiv 2604.09349).

Rather than adding a separate distillation loss, VGPO modulates the existing
GRPO advantage with visual grounding signals:

    Â^V_{i,t} = Â_i * (1 + ψ_{i,t}) * (1 + φ_i)

Where:
  - Â_i      : original GRPO advantage (task-reward-driven)
  - ψ_{i,t}  : intra-trajectory visual focus (token-level, zero-centered)
  - φ_i      : inter-trajectory visual grounding (rollout-level, zero-centered)

No extra loss term.  No coefficient tuning.  Visual signals amplify the
policy gradient on visually-grounded tokens without competing with the
task reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hidden_state_visual_focus import (
    compute_visual_focus_scores,
    extract_prediction_state_positions,
)


@dataclass(frozen=True)
class VGPOAdvantage:
    modulated: torch.Tensor                # [B, T]: Â^V = Â · (1+ψ) · (1+φ)
    psi: torch.Tensor                       # [B, T]: intra-trajectory token weights
    phi: torch.Tensor                       # [B]: inter-trajectory rollout weights
    vfs: torch.Tensor                       # [B, T]: raw visual focus scores ρ


def compute_vgpo_advantage(
    advantages: torch.Tensor,               # [B, T]: GRPO advantage Â_i
    visual_focus_scores: torch.Tensor,      # [B, T]: ρ_{i,t} ∈ [0,1]
    response_mask: torch.Tensor,            # [B, T]: valid token mask
    *,
    prompt_ids: torch.Tensor | None = None, # [B]: for inter-trajectory grouping
    beta: float = 0.3,                      # VAC: progressive visual amplification
    gamma: float = 0.5,                     # VAC: tail fraction
    kappa: float = 0.5,                     # VAC: top percentile in tail
    eps: float = 1e-8,
) -> VGPOAdvantage:
    """Compute visually-modulated GRPO advantage (§3.2-3.4).

    1. Visual Attention Compensation: amplify later tokens' visual focus
    2. Intra-trajectory re-weighting: ψ = zero-centered normalized scores
    3. Inter-trajectory re-weighting: φ = zero-centered aggregated scores
    4. Modulate: Â^V = Â · (1 + ψ) · (1 + φ)
    """
    mask = response_mask.bool()
    B, T = mask.shape
    device = advantages.device

    # ── 1. Visual Attention Compensation (VAC, §3.1) ──────────────────────
    # w_{i,t} = ρ_{i,t} · [1 + G(ρ, t) · β · t/T_i]  — progressive reweighting
    rho = visual_focus_scores.float()

    # Tail gate: G = 1 for tokens in top-κ of last γ fraction
    w_vac = rho.clone()
    for i in range(B):
        valid = mask[i]
        if not valid.any():
            continue
        T_i = int(valid.sum().item())
        if T_i <= 1:
            continue
        tail_start = int((1.0 - gamma) * T_i)
        if tail_start >= T_i:
            continue
        tail_positions = torch.arange(tail_start, T_i, device=device)
        tail_scores = rho[i, valid][tail_positions]
        if tail_scores.numel() <= 1:
            continue
        threshold_idx = max(1, int(tail_scores.numel() * kappa))
        threshold = torch.topk(tail_scores, k=threshold_idx, largest=True).values[-1]
        tail_gate = (rho[i, valid][tail_positions] >= threshold).float()

        # Progressive linear amplification per position
        t_frac = torch.arange(T_i, device=device).float() / max(T_i, 1)
        amplification = 1.0 + beta * t_frac

        # Apply gate to tail, no gate for early tokens
        full_gate = torch.zeros(T_i, device=device)
        full_gate[tail_start:] = tail_gate
        w_vac[i, valid] = rho[i, valid] * (1.0 + full_gate * beta * t_frac)

    # ── 2. Intra-trajectory re-weighting (§3.2) ───────────────────────────
    # ψ_{i,t} = ŵ_{i,t} - mean(ŵ_i)  — zero-centered within each trajectory
    w_norm = torch.zeros_like(w_vac)
    for i in range(B):
        valid = mask[i]
        if not valid.any():
            continue
        w_i = w_vac[i, valid]
        w_min, w_max = w_i.min(), w_i.max()
        if w_max - w_min < eps:
            w_hat = torch.zeros_like(w_i)
        else:
            w_hat = (w_i - w_min) / (w_max - w_min + eps)
        w_norm[i, valid] = w_hat

    psi = torch.zeros_like(w_norm)
    for i in range(B):
        valid = mask[i]
        if not valid.any():
            continue
        psi[i, valid] = w_norm[i, valid] - w_norm[i, valid].mean()

    # ── 3. Inter-trajectory re-weighting (§3.2) ───────────────────────────
    # φ_i = aggregated score normalized across sibling rollouts
    s = (w_vac * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)  # [B]

    groups = _prompt_groups(prompt_ids, B)
    phi = torch.zeros(B, device=device)
    for _, indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        if len(indices) <= 1:
            continue
        s_group = s[idx]
        s_min, s_max = s_group.min(), s_group.max()
        if s_max - s_min < eps:
            s_hat = torch.zeros_like(s_group)
        else:
            s_hat = (s_group - s_min) / (s_max - s_min + eps)
        phi[idx] = s_hat - s_hat.mean()

    # ── 4. Modulated advantage ────────────────────────────────────────────
    # Â^V_{i,t} = Â_i · (1 + ψ_{i,t}) · (1 + φ_i)
    modulated = advantages.float() * (1.0 + psi) * (1.0 + phi.unsqueeze(1))
    modulated = modulated * mask.float()  # zero out padding

    return VGPOAdvantage(
        modulated=modulated,
        psi=psi,
        phi=phi,
        vfs=rho,
    )


def _prompt_groups(
    prompt_ids: torch.Tensor | None, batch: int,
) -> dict[int, list[int]]:
    if prompt_ids is None:
        return {0: list(range(batch))}
    labels = [int(p) for p in prompt_ids.reshape(-1).tolist()]
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)
    return groups
