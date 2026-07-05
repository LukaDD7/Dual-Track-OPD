"""VA-OPD: Visual-Advantage On-Policy Distillation loss.

Faithful implementation of arXiv 2605.21924 §3.2-3.3.

Algorithm per prompt x with K sibling rollouts:
  1. a_t = max(log P_T(y_t|full) - log P_T(y_t|degraded), 0)         [VA, §3.1]
  2. ā^(k) = (1/T) Σ_t a_t^(k)                                        [trajectory mean]
  3. ẑ^(k) = (ā^(k) - μ) / (σ + ε)  where μ,σ over {ā^(j)}_{j=1..K}  [z-score]
  4. w^(k) = softmax(ẑ^(k) / τ)           (τ=1.0, sums to 1)           [rollout weight]
  5. HighVA = top= p_v of a_t,  LowVA = rest       (p_v = 0.20)        [token groups]
  6. L_group = λ·mean(KL_High) + (1-λ)·mean(KL_Low)  (λ = 0.50)       [grouped KL]
  7. L_VA-OPD(x) = Σ_k w^(k) * L_group^(k)                            [full objective]

  KL is REVERSE KL: KL(P_S || P_T) — mode-seeking, preferred for distillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .signal_decomposer import TeacherTopK, sampled_token_log_prob
from .verl_sparse_kd import compute_verl_sparse_reverse_kl


# ── VA-OPD defaults (from paper) ─────────────────────────────────────────
VA_LAMBDA = 0.50          # §3.3: λ = 0.5 (equal weighting of HighVA / LowVA)
VA_TOP_Q = 0.20           # §3.3: top p_v = 0.20 for HighVA split
VA_TAU = 1.0              # §3.2: rollout softmax temperature τ


@dataclass(frozen=True)
class VAOPDLossResult:
    loss: torch.Tensor                          # scalar: L_VA-OPD
    token_mean_loss: torch.Tensor               # scalar: token-mean KL (diagnostic)
    per_token_kl: torch.Tensor                  # [B, T]: raw per-token reverse KL
    per_rollout_loss: torch.Tensor              # [B]: rollout-weighted grouped KL
    va_pos: torch.Tensor                        # [B, T]: rectified visual advantage
    rollout_weights: torch.Tensor               # [B]: per-rollout VA softmax weight
    metrics: dict[str, torch.Tensor]


def compute_va(
    teacher_full: TeacherTopK,
    teacher_degraded: TeacherTopK,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Signed teacher visual advantage for sampled rollout tokens (§3.1).

    ``VA_t = log P_T(y_t | full) - log P_T(y_t | degraded)`` — NOT rectified.

    Uses exact teacher log-probs via full-vocab gather when available (no tail
    approximation).  Falls back to sampled_token_log_prob for backward compat.
    """
    if teacher_full.sampled_log_probs is not None and teacher_degraded.sampled_log_probs is not None:
        return teacher_full.sampled_log_probs - teacher_degraded.sampled_log_probs
    return sampled_token_log_prob(teacher_full, sampled_token_ids) - sampled_token_log_prob(
        teacher_degraded, sampled_token_ids
    )


def compute_rollout_va_weights(
    va_pos: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    prompt_ids: Sequence[int | str] | None = None,
    tau: float = VA_TAU,
) -> torch.Tensor:
    """Compute sibling-rollout VA softmax weights that sum to 1 (§3.2).

    Steps per prompt group with K rollouts:
      ā^(k) = (1/T) Σ_t a_t^(k)                    [mean over ALL tokens]
      ẑ^(k) = (ā^(k) - μ) / (σ + ε)                [z-score within group]
      w^(k) = softmax(ẑ^(k) / τ)                   [sums to 1]
    """
    if va_pos.ndim != 2:
        raise ValueError("va_pos must have shape [batch, seq]")
    mask = response_mask if response_mask is not None else torch.ones(
        va_pos.shape[:2], dtype=torch.bool, device=va_pos.device
    )

    # ā^(k) = mean over ALL valid tokens (not top-q)
    summaries = torch.tensor(
        [va_pos[i, mask[i]].float().mean().item() if mask[i].any() else 0.0
         for i in range(va_pos.shape[0])],
        dtype=torch.float32, device=va_pos.device,
    )

    groups = _prompt_groups(prompt_ids, va_pos.shape[0])
    weights = torch.ones_like(summaries)
    eps = 1e-8
    for _, indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long, device=va_pos.device)
        group_scores = summaries[idx]  # [K]
        K = float(len(indices))

        if K <= 1:
            # Solo rollout: weight = 1.0
            weights[idx] = torch.ones_like(group_scores)
        else:
            # ẑ^(k) = (ā - μ) / (σ + ε)  — z-score within group
            mu = group_scores.mean()
            sigma = group_scores.std()
            if sigma < eps:
                # All rollouts have nearly identical VA → uniform weights 1/K
                weights[idx] = torch.full_like(group_scores, 1.0 / K)
            else:
                z_scores = (group_scores - mu) / sigma
                # w^(k) = softmax(ẑ / τ)  — sums to 1
                weights[idx] = torch.softmax(z_scores / tau, dim=0)
    return weights


def split_high_low_va(
    va_pos: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    top_q: float = VA_TOP_Q,
    min_high_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each batch row into HighVA (top q) and LowVA token masks (§3.3)."""
    if mask is None:
        mask = torch.ones_like(va_pos, dtype=torch.bool)
    mask = mask.bool()
    high = torch.zeros_like(mask, dtype=torch.bool)
    for batch_index in range(va_pos.shape[0]):
        idx = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        count = max(min_high_tokens, int(math.ceil(float(idx.numel()) * top_q)))
        count = min(count, int(idx.numel()))
        _, order = torch.topk(va_pos[batch_index, idx], k=count, largest=True, sorted=False)
        high[batch_index, idx[order]] = True
    low = mask & ~high
    return high, low


def compute_va_opd_loss(
    student_logits: torch.Tensor,                    # [B, T, V]
    teacher_full: TeacherTopK,                       # teacher full-image top-k
    teacher_degraded: TeacherTopK,                   # teacher degraded-image top-k
    sampled_token_ids: torch.Tensor,                 # [B, T]
    *,
    response_mask: torch.Tensor | None = None,       # [B, T]
    prompt_ids: Sequence[int | str] | None = None,   # [B]
    top_q: float = VA_TOP_Q,
    tau_rollout: float = VA_TAU,
    lambda_high: float = VA_LAMBDA,
    min_high_tokens: int = 1,
    rollout_weights: torch.Tensor | None = None,
    renormalize_topk: bool = True,
    include_tail: bool = True,
    eps: float = 1e-8,
) -> VAOPDLossResult:
    """Compute VA-OPD grouped reverse-KL loss (§3.2-3.3).

    L_VA-OPD(x) = Σ_k w^(k) * [λ·mean(KL_rev_High) + (1-λ)·mean(KL_rev_Low)]
    where KL_rev = KL(P_S || P_T) is reverse (mode-seeking) KL.
    """
    if response_mask is None:
        response_mask = torch.ones(
            student_logits.shape[:2], dtype=torch.bool, device=student_logits.device
        )
    response_mask = response_mask.bool()
    B, T = response_mask.shape

    # ── 1. Per-token REVERSE KL (P_S || P_T) — mode-seeking, §3.3 ────────
    # Reverse KL = KL(P_student || P_teacher), preferred for distillation.
    reverse_kl_output = compute_verl_sparse_reverse_kl(
        student_logits=student_logits,
        teacher_topk_indices=teacher_full.token_ids.unsqueeze(1),   # [B, 1, T, K]
        teacher_topk_log_probs=teacher_full.log_probs.unsqueeze(1), # [B, 1, T, K]
        condition_weights=torch.ones(B, 1, T, device=student_logits.device),
        response_mask=response_mask,
        teacher_tail_log_prob=(
            teacher_full.tail_log_prob.unsqueeze(1)                 # [B, 1, T]
            if teacher_full.tail_log_prob is not None else None
        ),
        renormalize_topk=renormalize_topk,
        include_tail=include_tail,
        eps=eps,
    )
    per_token_kl = reverse_kl_output.per_token_loss  # [B, T] — already masked

    # ── 2. Visual advantage: a_t = max(log p_full - log p_deg, 0) §3.1 ──
    va_raw = compute_va(teacher_full, teacher_degraded, sampled_token_ids)
    va_pos = va_raw.clamp_min(0.0)

    if rollout_weights is None and prompt_ids is None and B > 1:
        raise ValueError("prompt_ids are required for VA-OPD batches with multiple rollouts")

    # ── 3. Rollout weights: ā → ẑ → w = softmax(ẑ/τ) §3.2 ───────────────
    if rollout_weights is None:
        rollout_weights = compute_rollout_va_weights(
            va_pos, response_mask=response_mask, prompt_ids=prompt_ids, tau=tau_rollout,
        )  # [B]
    else:
        rollout_weights = rollout_weights.to(device=student_logits.device, dtype=torch.float32)
        if rollout_weights.shape != (B,):
            raise ValueError("rollout_weights must have shape [B]")

    # ── 4. HighVA / LowVA grouped reverse KL §3.3 ────────────────────────
    high, low = split_high_low_va(va_pos, mask=response_mask, top_q=top_q,
                                   min_high_tokens=min_high_tokens)

    numerator = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    token_sum = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    token_count = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    per_rollout_loss = torch.zeros((B,), dtype=torch.float32, device=student_logits.device)

    for i in range(B):
        w_r = rollout_weights[i].float()
        if w_r <= 0:
            continue

        high_loss = _masked_mean(per_token_kl[i], high[i])
        low_loss = _masked_mean(per_token_kl[i], low[i])

        # L_group = λ·mean(KL_High) + (1-λ)·mean(KL_Low)  (λ=0.50)
        L_group = lambda_high * high_loss + (1.0 - lambda_high) * low_loss
        weighted_group_loss = w_r * L_group
        per_rollout_loss[i] = weighted_group_loss
        numerator = numerator + weighted_group_loss

        n_high = high[i].sum()
        n_low = low[i].sum()
        token_sum = token_sum + w_r * (lambda_high * n_high * high_loss +
                                        (1.0 - lambda_high) * n_low * low_loss)
        token_count = token_count + w_r * (lambda_high * n_high +
                                            (1.0 - lambda_high) * n_low)

    # L_VA-OPD = Σ_k w^(k) · L_group^(k)  (formula 7, no denominator)
    loss = numerator
    token_mean = token_sum / token_count.clamp_min(eps)

    metrics = {
        "va_opd/loss": loss.detach(),
        "va_opd/token_mean_loss": token_mean.detach(),
        "va_opd/rollout_weight_sum": rollout_weights.sum().detach().item(),
        "va/mean": _masked_mean(va_pos, response_mask).detach(),
        "va/sparsity": ((va_pos <= 0) & response_mask).float().sum() /
                       response_mask.float().sum().clamp_min(1.0),
    }

    return VAOPDLossResult(
        loss=loss,
        token_mean_loss=token_mean,
        per_token_kl=per_token_kl,
        per_rollout_loss=per_rollout_loss,
        va_pos=va_pos,
        rollout_weights=rollout_weights,
        metrics=metrics,
    )


# ── helpers ────────────────────────────────────────────────────────────────

def _top_mean(values: torch.Tensor, *, top_q: float) -> float:
    if values.numel() == 0:
        return 0.0
    count = max(1, int(math.ceil(float(values.numel()) * top_q)))
    top = torch.topk(values.float(), k=min(count, int(values.numel())), largest=True).values
    return float(top.mean().item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return selected.float().mean()


def _prompt_groups(
    prompt_ids: Sequence[int | str] | None, batch: int,
) -> dict[int | str, list[int]]:
    if prompt_ids is None:
        return {"__all__": list(range(batch))}
    if len(prompt_ids) != batch:
        raise ValueError("prompt_ids must have one item per batch row")
    groups: dict[int | str, list[int]] = {}
    for idx, label in enumerate(prompt_ids):
        groups.setdefault(label, []).append(idx)
    return groups
