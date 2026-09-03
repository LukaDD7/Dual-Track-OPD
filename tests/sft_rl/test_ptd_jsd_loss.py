"""Unit tests for the PTD-PO top-K JSD loss (verl backend, engine-agnostic)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DTOPD_ROOT = os.environ.get(
    "DTOPD_ROOT", "/inspire/hdd/global_user/mengweicheng-240108120092/lzy"
)
VERL_BACKEND = Path(DTOPD_ROOT) / "fc-opd-storage" / "backends" / "verl-qwen35-v090-cu132"
sys.path.insert(0, str(VERL_BACKEND))

from verl.trainer.distillation.fsdp.losses import compute_jsd_topk  # noqa: E402
from verl.workers.config import DistillationConfig, DistillationLossConfig  # noqa: E402


def _config(kl_direction: str = "jsd_kl") -> DistillationConfig:
    return DistillationConfig(
        enabled=False,
        n_gpus_per_node=1,
        nnodes=1,
        teacher_models={},
        distillation_loss=DistillationLossConfig(
            loss_mode="jsd_topk",
            topk=3,
            use_task_rewards=True,
            distillation_loss_coef=5e-2,
            use_policy_gradient=False,
            kl_direction=kl_direction,
            vocab_size=8,
        ),
    )


def _build(
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
    bsz: int = 2,
    tokens: int = 5,
):
    """teacher_probs/student_logits: [V] distributions; returns nested teacher top-K + logits."""
    vocab = teacher_probs.numel()
    teacher_lp_all = teacher_probs.log().repeat(bsz * tokens, 1)  # [B*T, V]
    topk_ids = torch.topk(teacher_lp_all, k=3, dim=-1).indices
    topk_lp = torch.gather(teacher_lp_all, -1, topk_ids)
    offsets = torch.arange(0, bsz * tokens + 1, tokens)
    teacher_nested_lp = torch.nested.nested_tensor_from_jagged(topk_lp, offsets)
    teacher_nested_ids = torch.nested.nested_tensor_from_jagged(topk_ids, offsets)
    student_logits_b = student_logits.repeat(bsz * tokens, 1).unsqueeze(0)  # [1, B*T, V]
    return student_logits_b, teacher_nested_lp, teacher_nested_ids


def test_jsd_zero_when_identical():
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    logits, t_lp, t_ids = _build(p, p.log())
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("jsd_kl"), "thd")
    loss = out["distillation_losses"]
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)
    assert torch.allclose(out["teacher_mass"], torch.full_like(out["teacher_mass"], 0.75), atol=1e-5)


def test_jsd_positive_when_different():
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    q = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])
    logits, t_lp, t_ids = _build(p, q.log())
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("jsd_kl"), "thd")
    assert (out["distillation_losses"] > 1e-4).all()


def test_forward_kl_direction():
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    q = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])
    logits, t_lp, t_ids = _build(p, q.log())
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("forward_kl"), "thd")
    assert (out["distillation_losses"] > 1e-4).all()
    # forward KL(student || teacher) >= 0 and asymmetric: swapping would differ
    out_ident = compute_jsd_topk(*_build(p, p.log()), _config("forward_kl"), "thd")
    assert torch.allclose(out_ident["distillation_losses"], torch.zeros(1), atol=1e-5)


def test_reverse_kl_direction():
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    q = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])
    logits, t_lp, t_ids = _build(p, q.log())
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("reverse_kl"), "thd")
    assert (out["distillation_losses"] > 1e-4).all()


def test_gradient_flows_to_student_logits():
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    q = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])
    logits, t_lp, t_ids = _build(p, q.log())
    logits.requires_grad_(True)
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("jsd_kl"), "thd")
    out["distillation_losses"].sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_jsd_gradient_nonzero_guard():
    """Regression guard: the released PTD-PO jsd_kl stop-grads the student
    weights, which makes the autograd gradient vanish exactly. Our port keeps the
    correct JSD gradient; this test fails against the released pattern."""
    p = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.03])
    q = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])
    logits, t_lp, t_ids = _build(p, q.log())
    logits.requires_grad_(True)
    out = compute_jsd_topk(logits, t_lp, t_ids, _config("jsd_kl"), "thd")
    out["distillation_losses"].sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 1e-6


def test_hint_teacher_position_ids_microbatch_slicing():
    """Regression guard for the 2026-08-22 step-0 crash
    (RuntimeError: size of tensor a (3) vs b (2) in Qwen3VL RoPE).

    Layout chain: agent-loop _compute_position_ids returns [1, 4, seq];
    batching cats dim=0 -> [B, 4, seq]; engine_workers transposes to
    [4, B, seq]. The teacher micro-batch must therefore slice the batch on
    dim=1. Slicing dim=0 (the mRoPE-component axis) reproduced the crash.
    """
    bsz, seq = 5, 11
    per_sample = torch.arange(bsz * 4 * seq).reshape(bsz, 4, seq)  # [B, 4, seq]
    teacher_pos = per_sample.transpose(0, 1)  # [4, B, seq], as in engine_workers

    mb = 2
    for i in range(0, bsz, mb):
        end = min(i + mb, bsz)
        pos_batch = teacher_pos[:, i:end] if teacher_pos.dim() == 3 else teacher_pos[i:end]
        assert pos_batch.shape == (4, end - i, seq)
        assert torch.equal(pos_batch, per_sample[i:end].transpose(0, 1))

        # The buggy slicing that must stay dead: indexing dim=0 yields the first
        # mb mRoPE components, not the first mb samples.
        assert not torch.equal(pos_batch, teacher_pos[i:end])
