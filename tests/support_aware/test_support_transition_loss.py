"""CPU tests for the token-level STP-OPD regional objective."""

import torch

from dual_track_opd.support_aware.prefix_scaffold import region_masks
from dual_track_opd.support_aware.support_transition_loss import (
    prefix_fkl_ce,
    stp_opd_loss,
    suffix_rkl_k1,
    suffix_rkl_k1_sampled,
    suffix_task_grpo,
    topk_tail_fkl,
)


def _tensors(batch=2, seq=8, vocab=16, prefix_length=3):
    prefix_mask, suffix_mask = region_masks(seq, prefix_length, scaffolded=True)
    prefix_mask = prefix_mask.unsqueeze(0).expand(batch, -1)
    suffix_mask = suffix_mask.unsqueeze(0).expand(batch, -1)
    student_logits = torch.randn(batch, seq, vocab, requires_grad=True)
    teacher_logits = torch.randn(batch, seq, vocab)
    prefix_ids = torch.randint(0, vocab, (batch, seq))
    sampled_ids = torch.randint(0, vocab, (batch, seq))
    advantages = torch.randn(batch, seq)
    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "prefix_ids": prefix_ids,
        "sampled_ids": sampled_ids,
        "advantages": advantages,
        "prefix_mask": prefix_mask,
        "suffix_mask": suffix_mask,
    }


def test_prefix_fkl_ce_is_differentiable_and_region_normalized():
    tensors = _tensors()
    loss = prefix_fkl_ce(
        tensors["student_logits"],
        tensors["prefix_ids"],
        tensors["prefix_mask"],
    )
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert tensors["student_logits"].grad is not None
    assert torch.isfinite(tensors["student_logits"].grad).all()
    assert bool((tensors["student_logits"].grad != 0).any())


def test_topk_tail_fkl_uses_soft_teacher_distribution() -> None:
    tensors = _tensors(vocab=7)
    loss = topk_tail_fkl(
        tensors["student_logits"],
        tensors["teacher_logits"],
        tensors["prefix_mask"],
        top_k=3,
    )
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert tensors["student_logits"].grad is not None
    assert torch.isfinite(tensors["student_logits"].grad).all()


def test_topk_tail_fkl_matches_exact_fkl_when_k_covers_vocab() -> None:
    tensors = _tensors(vocab=5)
    actual = topk_tail_fkl(
        tensors["student_logits"],
        tensors["teacher_logits"],
        tensors["prefix_mask"],
        top_k=100,
    )
    teacher_logp = torch.log_softmax(tensors["teacher_logits"].float(), dim=-1)
    student_logp = torch.log_softmax(tensors["student_logits"].float(), dim=-1)
    exact_tokens = torch.sum(
        teacher_logp.exp() * (teacher_logp - student_logp),
        dim=-1,
    )
    expected = exact_tokens[tensors["prefix_mask"]].mean()
    assert torch.allclose(actual, expected, atol=1e-6)


def test_suffix_rkl_k1_is_differentiable():
    tensors = _tensors()
    loss = suffix_rkl_k1(
        tensors["student_logits"],
        tensors["teacher_logits"],
        tensors["suffix_mask"],
    )
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert tensors["student_logits"].grad is not None
    assert bool((tensors["student_logits"].grad != 0).any())


def test_suffix_rkl_k1_sampled_is_minus_mean_teacher_log_prob():
    tensors = _tensors()
    teacher_k1 = torch.randn_like(tensors["suffix_mask"].float())
    loss = suffix_rkl_k1_sampled(teacher_k1, tensors["suffix_mask"])
    expected = -(teacher_k1[tensors["suffix_mask"]].mean())
    assert torch.allclose(loss, expected)
    assert not torch.isnan(loss)
    # empty suffix -> zero tensor, no NaN
    empty = torch.zeros_like(tensors["suffix_mask"])
    assert suffix_rkl_k1_sampled(teacher_k1, empty).item() == 0.0


def test_suffix_task_grpo_excludes_prefix_positions():
    tensors = _tensors()
    loss = suffix_task_grpo(
        tensors["student_logits"],
        tensors["sampled_ids"],
        tensors["advantages"],
        tensors["suffix_mask"],
    )
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    grad = tensors["student_logits"].grad
    assert grad is not None
    # prefix positions must have zero PG gradient (excluded from suffix region)
    assert bool((grad[:, :3, :] == 0).all())
    assert bool((grad[:, 3:, :] != 0).any())


def test_stp_opd_loss_regions_disjoint_and_lamdas_scale():
    tensors = _tensors()
    total, terms = stp_opd_loss(
        tensors["student_logits"],
        tensors["teacher_logits"],
        prefix_ids=tensors["prefix_ids"],
        sampled_ids=tensors["sampled_ids"],
        advantages=tensors["advantages"],
        prefix_mask=tensors["prefix_mask"],
        suffix_mask=tensors["suffix_mask"],
    )
    assert torch.isfinite(total)
    base, _ = stp_opd_loss(
        tensors["student_logits"],
        tensors["teacher_logits"],
        prefix_ids=tensors["prefix_ids"],
        sampled_ids=tensors["sampled_ids"],
        advantages=tensors["advantages"],
        prefix_mask=tensors["prefix_mask"],
        suffix_mask=tensors["suffix_mask"],
        lambda_prefix=2.0,
    )
    # doubling lambda_prefix scales only the prefix term
    assert torch.allclose(
        base - total,
        terms["prefix_fkl"],
        atol=1e-5,
    )


def test_stp_opd_loss_valid_mask_and_empty_regions():
    tensors = _tensors(prefix_length=8)  # suffix empty
    valid = torch.ones_like(tensors["prefix_mask"], dtype=torch.bool)
    total, terms = stp_opd_loss(
        tensors["student_logits"],
        tensors["teacher_logits"],
        prefix_ids=tensors["prefix_ids"],
        sampled_ids=tensors["sampled_ids"],
        advantages=tensors["advantages"],
        prefix_mask=tensors["prefix_mask"],
        suffix_mask=tensors["suffix_mask"],
        valid_mask=valid,
    )
    assert not torch.isnan(total)
    assert terms["suffix_rkl"].item() == 0.0
    assert terms["suffix_pg"].item() == 0.0
    total.backward()
    grad = tensors["student_logits"].grad
    assert grad is not None and torch.isfinite(grad).all()
    # invalid alignment mask empties the prefix region too
    invalid = torch.zeros_like(valid)
    total_invalid, _ = stp_opd_loss(
        tensors["student_logits"],
        tensors["teacher_logits"],
        prefix_ids=tensors["prefix_ids"],
        sampled_ids=tensors["sampled_ids"],
        advantages=tensors["advantages"],
        prefix_mask=tensors["prefix_mask"],
        suffix_mask=tensors["suffix_mask"],
        valid_mask=invalid,
    )
    assert total_invalid.item() == 0.0
