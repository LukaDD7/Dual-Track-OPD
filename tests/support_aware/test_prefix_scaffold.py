"""CPU unit tests for STP-OPD scaffold primitives."""

import torch

from dual_track_opd.support_aware.prefix_scaffold import (
    assign_scaffold,
    composed_loss,
    hybrid_prefix_ids,
    masked_mean,
    paired_batch,
    realized_scaffold_share,
    region_masks,
    scaffold_fraction,
    validate_masks,
)


def test_region_masks_scaffolded():
    prefix_mask, suffix_mask = region_masks(response_length=10, prefix_length=3, scaffolded=True)
    assert prefix_mask[:3].all() and not prefix_mask[3:].any()
    assert not suffix_mask[:3].any() and suffix_mask[3:].all()
    validate_masks(prefix_mask, suffix_mask)


def test_region_masks_unscaffolded():
    prefix_mask, suffix_mask = region_masks(response_length=8, prefix_length=0, scaffolded=False)
    assert not prefix_mask.any()
    assert suffix_mask.all()


def test_masked_mean_empty_returns_zero_tensor():
    values = torch.tensor([1.0, 2.0])
    empty = torch.zeros(2, dtype=torch.bool)
    result = masked_mean(values, empty)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 0.0
    assert not torch.isnan(result)


def test_composed_loss_region_independent():
    response_length = 6
    prefix_length = 2
    prefix_mask, suffix_mask = region_masks(response_length, prefix_length, scaffolded=True)
    prefix_fkl = torch.tensor([0.1, 0.3, 9.9, 9.9, 9.9, 9.9])
    suffix_rkl = torch.tensor([9.9, 9.9, 0.2, 0.4, 0.6, 0.8])
    suffix_pg = torch.zeros(response_length)
    total, terms = composed_loss(
        prefix_fkl, suffix_rkl, suffix_pg, prefix_mask, suffix_mask,
        lambda_prefix=1.0, lambda_distill=1.0, lambda_task=1.0,
    )
    assert isinstance(total, torch.Tensor)
    assert abs(terms.prefix_fkl.item() - 0.2) < 1e-6
    assert abs(terms.suffix_rkl.item() - 0.5) < 1e-6
    assert abs(total.item() - 0.7) < 1e-6
    # prefix term must not depend on suffix values
    prefix_fkl_other = torch.tensor([0.1, 0.3, 1.0, 1.0, 1.0, 1.0])
    _, terms_other = composed_loss(
        prefix_fkl_other, suffix_rkl, suffix_pg, prefix_mask, suffix_mask
    )
    assert abs(terms_other.prefix_fkl.item() - terms.prefix_fkl.item()) < 1e-9


def test_composed_loss_gradients_isolated_by_region():
    response_length = 6
    prefix_length = 2
    prefix_mask, suffix_mask = region_masks(response_length, prefix_length, scaffolded=True)
    prefix_param = torch.tensor([0.1, 0.3], requires_grad=True)
    suffix_param = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    # prefix term depends only on prefix_param; suffix terms only on suffix_param
    prefix_fkl = torch.zeros(response_length)
    suffix_rkl = torch.zeros(response_length)
    suffix_pg = torch.zeros(response_length)
    prefix_fkl = prefix_fkl.index_put(
        (prefix_mask,), prefix_param, accumulate=True
    )
    suffix_rkl = suffix_rkl.index_put(
        (suffix_mask,), suffix_param, accumulate=True
    )
    total, _ = composed_loss(
        prefix_fkl, suffix_rkl, suffix_pg, prefix_mask, suffix_mask,
        lambda_prefix=1.0, lambda_distill=1.0, lambda_task=1.0,
    )
    assert total.requires_grad
    total.backward()
    assert prefix_param.grad is not None
    assert suffix_param.grad is not None
    assert torch.isfinite(prefix_param.grad).all()
    assert torch.isfinite(suffix_param.grad).all()
    assert bool((prefix_param.grad != 0).any())
    assert bool((suffix_param.grad != 0).any())


def test_composed_loss_empty_region_has_zero_gradient_not_nan():
    response_length = 4
    prefix_mask, _ = region_masks(response_length, prefix_length=4, scaffolded=True)
    # suffix mask empty: suffix terms must contribute 0 (never NaN)
    empty_suffix = ~prefix_mask
    param = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    prefix_fkl = param * 2.0
    suffix_rkl = param * 3.0
    suffix_pg = torch.zeros(response_length)
    total, terms = composed_loss(
        prefix_fkl, suffix_rkl, suffix_pg, prefix_mask, empty_suffix,
        lambda_prefix=1.0, lambda_distill=1.0, lambda_task=1.0,
    )
    assert not torch.isnan(total)
    assert terms.suffix_rkl.item() == 0.0
    total.backward()
    assert param.grad is not None and torch.isfinite(param.grad).all()
    assert torch.allclose(param.grad, torch.full_like(param.grad, 0.5))


def test_scaffold_fraction_schedule():
    assert scaffold_fraction(1) == 0.75
    assert scaffold_fraction(20) == 0.75
    assert scaffold_fraction(21) == 0.50
    assert scaffold_fraction(40) == 0.50
    assert scaffold_fraction(41) == 0.25
    assert scaffold_fraction(60) == 0.25
    assert scaffold_fraction(61) == 0.0
    assert scaffold_fraction(0) == 0.0


def test_assign_scaffold_deterministic():
    prompts = [f"geo3k:{index}" for index in range(20)]
    first = assign_scaffold(prompts, step=10, seed=42)
    second = assign_scaffold(prompts, step=10, seed=42)
    assert first == second
    fraction = sum(first.values()) / len(first)
    assert 0.4 < fraction < 0.95  # 0.75 fraction with n=20 is well-separated


def test_paired_batch_covers_both_arms():
    prompts = ["geo3k:1", "geo3k:2", "geo3k:3"]
    records = paired_batch(prompts, step=10, seed=7)
    assert len(records) == 2 * len(prompts)
    by_prompt: dict[str, list[bool]] = {}
    for record in records:
        by_prompt.setdefault(str(record["prompt_id"]), []).append(record["scaffolded"])
    assert len(by_prompt) == len(prompts)
    for flags in by_prompt.values():
        assert sorted(flags) == [False, True]
    # deterministic for the same (step, seed)
    assert [r["scaffolded"] for r in records] == [
        r["scaffolded"] for r in paired_batch(prompts, step=10, seed=7)
    ]
    # assignment flags are surfaced and consistent per prompt
    assert all("assigned_scaffolded" in record for record in records)
    assert len({r["assigned_scaffolded"] for r in records}) == 2


def test_paired_batch_realized_share_follows_schedule():
    prompts = [f"geo3k:{index}" for index in range(200)]
    for step, expected_fraction in ((10, 0.75), (30, 0.50), (50, 0.25), (70, 0.0)):
        shares = [
            realized_scaffold_share(paired_batch(prompts, step=step, seed=seed))
            for seed in range(30)
        ]
        mean_share = sum(shares) / len(shares)
        assert abs(mean_share - expected_fraction) < 0.08


def test_hybrid_prefix_ids():
    assert hybrid_prefix_ids([1, 2], [3, 4, 5]) == (1, 2, 3, 4, 5)
