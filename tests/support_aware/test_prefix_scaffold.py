"""CPU unit tests for STP-OPD scaffold primitives."""

import torch

from dual_track_opd.support_aware.prefix_scaffold import (
    assign_scaffold,
    composed_loss,
    hybrid_prefix_ids,
    masked_mean,
    paired_batch,
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


def test_masked_mean_empty_returns_none():
    values = torch.tensor([1.0, 2.0])
    empty = torch.zeros(2, dtype=torch.bool)
    assert masked_mean(values, empty) is None


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
    assert terms.prefix_fkl is not None and abs(terms.prefix_fkl - 0.2) < 1e-6
    assert terms.suffix_rkl is not None and abs(terms.suffix_rkl - 0.5) < 1e-6
    # prefix term must not depend on suffix values
    prefix_fkl_other = torch.tensor([0.1, 0.3, 1.0, 1.0, 1.0, 1.0])
    _, terms_other = composed_loss(
        prefix_fkl_other, suffix_rkl, suffix_pg, prefix_mask, suffix_mask
    )
    assert abs(terms_other.prefix_fkl - terms.prefix_fkl) < 1e-9


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
    by_prompt: dict[str, set[bool]] = {}
    for record in records:
        by_prompt.setdefault(str(record["prompt_id"]), set()).add(bool(record["scaffolded"]))
    for prompt_id in prompts:
        assert by_prompt[prompt_id] == {True, False}


def test_hybrid_prefix_ids():
    assert hybrid_prefix_ids([1, 2], [3, 4, 5]) == (1, 2, 3, 4, 5)
