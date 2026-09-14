"""Unit tests for TailSFT online sequence-level filtering (arXiv:2608.25756).

Covers, engine- and process-group-free:
- tailsft_keep_mask floor semantics + tie handling + degenerate fractions
- tailsft_filter_fraction static / ramp / clamping
- _tailsft_per_example_ce against a hand-computed reference
- init_ce_from_logits roll alignment (annotate-side ℓ0 == loss-side ℓt on
  identical log_probs)
- tailsft_loss fallback paths (no init_ce -> plain sft_loss; gamma<=0 ->
  plain sft_loss; pad_mode != NO_PADDING -> plain sft_loss)
- tailsft_loss filtering math on a synthetic jagged micro-batch (dropped
  sequences' tokens leave the gradient; retained-token denominator)
- γ_t / init_ce meta threading through get_tensordict +
  index_select_tensor_dict (the micro-batch sub-slicing path)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

DTOPD_ROOT = os.environ.get(
    "DTOPD_ROOT", "/inspire/hdd/global_user/mengweicheng-240108120092/lzy"
)
VERL_BACKEND = Path(DTOPD_ROOT) / "fc-opd-storage" / "backends" / "verl-qwen35-v090-cu132"
sys.path.insert(0, str(VERL_BACKEND))

from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.utils.dataset.dataset_utils import DatasetPadMode  # noqa: E402
from verl.workers.utils.losses import (  # noqa: E402
    _tailsft_per_example_ce,
    sft_loss,
    tailsft_filter_fraction,
    tailsft_keep_mask,
    tailsft_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sft_rl"))

from annotate_tailsft_init_ce import init_ce_from_logits  # noqa: E402


# --- helpers -----------------------------------------------------------------


def _jagged_micro_batch(seq_logprobs, seq_masks, init_ce=None, gamma=None, dp_size=1):
    """Build a no_padding TensorDict mirroring the sft dynamic-bsz packing.

    seq_logprobs / seq_masks: per-sequence 1-D lists (values / rolled mask
    semantics are applied by the caller where needed). Returns
    (tensordict, model_output) usable by both sft_loss and tailsft_loss.
    """
    lengths = [len(m) for m in seq_masks]
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int64)
    flat_lp = torch.cat([torch.as_tensor(lp, dtype=torch.float32) for lp in seq_logprobs])
    flat_mask = torch.cat([torch.as_tensor(m, dtype=torch.float32) for m in seq_masks])
    log_probs = torch.nested.nested_tensor_from_jagged(flat_lp, offsets)
    loss_mask = torch.nested.nested_tensor_from_jagged(flat_mask, offsets)
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(sum(lengths), dtype=torch.long), offsets
    )

    tensor_dict = {
        "log_probs": log_probs,
        "loss_mask": loss_mask,
        "input_ids": input_ids,
    }
    # scalars live in non_tensor_dict (the real path assigns dp_size /
    # batch_num_tokens via tu.assign_non_tensor after an all-reduce)
    non_tensor = {
        "pad_mode": DatasetPadMode.NO_PADDING,
        "dp_size": dp_size,
        "batch_num_tokens": float(flat_mask.sum().item()),
    }
    if init_ce is not None:
        tensor_dict["init_ce"] = list(init_ce)
    if gamma is not None:
        non_tensor["tailsft_filter_fraction"] = gamma
    data = tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor)
    model_output = {"log_probs": log_probs}
    return data, model_output


# --- keep-mask ---------------------------------------------------------------


def test_keep_mask_drops_most_negative_half():
    margins = torch.tensor([0.1, -2.0, 0.3, -1.5, 0.0, -0.2])
    keep = tailsft_keep_mask(margins, 0.5)
    # k = floor(6 * 0.5) = 3 dropped: -2.0, -1.5, -0.2
    assert keep.tolist() == [True, False, True, False, True, False]


def test_keep_mask_floor_semantics():
    margins = torch.tensor([1.0, 2.0, 3.0])
    keep = tailsft_keep_mask(margins, 0.5)
    # k = floor(3 * 0.5) = 1 dropped
    assert keep.tolist() == [False, True, True]


def test_keep_mask_ties_broken_deterministically():
    """All margins equal -> stable argsort drops exactly k (by position)."""
    margins = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    keep = tailsft_keep_mask(margins, 0.5)
    # k = 2; stable argsort ranks [0,1,2,3] -> drops first 2 by position
    assert keep.tolist() == [False, False, True, True]
    assert int((~keep).sum()) == 2

    # ties spanning the cutoff: k=2 drops both -1.0 entries, keeps the rest
    margins2 = torch.tensor([-1.0, -1.0, 0.5, 2.0])
    keep2 = tailsft_keep_mask(margins2, 0.5)
    assert int((~keep2).sum()) == 2
    assert keep2.tolist() == [False, False, True, True]


def test_keep_mask_degenerate_fractions():
    margins = torch.tensor([1.0, 2.0, 3.0])
    assert tailsft_keep_mask(margins, 0.0).all()
    assert not tailsft_keep_mask(margins, 1.0).any()
    assert tailsft_keep_mask(torch.empty(0), 0.5).numel() == 0


# --- gamma schedule ----------------------------------------------------------


def test_filter_fraction_static():
    assert tailsft_filter_fraction(0, "static", 0.5, 800) == 0.5
    assert tailsft_filter_fraction(12345, "static", 0.5, 800) == 0.5


def test_filter_fraction_ramp():
    assert tailsft_filter_fraction(0, "ramp", 0.5, 800) == 0.0
    assert abs(tailsft_filter_fraction(400, "ramp", 0.5, 800) - 0.25) < 1e-9
    assert abs(tailsft_filter_fraction(800, "ramp", 0.5, 800) - 0.5) < 1e-9
    # clamped after ramp end
    assert abs(tailsft_filter_fraction(5000, "ramp", 0.5, 800) - 0.5) < 1e-9


def test_filter_fraction_ramp_zero_steps():
    assert tailsft_filter_fraction(10, "ramp", 0.5, 0) == 0.5


def test_filter_fraction_unknown_schedule_raises():
    with pytest.raises(ValueError):
        tailsft_filter_fraction(0, "linear", 0.5, 800)


# --- per-example CE ----------------------------------------------------------


def test_per_example_ce_matches_hand_computation():
    # two sequences; flat tensors with offsets [0, 3, 7]
    lp = torch.tensor([math_log(0.5), math_log(0.5), math_log(0.25), math_log(0.5), math_log(0.25), math_log(0.25), 0.0])
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    offsets = torch.tensor([0, 3, 7])

    ce, n_tok = _tailsft_per_example_ce(lp, mask, offsets)
    # seq 0: targets at rolled positions 1,2 -> -mean(log .5, log .25)
    exp0 = -(math_log(0.5) + math_log(0.25)) / 2
    # seq 1: targets at rolled positions 4,5 -> -mean(log .25, log .25); pos 6 masked
    exp1 = -(math_log(0.25) + math_log(0.25)) / 2
    assert torch.allclose(ce, torch.tensor([exp0, exp1]), atol=1e-6)
    assert n_tok.tolist() == [2.0, 2.0]


def math_log(x):
    import math

    return math.log(x)


def test_per_example_ce_zero_target_tokens():
    # a sequence with an all-zero rolled mask yields CE 0 with n_tok clamped
    lp = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([0.0, 0.0, 0.0, 0.0])
    offsets = torch.tensor([0, 4])
    ce, n_tok = _tailsft_per_example_ce(lp, mask, offsets)
    assert torch.allclose(ce, torch.zeros(1), atol=1e-9)
    assert n_tok.tolist() == [0.0]


# --- annotate-side l0 roll alignment -----------------------------------------


def test_init_ce_from_logits_roll_alignment():
    """l0 from (logits -> target logprobs) must equal the loss-side per-example
    CE computed on the same target logprobs, when both use the same mask."""
    torch.manual_seed(0)
    seq_len, vocab = 9, 7
    input_ids = torch.randint(0, vocab, (seq_len,))
    loss_mask = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=torch.float32)
    logits = torch.randn(seq_len, vocab)

    ce, n_valid = init_ce_from_logits(logits, input_ids, loss_mask)
    # manual: targets = input_ids[1:], valid = loss_mask[1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt_lp = lp.gather(-1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
    valid = loss_mask[1:].to(torch.bool)
    manual = -(tgt_lp[valid].float()).mean().item()
    assert abs(ce - manual) < 1e-6
    assert n_valid == int(valid.sum().item())

    # and the loss-side functional on the SAME logprobs gives the same CE.
    # Real-path shapes: log_prob_flatten is the UNROLLED length (seq_len) —
    # entry i holds log P(x_{i+1}); rolled_mask = roll(loss_mask, -1) also has
    # length seq_len (position seq_len-1 wraps to loss_mask[0]). The wrapped
    # tail position is mask-excluded here, so any value works there.
    lp_values = torch.cat([tgt_lp, torch.zeros(1)])  # length 9, pos 8 unused
    rolled_mask = torch.roll(loss_mask, shifts=-1, dims=0)
    ce_loss_side, _ = _tailsft_per_example_ce(
        lp_values, rolled_mask, torch.tensor([0, seq_len])
    )
    assert abs(ce_loss_side[0].item() - ce) < 1e-6
    assert abs(ce_loss_side[0].item() - manual) < 1e-6


# --- tailsft_loss fallbacks --------------------------------------------------


def test_tailsft_loss_no_init_ce_falls_back_to_sft():
    lp = [[math_log(0.5), math_log(0.25)], [math_log(0.5)]]
    masks = [[1.0, 1.0], [1.0]]
    data, out = _jagged_micro_batch(lp, masks, init_ce=None, gamma=0.5)
    l_ts, m_ts = tailsft_loss(None, out, data, dp_group=None)
    l_sft, _ = sft_loss(None, out, data, dp_group=None)
    assert torch.allclose(l_ts, l_sft)
    assert m_ts == {}


def test_tailsft_loss_gamma_zero_falls_back_to_sft():
    lp = [[math_log(0.5)], [math_log(0.25)]]
    masks = [[1.0], [1.0]]
    data, out = _jagged_micro_batch(lp, masks, init_ce=[0.5, 0.4], gamma=0.0)
    l_ts, _ = tailsft_loss(None, out, data, dp_group=None)
    l_sft, _ = sft_loss(None, out, data, dp_group=None)
    assert torch.allclose(l_ts, l_sft)


def test_tailsft_loss_none_init_ce_falls_back():
    lp = [[math_log(0.5)], [math_log(0.25)]]
    masks = [[1.0], [1.0]]
    # init_ce present but None values (non-annotated rows) -> plain SFT
    data, out = _jagged_micro_batch(lp, masks, init_ce=[None, None], gamma=0.5)
    l_ts, _ = tailsft_loss(None, out, data, dp_group=None)
    l_sft, _ = sft_loss(None, out, data, dp_group=None)
    assert torch.allclose(l_ts, l_sft)


def test_tailsft_loss_pad_mode_falls_back():
    # padded layout: tailsft routes to sft_loss's padding branch, which needs
    # plain (padded) log_probs + response_mask instead of nested tensors.
    lp = torch.tensor([[math_log(0.5), math_log(0.25)], [math_log(0.2), 0.0]])
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    data = tu.get_tensordict(
        tensor_dict={
            "log_probs": lp,
            "response_mask": response_mask,
            "input_ids": torch.zeros(2, 2, dtype=torch.long),
        },
        non_tensor_dict={
            "pad_mode": DatasetPadMode.RIGHT,
            "dp_size": 1,
            "batch_num_tokens": 3.0,  # mask.sum() over the padded batch
            "init_ce": [0.05, 1.5],
            "tailsft_filter_fraction": 0.5,
        },
    )
    l_ts, m_ts = tailsft_loss(None, {"log_probs": lp}, data, dp_group=None)
    l_sft, _ = sft_loss(None, {"log_probs": lp}, data, dp_group=None)
    assert torch.allclose(l_ts, l_sft)
    assert m_ts == {}


# --- tailsft_loss filtering math ---------------------------------------------


def test_tailsft_loss_drops_smallest_margin_sequence():
    """2 sequences, gamma=0.5 -> 1 dropped. The dropped one is the sequence
    whose current CE improved MOST vs init_ce (most negative margin)."""
    lp = [[math_log(0.5), math_log(0.5)], [math_log(0.2), math_log(0.2)]]
    masks = [[1.0, 1.0], [1.0, 1.0]]
    # seq0: lt = -log(0.5) ~ 0.693; seq1: lt = -log(0.2) ~ 1.609
    lt = [0.6931, 1.6094]
    # margins: seq0: 0.6931 - 0.05 = 0.6431 (improved a lot); seq1: 1.6094 - 1.5 = 0.1094
    # smallest margin = seq1 -> dropped. Retained tokens = 2 (seq0 only)
    data, out = _jagged_micro_batch(lp, masks, init_ce=[0.05, 1.5], gamma=0.5)
    loss, metrics = tailsft_loss(None, out, data, dp_group=None)

    expected = -(math_log(0.5) + math_log(0.5)) / 2.0  # token-mean over seq0
    assert abs(loss.item() - expected) < 1e-6
    assert metrics["tailsft/dropped"] == 1.0
    assert metrics["tailsft/batch"] == 2.0
    assert metrics["tailsft/retained_tokens"] == 2.0
    assert abs(metrics["tailsft/margin_mean"] - (0.6431 + 0.1094) / 2) < 1e-3


def test_tailsft_loss_keeps_all_when_margins_equal():
    """All margins equal -> exactly k dropped by stable rank; the rest are
    averaged, so the loss equals the token-mean over the retained half."""
    lp = [[math_log(0.5)], [math_log(0.5)], [math_log(0.5)], [math_log(0.5)]]
    masks = [[1.0], [1.0], [1.0], [1.0]]
    data, out = _jagged_micro_batch(lp, masks, init_ce=[0.5, 0.5, 0.5, 0.5], gamma=0.5)
    loss, metrics = tailsft_loss(None, out, data, dp_group=None)
    expected = -math_log(0.5)  # uniform per-token CE -> same mean on any subset
    assert abs(loss.item() - expected) < 1e-6
    assert metrics["tailsft/dropped"] == 2.0
    assert metrics["tailsft/retained_tokens"] == 2.0


# --- engine multi-micro-batch aggregation (the 2x-loss regression) ------------


def test_tailsft_loss_step_level_denominator_survives_microbatch_sum():
    """Regression for the dynamic-bsz 2x-loss/grad bug (qwen3vl_sft_tailsft
    step-24 class): forward_backward_batch calls the loss per micro-batch and
    postprocess_batch_func SUMS the losses, so each micro-batch must divide
    by the same step-global constant for the sum to reconstruct the global
    retained-token mean. With the step-level denominator
    batch_num_tokens*(1-γ), M micro-batch losses sum exactly to the mean over
    retained tokens (dp_size=1).

    The filter itself is per-micro-batch (k = floor(n_mb * γ) drops the
    smallest-margin sequences of THAT micro-batch). The OLD code divided each
    micro-batch by its own retained-token count instead, making every
    micro-batch loss a complete mean: M=2 micro-batches summed to an exact
    2x loss AND gradient at heavy-token steps. Old code: total = 1.2037
    (retained sum, no averaging); new code: total = 0.6018 (mean over the
    2 retained tokens). This test pins the correct value."""
    # step: 4 sequences, 1 target token each, γ=0.5, split into 2
    # micro-batches of 2. Each micro-batch drops its smaller-margin sequence.
    # seq margins: mb0 = [0.643, 0.109] (drop seq1), mb1 = [0.501, 0.204]
    # (drop seq3). Retained: seq0 (-log .5), seq2 (-log .6).
    mb0_data, mb0_out = _jagged_micro_batch(
        [[math_log(0.5)], [math_log(0.2)]], [[1.0], [1.0]], init_ce=[0.05, 1.5], gamma=0.5
    )
    mb1_data, mb1_out = _jagged_micro_batch(
        [[math_log(0.6)], [math_log(0.3)]], [[1.0], [1.0]], init_ce=[0.01, 1.0], gamma=0.5
    )
    # step-global batch_num_tokens = 4 (all-reduced over the whole mini-batch
    # BEFORE micro-batching; identical constant injected into every
    # micro-batch, exactly as transformer_impl does for sft_loss)
    tu.assign_non_tensor(mb0_data, batch_num_tokens=4.0)
    tu.assign_non_tensor(mb1_data, batch_num_tokens=4.0)

    l0, m0 = tailsft_loss(None, mb0_out, mb0_data, dp_group=None)
    l1, m1 = tailsft_loss(None, mb1_out, mb1_data, dp_group=None)
    assert m0["tailsft/dropped"] == 1.0 and m1["tailsft/dropped"] == 1.0
    assert m0["tailsft/retained_tokens"] == 1.0 and m1["tailsft/retained_tokens"] == 1.0

    # engine total = SUM of micro-batch losses
    total = l0.item() + l1.item()
    retained_sum = -(math_log(0.5) + math_log(0.6))  # seq0 + seq2 target CE
    expected = retained_sum / 2.0  # mean over the step's 2 retained tokens
    assert abs(total - expected) < 1e-6, (total, expected)
    # the old buggy code returned each micro-batch as a complete mean over its
    # own retained token, summing to retained_sum (exactly 2x expected here)
    assert abs(total - retained_sum) > 1e-3


def test_tailsft_loss_gradient_only_on_retained():
    """Gradient must vanish on dropped-sequence logprobs."""
    lp0 = [math_log(0.5), math_log(0.5)]
    lp1 = [math_log(0.2), math_log(0.2)]
    masks = [[1.0, 1.0], [1.0, 1.0]]
    flat = torch.tensor([*lp0, *lp1], dtype=torch.float32, requires_grad=True)
    offsets = torch.tensor([0, 2, 4])
    log_probs = torch.nested.nested_tensor_from_jagged(flat, offsets)
    loss_mask = torch.nested.nested_tensor_from_jagged(
        torch.tensor([1.0, 1.0, 1.0, 1.0]), offsets
    )
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(4, dtype=torch.long), offsets
    )
    data = tu.get_tensordict(
        tensor_dict={
            "log_probs": log_probs,
            "loss_mask": loss_mask,
            "input_ids": input_ids,
            "init_ce": [0.05, 1.5],  # seq1 dropped (smallest margin)
        },
        non_tensor_dict={
            "pad_mode": DatasetPadMode.NO_PADDING,
            "dp_size": 1,
            "batch_num_tokens": 4.0,
            "tailsft_filter_fraction": 0.5,
        },
    )
    loss, metrics = tailsft_loss(None, {"log_probs": log_probs}, data, dp_group=None)
    loss.backward()
    # seq0 logprobs receive gradient; seq1 logprobs receive none
    g = flat.grad
    assert g[:2].abs().sum().item() > 0
    assert g[2:].abs().sum().item() == 0.0


def test_tailsft_loss_retained_denominator_global():
    """With dp_size=2 (single-process stand-in), loss = token-mean over
    retained tokens * dp_size, matching the sft_loss scaling convention."""
    lp = [[math_log(0.5), math_log(0.5)], [math_log(0.2), math_log(0.2)]]
    masks = [[1.0, 1.0], [1.0, 1.0]]
    data, out = _jagged_micro_batch(lp, masks, init_ce=[0.05, 1.5], gamma=0.5, dp_size=2)
    loss, _ = tailsft_loss(None, out, data, dp_group=None)
    expected = -(math_log(0.5) + math_log(0.5)) / 2.0 * 2
    assert abs(loss.item() - expected) < 1e-6


# --- meta threading (gamma + init_ce through micro-batch slicing) ------------


def test_gamma_and_init_ce_survive_tensordict_and_slicing():
    lp = [[math_log(0.5)], [math_log(0.25)], [math_log(0.6)]]
    masks = [[1.0], [1.0], [1.0]]
    data, out = _jagged_micro_batch(lp, masks, init_ce=[0.1, 0.2, 0.3], gamma=0.3)
    # gamma: NonTensorData scalar -> preserved through slicing
    sliced = tu.index_select_tensor_dict(data, torch.tensor([1, 2]))
    assert tu.get_non_tensor_data(sliced, "tailsft_filter_fraction", None) == 0.3
    assert tu.get(sliced, "init_ce") == [0.2, 0.3]
    # sliced log_probs follow the same reindexing (row 1, 2 of the jagged batch)
    sliced_lp = sliced["log_probs"].values()
    assert torch.allclose(sliced_lp, torch.tensor([math_log(0.25), math_log(0.6)]))
    # and init_ce rows align with sliced rows: run the real loss on the slice
    # (γ=0.3, n=2 → k=0 dropped; batch metric must be 2)
    l_sliced, m_sliced = tailsft_loss(None, {"log_probs": sliced["log_probs"]}, sliced, dp_group=None)
    assert m_sliced["tailsft/batch"] == 2.0
    assert m_sliced["tailsft/dropped"] == 0.0


# --- trainer-side metric reduction (allgather-nested lists) ------------------


def _reduce_tailsft_metrics(metrics):
    """Mirror of the reduction loop in sft_trainer.py (keptr in sync by
    this test): flattens the allgather-NESTED per-rank lists before
    summing/averaging."""
    for k in sorted(metrics.keys()):
        if not k.startswith("tailsft/"):
            continue
        v = metrics[k]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            flat = []
            for item in v:
                if isinstance(item, (list, tuple)):
                    flat.extend(float(x) for x in item)
                else:
                    flat.append(float(item))
            if len(flat) == 0:
                continue
            if k == "tailsft/margin_mean":
                metrics[k] = float(sum(flat)) / len(flat)
            else:
                metrics[k] = float(sum(flat))
    return metrics


def test_tailsft_metric_reduction_flattens_allgather_nested_lists():
    """Regression for the 2026-09-05 step-1 crash: allgather_dict_into_dict
    appends each dp rank's already-flat per-micro-batch list, so the trainer
    sees [[f, ...], [f, ...]]. sum() over that raises TypeError (int+list);
    the reduction must flatten one level first."""
    nested = {
        "tailsft/dropped": [[1.0, 2.0], [3.0]],
        "tailsft/batch": [[2.0, 2.0], [2.0]],
        "tailsft/margin_mean": [[0.5, 0.7], [0.9]],
        "train/loss": 1.23,  # untouched: no tailsft/ prefix
    }
    out = _reduce_tailsft_metrics(nested)
    assert out["tailsft/dropped"] == 6.0
    assert out["tailsft/batch"] == 6.0
    assert abs(out["tailsft/margin_mean"] - 0.7) < 1e-9
    assert out["train/loss"] == 1.23
    # the old (flat-assuming) code crashes on exactly this shape
    with pytest.raises(TypeError):
        float(sum(nested["tailsft/dropped"] if False else [[1.0, 2.0], [3.0]]))


def test_tailsft_metric_reduction_handles_flat_single_rank_and_empty():
    """dp=1 still nests one level; flat lists keep working; empty shapes
    are left alone instead of raising."""
    # dp=1: allgather still wraps the single rank's list -> [[...]]
    single = {"tailsft/dropped": [[4.0]], "tailsft/margin_mean": [[0.4]]}
    out = _reduce_tailsft_metrics(single)
    assert out["tailsft/dropped"] == 4.0
    assert abs(out["tailsft/margin_mean"] - 0.4) < 1e-9
    # genuinely flat (pre-allgather shape) is unchanged behavior
    flat = {"tailsft/dropped": [1.0, 2.0, 3.0], "tailsft/margin_mean": [0.5, 0.7, 0.9]}
    out = _reduce_tailsft_metrics(flat)
    assert out["tailsft/dropped"] == 6.0
    assert abs(out["tailsft/margin_mean"] - 0.7) < 1e-9
    # empty list / empty inner lists: no crash, left as-is
    for m in (
        {"tailsft/dropped": [], "tailsft/margin_mean": []},
        {"tailsft/dropped": [[], []], "tailsft/margin_mean": [[], []]},
    ):
        out = _reduce_tailsft_metrics(m)
        assert out["tailsft/dropped"] in ([], [[], []])
