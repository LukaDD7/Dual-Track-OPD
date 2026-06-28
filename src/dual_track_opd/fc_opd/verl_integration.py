"""Adapters between online FC-OPD outputs and verl tensor batches.

This file contains no imports from ``third_party/verl``. The trainer-side patch
can call these helpers and then attach the returned tensors to
``DataProto.batch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .conditions import Condition
from .online_batch import OnlineFCOPDBatchOutput, OnlineFCOPDSampleOutput


DEFAULT_VERL_CONDITION_ORDER: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK_VISIBLE,
    Condition.TASK_INFER,
    Condition.TASK_SOLVE,
)
VERL_CONDITION_IDS: Mapping[Condition, int] = {
    condition: index for index, condition in enumerate(DEFAULT_VERL_CONDITION_ORDER)
}


@dataclass(frozen=True)
class VerlFCOPDTensors:
    teacher_topk_indices: torch.Tensor
    teacher_topk_log_probs: torch.Tensor
    condition_weights: torch.Tensor
    condition_ids: torch.Tensor
    teacher_tail_log_prob: torch.Tensor | None = None

    def as_batch_dict(self) -> dict[str, torch.Tensor]:
        tensors = {
            "fc_teacher_topk_indices": self.teacher_topk_indices,
            "fc_teacher_topk_log_probs": self.teacher_topk_log_probs,
            "fc_condition_weights": self.condition_weights,
            "fc_condition_ids": self.condition_ids,
        }
        if self.teacher_tail_log_prob is not None:
            tensors["fc_teacher_tail_log_prob"] = self.teacher_tail_log_prob
        return tensors


def online_batch_output_to_verl_tensors(
    output: OnlineFCOPDBatchOutput,
    *,
    condition_order: Sequence[Condition | str] = DEFAULT_VERL_CONDITION_ORDER,
) -> VerlFCOPDTensors:
    """Stack online FC-OPD sample outputs into verl actor tensor fields."""

    return online_sample_outputs_to_verl_tensors(output.samples, condition_order=condition_order)


def online_sample_outputs_to_verl_tensors(
    samples: Sequence[OnlineFCOPDSampleOutput],
    *,
    condition_order: Sequence[Condition | str] = DEFAULT_VERL_CONDITION_ORDER,
) -> VerlFCOPDTensors:
    if not samples:
        raise ValueError("at least one online FC-OPD sample output is required")
    normalized_order = tuple(Condition(condition) for condition in condition_order)
    if not normalized_order:
        raise ValueError("condition_order must be non-empty")

    seq_len = samples[0].response_token_ids.shape[-1]
    top_k = _top_k_for_sample(samples[0], normalized_order[0])
    device = samples[0].response_token_ids.device

    topk_ids = []
    topk_log_probs = []
    condition_weights = []
    tail_blocks = []
    has_any_tail = False
    for sample in samples:
        _validate_sample_shape(sample, seq_len=seq_len)
        sample_ids = []
        sample_log_probs = []
        sample_weights = []
        sample_tails = []
        for condition in normalized_order:
            if condition not in sample.teacher_scores:
                raise ValueError(f"{sample.sample_uid}: missing teacher scores for {condition.value}")
            if condition not in sample.condition_weights:
                raise ValueError(f"{sample.sample_uid}: missing condition weights for {condition.value}")
            teacher = sample.teacher_scores[condition]
            teacher.validate()
            if teacher.token_ids.shape != (1, seq_len, top_k):
                raise ValueError(
                    f"{sample.sample_uid}: {condition.value} teacher top-k shape "
                    f"{tuple(teacher.token_ids.shape)} != {(1, seq_len, top_k)}"
                )
            weight = sample.condition_weights[condition]
            if weight.shape != (1, seq_len):
                raise ValueError(
                    f"{sample.sample_uid}: {condition.value} weight shape {tuple(weight.shape)} != {(1, seq_len)}"
                )
            sample_ids.append(teacher.token_ids.squeeze(0).to(device=device, dtype=torch.long))
            sample_log_probs.append(teacher.log_probs.squeeze(0).to(device=device, dtype=torch.float32))
            sample_weights.append(weight.squeeze(0).to(device=device, dtype=torch.float32))
            if teacher.tail_log_prob is None:
                sample_tails.append(torch.full((seq_len,), float("-inf"), dtype=torch.float32, device=device))
            else:
                has_any_tail = True
                sample_tails.append(teacher.tail_log_prob.squeeze(0).to(device=device, dtype=torch.float32))
        topk_ids.append(torch.stack(sample_ids, dim=0))
        topk_log_probs.append(torch.stack(sample_log_probs, dim=0))
        condition_weights.append(torch.stack(sample_weights, dim=0))
        tail_blocks.append(torch.stack(sample_tails, dim=0))

    condition_ids = torch.tensor(
        [VERL_CONDITION_IDS.get(condition, -1) for condition in normalized_order],
        dtype=torch.long,
        device=device,
    )
    if torch.any(condition_ids < 0):
        missing = [condition.value for condition in normalized_order if condition not in VERL_CONDITION_IDS]
        raise ValueError(f"condition_order includes unsupported conditions: {', '.join(missing)}")

    return VerlFCOPDTensors(
        teacher_topk_indices=torch.stack(topk_ids, dim=0),
        teacher_topk_log_probs=torch.stack(topk_log_probs, dim=0),
        condition_weights=torch.stack(condition_weights, dim=0),
        condition_ids=condition_ids,
        teacher_tail_log_prob=torch.stack(tail_blocks, dim=0) if has_any_tail else None,
    )


def _top_k_for_sample(sample: OnlineFCOPDSampleOutput, condition: Condition) -> int:
    if condition not in sample.teacher_scores:
        raise ValueError(f"{sample.sample_uid}: missing teacher scores for {condition.value}")
    return int(sample.teacher_scores[condition].token_ids.shape[-1])


def _validate_sample_shape(sample: OnlineFCOPDSampleOutput, *, seq_len: int) -> None:
    if sample.response_token_ids.shape != (1, seq_len):
        raise ValueError(f"{sample.sample_uid}: response_token_ids must have shape [1, {seq_len}]")
