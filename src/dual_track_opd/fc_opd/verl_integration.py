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
from .signal_decomposer import TeacherTopK


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
    teacher_sampled_log_probs: torch.Tensor | None = None  # [B,C,T] exact log P_T(y_t|cond)
    teacher_valid_mask: torch.Tensor | None = None  # [B,C,T] reliable teacher-token positions

    def as_batch_dict(self) -> dict[str, torch.Tensor]:
        tensors = {
            "fc_teacher_topk_indices": self.teacher_topk_indices,
            "fc_teacher_topk_log_probs": self.teacher_topk_log_probs,
            "fc_condition_weights": self.condition_weights,
            "fc_condition_ids": self.condition_ids,
        }
        if self.teacher_tail_log_prob is not None:
            tensors["fc_teacher_tail_log_prob"] = self.teacher_tail_log_prob
        if self.teacher_sampled_log_probs is not None:
            tensors["fc_teacher_sampled_log_probs"] = self.teacher_sampled_log_probs
        if self.teacher_valid_mask is not None:
            tensors["fc_teacher_valid_mask"] = self.teacher_valid_mask
        return tensors


def online_batch_output_to_verl_tensors(
    output: OnlineFCOPDBatchOutput,
    *,
    condition_order: Sequence[Condition | str] = DEFAULT_VERL_CONDITION_ORDER,
    target_seq_len: int | None = None,
    response_mask: torch.Tensor | None = None,
) -> VerlFCOPDTensors:
    """Stack online FC-OPD sample outputs into verl actor tensor fields."""

    return online_sample_outputs_to_verl_tensors(
        output.samples,
        condition_order=condition_order,
        target_seq_len=target_seq_len,
        response_mask=response_mask,
    )


def online_sample_outputs_to_verl_tensors(
    samples: Sequence[OnlineFCOPDSampleOutput],
    *,
    condition_order: Sequence[Condition | str] = DEFAULT_VERL_CONDITION_ORDER,
    target_seq_len: int | None = None,
    response_mask: torch.Tensor | None = None,
) -> VerlFCOPDTensors:
    if not samples:
        raise ValueError("at least one online FC-OPD sample output is required")
    normalized_order = tuple(Condition(condition) for condition in condition_order)
    if not normalized_order:
        raise ValueError("condition_order must be non-empty")

    first_seq_len = samples[0].response_token_ids.shape[-1]
    seq_len = int(target_seq_len or first_seq_len)
    if seq_len < first_seq_len:
        raise ValueError("target_seq_len must be at least the longest sample response length")
    top_k = _top_k_for_sample(samples[0], normalized_order[0])
    device = samples[0].response_token_ids.device
    if response_mask is not None:
        if response_mask.shape != (len(samples), seq_len):
            raise ValueError(f"response_mask must have shape {(len(samples), seq_len)}")
        response_mask = response_mask.to(device=device, dtype=torch.float32)

    topk_ids = []
    topk_log_probs = []
    condition_weights = []
    tail_blocks = []
    sampled_lp_blocks = []
    valid_mask_blocks = []
    has_any_tail = False
    for sample in samples:
        sample_seq_len = sample.response_token_ids.shape[-1]
        if sample_seq_len > seq_len:
            raise ValueError(f"{sample.sample_uid}: sample response length exceeds target_seq_len")
        sample_ids = []
        sample_log_probs = []
        sample_weights = []
        sample_tails = []
        sample_sampled_lp = []
        sample_valid_masks = []
        for condition in normalized_order:
            if condition not in sample.teacher_scores:
                raise ValueError(f"{sample.sample_uid}: missing teacher scores for {condition.value}")
            if condition not in sample.condition_weights:
                raise ValueError(f"{sample.sample_uid}: missing condition weights for {condition.value}")
            teacher = sample.teacher_scores[condition]
            teacher.validate()
            if teacher.token_ids.shape != (1, sample_seq_len, top_k):
                raise ValueError(
                    f"{sample.sample_uid}: {condition.value} teacher top-k shape "
                    f"{tuple(teacher.token_ids.shape)} != {(1, sample_seq_len, top_k)}"
                )
            weight = sample.condition_weights[condition]
            if weight.shape != (1, sample_seq_len):
                raise ValueError(
                    f"{sample.sample_uid}: {condition.value} weight shape {tuple(weight.shape)} != {(1, sample_seq_len)}"
                )
            sample_ids.append(_pad_topk_ids(teacher.token_ids.squeeze(0), seq_len).to(device=device, dtype=torch.long))
            sample_log_probs.append(_pad_topk_log_probs(teacher.log_probs.squeeze(0), seq_len).to(device=device, dtype=torch.float32))
            sample_weight = _pad_vector(weight.squeeze(0), seq_len).to(device=device, dtype=torch.float32)
            sample_weights.append(sample_weight)
            if teacher.tail_log_prob is None:
                sample_tails.append(torch.full((seq_len,), -30.0, dtype=torch.float32, device=device))
            else:
                has_any_tail = True
                sample_tails.append(_pad_vector(teacher.tail_log_prob.squeeze(0), seq_len, pad_value=0.0).to(device=device, dtype=torch.float32))
            # Exact teacher log P_T(y_t | condition_c) per response token
            if teacher.sampled_log_probs is not None:
                sample_slp = _pad_vector(teacher.sampled_log_probs.squeeze(0), seq_len, pad_value=-30.0)
                sample_sampled_lp.append(sample_slp.to(device=device, dtype=torch.float32))
            else:
                sample_sampled_lp.append(torch.full((seq_len,), -30.0, dtype=torch.float32, device=device))
            valid_mask = _teacher_valid_mask(teacher).squeeze(0)
            valid_mask = _pad_bool_vector(valid_mask, seq_len, pad_value=False).to(device=device)
            sample_valid_masks.append(valid_mask)
        topk_ids.append(torch.stack(sample_ids, dim=0))
        topk_log_probs.append(torch.stack(sample_log_probs, dim=0))
        stacked_sample_valid = torch.stack(sample_valid_masks, dim=0).float()
        condition_weights.append(torch.stack(sample_weights, dim=0) * stacked_sample_valid)
        tail_blocks.append(torch.stack(sample_tails, dim=0))
        sampled_lp_blocks.append(torch.stack(sample_sampled_lp, dim=0))
        valid_mask_blocks.append(torch.stack(sample_valid_masks, dim=0))

    condition_ids = torch.tensor(
        [VERL_CONDITION_IDS.get(condition, -1) for condition in normalized_order],
        dtype=torch.long,
        device=device,
    )
    if torch.any(condition_ids < 0):
        missing = [condition.value for condition in normalized_order if condition not in VERL_CONDITION_IDS]
        raise ValueError(f"condition_order includes unsupported conditions: {', '.join(missing)}")

    stacked_weights = torch.stack(condition_weights, dim=0)
    if response_mask is not None:
        stacked_weights = stacked_weights * response_mask.unsqueeze(1)

    return VerlFCOPDTensors(
        teacher_topk_indices=torch.stack(topk_ids, dim=0),
        teacher_topk_log_probs=torch.stack(topk_log_probs, dim=0),
        condition_weights=stacked_weights,
        condition_ids=condition_ids,
        teacher_tail_log_prob=torch.stack(tail_blocks, dim=0) if has_any_tail else None,
        teacher_sampled_log_probs=torch.stack(sampled_lp_blocks, dim=0),  # [B,C,T]
        teacher_valid_mask=torch.stack(valid_mask_blocks, dim=0),
    )


def _top_k_for_sample(sample: OnlineFCOPDSampleOutput, condition: Condition) -> int:
    if condition not in sample.teacher_scores:
        raise ValueError(f"{sample.sample_uid}: missing teacher scores for {condition.value}")
    return int(sample.teacher_scores[condition].token_ids.shape[-1])


def _pad_topk_ids(values: torch.Tensor, seq_len: int) -> torch.Tensor:
    if values.shape[0] == seq_len:
        return values
    pad = torch.zeros((seq_len - values.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    return torch.cat([values, pad], dim=0)


def _pad_topk_log_probs(values: torch.Tensor, seq_len: int) -> torch.Tensor:
    if values.shape[0] == seq_len:
        return values
    pad = torch.zeros((seq_len - values.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    return torch.cat([values, pad], dim=0)


def _pad_vector(values: torch.Tensor, seq_len: int, *, pad_value: float = 0.0) -> torch.Tensor:
    if values.shape[0] == seq_len:
        return values
    pad = torch.full((seq_len - values.shape[0],), pad_value, dtype=values.dtype, device=values.device)
    return torch.cat([values, pad], dim=0)


def _pad_bool_vector(values: torch.Tensor, seq_len: int, *, pad_value: bool) -> torch.Tensor:
    values = values.bool()
    if values.shape[0] == seq_len:
        return values
    pad = torch.full((seq_len - values.shape[0],), pad_value, dtype=torch.bool, device=values.device)
    return torch.cat([values, pad], dim=0)


def _teacher_valid_mask(teacher: TeacherTopK) -> torch.Tensor:
    if teacher.valid_mask is not None:
        return teacher.valid_mask.bool()
    return torch.ones(teacher.token_ids.shape[:2], dtype=torch.bool, device=teacher.token_ids.device)
