"""Hidden-state visual focus utilities for Visual Grounding Gap OPD.

The functions here are deliberately model-agnostic.  Callers are responsible
for running the VLM forced-forward pass with hidden states enabled and for
providing token positions that are already aligned to the model's final input
sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualGapScores:
    teacher_rank: torch.Tensor
    student_rank: torch.Tensor
    gap_raw: torch.Tensor
    gap_pos: torch.Tensor


def select_hidden_state_layer(
    hidden_states: torch.Tensor | Sequence[torch.Tensor],
    *,
    layer: int | str = "last",
    last_n: int = 4,
) -> torch.Tensor:
    """Select or average hidden-state layers into ``[batch, seq, hidden]``."""

    if isinstance(hidden_states, torch.Tensor):
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states tensor must have shape [batch, seq, hidden]")
        return hidden_states
    layers = tuple(hidden_states)
    if not layers:
        raise ValueError("hidden_states sequence must not be empty")
    for item in layers:
        if item.ndim != 3:
            raise ValueError("each hidden-state layer must have shape [batch, seq, hidden]")
    if layer == "last":
        return layers[-1]
    if layer == "mean_last_n":
        if last_n <= 0:
            raise ValueError("last_n must be positive")
        selected = layers[-last_n:]
        return torch.stack([item.float() for item in selected], dim=0).mean(dim=0).to(layers[-1].dtype)
    if isinstance(layer, int):
        return layers[layer]
    raise ValueError("layer must be 'last', 'mean_last_n', or an integer index")


def extract_visual_token_positions(
    input_ids: torch.Tensor,
    *,
    image_token_id: int | None = None,
    image_token_ids: Iterable[int] | None = None,
    prompt_lengths: torch.Tensor | Sequence[int] | None = None,
) -> list[torch.Tensor]:
    """Return visual-token positions for each batch row.

    ``prompt_lengths`` can be supplied to ensure response text that happens to
    contain an image-token id is not treated as visual context.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    ids = set(int(item) for item in (image_token_ids or ()))
    if image_token_id is not None:
        ids.add(int(image_token_id))
    if not ids:
        raise ValueError("image_token_id or image_token_ids must be provided")
    prompt_tensor = _as_length_tensor(prompt_lengths, batch=input_ids.shape[0], device=input_ids.device)

    positions: list[torch.Tensor] = []
    for batch_index in range(input_ids.shape[0]):
        mask = torch.zeros_like(input_ids[batch_index], dtype=torch.bool)
        for token_id in ids:
            mask |= input_ids[batch_index] == token_id
        if prompt_tensor is not None:
            mask &= torch.arange(input_ids.shape[1], device=input_ids.device) < prompt_tensor[batch_index]
        positions.append(torch.nonzero(mask, as_tuple=False).flatten())
    return positions


def extract_prediction_state_positions(
    *,
    response_start: int | Sequence[int] | torch.Tensor,
    response_length: int,
    batch_size: int = 1,
    seq_len: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return hidden-state positions that predict each response token.

    For a causal LM, token ``y_t`` is predicted by the state immediately before
    it is consumed.  If response tokens start at absolute position ``P``, the
    aligned prediction positions are ``P-1, P, ..., P+T-2``.
    """

    if response_length < 0:
        raise ValueError("response_length must be non-negative")
    starts = _as_length_tensor(response_start, batch=batch_size, device=device)
    if torch.any(starts <= 0):
        raise ValueError("response_start must be positive so y_0 has a prefix state")
    offsets = torch.arange(response_length, device=starts.device, dtype=starts.dtype)
    positions = starts.unsqueeze(1) - 1 + offsets.unsqueeze(0)
    if seq_len is not None and torch.any(positions >= seq_len):
        raise ValueError("prediction positions exceed hidden-state sequence length")
    return positions.long()


def compute_visual_prototype(
    hidden_states: torch.Tensor,
    visual_positions: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean-pool visual-token hidden states for each batch row."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, seq, hidden]")
    if len(visual_positions) != hidden_states.shape[0]:
        raise ValueError("visual_positions length must match batch size")
    prototypes = []
    for batch_index, positions in enumerate(visual_positions):
        positions = positions.to(hidden_states.device).long()
        if positions.numel() == 0:
            raise ValueError(f"batch row {batch_index} has no visual token positions")
        if torch.any(positions < 0) or torch.any(positions >= hidden_states.shape[1]):
            raise ValueError("visual token position out of bounds")
        prototypes.append(hidden_states[batch_index, positions].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def compute_visual_focus_scores(
    hidden_states: torch.Tensor | Sequence[torch.Tensor],
    visual_positions: Sequence[torch.Tensor],
    prediction_positions: torch.Tensor,
    *,
    layer: int | str = "last",
    last_n: int = 4,
) -> torch.Tensor:
    """Compute VGPO-like visual focus scores in ``[0, 1]`` for response tokens."""

    selected = select_hidden_state_layer(hidden_states, layer=layer, last_n=last_n)
    if prediction_positions.ndim != 2:
        raise ValueError("prediction_positions must have shape [batch, response_len]")
    if prediction_positions.shape[0] != selected.shape[0]:
        raise ValueError("prediction_positions batch must match hidden_states batch")
    if torch.any(prediction_positions < 0) or torch.any(prediction_positions >= selected.shape[1]):
        raise ValueError("prediction position out of bounds")

    prototype = compute_visual_prototype(selected, visual_positions)
    batch_indices = torch.arange(selected.shape[0], device=selected.device).unsqueeze(1)
    token_hidden = selected[batch_indices, prediction_positions.to(selected.device)]
    cosine = F.cosine_similarity(token_hidden.float(), prototype[:, None, :].float(), dim=-1)
    return (0.5 * (cosine + 1.0)).clamp(0.0, 1.0).to(selected.dtype)


def rank_normalize_scores(scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Within-row percentile ranks in ``[0, 1]`` using average ranks for ties."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, seq]")
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape:
        raise ValueError("mask must match scores shape")
    ranks = torch.zeros_like(scores, dtype=torch.float32)
    for batch_index in range(scores.shape[0]):
        valid = mask[batch_index].bool()
        values = scores[batch_index, valid].float()
        if values.numel() == 0:
            continue
        if values.numel() == 1:
            ranks[batch_index, valid] = 1.0
            continue
        row_ranks = torch.empty_like(values)
        for value in torch.unique(values, sorted=True):
            tied = values == value
            less = (values < value).sum().float()
            equal = tied.sum().float()
            row_ranks[tied] = (less + 0.5 * (equal - 1.0)) / float(values.numel() - 1)
        ranks[batch_index, valid] = row_ranks
    return ranks


def compute_teacher_student_visual_gap(
    teacher_scores: torch.Tensor,
    student_scores: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    inputs_are_ranked: bool = False,
) -> VisualGapScores:
    """Compute student grounding gap without comparing raw hidden spaces."""

    if teacher_scores.shape != student_scores.shape:
        raise ValueError("teacher_scores and student_scores must have the same shape")
    if teacher_scores.ndim != 2:
        raise ValueError("visual focus scores must have shape [batch, seq]")
    if mask is not None and mask.shape != teacher_scores.shape:
        raise ValueError("mask must match score shape")
    teacher_rank = teacher_scores.float() if inputs_are_ranked else rank_normalize_scores(teacher_scores, mask)
    student_rank = student_scores.float() if inputs_are_ranked else rank_normalize_scores(student_scores, mask)
    if torch.any(teacher_rank < -1e-6) or torch.any(teacher_rank > 1.0 + 1e-6):
        raise ValueError("teacher rank-normalized scores must be in [0, 1]")
    if torch.any(student_rank < -1e-6) or torch.any(student_rank > 1.0 + 1e-6):
        raise ValueError("student rank-normalized scores must be in [0, 1]")
    gap_raw = teacher_rank - student_rank
    if mask is not None:
        gap_raw = torch.where(mask.bool(), gap_raw, torch.zeros_like(gap_raw))
    return VisualGapScores(
        teacher_rank=teacher_rank,
        student_rank=student_rank,
        gap_raw=gap_raw,
        gap_pos=gap_raw.clamp_min(0.0),
    )


def _as_length_tensor(
    value: int | Sequence[int] | torch.Tensor | None,
    *,
    batch: int,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device).long() if device is not None else value.long()
    elif isinstance(value, int):
        tensor = torch.full((batch,), int(value), dtype=torch.long, device=device)
    else:
        tensor = torch.tensor(list(value), dtype=torch.long, device=device)
    if tensor.numel() != batch:
        raise ValueError("length tensor must have one value per batch row")
    return tensor.reshape(batch)
