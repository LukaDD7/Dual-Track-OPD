"""Chunk-level visual anchor utilities."""

from __future__ import annotations

import torch


def _validate_token_chunk_shapes(token_scores: torch.Tensor, chunk_ids: torch.Tensor) -> None:
    if token_scores.ndim != 2:
        raise ValueError(f"token_scores must have shape [batch, seq]; got {tuple(token_scores.shape)}")
    if chunk_ids.shape != token_scores.shape:
        raise ValueError(
            f"chunk_ids must have shape {tuple(token_scores.shape)}; got {tuple(chunk_ids.shape)}"
        )


def pool_token_scores_to_chunks(
    token_scores: torch.Tensor,
    chunk_ids: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Pool token scores into padded per-batch chunk scores.

    Negative chunk IDs are ignored. Output shape is ``[batch, max_chunk_id + 1]``.
    Missing chunks receive zero.
    """

    _validate_token_chunk_shapes(token_scores, chunk_ids)
    if reduce not in {"mean", "max"}:
        raise ValueError("reduce must be 'mean' or 'max'")

    scores = token_scores.float()
    ids = chunk_ids.long()
    valid = ids >= 0
    if not torch.any(valid):
        return torch.zeros((scores.shape[0], 0), dtype=scores.dtype, device=scores.device)

    num_chunks = int(ids[valid].max().item()) + 1
    output = torch.zeros((scores.shape[0], num_chunks), dtype=scores.dtype, device=scores.device)

    for batch_idx in range(scores.shape[0]):
        for chunk_id in range(num_chunks):
            positions = ids[batch_idx] == chunk_id
            if not torch.any(positions):
                continue
            values = scores[batch_idx, positions]
            output[batch_idx, chunk_id] = values.mean() if reduce == "mean" else values.max()
    return output


def expand_chunk_weights_to_tokens(
    chunk_weights: torch.Tensor,
    chunk_ids: torch.Tensor,
) -> torch.Tensor:
    """Expand per-chunk weights back to token positions."""

    if chunk_weights.ndim != 2:
        raise ValueError(
            "chunk_weights must have shape [batch, chunks]; "
            f"got {tuple(chunk_weights.shape)}"
        )
    if chunk_ids.ndim != 2:
        raise ValueError(f"chunk_ids must have shape [batch, seq]; got {tuple(chunk_ids.shape)}")
    if chunk_ids.shape[0] != chunk_weights.shape[0]:
        raise ValueError("chunk_weights and chunk_ids must share batch size")

    ids = chunk_ids.long()
    expanded = torch.zeros(ids.shape, dtype=chunk_weights.dtype, device=chunk_weights.device)
    valid = (ids >= 0) & (ids < chunk_weights.shape[1])
    if torch.any(valid):
        expanded[valid] = chunk_weights[torch.arange(ids.shape[0], device=ids.device).unsqueeze(1), ids.clamp_min(0)][valid]
    return expanded

