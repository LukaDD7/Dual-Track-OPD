import torch

from dual_track_opd.losses.chunk_anchor import (
    expand_chunk_weights_to_tokens,
    pool_token_scores_to_chunks,
)


def test_pool_token_scores_to_chunks_mean_and_max():
    scores = torch.tensor([[1.0, 3.0, 10.0, 4.0]])
    chunk_ids = torch.tensor([[0, 0, 1, -1]])
    mean = pool_token_scores_to_chunks(scores, chunk_ids, reduce="mean")
    maxed = pool_token_scores_to_chunks(scores, chunk_ids, reduce="max")
    assert torch.allclose(mean, torch.tensor([[2.0, 10.0]]))
    assert torch.allclose(maxed, torch.tensor([[3.0, 10.0]]))


def test_expand_chunk_weights_to_tokens():
    chunk_weights = torch.tensor([[2.0, 5.0]])
    chunk_ids = torch.tensor([[0, 1, -1, 0]])
    expanded = expand_chunk_weights_to_tokens(chunk_weights, chunk_ids)
    assert torch.allclose(expanded, torch.tensor([[2.0, 5.0, 0.0, 2.0]]))

