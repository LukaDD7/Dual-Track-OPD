"""A single full forward must equal per-chunk forwards for fixed trajectories.

The causal-state probe scores fixed response token IDs with
``response_chunk_logits``.  The old statistics loops forwarded the whole
growing prefix once per 64-token chunk (O(T^2/chunk_size) model work); the
optimization forwards once per model/condition and slices the vocabulary
math.  Because scoring is causal, both must yield identical logits.

This test uses a tiny deterministic causal LM that mirrors the transformers
``logits_to_keep`` contract (last-k positions returned) and checks both the
``logits_to_keep`` path and the full-logits fallback path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dual_track_opd.support_aware.causal_runtime import response_chunk_logits


class _ToyAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        scores = self.q(h) @ self.k(h).transpose(-2, -1) / (h.shape[-1] ** 0.5)
        causal = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        attention = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
        x = x + attention @ self.v(h)
        return x + self.mlp(x)


class _ToyCausalLM(nn.Module):
    """Minimal causal LM honoring the transformers logits_to_keep contract."""

    def __init__(self, vocab: int, dim: int, layers: int, supports_keep: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(_ToyAttentionBlock(dim) for _ in range(layers))
        self.head = nn.Linear(dim, vocab, bias=False)
        self.supports_keep = supports_keep

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> object:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        if self.supports_keep and logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]

        class _Outputs:
            pass

        outputs = _Outputs()
        outputs.logits = logits
        return outputs


def _chunked_logits(model, prompt_inputs, response_ids, prefer_logits_to_keep):
    chunks = []
    for start in range(0, len(response_ids), 64):
        end = min(len(response_ids), start + 64)
        logits, _ = response_chunk_logits(
            model,
            prompt_inputs,
            response_ids,
            start=start,
            end=end,
            device="cpu",
            prefer_logits_to_keep=prefer_logits_to_keep,
        )
        chunks.append(logits)
    return torch.cat(chunks, dim=0)


def _run_case(supports_keep: bool) -> None:
    torch.manual_seed(1234)
    vocab, dim, layers, prompt_len, response_len = 997, 32, 2, 17, 137
    model = _ToyCausalLM(vocab, dim, layers, supports_keep)
    model.eval()
    prompt_inputs = {
        "input_ids": torch.randint(0, vocab, (1, prompt_len)),
        "attention_mask": torch.ones((1, prompt_len), dtype=torch.long),
    }
    response_ids = [int(value) for value in torch.randint(0, vocab, (response_len,))]

    with torch.inference_mode():
        chunked = _chunked_logits(model, prompt_inputs, response_ids, supports_keep)
        full, used_keep = response_chunk_logits(
            model,
            prompt_inputs,
            response_ids,
            start=0,
            end=response_len,
            device="cpu",
            prefer_logits_to_keep=supports_keep,
        )

    assert used_keep is supports_keep
    assert tuple(full.shape) == (response_len, vocab)
    assert tuple(chunked.shape) == (response_len, vocab)
    torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)


def test_full_forward_equals_chunked_with_logits_to_keep() -> None:
    _run_case(supports_keep=True)


def test_full_forward_equals_chunked_without_logits_to_keep() -> None:
    _run_case(supports_keep=False)
