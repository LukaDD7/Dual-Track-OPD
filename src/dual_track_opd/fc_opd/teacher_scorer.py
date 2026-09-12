"""Teacher scorer interfaces and a deterministic CPU protocol backend."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import torch

from .teacher_protocol import (
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
    ensure_exact_token_alignment,
)
from .teacher_prompts import render_teacher_prompt


class TeacherScorer(ABC):
    @property
    @abstractmethod
    def metadata(self) -> TeacherMetadata: ...

    @abstractmethod
    def score_batch(
        self,
        requests: Sequence[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]: ...


class SyntheticTeacherScorer(TeacherScorer):
    """Deterministic scorer for CPU protocol and integration smoke tests."""

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        top_k: int = 8,
        tokenizer_hash: str = "synthetic-tokenizer-v1",
        git_revision: str = "synthetic",
    ):
        if not 1 <= top_k < vocab_size:
            raise ValueError("top_k must be positive and smaller than vocab_size")
        self._metadata = TeacherMetadata(
            model_id="synthetic/fc-opd-teacher",
            tokenizer_hash=tokenizer_hash,
            vocab_size=vocab_size,
            top_k=top_k,
            dtype="float32",
            git_revision=git_revision,
        )

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def _score_one(self, request: TeacherScoreRequest) -> TeacherScoreResponse:
        if request.tokenizer_hash != self.metadata.tokenizer_hash:
            raise ValueError("request tokenizer hash does not match teacher metadata")
        render_teacher_prompt(request.condition, request.question, request.condition_inputs)
        if any(token_id < 0 or token_id >= self.metadata.vocab_size for token_id in request.response_token_ids):
            raise ValueError("response token ID is outside the synthetic vocabulary")

        condition_seed = int.from_bytes(
            hashlib.sha256(request.condition.value.encode()).digest()[:4],
            "big",
        )
        topk_ids: list[tuple[int, ...]] = []
        topk_log_probs: list[tuple[float, ...]] = []
        tails: list[float] = []
        entropies: list[float] = []
        sampled_lps: list[float] = []

        for position, response_token_id in enumerate(request.response_token_ids):
            vocab = torch.arange(self.metadata.vocab_size, dtype=torch.float32)
            center = (response_token_id + condition_seed + position * 7) % self.metadata.vocab_size
            distance = torch.minimum(
                (vocab - center).abs(),
                self.metadata.vocab_size - (vocab - center).abs(),
            )
            condition_bias = ((condition_seed % 17) + 1) / 50.0
            logits = -distance * (0.25 + condition_bias)
            log_probs = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, k=self.metadata.top_k)
            topk_ids.append(tuple(int(item) for item in indices.tolist()))
            topk_log_probs.append(tuple(float(item) for item in values.tolist()))
            tail_mass = (1.0 - values.exp().sum()).clamp_min(torch.finfo(torch.float32).tiny)
            tails.append(float(tail_mass.log().item()))
            probabilities = log_probs.exp()
            entropies.append(float((-(probabilities * log_probs).sum()).item()))
            # Exact log P_T(y_t|condition) — gather at the sampled token
            sampled_lps.append(float(log_probs[response_token_id].item()))

        response = TeacherScoreResponse(
            request_id=request.request_id,
            condition=request.condition,
            token_ids=request.response_token_ids,
            topk_token_ids=tuple(topk_ids),
            topk_log_probs=tuple(topk_log_probs),
            tail_log_prob=tuple(tails),
            teacher_entropy=tuple(entropies),
            sampled_token_log_probs=tuple(sampled_lps),
        )
        ensure_exact_token_alignment(
            request.response_token_ids,
            response.token_ids,
            context="synthetic teacher scoring",
        )
        return response

    def score_batch(
        self,
        requests: Sequence[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        return [self._score_one(request) for request in requests]
