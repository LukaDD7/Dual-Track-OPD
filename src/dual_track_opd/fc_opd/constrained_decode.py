"""Constrained decoding helpers for evidence generation."""

from __future__ import annotations

from typing import Sequence

import torch

DEFAULT_FORBID_STRINGS: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    " A",
    " B",
    " C",
    " D",
    "A.",
    "B.",
    "C.",
    "D.",
    " A.",
    " B.",
    " C.",
    " D.",
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "a",
    "b",
    "c",
    "d",
    "answer",
    "Answer",
    "ANSWER",
    "答案",
    "故选",
    "因此",
)


class ForbidAnswerTokensLogitsProcessor:
    """Set logits for answer/option-like token IDs to negative infinity."""

    def __init__(self, forbidden_token_ids: Sequence[int]):
        self.forbidden_token_ids = tuple(sorted({int(token_id) for token_id in forbidden_token_ids if int(token_id) >= 0}))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if not self.forbidden_token_ids:
            return scores
        token_ids = torch.tensor(self.forbidden_token_ids, dtype=torch.long, device=scores.device)
        token_ids = token_ids[token_ids < scores.shape[-1]]
        if token_ids.numel() > 0:
            scores[:, token_ids] = -float("inf")
        return scores


def build_forbidden_token_ids(
    tokenizer: object,
    extra_strings: Sequence[str] | None = None,
) -> list[int]:
    forbid_strings = [*DEFAULT_FORBID_STRINGS, *(extra_strings or ())]
    token_ids: set[int] = set()
    encode = getattr(tokenizer, "encode")
    for text in forbid_strings:
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
        token_ids.update(int(token_id) for token_id in ids if int(token_id) >= 0)
    return sorted(token_ids)
