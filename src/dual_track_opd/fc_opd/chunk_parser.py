"""Token-aligned parser for structured VLM student responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class DecodeTokenizer(Protocol):
    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str: ...


CHUNK_NAMES = ("visual_evidence", "reasoning", "answer")
OPEN_TAGS = {name: f"<{name}>" for name in CHUNK_NAMES}
CLOSE_TAGS = {name: f"</{name}>" for name in CHUNK_NAMES}


@dataclass(frozen=True)
class ChunkMasks:
    visual_evidence_mask: torch.Tensor
    reasoning_mask: torch.Tensor
    answer_mask: torch.Tensor
    format_valid: bool
    errors: tuple[str, ...]
    fallback: str | None

    @property
    def token_counts(self) -> dict[str, int]:
        return {
            "visual_evidence": int(self.visual_evidence_mask.sum().item()),
            "reasoning": int(self.reasoning_mask.sum().item()),
            "answer": int(self.answer_mask.sum().item()),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "visual_evidence_mask": self.visual_evidence_mask,
            "reasoning_mask": self.reasoning_mask,
            "answer_mask": self.answer_mask,
            "format_valid": self.format_valid,
            "errors": self.errors,
            "fallback": self.fallback,
            "token_counts": self.token_counts,
        }


def _decode(tokenizer: DecodeTokenizer, token_ids: Sequence[int]) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return tokenizer.decode(token_ids, **kwargs)
    except TypeError:
        return tokenizer.decode(token_ids)


def _token_char_spans(token_ids: Sequence[int], tokenizer: DecodeTokenizer) -> tuple[str, list[tuple[int, int]]]:
    """Map response tokens to decoded character spans using prefix decoding."""

    prefixes = [_decode(tokenizer, token_ids[:index]) for index in range(len(token_ids) + 1)]
    full_text = prefixes[-1]
    boundaries = []
    previous = 0
    for prefix in prefixes:
        common = 0
        for left, right in zip(prefix, full_text, strict=False):
            if left != right:
                break
            common += 1
        previous = max(previous, common)
        boundaries.append(previous)
    boundaries[-1] = len(full_text)

    spans = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    expanded_spans: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(spans):
        if end > start:
            expanded_spans.append((start, end))
            continue
        next_end = next(
            (candidate_end for _, candidate_end in spans[index + 1 :] if candidate_end > start),
            start,
        )
        if next_end > start:
            expanded_spans.append((start, next_end))
        elif expanded_spans:
            expanded_spans.append(expanded_spans[-1])
        else:
            expanded_spans.append((start, end))
    return full_text, expanded_spans


def _find_tag_intervals(text: str) -> tuple[dict[str, tuple[int, int]], list[tuple[int, int]], tuple[str, ...]]:
    semantic: dict[str, tuple[int, int]] = {}
    tag_intervals: list[tuple[int, int]] = []
    errors: list[str] = []
    cursor = 0

    for name in CHUNK_NAMES:
        open_tag = OPEN_TAGS[name]
        close_tag = CLOSE_TAGS[name]
        open_positions = [index for index in range(len(text)) if text.startswith(open_tag, index)]
        close_positions = [index for index in range(len(text)) if text.startswith(close_tag, index)]

        if len(open_positions) != 1:
            errors.append(f"{name}:open_tag_count={len(open_positions)}")
        if len(close_positions) != 1:
            errors.append(f"{name}:close_tag_count={len(close_positions)}")
        if len(open_positions) != 1 or len(close_positions) != 1:
            continue

        open_start = open_positions[0]
        open_end = open_start + len(open_tag)
        close_start = close_positions[0]
        close_end = close_start + len(close_tag)

        if open_start < cursor:
            errors.append(f"{name}:out_of_order")
        if close_start < open_end:
            errors.append(f"{name}:nested_or_reversed")
        if errors:
            continue

        semantic[name] = (open_end, close_start)
        tag_intervals.extend(((open_start, open_end), (close_start, close_end)))
        cursor = close_end

    if len(semantic) == len(CHUNK_NAMES):
        for name, (start, end) in semantic.items():
            inner = text[start:end]
            if any(tag in inner for tag in (*OPEN_TAGS.values(), *CLOSE_TAGS.values())):
                errors.append(f"{name}:nested_tag")
        ordered_tags = sorted(tag_intervals)
        outside_intervals = (
            (0, ordered_tags[0][0]),
            (ordered_tags[1][1], ordered_tags[2][0]),
            (ordered_tags[3][1], ordered_tags[4][0]),
            (ordered_tags[5][1], len(text)),
        )
        if any(text[start:end].strip() for start, end in outside_intervals):
            errors.append("unexpected_text_outside_chunks")

    return semantic, tag_intervals, tuple(dict.fromkeys(errors))


def _overlaps(span: tuple[int, int], interval: tuple[int, int]) -> bool:
    start, end = span
    interval_start, interval_end = interval
    return end > interval_start and start < interval_end


def _mask_for_interval(
    token_spans: Sequence[tuple[int, int]],
    interval: tuple[int, int],
    tag_intervals: Sequence[tuple[int, int]],
) -> torch.Tensor:
    return torch.tensor(
        [
            _overlaps(span, interval) and not any(_overlaps(span, tag) for tag in tag_intervals)
            for span in token_spans
        ],
        dtype=torch.bool,
    )


def _fallback_masks(length: int, fallback: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = torch.zeros(length, dtype=torch.bool)
    if fallback == "all_reasoning":
        return empty.clone(), torch.ones(length, dtype=torch.bool), empty.clone()
    if fallback == "all_answer":
        return empty.clone(), empty.clone(), torch.ones(length, dtype=torch.bool)
    if fallback == "empty":
        return empty.clone(), empty.clone(), empty.clone()
    raise ValueError("fallback must be one of: all_reasoning, all_answer, empty")


def parse_response_chunks(
    token_ids: Sequence[int] | torch.Tensor,
    decoded_text: str,
    tokenizer: DecodeTokenizer,
    *,
    fallback: str = "all_reasoning",
) -> ChunkMasks:
    """Parse structured chunks into masks aligned to response token positions.

    Invalid formatting is reported and uses an explicit fallback. Tag tokens are
    excluded from semantic masks when the format is valid.
    """

    ids = token_ids.detach().cpu().tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    decoded_from_tokens, token_spans = _token_char_spans(ids, tokenizer)
    errors: list[str] = []
    if decoded_text != decoded_from_tokens:
        errors.append("decoded_text_mismatch")

    semantic, tag_intervals, syntax_errors = _find_tag_intervals(decoded_from_tokens)
    errors.extend(syntax_errors)

    if errors or len(semantic) != len(CHUNK_NAMES):
        visual, reasoning, answer = _fallback_masks(len(ids), fallback)
        return ChunkMasks(
            visual_evidence_mask=visual,
            reasoning_mask=reasoning,
            answer_mask=answer,
            format_valid=False,
            errors=tuple(dict.fromkeys(errors or ["missing_chunks"])),
            fallback=fallback,
        )

    visual = _mask_for_interval(token_spans, semantic["visual_evidence"], tag_intervals)
    reasoning = _mask_for_interval(token_spans, semantic["reasoning"], tag_intervals)
    answer = _mask_for_interval(token_spans, semantic["answer"], tag_intervals)
    if torch.any((visual.int() + reasoning.int() + answer.int()) > 1):
        visual, reasoning, answer = _fallback_masks(len(ids), fallback)
        return ChunkMasks(
            visual_evidence_mask=visual,
            reasoning_mask=reasoning,
            answer_mask=answer,
            format_valid=False,
            errors=("overlapping_chunk_masks",),
            fallback=fallback,
        )

    return ChunkMasks(
        visual_evidence_mask=visual,
        reasoning_mask=reasoning,
        answer_mask=answer,
        format_valid=True,
        errors=(),
        fallback=None,
    )
