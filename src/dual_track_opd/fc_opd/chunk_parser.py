"""Token-aligned parser for structured VLM student responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import torch


class DecodeTokenizer(Protocol):
    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str: ...


CANONICAL_CHUNK_NAMES = ("visible_evidence", "diagram_inference", "reasoning", "answer")
LEGACY_CHUNK_NAMES = ("visual_evidence", "reasoning", "answer")
CHUNK_NAMES = CANONICAL_CHUNK_NAMES
LEGACY_TO_CANONICAL = {"visual_evidence": "visible_evidence"}


@dataclass(frozen=True)
class ChunkMasks:
    visible_evidence_mask: torch.Tensor
    diagram_inference_mask: torch.Tensor
    reasoning_mask: torch.Tensor
    answer_mask: torch.Tensor
    format_valid: bool
    errors: tuple[str, ...]
    fallback: str | None
    chunk_labels: tuple[str, ...] = field(default_factory=tuple)
    chunk_spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def visual_evidence_mask(self) -> torch.Tensor:
        """Backward-compatible alias for the old chunk name."""

        return self.visible_evidence_mask

    @property
    def token_counts(self) -> dict[str, int]:
        return {
            "visible_evidence": int(self.visible_evidence_mask.sum().item()),
            "diagram_inference": int(self.diagram_inference_mask.sum().item()),
            "reasoning": int(self.reasoning_mask.sum().item()),
            "answer": int(self.answer_mask.sum().item()),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "visible_evidence_mask": self.visible_evidence_mask,
            "visual_evidence_mask": self.visible_evidence_mask,
            "diagram_inference_mask": self.diagram_inference_mask,
            "reasoning_mask": self.reasoning_mask,
            "answer_mask": self.answer_mask,
            "format_valid": self.format_valid,
            "errors": self.errors,
            "fallback": self.fallback,
            "token_counts": self.token_counts,
            "chunk_labels": self.chunk_labels,
            "chunk_spans": self.chunk_spans,
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


def _tag(name: str, *, close: bool = False) -> str:
    return f"</{name}>" if close else f"<{name}>"


def _positions(text: str, needle: str) -> list[int]:
    return [index for index in range(len(text)) if text.startswith(needle, index)]


def _canonical_name(name: str) -> str:
    return LEGACY_TO_CANONICAL.get(name, name)


def _find_tag_intervals(
    text: str,
    chunk_names: Sequence[str],
) -> tuple[dict[str, tuple[int, int]], list[tuple[int, int]], tuple[str, ...]]:
    semantic: dict[str, tuple[int, int]] = {}
    tag_intervals: list[tuple[int, int]] = []
    errors: list[str] = []
    cursor = 0
    close_cursor = 0

    for name in chunk_names:
        open_tag = _tag(name)
        close_tag = _tag(name, close=True)
        open_positions = _positions(text, open_tag)
        close_positions = _positions(text, close_tag)

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
        if close_start < close_cursor:
            errors.append(f"{name}:out_of_order")
        if any(error.startswith(f"{name}:") for error in errors):
            continue

        semantic[_canonical_name(name)] = (open_end, close_start)
        tag_intervals.extend(((open_start, open_end), (close_start, close_end)))
        cursor = close_end
        close_cursor = close_end

    if len(semantic) == len(chunk_names):
        all_tag_text = tuple(
            tag
            for chunk in (*CANONICAL_CHUNK_NAMES, *LEGACY_CHUNK_NAMES)
            for tag in (_tag(chunk), _tag(chunk, close=True))
        )
        for name, (start, end) in semantic.items():
            inner = text[start:end]
            if any(tag in inner for tag in all_tag_text):
                errors.append(f"{name}:nested_tag")
        ordered_tags = sorted(tag_intervals)
        outside_intervals: list[tuple[int, int]] = [(0, ordered_tags[0][0])]
        for index in range(1, len(ordered_tags) - 1, 2):
            outside_intervals.append((ordered_tags[index][1], ordered_tags[index + 1][0]))
        outside_intervals.append((ordered_tags[-1][1], len(text)))
        if any(text[start:end].strip() for start, end in outside_intervals):
            errors.append("unexpected_text_outside_chunks")

    return semantic, tag_intervals, tuple(dict.fromkeys(errors))


def _parse_intervals(text: str) -> tuple[dict[str, tuple[int, int]], list[tuple[int, int]], tuple[str, ...]]:
    canonical = _find_tag_intervals(text, CANONICAL_CHUNK_NAMES)
    if not canonical[2] and len(canonical[0]) == len(CANONICAL_CHUNK_NAMES):
        return canonical
    legacy = _find_tag_intervals(text, LEGACY_CHUNK_NAMES)
    if not legacy[2] and len(legacy[0]) == len(LEGACY_CHUNK_NAMES):
        semantic = dict(legacy[0])
        semantic.setdefault("diagram_inference", (0, 0))
        return semantic, legacy[1], ()
    if "<visual_evidence" in text or "</visual_evidence>" in text:
        return legacy
    return canonical if len(canonical[0]) >= len(legacy[0]) else legacy


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


def _fallback_masks(length: int, fallback: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = torch.zeros(length, dtype=torch.bool)
    if fallback == "all_reasoning":
        return empty.clone(), empty.clone(), torch.ones(length, dtype=torch.bool), empty.clone()
    if fallback == "all_answer":
        return empty.clone(), empty.clone(), empty.clone(), torch.ones(length, dtype=torch.bool)
    if fallback == "empty":
        return empty.clone(), empty.clone(), empty.clone(), empty.clone()
    raise ValueError("fallback must be one of: all_reasoning, all_answer, empty")


def _labels_from_masks(
    masks: dict[str, torch.Tensor],
    *,
    length: int,
    default: str = "outside",
) -> tuple[str, ...]:
    labels = [default for _ in range(length)]
    for name in CANONICAL_CHUNK_NAMES:
        mask = masks[name].bool().tolist()
        for index, selected in enumerate(mask):
            if selected:
                labels[index] = name
    return tuple(labels)


def _span_from_mask(mask: torch.Tensor) -> tuple[int, int]:
    indices = torch.nonzero(mask.bool(), as_tuple=False).flatten()
    if indices.numel() == 0:
        return (0, 0)
    return (int(indices[0].item()), int(indices[-1].item()) + 1)


def _chunk_spans_from_masks(masks: dict[str, torch.Tensor]) -> dict[str, tuple[int, int]]:
    return {name: _span_from_mask(mask) for name, mask in masks.items()}


def parse_response_chunks(
    token_ids: Sequence[int] | torch.Tensor,
    decoded_text: str,
    tokenizer: DecodeTokenizer,
    *,
    fallback: str = "all_reasoning",
) -> ChunkMasks:
    """Parse structured chunks into masks aligned to response token positions.

    The preferred schema is ``visible_evidence`` / ``diagram_inference`` /
    ``reasoning`` / ``answer``. The legacy ``visual_evidence`` / ``reasoning`` /
    ``answer`` schema is accepted and mapped to ``visible_evidence`` with an
    empty ``diagram_inference`` span.
    """

    ids = token_ids.detach().cpu().tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    decoded_from_tokens, token_spans = _token_char_spans(ids, tokenizer)
    errors: list[str] = []
    if decoded_text != decoded_from_tokens:
        errors.append("decoded_text_mismatch")

    semantic, tag_intervals, syntax_errors = _parse_intervals(decoded_from_tokens)
    errors.extend(syntax_errors)

    if errors or not {"visible_evidence", "diagram_inference", "reasoning", "answer"}.issubset(semantic):
        visible, diagram, reasoning, answer = _fallback_masks(len(ids), fallback)
        masks = {
            "visible_evidence": visible,
            "diagram_inference": diagram,
            "reasoning": reasoning,
            "answer": answer,
        }
        return ChunkMasks(
            visible_evidence_mask=visible,
            diagram_inference_mask=diagram,
            reasoning_mask=reasoning,
            answer_mask=answer,
            format_valid=False,
            errors=tuple(dict.fromkeys(errors or ["missing_chunks"])),
            fallback=fallback,
            chunk_labels=_labels_from_masks(masks, length=len(ids)),
            chunk_spans=_chunk_spans_from_masks(masks),
        )

    masks = {
        "visible_evidence": _mask_for_interval(token_spans, semantic["visible_evidence"], tag_intervals),
        "diagram_inference": _mask_for_interval(token_spans, semantic["diagram_inference"], tag_intervals),
        "reasoning": _mask_for_interval(token_spans, semantic["reasoning"], tag_intervals),
        "answer": _mask_for_interval(token_spans, semantic["answer"], tag_intervals),
    }
    if torch.any(sum(mask.int() for mask in masks.values()) > 1):
        visible, diagram, reasoning, answer = _fallback_masks(len(ids), fallback)
        fallback_masks = {
            "visible_evidence": visible,
            "diagram_inference": diagram,
            "reasoning": reasoning,
            "answer": answer,
        }
        return ChunkMasks(
            visible_evidence_mask=visible,
            diagram_inference_mask=diagram,
            reasoning_mask=reasoning,
            answer_mask=answer,
            format_valid=False,
            errors=("overlapping_chunk_masks",),
            fallback=fallback,
            chunk_labels=_labels_from_masks(fallback_masks, length=len(ids)),
            chunk_spans=_chunk_spans_from_masks(fallback_masks),
        )

    return ChunkMasks(
        visible_evidence_mask=masks["visible_evidence"],
        diagram_inference_mask=masks["diagram_inference"],
        reasoning_mask=masks["reasoning"],
        answer_mask=masks["answer"],
        format_valid=True,
        errors=(),
        fallback=None,
        chunk_labels=_labels_from_masks(masks, length=len(ids)),
        chunk_spans=_chunk_spans_from_masks(masks),
    )
