"""Deterministic reasoning-block segmentation for teacher traces (frozen plan A2).

Splits a natural teacher response into semantic reasoning blocks:

- natural-language sentence (split on sentence-ending punctuation)
- newline-separated derivation line
- complete LaTeX / equation block (never split inside ``$$..$$``,
  ``\\[..\\]``, ``\\(..\\)`` or inline ``$..$`` math)
- coherent therefore/hence/thus step (kept as the sentence that starts it)
- final answer block (``\\boxed`` / answer line)

Every block maps to an exact token span.  Character offsets are computed on
the tokenizer-decode of the given token IDs so alignment is guaranteed even
when the stored response text has rendering differences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReasoningBlock:
    start_char: int
    end_char: int
    start_token: int
    end_token: int  # exclusive
    block_type: str
    text: str


_MATH_PATTERNS = (
    (re.compile(r"\$\$.*?\$\$", re.DOTALL), True),
    (re.compile(r"\\\[.*?\\\]", re.DOTALL), True),
    (re.compile(r"\\\(.*?\\\)", re.DOTALL), False),
    (re.compile(r"\$[^$\n]+?\$"), False),
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[^a-z])|(?<=[.!?])(?=\s*(?:[A-Z]|\\|$))")
_ANSWER_MARKERS = (
    "\\boxed",
    "the answer is",
    "answer:",
    "final answer",
)


def _protect_math(text: str) -> tuple[str, list[tuple[int, int, bool]]]:
    """Replace math spans with same-length placeholders; return masked text."""

    masked = list(text)
    spans: list[tuple[int, int, bool]] = []
    for pattern, is_display in _MATH_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(not (end <= s or start >= e) for s, e, _ in spans):
                continue
            spans.append((start, end, is_display))
            masked[start:end] = ["x"] * (end - start)
    spans.sort()
    return "".join(masked), spans


def _line_math_fraction(line: str, spans: Sequence[tuple[int, int, bool]], line_start: int) -> float:
    """Fraction of a line's non-whitespace chars covered by protected math."""

    total = len(line.strip())
    if total == 0:
        return 0.0
    covered = 0
    for start, end, _ in spans:
        left = max(start, line_start)
        right = min(end, line_start + len(line))
        if right > left:
            covered += len(line[left - line_start : right - line_start].strip())
    return covered / total


def _classify(text: str, *, is_last: bool) -> str:
    lowered = text.strip().lower()
    if is_last and any(marker in lowered for marker in _ANSWER_MARKERS):
        return "answer"
    if lowered.startswith(("\\", "$", "$$")):
        return "equation"
    return "sentence"


def segment_reasoning_blocks(text: str) -> list[ReasoningBlock]:
    """Split ``text`` into blocks with character spans (token span left unset)."""

    masked, spans = _protect_math(text)
    blocks: list[ReasoningBlock] = []
    lines = text.split("\n")
    cursor = 0
    for line_index, line in enumerate(lines):
        line_start = cursor
        cursor += len(line) + 1  # +1 for the newline
        stripped = line.strip()
        if not stripped:
            continue
        masked_line = masked[line_start : line_start + len(line)]
        if _line_math_fraction(masked_line, spans, line_start) > 0.5:
            start = line_start + len(line) - len(line.lstrip())
            end = line_start + len(line.rstrip())
            blocks.append(
                ReasoningBlock(
                    start_char=start,
                    end_char=end,
                    start_token=-1,
                    end_token=-1,
                    block_type="equation",
                    text=text[start:end],
                )
            )
            continue
        # Walk the line: display math becomes its own equation block; inline
        # math stays embedded; plain text is sentence-split (boundaries inside
        # protected spans are skipped).
        segments: list[tuple[int, int, bool]] = []
        position = 0
        for start, end, is_display in spans:
            if end <= line_start or start >= line_start + len(line):
                continue
            if start > line_start + position:
                segments.append((position, start - line_start, False))
            if is_display:
                segments.append((start - line_start, end - line_start, True))
            else:
                segments.append((start - line_start, end - line_start, False))
            position = max(position, end - line_start)
        if position < len(line):
            segments.append((position, len(line), False))
        for part_start, part_end, is_equation in segments:
            if is_equation:
                abs_start = line_start + part_start
                abs_end = line_start + part_end
                blocks.append(
                    ReasoningBlock(
                        start_char=abs_start,
                        end_char=abs_end,
                        start_token=-1,
                        end_token=-1,
                        block_type="equation",
                        text=text[abs_start:abs_end],
                    )
                )
                continue
            sub_parts: list[tuple[int, int]] = []
            search_from = part_start
            for match in _SENTENCE_BOUNDARY.finditer(line[part_start:part_end]):
                boundary = part_start + match.start()
                if any(start < line_start + boundary < end for start, end, _ in spans):
                    continue
                sub_parts.append((search_from, boundary))
                search_from = boundary
            if search_from < part_end:
                sub_parts.append((search_from, part_end))
            for sub_start, sub_end in sub_parts:
                fragment = line[sub_start:sub_end]
                if not fragment.strip():
                    continue
                abs_start = line_start + sub_start
                abs_end = line_start + sub_end
                blocks.append(
                    ReasoningBlock(
                        start_char=abs_start,
                        end_char=abs_end,
                        start_token=-1,
                        end_token=-1,
                        block_type=_classify(
                            text[abs_start:abs_end],
                            is_last=line_index == len(lines) - 1,
                        ),
                        text=text[abs_start:abs_end],
                    )
                )
    # Re-classify the very last block as answer if it contains an answer marker.
    if blocks:
        last = blocks[-1]
        lowered = last.text.strip().lower()
        if any(marker in lowered for marker in _ANSWER_MARKERS):
            blocks[-1] = ReasoningBlock(
                start_char=last.start_char,
                end_char=last.end_char,
                start_token=-1,
                end_token=-1,
                block_type="answer",
                text=last.text,
            )
    return blocks


def _token_char_starts(tokenizer: object, token_ids: Sequence[int]) -> tuple[str, list[int]]:
    """Decode token-by-token; return full text and the char start of each token."""

    decode = tokenizer.decode
    text = decode(list(token_ids), skip_special_tokens=False)
    starts: list[int] = []
    for index in range(len(token_ids)):
        starts.append(len(decode(list(token_ids[:index]), skip_special_tokens=False)))
    return text, starts


def blocks_with_token_spans(
    tokenizer: object,
    token_ids: Sequence[int],
) -> tuple[str, list[ReasoningBlock]]:
    """Segment a tokenized teacher response; every block gets exact token spans."""

    text, char_starts = _token_char_starts(tokenizer, token_ids)
    blocks = segment_reasoning_blocks(text)
    resolved: list[ReasoningBlock] = []
    for block in blocks:
        start_token = _first_token_at_or_after(char_starts, block.start_char)
        end_token = _first_token_at_or_after(char_starts, block.end_char)
        resolved.append(
            ReasoningBlock(
                start_char=block.start_char,
                end_char=block.end_char,
                start_token=start_token,
                end_token=end_token,
                block_type=block.block_type,
                text=block.text,
            )
        )
    return text, resolved


def _first_token_at_or_after(char_starts: Sequence[int], char_offset: int) -> int:
    """First token index whose char start is >= char_offset."""

    low, high = 0, len(char_starts)
    while low < high:
        mid = (low + high) // 2
        if char_starts[mid] < char_offset:
            low = mid + 1
        else:
            high = mid
    return low


if __name__ == "__main__":
    import sys

    sample = (
        "We are given a triangle. AB = 3.5. \\[x^2 + y^2 = 25.\\] "
        "Therefore, x is 4. The final answer is \\boxed{4}."
    )
    if len(sys.argv) > 1:
        sample = sys.argv[1]
    for block in segment_reasoning_blocks(sample):
        print(f"[{block.block_type}] {block.start_char}:{block.end_char} {block.text!r}")
