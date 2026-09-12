"""Unit tests for reasoning-block segmentation."""

from __future__ import annotations

from dual_track_opd.support_aware.reasoning_blocks import (
    _first_token_at_or_after,
    blocks_with_token_spans,
    segment_reasoning_blocks,
)


def test_equation_not_split_inside_math() -> None:
    text = "We are given a triangle. \\[x^2 + y^2 = 25.\\] Therefore, x is 4."
    blocks = segment_reasoning_blocks(text)
    equation = next(block for block in blocks if block.block_type == "equation")
    assert equation.text.strip() == "\\[x^2 + y^2 = 25.\\]"
    assert "Therefore" not in equation.text


def test_final_answer_block() -> None:
    text = "The angle is 30 degrees. The final answer is \\boxed{30}."
    blocks = segment_reasoning_blocks(text)
    assert blocks[-1].block_type == "answer"
    assert "boxed" in blocks[-1].text


def test_newline_derivation_lines_kept() -> None:
    text = "Step 1: sum angles.\nStep 2: x = 180 - 60 = 120.\nAnswer: 120"
    blocks = segment_reasoning_blocks(text)
    assert len(blocks) == 3
    assert "Step 1" in blocks[0].text
    assert "Step 2" in blocks[1].text


class _StubTokenizer:
    """Each token decodes to one character; used for span alignment checks."""

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(97 + int(value)) for value in ids)


def test_blocks_with_token_spans_are_exact() -> None:
    tokenizer = _StubTokenizer()
    token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    text, blocks = blocks_with_token_spans(tokenizer, token_ids)
    assert text == "abcdefghij"
    assert blocks[0].start_token == 0
    assert blocks[-1].end_token == len(token_ids)
    for block in blocks:
        assert block.end_token > block.start_token


def test_first_token_at_or_after() -> None:
    starts = [0, 2, 5, 9]
    assert _first_token_at_or_after(starts, 0) == 0
    assert _first_token_at_or_after(starts, 2) == 1
    assert _first_token_at_or_after(starts, 6) == 3
    assert _first_token_at_or_after(starts, 99) == 4
