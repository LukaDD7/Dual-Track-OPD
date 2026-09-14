import pytest
import torch

from dual_track_opd.fc_opd.chunk_parser import parse_response_chunks


class CharacterTokenizer:
    """A deterministic tokenizer that deliberately splits every XML tag."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


TOKENIZER = CharacterTokenizer()


class ByteChangingTokenizer(CharacterTokenizer):
    """Simulate a UTF-8 character whose partial prefix decodes as replacement text."""

    CHINESE_FIRST = 1_000_001
    CHINESE_SECOND = 1_000_002

    def encode(self, text: str) -> list[int]:
        ids = []
        for character in text:
            if character == "你":
                ids.extend((self.CHINESE_FIRST, self.CHINESE_SECOND))
            else:
                ids.append(ord(character))
        return ids

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        output = []
        index = 0
        while index < len(token_ids):
            token_id = token_ids[index]
            if token_id == self.CHINESE_FIRST:
                if index + 1 < len(token_ids) and token_ids[index + 1] == self.CHINESE_SECOND:
                    output.append("你")
                    index += 2
                    continue
                output.append("�")
            elif token_id == self.CHINESE_SECOND:
                output.append("�")
            else:
                output.append(chr(token_id))
            index += 1
        return "".join(output)


def _parse(text: str, **kwargs):
    ids = TOKENIZER.encode(text)
    return parse_response_chunks(ids, text, TOKENIZER, **kwargs)


def test_valid_response_excludes_split_tag_tokens():
    text = (
        "<visual_evidence>two red circles</visual_evidence>"
        "<reasoning>two objects imply two</reasoning>"
        "<answer>2</answer>"
    )
    parsed = _parse(text)
    assert parsed.format_valid
    assert parsed.errors == ()
    assert parsed.token_counts == {
        "visible_evidence": len("two red circles"),
        "diagram_inference": 0,
        "reasoning": len("two objects imply two"),
        "answer": 1,
    }
    tag_position = text.index("<visual_evidence>")
    assert not parsed.visual_evidence_mask[tag_position]


def test_empty_evidence_is_valid():
    parsed = _parse(
        "<visual_evidence></visual_evidence>"
        "<reasoning>language-only</reasoning>"
        "<answer>A</answer>"
    )
    assert parsed.format_valid
    assert parsed.token_counts["visible_evidence"] == 0


@pytest.mark.parametrize(
    ("text", "error_fragment"),
    [
        (
            "<visual_evidence>x<reasoning>y</reasoning><answer>A</answer>",
            "visual_evidence:close_tag_count=0",
        ),
        (
            "<visual_evidence>x</visual_evidence><visual_evidence>z</visual_evidence>"
            "<reasoning>y</reasoning><answer>A</answer>",
            "visual_evidence:open_tag_count=2",
        ),
        (
            "<reasoning>y</reasoning><visual_evidence>x</visual_evidence><answer>A</answer>",
            "reasoning:out_of_order",
        ),
    ],
)
def test_invalid_formats_use_documented_fallback(text: str, error_fragment: str):
    parsed = _parse(text)
    assert not parsed.format_valid
    assert error_fragment in parsed.errors
    assert parsed.fallback == "all_reasoning"
    assert torch.all(parsed.reasoning_mask)
    assert not torch.any(parsed.visual_evidence_mask)
    assert not torch.any(parsed.answer_mask)


def test_chinese_text_is_token_aligned():
    parsed = _parse(
        "<visual_evidence>图中有三个蓝色方块</visual_evidence>"
        "<reasoning>逐个计数得到三</reasoning>"
        "<answer>三</answer>"
    )
    assert parsed.format_valid
    assert parsed.token_counts["visible_evidence"] == len("图中有三个蓝色方块")
    assert parsed.token_counts["answer"] == 1


def test_byte_split_chinese_tokens_share_semantic_character_span():
    tokenizer = ByteChangingTokenizer()
    text = (
        "<visual_evidence>你</visual_evidence>"
        "<reasoning>看图</reasoning><answer>A</answer>"
    )
    ids = tokenizer.encode(text)
    parsed = parse_response_chunks(ids, text, tokenizer)
    assert parsed.format_valid
    assert parsed.token_counts["visible_evidence"] == 2


def test_long_ocr_string_remains_in_evidence_chunk():
    ocr = "SERIAL-" + "0123456789" * 100
    parsed = _parse(
        f"<visual_evidence>{ocr}</visual_evidence>"
        "<reasoning>read the serial</reasoning><answer>valid</answer>"
    )
    assert parsed.format_valid
    assert parsed.token_counts["visible_evidence"] == len(ocr)


def test_v2_response_returns_four_chunk_labels():
    text = (
        "<visible_evidence>labels 13 and 10</visible_evidence>"
        "<diagram_inference>altitude bisects the base</diagram_inference>"
        "<reasoning>use the inferred right triangle</reasoning>"
        "<answer>B</answer>"
    )
    parsed = _parse(text)
    assert parsed.format_valid
    assert len(parsed.chunk_labels) == len(TOKENIZER.encode(text))
    assert parsed.token_counts["visible_evidence"] == len("labels 13 and 10")
    assert parsed.token_counts["diagram_inference"] == len("altitude bisects the base")


def test_truncated_v2_response_reports_missing_tag_names():
    text = (
        "<visible_evidence>labels 13 and 10</visible_evidence>"
        "<diagram_inference>altitude bisects the base</diagram_inference>"
        "<reasoning>starts reasoning but never closes"
    )
    parsed = _parse(text)

    assert not parsed.format_valid
    assert "reasoning:close_tag_count=0" in parsed.errors
    assert "answer:open_tag_count=0" in parsed.errors
    assert "answer:close_tag_count=0" in parsed.errors
    assert len(parsed.chunk_labels) == len(TOKENIZER.encode(text))


def test_truncated_response_is_invalid():
    parsed = _parse(
        "<visual_evidence>x</visual_evidence><reasoning>unfinished",
        fallback="all_answer",
    )
    assert not parsed.format_valid
    assert parsed.fallback == "all_answer"
    assert torch.all(parsed.answer_mask)


def test_decoded_text_mismatch_is_hard_parse_failure():
    text = "<visual_evidence>x</visual_evidence><reasoning>y</reasoning><answer>A</answer>"
    ids = TOKENIZER.encode(text)
    parsed = parse_response_chunks(ids, text + "!", TOKENIZER)
    assert not parsed.format_valid
    assert "decoded_text_mismatch" in parsed.errors


def test_non_whitespace_text_outside_chunks_is_invalid():
    parsed = _parse(
        "preface"
        "<visual_evidence>x</visual_evidence>"
        "<reasoning>y</reasoning>"
        "<answer>A</answer>"
    )
    assert not parsed.format_valid
    assert "unexpected_text_outside_chunks" in parsed.errors
