"""Unit tests for full-scale PTD-PO hint pool preparation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sft_rl"))

from prep_hint_pools import (  # noqa: E402
    extract_turn_pairs,
    gt_type,
    normalize_placeholders,
)


def test_gt_type_classification():
    assert gt_type("(B)") == "letter"
    assert gt_type("yes") == "yesno"
    assert gt_type("-3.25") == "pure_number"
    assert gt_type("50, 57.6, 292") == "has_number"
    assert gt_type("the shaded rectangle") == "other"


def test_normalize_placeholders_for_expanded_turns():
    # first turn already carries the placeholder -> unchanged body, one placeholder
    assert normalize_placeholders("<image>\nWhat color?", 1) == "<image>\nWhat color?"
    # later turns share the image but carry no placeholder -> placeholder added
    assert normalize_placeholders("What is on top?", 1) == "<image>\nWhat is on top?"
    # duplicated placeholders are normalized to exactly n_images
    assert normalize_placeholders("<image><image>\nWhat color?", 1) == "<image>\nWhat color?"
    assert normalize_placeholders("text-only question", 0) == "text-only question"


def test_extract_turn_pairs_skips_unpaired():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    assert extract_turn_pairs(msgs) == [("q1", "a1"), ("q3", "a3")]
