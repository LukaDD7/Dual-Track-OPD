"""Unit tests for the mixed-difficulty RL pool preparation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sft_rl"))

from prep_mixed_rl_pools import (  # noqa: E402
    gt_rule_kind,
    inline_cauldron_bytes,
    select_stratified,
    select_stratified_by_share,
)

import random


# --- gt_rule_kind must mirror mmf_reward's fixed gates ----------------------


def test_gt_rule_kind_cauldron_forms():
    # trailing periods (vsr / raven yesno, tallyqa numbers, MCQ letters)
    assert gt_rule_kind("Yes.") == "yesno"
    assert gt_rule_kind("No.") == "yesno"
    assert gt_rule_kind("4.") == "number"
    assert gt_rule_kind("B.") == "letter"
    assert gt_rule_kind("(F)") == "letter"  # raven F..H
    assert gt_rule_kind("H") == "letter"
    # clean forms
    assert gt_rule_kind("yes") == "yesno"
    assert gt_rule_kind("-3.25") == "number"
    assert gt_rule_kind("D") == "letter"


def test_gt_rule_kind_mmf_forms_unchanged():
    assert gt_rule_kind("65.12") == "number"
    assert gt_rule_kind("50, 57.6, 292") == "latex"  # digits but not pure number
    assert gt_rule_kind("the shaded rectangle") == "other"
    assert gt_rule_kind(None) is None
    assert gt_rule_kind("") is None
    assert gt_rule_kind(".") is None


# --- select_stratified: deterministic, per-source caps ----------------------


def _mkrow(src: str, uid: str) -> dict:
    return {"data_source": src, "sample_uid": uid}


def test_select_stratified_caps_and_determinism():
    rows = [_mkrow("a", f"a{i}") for i in range(10)] + [_mkrow("b", f"b{i}") for i in range(3)]
    sel1, taken1 = select_stratified(rows, 4, random.Random(42))
    sel2, taken2 = select_stratified(rows, 4, random.Random(42))
    assert taken1 == {"a": 4, "b": 3}
    assert [r["sample_uid"] for r in sel1] == [r["sample_uid"] for r in sel2]
    assert len(sel1) == 7


def test_select_stratified_min_group_drops_small_sources():
    rows = [_mkrow("a", f"a{i}") for i in range(10)] + [_mkrow("tiny", "t0")]
    sel, taken = select_stratified(rows, 4, random.Random(0), min_group=2)
    assert "tiny" not in taken
    assert len(sel) == 4


# --- select_stratified_by_share: proportional MMF stratification -------------


def test_select_stratified_by_share_hits_target_and_keeps_large_source():
    # MMR1-like dominance: one source is ~47% of the pool; the flat cap would
    # under-fill (8644 < 12000) and crush it to the same size as a tiny source.
    # total eligible 7000+5000+3000 = 15000 >= target, so no clamping.
    rows = (
        [_mkrow("dominant", f"d{i}") for i in range(7000)]
        + [_mkrow("med", f"m{i}") for i in range(5000)]
        + [_mkrow("small", f"s{i}") for i in range(3000)]
    )
    sel, taken = select_stratified_by_share(rows, 12000, random.Random(42))
    assert len(sel) == 12000
    assert taken == {"dominant": 5600, "med": 4000, "small": 2400}


def test_select_stratified_by_share_determinism():
    rows = [_mkrow("a", f"a{i}") for i in range(7000)] + [_mkrow("b", f"b{i}") for i in range(5000)]
    sel1, t1 = select_stratified_by_share(rows, 12000, random.Random(7))
    sel2, t2 = select_stratified_by_share(rows, 12000, random.Random(7))
    assert [r["sample_uid"] for r in sel1] == [r["sample_uid"] for r in sel2]
    assert t1 == t2


def test_select_stratified_by_share_caps_at_eligibility():
    # Only 5 eligible rows exist; a 12k target must clamp to 5, not crash.
    rows = [_mkrow("a", f"a{i}") for i in range(5)]
    sel, taken = select_stratified_by_share(rows, 12000, random.Random(1))
    assert len(sel) == 5
    assert taken == {"a": 5}


def test_select_stratified_by_share_empty():
    sel, taken = select_stratified_by_share([], 12000, random.Random(1))
    assert sel == [] and taken == {}


# --- Cauldron image inlining ------------------------------------------------


def test_inline_cauldron_bytes(tmp_path):
    img_file = tmp_path / "abc.img"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    row = {"images": [{"image_url": str(img_file)}]}
    assert inline_cauldron_bytes(row) is True
    assert row["images"][0]["bytes"] == b"\x89PNG\r\n\x1a\nfakepng"
    assert row["images"][0]["path"] == str(img_file)


def test_inline_cauldron_bytes_missing_file_drops_row(tmp_path):
    row = {"images": [{"image_url": str(tmp_path / "missing.img")}]}
    assert inline_cauldron_bytes(row) is False


def test_inline_cauldron_bytes_passes_inline_rows_through():
    row = {"images": [{"bytes": b"pngdata", "path": None}]}
    assert inline_cauldron_bytes(row) is True
    assert row["images"] == [{"bytes": b"pngdata", "path": None}]


def test_inline_cauldron_bytes_no_images_drops_row():
    assert inline_cauldron_bytes({"images": []}) is False
