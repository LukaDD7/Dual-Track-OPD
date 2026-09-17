"""Unit tests for region/path-level acquisition selection logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dual_track_opd.support_aware.region_acquisition import select_regions


def _write_blocks(path: Path) -> None:
    rows = []
    for uid, h_star in (("p1", 40), ("p2", 80)):
        for index in range(8):
            start = index * 10
            end = start + 10
            rows.append(
                {
                    "prompt_id": uid,
                    "block_index": index,
                    "start_token": start,
                    "end_token": end,
                    "h_star": h_star,
                    "pre_hstar": end <= h_star,
                    "C_i": 0.99,
                    "R_i": 0.1,
                    "F_i": 0.01,
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_full_prefix_region() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blocks.jsonl"
        _write_blocks(path)
        regions = select_regions(
            block_table_path=str(path),
            rescue_uids=["p1", "p2"],
            region_mode="full_prefix",
            window_blocks=None,
            min_tokens=20,
        )
        by_uid = {region["prompt_id"]: region for region in regions}
        # p1: blocks 0..3 pre-h* (end 10..40), start_token 0, end_token 40
        assert by_uid["p1"]["start_token"] == 0
        assert by_uid["p1"]["end_token"] == 40
        assert by_uid["p1"]["token_count"] == 40
        assert by_uid["p1"]["h_star"] == 40
        # p2: blocks 0..7 (end 80 <= 80), region 0..80
        assert by_uid["p2"]["end_token"] == 80


def test_window_before_hstar() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blocks.jsonl"
        _write_blocks(path)
        regions = select_regions(
            block_table_path=str(path),
            rescue_uids=["p1"],
            region_mode="window_before_hstar",
            window_blocks=2,
            min_tokens=10,
        )
        assert len(regions) == 1
        region = regions[0]
        assert region["start_block"] == 2
        assert region["end_block"] == 3
        assert region["start_token"] == 20
        assert region["end_token"] == 40
