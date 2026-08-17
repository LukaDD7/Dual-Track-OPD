"""Unit tests for the CPU parts of the micro-operator experiment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dual_track_opd.support_aware.micro_operator import (
    _summarize,
    select_candidate_blocks,
)


def _write_blocks(path: Path) -> None:
    rows = []
    for uid in ("p1", "p2"):
        for index in range(5):
            rows.append(
                {
                    "prompt_id": uid,
                    "block_index": index,
                    "block_type": "sentence",
                    "start_token": index * 4,
                    "end_token": index * 4 + 4,
                    "token_count": 4,
                    "F_i": [0.1, 0.3, 0.5, 0.7, 0.9][index] if uid == "p1" else 0.0,
                    "R_i": [0.0, 0.2, 0.8, 0.3, 0.1][index] if uid == "p1" else 0.0,
                    "C_i": 0.99,
                    "Q_i": 0.5,
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_select_candidate_blocks_picks_extremes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blocks.jsonl"
        _write_blocks(path)
        candidates = select_candidate_blocks(
            block_table_path=str(path),
            rescue_uids=["p1", "p2"],
            max_candidates=60,
        )
        by_uid: dict[str, dict[str, int]] = {}
        for candidate in candidates:
            by_uid.setdefault(candidate["prompt_id"], {})[candidate["operator"]] = candidate[
                "block_index"
            ]
        assert by_uid["p1"]["FKL"] == 4  # highest F_i
        assert by_uid["p1"]["RKL"] == 2  # highest R_i
        assert by_uid["p1"]["control"] == 0  # lowest |F|+|R|
        assert by_uid["p2"]["FKL"] == 0  # ties resolve to lowest block index
        assert len(candidates) <= 3 * 2


def test_select_candidate_blocks_respects_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blocks.jsonl"
        _write_blocks(path)
        candidates = select_candidate_blocks(
            block_table_path=str(path),
            rescue_uids=["p1", "p2"],
            max_candidates=3,
        )
        assert len(candidates) == 3


def test_summarize_correlation() -> None:
    results = []
    for i in range(1, 6):
        f = 0.1 * i
        r = 0.05 * i
        g_fkl = -0.1 + 0.2 * i  # grows with F
        g_rkl = 0.05 * i        # grows with R, smaller than g_fkl at high i
        results.extend(
            [
                {
                    "prompt_id": f"p{i}",
                    "block_index": i,
                    "role": "FKL",
                    "operator": "FKL",
                    "F_i": f,
                    "R_i": r,
                    "G_i": g_fkl,
                },
                {
                    "prompt_id": f"p{i}",
                    "block_index": i,
                    "role": "FKL",
                    "operator": "RKL",
                    "F_i": f,
                    "R_i": r,
                    "G_i": g_rkl,
                },
            ]
        )
    summary = _summarize(results)
    assert summary["n_candidates"] == 5
    assert summary["n_results"] == 10
    assert summary["n_fkl"] == 5
    assert summary["n_rkl"] == 5
    assert summary["corr_F_vs_G_fkl_minus_rkl"] is not None
    assert summary["corr_R_vs_G_rkl_minus_fkl"] is not None
    assert summary["corr_F_vs_G_fkl_minus_rkl"] > 0.9
    assert summary["corr_R_vs_G_rkl_minus_fkl"] < -0.9
