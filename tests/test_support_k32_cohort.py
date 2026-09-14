from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from dual_track_opd.support_aware import diagnostic as diagnostic_module
from dual_track_opd.support_aware.diagnostic import chunk_ranges
from dual_track_opd.support_aware.k32_cohort import (
    STRATUM_ORDER,
    build_candidates,
    observed_stratum,
    prepare_cohort,
    select_balanced_cohort,
)


def _summary(uid: str, correct_count: int, K: int = 8) -> dict:
    return {
        "sample_uid": uid,
        "K": K,
        "correct_count": correct_count,
        "greedy_correct": False,
        "support_state": "screening_only",
    }


def _rollouts(uid: str, *, K: int = 8, tokens: int = 1000) -> list[dict]:
    return [
        {
            "sample_uid": uid,
            "is_greedy": False,
            "rollout_id": index,
            "response_token_count": tokens,
            "finish_reason": "length" if tokens > 2048 else "stop",
            "malformed": False,
        }
        for index in range(1, K + 1)
    ]


@pytest.mark.parametrize(
    ("correct_count", "expected"),
    [
        (0, "no_correct_observed"),
        (1, "rare_success"),
        (2, "rare_success"),
        (3, "mixed_support"),
        (7, "mixed_support"),
        (8, "all_correct_observed"),
    ],
)
def test_observed_stratum_uses_stochastic_count_only(correct_count: int, expected: str):
    assert observed_stratum(correct_count, 8) == expected


def test_build_and_select_cohort_balances_stratum_and_length():
    summaries: list[dict] = []
    rollouts: list[dict] = []
    correct_counts = {
        "no_correct_observed": 0,
        "rare_success": 1,
        "mixed_support": 4,
        "all_correct_observed": 8,
    }
    for stratum, correct_count in correct_counts.items():
        for index in range(4):
            uid = f"{stratum}:{index}"
            summaries.append(_summary(uid, correct_count))
            rollouts.extend(_rollouts(uid, tokens=1000 if index < 2 else 3000))

    candidates = build_candidates(summaries, rollouts, seed=7)
    selected = select_balanced_cohort(candidates, per_stratum=4)

    assert len(selected) == 16
    for stratum in STRATUM_ORDER:
        rows = [row for row in selected if row["observed_stratum"] == stratum]
        assert len(rows) == 4
        assert {row["length_bucket"] for row in rows} == {"le_cutoff", "gt_cutoff"}


def test_select_balanced_cohort_backfills_missing_length_bucket():
    candidates = [
        {
            "sample_uid": f"{stratum}:{index}",
            "observed_stratum": stratum,
            "length_bucket": "le_cutoff",
            "selection_key": f"{index:03d}",
        }
        for stratum in STRATUM_ORDER
        for index in range(3)
    ]
    selected = select_balanced_cohort(candidates, per_stratum=2)
    assert len(selected) == 8


def test_build_candidates_rejects_incomplete_screening_rollouts():
    with pytest.raises(ValueError, match="expected 8 stochastic rollouts"):
        build_candidates([_summary("x", 0)], _rollouts("x", K=7))


def test_chunk_ranges_preserve_order_and_cover_batch():
    assert chunk_ranges(0, 4) == []
    assert chunk_ranges(9, 4) == [(0, 4), (4, 8), (8, 9)]
    with pytest.raises(ValueError, match="positive"):
        chunk_ranges(9, 0)


def test_prepare_cohort_materializes_manifest_and_parquet(tmp_path):
    run = tmp_path / "screening"
    run.mkdir()
    summaries: list[dict] = []
    rollouts: list[dict] = []
    source_rows: list[dict] = []
    correct_counts = (0, 1, 4, 8)
    for stratum_index, correct_count in enumerate(correct_counts):
        for index in range(2):
            uid = f"uid-{stratum_index}-{index}"
            summaries.append(_summary(uid, correct_count))
            rollouts.extend(_rollouts(uid, tokens=1000 if index == 0 else 3000))
            source_rows.append({"sample_uid": uid, "question": uid, "image": b"image"})

    for filename, rows in (
        ("prompt_support_summary.jsonl", summaries),
        ("rollouts.jsonl", rollouts),
    ):
        (run / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (run / "summary.json").write_text(json.dumps({
        "exact_token_alignment_rate": 1.0,
        "prompt_token_hash_rate": 1.0,
        "shard_completeness_rate": 1.0,
        "missing_image_count": 0,
        "non_finite_score_count": 0,
    }) + "\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({
        "exit_status": "GATE_FAIL",
        "git_commit": "screening-commit",
        "git_dirty": False,
    }) + "\n", encoding="utf-8")
    source = tmp_path / "source.parquet"
    pd.DataFrame(source_rows).to_parquet(source, index=False)
    output = tmp_path / "cohort"

    manifest = prepare_cohort(
        run,
        source,
        output,
        per_stratum=2,
        seed=9,
        num_shards=4,
    )

    assert manifest["num_prompts"] == 8
    assert len(manifest["shard_plan"]) == 4
    assert (output / "cohort.parquet").is_file()
    assert (output / "cohort_manifest.json").is_file()
    assert len(pd.read_parquet(output / "cohort.parquet")) == 8
    with pytest.raises(FileExistsError):
        prepare_cohort(run, source, output, per_stratum=2)


def test_exit_zero_on_complete_does_not_hide_gate_failures(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"dataset_path": "/not-loaded.parquet", "num_prompts": 2},
        "models": {"student": "/not-loaded-model"},
        "output": {"root": str(tmp_path)},
    }), encoding="utf-8")

    def fake_run(resolved):
        run_id = "complete-gate-fail"
        output = tmp_path / "support_aware_opd" / run_id
        output.mkdir(parents=True)
        for name in (
            "rollouts.jsonl",
            "prompt_support_summary.jsonl",
            "summary.json",
            "run_manifest.json",
            "resolved_config.yaml",
        ):
            (output / name).write_text("{}\n", encoding="utf-8")
        return {
            "run_id": run_id,
            "acceptance_gates": [["scientific_signal", False, "expected result"]],
        }

    monkeypatch.setattr(diagnostic_module, "run_diagnostic", fake_run)
    assert diagnostic_module.main([
        "--config", str(config),
        "--mode", "full",
        "--exit-zero-on-complete",
    ]) == 0
