import math
import json

import pandas as pd

import pytest

from dual_track_opd.support_aware.proposal_feasibility import (
    ProposalConfig,
    merge_shards,
    mark_reachable_proposals,
    select_prompt_records,
    trimmed_length_normalized_nll,
)


def test_trimmed_nll_removes_low_boilerplate_and_high_outliers():
    # Losses are 0..99.  A 10%/2% trim removes 10 lowest and 2 highest.
    result = trimmed_length_normalized_nll(
        [-float(loss) for loss in range(100)],
        low_trim_fraction=0.10,
        high_trim_fraction=0.02,
    )
    assert result["low_trim_count"] == 10
    assert result["high_trim_count"] == 2
    assert result["retained_token_count"] == 88
    assert result["trimmed_length_normalized_nll"] == pytest.approx(sum(range(10, 98)) / 88)


def test_trimmed_nll_respects_content_mask_and_rejects_empty():
    result = trimmed_length_normalized_nll(
        [-1.0, -2.0, -999.0],
        content_mask=[True, True, False],
        low_trim_fraction=0.0,
        high_trim_fraction=0.0,
    )
    assert result["trimmed_length_normalized_nll"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="no finite"):
        trimmed_length_normalized_nll([math.nan], content_mask=[True])


def test_reachability_selection_keeps_only_correct_finite_top_r():
    rows = [
        {"proposal_id": 1, "correct": True, "trimmed_length_normalized_nll": 2.0},
        {"proposal_id": 2, "correct": False, "trimmed_length_normalized_nll": 0.1},
        {"proposal_id": 3, "correct": True, "trimmed_length_normalized_nll": 1.0},
        {"proposal_id": 4, "correct": True, "trimmed_length_normalized_nll": 3.0},
    ]
    selected = mark_reachable_proposals(rows, retain_count=2)
    by_id = {row["proposal_id"]: row for row in selected}
    assert by_id[3]["reachability_rank"] == 1
    assert by_id[1]["reachability_rank"] == 2
    assert by_id[4]["reachability_rank"] == 3
    assert by_id[2]["reachability_rank"] is None
    assert {row["proposal_id"] for row in selected if row["retained_for_fkl"]} == {1, 3}


def test_select_prompt_records_uses_k32_state_and_deterministic_shards(tmp_path):
    run = tmp_path / "run"
    cohort = tmp_path / "cohort"
    run.mkdir()
    cohort.mkdir()
    (run / "k32_validation.json").write_text(
        json.dumps({"valid": True, "K": 32}), encoding="utf-8"
    )
    summaries = [
        {"sample_uid": "zero", "K": 32, "correct_count": 0},
        {"sample_uid": "rare", "K": 32, "correct_count": 4},
        {"sample_uid": "mixed", "K": 32, "correct_count": 16},
        {"sample_uid": "all", "K": 32, "correct_count": 32},
    ]
    (run / "prompt_support_summary.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in summaries), encoding="utf-8"
    )
    pd.DataFrame(
        {"sample_uid": ["zero", "rare", "mixed", "all"], "question": ["q"] * 4}
    ).to_parquet(cohort / "cohort.parquet", index=False)
    base = dict(
        k32_run_dir=str(run),
        cohort_dir=str(cohort),
        output_dir=str(tmp_path / "out"),
        student_model_path="student",
        teacher_model_path="teacher",
        num_shards=2,
    )
    shard_zero, provenance_zero = select_prompt_records(
        ProposalConfig(**base, shard_index=0)
    )
    shard_one, provenance_one = select_prompt_records(
        ProposalConfig(**base, shard_index=1)
    )
    selected = [row["sample_uid"] for row in shard_zero + shard_one]
    assert set(selected) == {"zero", "rare"}
    assert provenance_zero["all_selected_uids"] == provenance_one["all_selected_uids"]
    assert not set(provenance_zero["expected_uids"]).intersection(
        provenance_one["expected_uids"]
    )


def test_merge_shards_requires_clean_complete_uid_coverage(tmp_path):
    all_uids = ["zero", "rare"]
    shard_dirs = []
    common_config = {
        "k32_run_dir": "run",
        "cohort_dir": "cohort",
        "student_model_path": "student",
        "teacher_model_path": "teacher",
        "states": ["no_correct_observed", "rare_success"],
        "proposals_per_prompt": 4,
        "retain_count": 2,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 4096,
        "seed": 1,
        "response_format": "legacy_answer",
        "low_trim_fraction": 0.1,
        "high_trim_fraction": 0.02,
        "dtype": "bfloat16",
        "teacher_device": "cuda:0",
        "student_device": "cuda:1",
        "num_shards": 2,
        "max_prompts": None,
    }
    for shard_index, uid in enumerate(all_uids):
        shard = tmp_path / f"shard-{shard_index}"
        result_dir = shard / "prompt_results"
        result_dir.mkdir(parents=True)
        config = {
            **common_config,
            "output_dir": str(shard),
            "shard_index": shard_index,
        }
        manifest = {
            "status": "completed",
            "git_commit": "abc123",
            "git_dirty": False,
            "config": config,
            "provenance": {
                "all_selected_uids": all_uids,
                "expected_uids": [uid],
            },
        }
        (shard / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        prompt = {
            "sample_uid": uid,
            "observed_stratum": "no_correct_observed" if uid == "zero" else "rare_success",
            "correct_proposal_count": 0,
            "retained_proposal_count": 0,
            "proposals": [
                {
                    "sample_uid": uid,
                    "proposal_id": 1,
                    "correct": False,
                    "malformed": False,
                    "finish_reason": "stop",
                    "response_token_count": 3,
                    "retained_for_fkl": False,
                }
            ],
        }
        (result_dir / f"{uid}.json").write_text(json.dumps(prompt), encoding="utf-8")
        shard_dirs.append(shard)

    output = tmp_path / "merged"
    summary = merge_shards(shard_dirs, output)
    assert summary["complete"] is True
    assert summary["completed_prompt_count"] == 2
    assert summary["proposal_count"] == 2
    assert (output / "retained_proposals.jsonl").read_text(encoding="utf-8") == ""
