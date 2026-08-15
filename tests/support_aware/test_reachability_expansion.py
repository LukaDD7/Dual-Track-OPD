"""CPU-side unit tests for the Phase-5 expansion manifest and adaptive rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from dual_track_opd.support_aware.reachability_expansion import (
    ManifestSpec,
    freeze_manifest,
    verify_manifest,
)
from dual_track_opd.support_aware.proposal_feasibility import ProposalConfig, select_prompt_records
from dual_track_opd.support_aware.prefix_intervention import (
    InterventionConfig,
    _aggregate,
    _adaptive_candidate,
    _rescue_decision,
    load_config as load_intervention_config,
)
from dual_track_opd.support_aware.reachability_wrong_controls import (
    WrongControlConfig,
    select_targets,
)


def _frontier(pool256: Path, strata: dict[str, list[str]]) -> None:
    target = pool256 / "frontier_analysis"
    target.mkdir(parents=True)
    rows = []
    for stratum, uids in strata.items():
        for index, uid in enumerate(uids):
            rows.append(
                {
                    "sample_uid": uid,
                    "observed_support_stratum": stratum,
                    "K": 8,
                    "correct_count": 0,
                }
            )
    (target / "frontier_prompts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _cohort(cohort_dir: Path, uids: list[str]) -> None:
    cohort_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "sample_uid": uids,
            "question": [f"question {uid}" for uid in uids],
            "answer": ["42"] * len(uids),
            "images": [None] * len(uids),
        }
    )
    frame.to_parquet(cohort_dir / "cohort.parquet")


def test_freeze_manifest_deterministic_strata_and_verify(tmp_path: Path) -> None:
    pool256 = tmp_path / "pool256"
    rescue = tmp_path / "rescue"
    output = tmp_path / "out"
    rescue.mkdir()
    strata = {
        "rare_success": [f"geo3k:r{i}" for i in range(5)],
        "no_correct_observed": [f"geo3k:n{i}" for i in range(6)],
        "mixed_support": [f"geo3k:m{i}" for i in range(5)],
    }
    _frontier(pool256, strata)
    # Two already-rescued prompts must be excluded from the manifest.
    (rescue / "rescue_comparisons.jsonl").write_text(
        "\n".join(
            json.dumps({"sample_uid": uid, "horizon": 64, "meets_preregistered_rescue_rule": True})
            for uid in ("geo3k:r3", "geo3k:m2")
        )
        + "\n",
        encoding="utf-8",
    )
    spec = ManifestSpec(
        pool256_dir=str(pool256),
        rescue_dir=str(rescue),
        output_dir=str(output),
        seed=20260815,
        no_correct_count=3,
        mixed_control_count=2,
    )
    first = freeze_manifest(spec)
    second = freeze_manifest(spec)
    assert first["stratum_counts"] == {"rare_success": 4, "no_correct_observed": 3, "mixed_support": 2}
    assert first["total_prompts"] == 9
    assert first["manifest_sha256"] == second["manifest_sha256"]
    manifest_path = output / "expansion_manifest_20260815.jsonl"
    uids = [json.loads(line)["sample_uid"] for line in manifest_path.read_text().splitlines()]
    assert "geo3k:r3" not in uids and "geo3k:m2" not in uids
    assert verify_manifest(manifest_path, pool256)["verified"] is True
    # Strata must follow the frozen order.
    strata_order = [json.loads(line)["stratum"] for line in manifest_path.read_text().splitlines()]
    assert strata_order == sorted(strata_order, key=lambda s: ("rare_success", "no_correct_observed", "mixed_support").index(s))


def test_verify_manifest_rejects_mismatch(tmp_path: Path) -> None:
    pool256 = tmp_path / "pool256"
    _frontier(pool256, {"rare_success": ["geo3k:r1"]})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"sample_uid": "geo3k:r1", "stratum": "no_correct_observed"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        verify_manifest(manifest, pool256)


def _intervention_config(**overrides) -> InterventionConfig:
    base = dict(
        proposal_dir="proposals",
        k32_run_dir="k32",
        cohort_dir="cohort",
        output_dir="out",
        student_model_path="student",
    )
    base.update(overrides)
    return InterventionConfig(**base)


def test_rescue_decision_thresholds_stage_and_k() -> None:
    config = _intervention_config(
        stage1_k=4,
        stage2_k=8,
        adaptive_confirm=True,
        rescue_min_mean_lift=0.20,
        rescue_min_probability=0.90,
    )
    baseline = {"K": 4, "correct_count": 0}
    strong = {"K": 4, "correct_count": 3}
    weak_wrong = {"K": 4, "correct_count": 0}
    decision = _rescue_decision("p0", 128, strong, weak_wrong, baseline, config)
    assert decision["meets_preregistered_rescue_rule"] is True
    assert decision["rescue_K"] == 4
    assert decision["rescue_stage"] == "stage1"

    weak_teacher = {"K": 4, "correct_count": 1}
    weak_baseline = {"K": 4, "correct_count": 1}
    decision = _rescue_decision("p0", 128, weak_teacher, weak_wrong, weak_baseline, config)
    assert decision["meets_preregistered_rescue_rule"] is False

    stage2_decision = _rescue_decision(
        "p0",
        128,
        {"K": 8, "correct_count": 6},
        {"K": 8, "correct_count": 1},
        {"K": 8, "correct_count": 1},
        config,
    )
    assert stage2_decision["rescue_stage"] == "stage2"
    assert stage2_decision["rescue_K"] == 8

    with pytest.raises(ValueError):
        _rescue_decision(
            "p0", 128, {"K": 4, "correct_count": 2}, {"K": 8, "correct_count": 1}, baseline, config
        )


def test_intervention_config_loads_expansion_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "i.yaml"
    yaml_path.write_text(
        """
model:
  student: /models/student
data:
  proposal_dir: /data/proposals
  k32_run_dir: /data/k32
  cohort_dir: /data/cohort
  wrong_source_run_dir: /data/pool256
  prompt_manifest: /data/manifest.jsonl
intervention:
  horizons: [64, 128, 256]
  stage1_k: 4
  stage2_k: 8
  adaptive_confirm: true
generation:
  continuations_per_arm: 8
  max_continuation_tokens: 2048
  temperature: 0.7
  top_p: 0.95
  seed: 1
analysis:
  posterior_draws: 2000
  rescue_min_mean_lift: 0.20
  rescue_min_probability: 0.90
hardware:
  device: cuda:0
output:
  dir: /data/out
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        proposal_dir=None,
        k32_run_dir=None,
        cohort_dir=None,
        cohort_parquet_path=None,
        output_dir=None,
        horizons=None,
        continuations_per_arm=None,
        max_continuation_tokens=None,
        shard_index=0,
        num_shards=1,
        max_prompts=None,
        prompt_manifest=None,
        wrong_source_run_dir=None,
        stage1_k=None,
        stage2_k=None,
        adaptive_confirm=None,
    )
    config = load_intervention_config(yaml_path, args)
    assert config.wrong_source_run_dir == "/data/pool256"
    assert config.prompt_manifest == "/data/manifest.jsonl"
    assert config.stage1_k == 4 and config.stage2_k == 8
    assert config.adaptive_confirm is True


def test_proposal_selection_pool256_manifest(tmp_path: Path) -> None:
    pool256 = tmp_path / "pool256"
    cohort_dir = tmp_path / "cohort"
    manifest_path = tmp_path / "manifest.jsonl"
    strata = {
        "rare_success": ["geo3k:r1", "geo3k:r2"],
        "no_correct_observed": ["geo3k:n1", "geo3k:n2", "geo3k:n3"],
        "mixed_support": ["geo3k:m1"],
    }
    _frontier(pool256, strata)
    all_uids = [uid for uids in strata.values() for uid in uids]
    _cohort(cohort_dir, all_uids)
    manifest_uids = ["geo3k:n3", "geo3k:r1", "geo3k:m1"]
    manifest_path.write_text(
        "\n".join(
            json.dumps({"sample_uid": uid, "stratum": next(s for s, us in strata.items() if uid in us)})
            for uid in manifest_uids
        )
        + "\n",
        encoding="utf-8",
    )
    config = ProposalConfig(
        k32_run_dir=str(tmp_path / "k32"),
        cohort_dir=str(cohort_dir),
        output_dir=str(tmp_path / "out"),
        student_model_path="student",
        teacher_model_path="teacher",
        pool256_run_dir=str(pool256),
        prompt_manifest=str(manifest_path),
        states=("rare_success", "no_correct_observed", "mixed_support"),
    )
    records, provenance = select_prompt_records(config)
    assert [record["sample_uid"] for record in records] == manifest_uids
    assert provenance["selection_mode"] == "pool256_manifest"
    assert provenance["expected_uids"] == manifest_uids
    by_uid = {record["sample_uid"]: record for record in records}
    assert by_uid["geo3k:r1"]["k32_summary"]["observed_stratum"] == "rare_success"
    assert by_uid["geo3k:m1"]["k32_summary"]["observed_stratum"] == "mixed_support"


def test_aggregate_adaptive_stage_mix_matches_baseline_k(tmp_path: Path) -> None:
    """Stage-2 K=8 units must not corrupt stage-1 K=4 horizon decisions."""

    uid = "p0"
    def unit(arm: str, horizon: int, K: int, correct: int) -> dict:
        return {
            "sample_uid": uid,
            "arm": arm,
            "horizon": horizon,
            "K": K,
            "correct_count": correct,
            "pass_rate": correct / K,
            "expected_U8": 0.5,
            "skipped_reason": None,
        }

    units = [
        unit("unaided", 0, 4, 1),
        unit("teacher_prefix", 64, 4, 2),
        unit("wrong_student_prefix", 64, 4, 0),
        unit("teacher_prefix", 128, 4, 3),
        unit("wrong_student_prefix", 128, 4, 0),
        unit("unaided", 0, 8, 2),
        unit("teacher_prefix", 128, 8, 6),
        unit("wrong_student_prefix", 128, 8, 0),
    ]
    result_dir = tmp_path / "prompt_results"
    result_dir.mkdir()
    (result_dir / "p0.json").write_text(
        json.dumps(
            {"schema_version": "x", "sample_uid": uid, "intervention_units": units, "rollouts": []}
        ),
        encoding="utf-8",
    )
    config = _intervention_config(
        horizons=(64, 128),
        stage1_k=4,
        stage2_k=8,
        adaptive_confirm=True,
        rescue_min_mean_lift=0.20,
        rescue_min_probability=0.90,
        seed=7,
    )
    summary = _aggregate(tmp_path, [uid], config)
    assert summary["rescue_comparison_count"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "rescue_comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_horizon = {int(row["horizon"]): row for row in rows}
    assert by_horizon[64]["rescue_K"] == 4
    assert by_horizon[64]["rescue_stage"] == "stage1"
    assert by_horizon[64]["meets_preregistered_rescue_rule"] is False
    assert by_horizon[128]["rescue_K"] == 8
    assert by_horizon[128]["rescue_stage"] == "stage2"
    assert by_horizon[128]["meets_preregistered_rescue_rule"] is True
    minimal = [
        json.loads(line)
        for line in (tmp_path / "minimal_rescue_prefixes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["horizon"] for row in minimal] == [128]


def test_adaptive_candidate_ignores_skipped_units() -> None:
    """Skipped (K=0) arms must not reach the rescue decision."""

    config = _intervention_config(
        horizons=(64, 128, 256, 512),
        stage1_k=4,
        stage2_k=8,
        adaptive_confirm=True,
        rescue_min_mean_lift=0.20,
        rescue_min_probability=0.90,
        seed=7,
    )

    def unit(arm: str, horizon: int, K: int, correct: int, skipped: str | None = None) -> dict:
        return {
            "sample_uid": "p0",
            "arm": arm,
            "horizon": horizon,
            "K": K,
            "correct_count": correct,
            "pass_rate": correct / K if K else None,
            "expected_U8": 0.5 if K else None,
            "skipped_reason": skipped,
        }

    units = [
        unit("unaided", 0, 4, 1),
        unit("teacher_prefix", 64, 4, 2),
        unit("wrong_student_prefix", 64, 4, 0),
        unit("teacher_prefix", 128, 4, 3),
        unit("wrong_student_prefix", 128, 4, 0),
        unit("teacher_prefix", 256, 4, 2),
        unit("wrong_student_prefix", 256, 4, 1),
        # Trace shorter than the horizon: skipped units must be ignored.
        unit("teacher_prefix", 512, 0, 0, skipped="source_shorter_than_horizon"),
        unit("wrong_student_prefix", 512, 0, 0, skipped="source_shorter_than_horizon"),
    ]
    assert _adaptive_candidate(units, config, "p0") == 128

    # If every teacher arm is skipped, no candidate may be selected.
    skipped_only = [
        unit("unaided", 0, 4, 1),
        unit("teacher_prefix", 64, 0, 0, skipped="source_shorter_than_horizon"),
        unit("wrong_student_prefix", 64, 4, 0),
    ]
    assert _adaptive_candidate(skipped_only, config, "p0") is None


def test_wrong_control_target_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    proposal = tmp_path / "proposals"
    rescued = tmp_path / "rescued"
    proposal.mkdir()
    rescued.mkdir()
    manifest.write_text(
        "\n".join(
            json.dumps({"sample_uid": f"geo3k:{uid}"}) for uid in ("a", "b", "c")
        )
        + "\n",
        encoding="utf-8",
    )
    (proposal / "retained_proposals.jsonl").write_text(
        "\n".join(
            json.dumps({"sample_uid": f"geo3k:{uid}"}) for uid in ("a", "b", "c")
        )
        + "\n",
        encoding="utf-8",
    )
    (rescued / "rescue_comparisons.jsonl").write_text(
        json.dumps({"sample_uid": "geo3k:a", "horizon": 64}) + "\n",
        encoding="utf-8",
    )
    config = WrongControlConfig(
        manifest_path=str(manifest),
        proposal_dir=str(proposal),
        rescue_dirs=(str(rescued),),
        cohort_dir=str(tmp_path),
        cohort_parquet_path=None,
        output_dir=str(tmp_path / "out"),
        student_model_path="student",
    )
    targets, info = select_targets(config)
    assert targets == ["geo3k:b", "geo3k:c"]
    assert info["targets"] == 2
