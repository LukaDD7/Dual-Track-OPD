"""Focused unit tests for the offline reachability-proxy exporter."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from dual_track_opd.support_aware.reachability_proxy import (
    build_proxy_study_rows,
    coarse_topk_tail_fkl,
    horizon_aggregates,
    load_config,
    teacher_trace_id,
    write_proxy_study_csv,
    _selected_teacher_trace,
)


def _token_rows(trace_length: int, prompt_id: str = "p0") -> list[dict]:
    rows = []
    for t in range(trace_length):
        rows.append(
            {
                "prompt_id": prompt_id,
                "teacher_trace_id": f"{prompt_id}:proposal-1:abc",
                "position_index": t,
                "token_id": t,
                "student_nll": 0.5 + 0.01 * t,
                "d_coarse_fkl": 0.2 + 0.005 * t,
            }
        )
    return rows


def _trace(sample_uid: str, proposal_id: int, rank: int, length: int = 80) -> dict:
    ids = list(range(1000, 1000 + length))
    return {
        "sample_uid": sample_uid,
        "proposal_id": proposal_id,
        "reachability_rank": rank,
        "correct": True,
        "retained_for_fkl": True,
        "response_token_ids": ids,
        "response_token_hash": hashlib.sha256(
            repr(tuple(ids)).encode()
        ).hexdigest(),
    }


def test_coarse_fkl_matches_exact_full_vocabulary_reference() -> None:
    """Bucket KL equals exact KL when the tail conditional matches."""

    top_count = 100
    tail_count = 5
    vocab = top_count + tail_count
    q_logits = torch.randn(vocab) * 0.7
    p_logits = torch.randn(vocab) * 0.9
    q = torch.softmax(q_logits, dim=-1)
    p = torch.softmax(p_logits, dim=-1)
    teacher_values, teacher_ids = torch.topk(q, k=top_count)
    q_i = teacher_values
    q_tail = float(1.0 - q_i.sum())
    p_i = p[teacher_ids]
    p_tail = float(1.0 - p_i.sum())

    coarse = coarse_topk_tail_fkl(
        q_logp_topk=q_i.log().tolist(),
        q_tail_logp=math.log(q_tail),
        p_logp_topk=p_i.log().tolist(),
        p_tail_prob=p_tail,
        epsilon=1e-12,
    )
    exact = float((q * (q.log() - p.log())).sum())
    # Tail bucket with non-uniform conditionals differs by the within-bucket KL.
    teacher_id_set = set(int(value) for value in teacher_ids.tolist())
    tail_ids = [value for value in range(vocab) if value not in teacher_id_set]
    q_cond = q[tail_ids] / q_tail
    p_cond = p[tail_ids] / p_tail
    within_bucket = float((q_cond * (q_cond.log() - p_cond.log())).sum())
    assert math.isclose(coarse, exact - q_tail * within_bucket, rel_tol=1e-6)

    # With matching uniform tail conditionals the bucket KL is exact.
    q_tail_uniform = torch.full((tail_count,), 0.01)
    p_tail_uniform = torch.full((tail_count,), 0.02)
    coarse_uniform = coarse_topk_tail_fkl(
        q_logp_topk=q_i.log().tolist(),
        q_tail_logp=math.log(0.05),
        p_logp_topk=p_i.log().tolist(),
        p_tail_prob=0.10,
        epsilon=1e-12,
    )
    exact_uniform = float(
        (q_i * (q_i.log() - p_i.log())).sum()
        + (q_tail_uniform * (q_tail_uniform.log() - p_tail_uniform.log())).sum()
    )
    assert math.isclose(coarse_uniform, exact_uniform, rel_tol=1e-6)


def test_nll_gather_equals_full_vocab_log_softmax() -> None:
    torch.manual_seed(7)
    logits = torch.randn(5, 32100)
    teacher_ids = torch.tensor([1, 99, 321, 5000, 20000])
    student_logp = torch.log_softmax(logits.float(), dim=-1)
    gathered = student_logp[torch.arange(5), teacher_ids]
    for t in range(5):
        direct = torch.log_softmax(logits[t].float(), dim=-1)
        assert math.isclose(float(gathered[t]), float(direct[int(teacher_ids[t])]), rel_tol=1e-5)
        nll = -float(gathered[t])
        assert math.isclose(nll, -float(direct[int(teacher_ids[t])]), rel_tol=1e-5)


def test_mass_conservation_for_topk_plus_tail() -> None:
    torch.manual_seed(11)
    teacher_logits = torch.randn(1, 32100)
    student_logits = torch.randn(1, 32100)
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    values, ids = torch.topk(teacher_logp, k=100, dim=-1)
    q_tail = 1.0 - values.exp().sum()
    p_at_ids = student_logp[0, ids[0]].exp()
    p_tail = 1.0 - p_at_ids.sum()
    assert float(q_tail) >= -1e-6
    assert math.isclose(float(q_tail + values.exp().sum()), 1.0, abs_tol=1e-6)
    assert math.isclose(float(p_tail + p_at_ids.sum()), 1.0, abs_tol=1e-6)


def test_horizon_aggregation_exact_and_short_trajectory() -> None:
    rows = _token_rows(trace_length=80)
    aggregates = horizon_aggregates(token_rows=rows, trace_length=80, horizons=[64, 128])
    assert [row["horizon"] for row in aggregates] == [64]
    row = aggregates[0]
    assert row["position"] == 64 / 80
    assert math.isclose(row["cum_student_nll"], sum(r["student_nll"] for r in rows[:64]))
    assert math.isclose(
        row["cum_top100_fkl_tail"], sum(r["d_coarse_fkl"] for r in rows[:64])
    )

    with pytest.raises(ValueError):
        horizon_aggregates(token_rows=rows[:-1], trace_length=80, horizons=[64])
    with pytest.raises(ValueError):
        horizon_aggregates(token_rows=rows, trace_length=79, horizons=[64])


def test_strict_join_rejects_missing_and_duplicate_keys() -> None:
    trace = _trace("p0", proposal_id=1, rank=1)
    token_rows = _token_rows(80, "p0") + _token_rows(80, "p1")
    selected = {"p0": trace, "p1": _trace("p1", proposal_id=2, rank=1)}
    rescue = [
        {"sample_uid": "p0", "horizon": 64, "meets_preregistered_rescue_rule": True},
        {"sample_uid": "p1", "horizon": 64, "meets_preregistered_rescue_rule": False},
    ]
    study = build_proxy_study_rows(
        token_rows=token_rows, rescue_rows=rescue, selected_traces=selected, horizons=[64, 128]
    )
    assert len(study) == 2
    assert study[0]["prompt_id"] == "p0" and study[0]["rescue_gold"] is True

    missing = [
        {"sample_uid": "p2", "horizon": 64, "meets_preregistered_rescue_rule": False}
    ]
    with pytest.raises(ValueError):
        build_proxy_study_rows(
            token_rows=token_rows, rescue_rows=missing, selected_traces=selected, horizons=[64]
        )
    duplicate = rescue + [rescue[0]]
    with pytest.raises(ValueError):
        build_proxy_study_rows(
            token_rows=token_rows, rescue_rows=duplicate, selected_traces=selected, horizons=[64, 128]
        )


def test_analyze_partial_study_marks_missing_minimal_prompts() -> None:
    """Smoke analyze with a subset of prompts must not crash on other min rows."""

    from dual_track_opd.support_aware.reachability_proxy import ProxyConfig, run_analyze

    output = tmp_path / "out"
    output.mkdir()
    rescue_dir = tmp_path / "rescue"
    proposal_dir = tmp_path / "proposals"
    rescue_dir.mkdir()
    proposal_dir.mkdir()
    trace = _trace("p0", proposal_id=1, rank=1, length=80)
    (rescue_dir / "rescue_comparisons.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_uid": "p0", "horizon": 64, "meets_preregistered_rescue_rule": True},
                {"sample_uid": "p1", "horizon": 64, "meets_preregistered_rescue_rule": True},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (rescue_dir / "minimal_rescue_prefixes.jsonl").write_text(
        json.dumps({"sample_uid": "p0", "horizon": 64, "meets_preregistered_rescue_rule": True})
        + "\n"
        + json.dumps({"sample_uid": "p1", "horizon": 64, "meets_preregistered_rescue_rule": True})
        + "\n",
        encoding="utf-8",
    )
    (proposal_dir / "retained_proposals.jsonl").write_text(
        json.dumps(trace) + "\n",
        encoding="utf-8",
    )
    (output / "proxy_token_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _token_rows(80, "p0")) + "\n",
        encoding="utf-8",
    )
    config = ProxyConfig(
        output_dir=str(output),
        proposal_dir=str(proposal_dir),
        k32_run_dir=str(tmp_path),
        cohort_dir=str(tmp_path),
        pool256_dir=str(tmp_path),
        causal_dir=str(tmp_path),
        rescue_dir=str(rescue_dir),
        student_model_path="student",
        teacher_model_path="teacher",
        horizons=(64,),
    )
    analysis = run_analyze(config)
    coverage = analysis["coverage"]
    assert coverage["prompts"] == 1
    assert coverage["rescue_positive_prompts_in_study"] == 1
    assert coverage["minimal_horizons_reproduced"] is True
    assert analysis["minimal_horizon_reproduction"]["p0"]["reproduced_minimal_horizon"] == 64
    assert analysis["minimal_horizon_reproduction"]["p1"]["missing_from_study"] is True
    assert (output / "proxy_study.csv").is_file()


def test_csv_deterministic_ordering_and_hash(tmp_path: Path) -> None:
    rows = build_proxy_study_rows(
        token_rows=_token_rows(80, "p0") + _token_rows(80, "p1"),
        rescue_rows=[
            {"sample_uid": "p1", "horizon": 64, "meets_preregistered_rescue_rule": False},
            {"sample_uid": "p0", "horizon": 64, "meets_preregistered_rescue_rule": True},
        ],
        selected_traces={
            "p0": _trace("p0", proposal_id=1, rank=1),
            "p1": _trace("p1", proposal_id=2, rank=1),
        },
        horizons=[64],
    )
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    hash_a = write_proxy_study_csv(first, rows)
    hash_b = write_proxy_study_csv(second, rows)
    assert hash_a == hash_b
    header = first.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("prompt_id,teacher_trace_id,horizon,rescue_gold,position")


def test_selection_rule_lowest_reachability_rank() -> None:
    retained = [
        _trace("p0", proposal_id=3, rank=2),
        _trace("p0", proposal_id=1, rank=1),
        _trace("p0", proposal_id=2, rank=5),
        {**_trace("p0", proposal_id=9, rank=0), "correct": False},
        {**_trace("p0", proposal_id=8, rank=0), "retained_for_fkl": False},
    ]
    selected = _selected_teacher_trace(retained, "p0")
    assert selected["proposal_id"] == 1
    assert selected["reachability_rank"] == 1
    trace_id = teacher_trace_id(selected)
    assert trace_id.startswith("p0:proposal-1:")
    assert teacher_trace_id(selected) == trace_id


def test_load_config_expands_env_vars(tmp_path: Path) -> None:
    os.environ["DTOPD_TEST_ROOT"] = "/tmp/dtopd-test"
    config_path = tmp_path / "proxy.yaml"
    config_path.write_text(
        """
models:
  student: ${DTOPD_TEST_ROOT}/student
  teacher: ${DTOPD_TEST_ROOT}/teacher
data:
  proposal_dir: ${DTOPD_TEST_ROOT}/proposals
  k32_run_dir: ${DTOPD_TEST_ROOT}/k32
  cohort_dir: ${DTOPD_TEST_ROOT}/cohort
  pool256_dir: ${DTOPD_TEST_ROOT}/pool256
  causal_dir: ${DTOPD_TEST_ROOT}/causal
  rescue_dir: ${DTOPD_TEST_ROOT}/rescue
output:
  dir: ${DTOPD_TEST_ROOT}/out
proxy:
  top_k: 100
  horizons: [64, 128, 256, 512]
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.student_model_path == "/tmp/dtopd-test/student"
    assert config.teacher_model_path == "/tmp/dtopd-test/teacher"
    assert config.top_k == 100
    assert config.horizons == (64, 128, 256, 512)
    assert config.rescue_dir == "/tmp/dtopd-test/rescue"
