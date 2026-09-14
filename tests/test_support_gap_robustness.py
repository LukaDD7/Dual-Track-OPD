import json

import pytest

from dual_track_opd.support_aware.gap_robustness import (
    cross_fitted_length_residuals,
    exact_content_gaps,
    paired_prompt_auc,
    run_analysis,
    score_variants,
    trek_trim_indices,
)


def _raw(uid="p", rollout_id=1, correct=True, count=100):
    student = [-float(index + 1) for index in range(count)] + [-999.0]
    teacher = [value + (1.0 if correct else -1.0) for value in student[:-1]] + [-998.0]
    row = {
        "sample_uid": uid,
        "rollout_id": rollout_id,
        "is_greedy": False,
        "correct": correct,
        "response_token_ids": list(range(count + 1)),
        "content_mask": [True] * count + [False],
        "exact_token_alignment": True,
        "teacher_sampled_token_log_probs": teacher,
        "student_sampled_token_log_probs": student,
    }
    row["teacher_gap"] = 1.0 if correct else -1.0
    row["content_token_count"] = count
    row["response_token_count"] = count + 1
    row["finish_reason"] = "stop"
    return row


def test_exact_content_gaps_excludes_terminal_and_requires_alignment():
    gaps, nll = exact_content_gaps(_raw(count=3))
    assert gaps.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert nll.tolist() == pytest.approx([1.0, 2.0, 3.0])
    broken = _raw(count=3)
    broken["exact_token_alignment"] = False
    with pytest.raises(ValueError, match="alignment"):
        exact_content_gaps(broken)


def test_trek_trim_and_score_variants_use_student_nll_indices():
    retained = trek_trim_indices(list(range(100)), low_fraction=0.10, high_fraction=0.02)
    assert retained.tolist() == list(range(10, 98))
    scores = score_variants(_raw(count=600), prefix_horizons=(128, 512, 1024))
    assert scores["mean_gap"] == pytest.approx(1.0)
    assert scores["trek_trimmed_gap"] == pytest.approx(1.0)
    assert scores["prefix_128_gap"] == pytest.approx(1.0)
    assert scores["prefix_512_gap"] == pytest.approx(1.0)
    assert scores["prefix_1024_gap"] is None


def test_paired_prompt_auc_uses_identical_rows_for_metric_and_baseline():
    rows = [
        {"sample_uid": "p1", "correct": True, "metric_gap": 3.0, "mean_gap": 1.0},
        {"sample_uid": "p1", "correct": False, "metric_gap": 1.0, "mean_gap": 2.0},
        {"sample_uid": "p2", "correct": True, "metric_gap": None, "mean_gap": 3.0},
        {"sample_uid": "p2", "correct": False, "metric_gap": 1.0, "mean_gap": 1.0},
    ]
    result = paired_prompt_auc(rows, score_key="metric_gap", seed=3, resamples=100)
    assert result["eligible_prompt_count"] == 1
    assert result["metric_auc"] == pytest.approx(1.0)
    assert result["baseline_auc_on_same_rows"] == pytest.approx(0.0)
    assert result["paired_delta_auc"] == pytest.approx(1.0)


def test_cross_fitted_length_residuals_do_not_use_correctness():
    rows = []
    for index in range(20):
        length = 10 + index
        rows.append({
            "rollout_key": f"r{index}",
            "mean_gap": 2.0 + 3.0 * __import__("math").log1p(length),
            "content_token_count": length,
            "finish_reason": "stop",
            "fold": index % 5,
            "correct": index % 2 == 0,
        })
    residuals = cross_fitted_length_residuals(rows)
    assert len(residuals) == 20
    assert max(abs(value) for value in residuals.values()) < 1e-8


def test_run_analysis_writes_paired_outputs(tmp_path):
    source = tmp_path / "k32"
    source.mkdir()
    rows = [_raw("p", 1, True, 600), _raw("p", 2, False, 600)]
    (source / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (source / "summary.json").write_text("{}", encoding="utf-8")
    (source / "k32_validation.json").write_text(
        json.dumps({"valid": True, "K": 32}), encoding="utf-8"
    )
    output = tmp_path / "analysis"
    summary = run_analysis(source, output, bootstrap_resamples=100)
    assert summary["stochastic_rollout_count"] == 2
    assert summary["score_analyses"]["trek_trimmed_gap"]["metric_auc"] == 1.0
    assert (output / "gap_metric_summary.csv").is_file()
