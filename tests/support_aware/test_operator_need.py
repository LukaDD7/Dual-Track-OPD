"""Unit tests for operator-need token scalars (C_t, D_t, Q_t)."""

from __future__ import annotations

import math

from dual_track_opd.support_aware.operator_need import (
    _robust_unit_normalize,
    _token_scalars,
)


def test_robust_normalization_is_bounded_and_clips_outliers() -> None:
    values = [-100.0, 0.0, 1.0, 2.0, 3.0, 100.0]
    normalized = _robust_unit_normalize(values)
    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in normalized)
    assert normalized == sorted(normalized)


def test_robust_normalization_constant_input_is_zero() -> None:
    assert _robust_unit_normalize([0.9, 0.9, 0.9]) == [0.0, 0.0, 0.0]


def _row(
    *,
    student_top_ids: list[int],
    teacher_top_ids: list[int],
    student_logp: dict[int, float],
    teacher_logp: dict[int, float],
    k: int,
) -> dict:
    def _top100_logp(ids: list[int], probs: dict[int, float], vocab_size: int = 20) -> list[float]:
        values = []
        for token in ids:
            values.append(probs.get(token, -20.0))
        while len(values) < vocab_size:
            values.append(-20.0)
        return values

    student_top = student_top_ids[:k]
    teacher_top = teacher_top_ids[:k]
    return {
        "student_top100_ids": student_top,
        "student_top100_logp": _top100_logp(student_top, student_logp),
        "teacher_q_top100_ids": teacher_top,
        "teacher_q_logp_top100": _top100_logp(teacher_top, teacher_logp),
        "teacher_logp_at_student_top100_ids": _top100_logp(
            student_top, {**teacher_logp, **{t: teacher_logp.get(t, 1e-6) for t in student_top}}
        ),
        "student_p_logp_top100_at_teacher_ids": _top100_logp(
            teacher_top, {**student_logp, **{t: student_logp.get(t, 1e-6) for t in teacher_top}}
        ),
    }


def test_c_t_is_exact_cross_gather_sum() -> None:
    student_logp = {0: -0.5, 1: -1.0, 2: -2.0}
    teacher_logp = {0: -0.2, 1: -0.8, 2: -3.0, 3: -0.3}
    row = _row(
        student_top_ids=[0, 1, 2, 3],
        teacher_top_ids=[0, 3, 1, 2],
        student_logp=student_logp,
        teacher_logp=teacher_logp,
        k=3,
    )
    c_t, d_t, q_t, off_count, off_ids, _ = _token_scalars(row, 3)
    expected = sum(math.exp(teacher_logp[token]) for token in [0, 1, 2])
    assert math.isclose(c_t, expected, rel_tol=1e-6)
    assert off_ids == [3]
    assert off_count == 1
    assert q_t == 1.0
    assert d_t > 0.0


def test_q_t_low_when_off_support_diffuse() -> None:
    student_logp = {0: -0.1, 1: -0.2, 2: -0.5, 3: -0.9}
    teacher_logp = {0: -0.1, 1: -0.2, 2: -0.5, 3: -0.9, 4: -0.6, 5: -1.5}
    row = _row(
        student_top_ids=[0, 1, 2, 3],
        teacher_top_ids=[0, 4, 5, 1],
        student_logp=student_logp,
        teacher_logp=teacher_logp,
        k=4,
    )
    c_t, d_t, q_t, off_count, off_ids, _ = _token_scalars(row, 4)
    assert off_count == 2
    assert 0.0 < q_t < 1.0
    assert d_t > 0.0


def test_no_off_support_gives_q_zero() -> None:
    student_logp = {0: -0.1, 1: -0.2}
    teacher_logp = {0: -0.1, 1: -0.2}
    row = _row(
        student_top_ids=[0, 1],
        teacher_top_ids=[0, 1],
        student_logp=student_logp,
        teacher_logp=teacher_logp,
        k=2,
    )
    _, _, q_t, off_count, _, _ = _token_scalars(row, 2)
    assert off_count == 0
    assert q_t == 0.0
