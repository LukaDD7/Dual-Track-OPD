import math

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.signal_decomposer import (
    TeacherTopK,
    compute_condition_signals,
    jensen_shannon_topk,
)


def _scores(ids, probabilities, tail=None) -> TeacherTopK:
    tail_tensor = None if tail is None else torch.tensor([[math.log(tail)]])
    return TeacherTopK(
        token_ids=torch.tensor([[ids]], dtype=torch.int64),
        log_probs=torch.log(torch.tensor([[probabilities]], dtype=torch.float32)),
        tail_log_prob=tail_tensor,
    )


def test_js_is_zero_for_identical_distributions():
    scores = _scores([1, 2], [0.7, 0.2], tail=0.1)
    js = jensen_shannon_topk(scores, scores)
    assert torch.allclose(js, torch.zeros_like(js), atol=1e-7)


def test_js_handles_different_topk_supports_and_is_symmetric():
    left = _scores([1, 2], [0.8, 0.1], tail=0.1)
    right = _scores([2, 3], [0.1, 0.8], tail=0.1)
    forward = jensen_shannon_topk(left, right)
    backward = jensen_shannon_topk(right, left)
    assert forward.item() > 0
    assert torch.allclose(forward, backward, atol=1e-7)


def test_duplicate_ids_are_coalesced():
    duplicated = _scores([1, 1], [0.4, 0.4], tail=0.2)
    coalesced = _scores([1, 2], [0.8, 0.0 + 1e-12], tail=0.2)
    js = jensen_shannon_topk(duplicated, coalesced)
    assert js.item() < 1e-5


def test_compute_condition_signals_and_sampled_delta():
    full = _scores([1, 2], [0.8, 0.1], tail=0.1)
    degraded = _scores([1, 2], [0.4, 0.5], tail=0.1)
    blur = _scores([1, 2], [0.5, 0.4], tail=0.1)
    task = _scores([1, 2], [0.7, 0.2], tail=0.1)
    free = _scores([1, 2], [0.3, 0.6], tail=0.1)
    signals = compute_condition_signals(
        {
            Condition.FULL: full,
            Condition.DEGRADED: degraded,
            Condition.BLUR: blur,
            Condition.TASK: task,
            Condition.FREE: free,
        },
        sampled_token_ids=torch.tensor([[1]]),
    )
    assert set(signals) == {
        "visual_detail",
        "visual_detail_sampled_logprob_delta",
        "visual_detail_delta",
        "visual_detail_delta_sampled_logprob_delta",
        "visual_detail_blur",
        "visual_detail_blur_sampled_logprob_delta",
        "task_extraction",
        "task_extraction_sampled_logprob_delta",
    }
    assert signals["visual_detail"].item() > 0
    assert signals["visual_detail_sampled_logprob_delta"].item() > 0


def test_visual_detail_delta_uses_full_minus_degraded_not_blur():
    full = _scores([1, 2], [0.8, 0.1], tail=0.1)
    degraded = _scores([1, 2], [0.8, 0.1], tail=0.1)
    blur = _scores([1, 2], [0.1, 0.8], tail=0.1)
    signals = compute_condition_signals(
        {
            Condition.FULL: full,
            Condition.DEGRADED: degraded,
            Condition.BLUR: blur,
        }
    )

    assert torch.allclose(signals["visual_detail_delta"], torch.zeros_like(signals["visual_detail_delta"]))
    assert signals["visual_detail_blur"].item() > 0


def test_non_finite_teacher_scores_are_rejected():
    scores = TeacherTopK(
        token_ids=torch.tensor([[[1]]]),
        log_probs=torch.tensor([[[float("nan")]]]),
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        scores.validate()
