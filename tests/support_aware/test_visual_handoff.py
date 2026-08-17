"""Unit tests for the CPU parts of the visual handoff diagnostic."""

from __future__ import annotations

import torch

from dual_track_opd.support_aware.reasoning_blocks import ReasoningBlock
from dual_track_opd.support_aware.visual_handoff import (
    _block_containing_token,
    _block_signals,
    _handoff_block_index,
    _jensen_shannon_from_logits,
    _visual_change_point,
)


def test_change_point_detects_step_down() -> None:
    values = [1.0, 0.9, 1.1, 0.2, 0.1, 0.0, 0.15]
    result = _visual_change_point(values)
    assert result["significant"] is True
    assert 2 <= result["tau_block"] <= 3


def test_change_point_ignores_single_spike() -> None:
    # One noisy +5 block must not masquerade as a sustained regime change.
    values = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = _visual_change_point(values)
    assert result["significant"] is False
    assert result["tau_block"] is None


def test_change_point_flat_is_not_significant() -> None:
    result = _visual_change_point([0.5, 0.5, 0.5, 0.5, 0.5])
    assert result["significant"] is False
    assert result["tau_block"] is None


def test_change_point_requires_downward_direction() -> None:
    # Upward step must not be selected (mu_pre > mu_post is required).
    result = _visual_change_point([0.1, 0.2, 0.3, 1.0, 1.1, 1.2])
    assert result["significant"] is False


def test_handoff_is_last_pre_change_block_and_requires_significance() -> None:
    assert _handoff_block_index({"significant": True, "tau_block": 3}) == 2
    assert _handoff_block_index({"significant": False, "tau_block": 3}) is None
    assert _handoff_block_index({"significant": True, "tau_block": None}) is None


def test_full_vocab_js_is_zero_only_for_identical_distributions() -> None:
    first = torch.tensor([[8.0, -8.0, -8.0], [0.0, 0.0, 0.0]])
    second = torch.tensor([[-8.0, 8.0, -8.0], [0.0, 0.0, 0.0]])
    js = _jensen_shannon_from_logits(first, second)
    assert js.shape == (2,)
    assert js[0] > 0.6
    assert torch.allclose(js[1], torch.tensor(0.0), atol=1e-7)


def test_block_signals_delta_v() -> None:
    blocks = [
        ReasoningBlock(0, 4, 0, 4, "sentence", "abcd"),
        ReasoningBlock(4, 8, 4, 8, "sentence", "efgh"),
    ]
    signals = _block_signals(
        response_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        blocks=blocks,
        teacher_js={
            "degraded": [0.0] * 8,
            "null": [0.0] * 8,
        },
        student_js={
            "degraded": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "null": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
        null_control=True,
    )
    assert signals[0]["V_T_i"] == 0.0
    assert signals[0]["V_S_i"] == 1.0
    assert signals[0]["DeltaV_i"] == -1.0
    assert signals[1]["DeltaV_i"] == 0.0
    assert signals[0]["V_T_null_i"] == 0.0
    assert signals[0]["DeltaV_null_i"] == -1.0


def test_block_containing_token() -> None:
    blocks = [
        ReasoningBlock(0, 4, 0, 4, "sentence", "abcd"),
        ReasoningBlock(4, 8, 4, 8, "sentence", "efgh"),
    ]
    assert _block_containing_token(blocks, 0) == 0
    assert _block_containing_token(blocks, 3) == 0
    assert _block_containing_token(blocks, 4) == 1
    assert _block_containing_token(blocks, 7) == 1
    assert _block_containing_token(blocks, 9) is None
