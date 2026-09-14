import pytest
import torch

from dual_track_opd.fc_opd.hidden_state_visual_focus import (
    compute_teacher_student_visual_gap,
    compute_visual_focus_scores,
    compute_visual_prototype,
    extract_prediction_state_positions,
    extract_visual_token_positions,
    rank_normalize_scores,
)


def test_prediction_state_positions_shift_back_one_token():
    positions = extract_prediction_state_positions(response_start=5, response_length=4, batch_size=1, seq_len=9)
    assert positions.tolist() == [[4, 5, 6, 7]]


def test_prediction_state_positions_reject_no_prefix_state():
    with pytest.raises(ValueError, match="positive"):
        extract_prediction_state_positions(response_start=0, response_length=1, batch_size=1)


def test_visual_focus_uses_prediction_state_not_consumed_token_state():
    hidden = torch.zeros((1, 6, 2), dtype=torch.float32)
    hidden[0, 1] = torch.tensor([1.0, 0.0])  # image token prototype
    hidden[0, 2] = torch.tensor([1.0, 0.0])  # prefix state predicting y_0
    hidden[0, 3] = torch.tensor([-1.0, 0.0])  # after consuming y_0
    visual_positions = [torch.tensor([1])]
    prediction_positions = extract_prediction_state_positions(
        response_start=3, response_length=1, batch_size=1, seq_len=6
    )

    scores = compute_visual_focus_scores(hidden, visual_positions, prediction_positions)
    assert scores.tolist() == [[pytest.approx(1.0)]]


def test_extract_visual_token_positions_respects_prompt_length():
    input_ids = torch.tensor([[10, 99, 11, 99, 12]])
    positions = extract_visual_token_positions(input_ids, image_token_id=99, prompt_lengths=[3])
    assert [row.tolist() for row in positions] == [[1]]


def test_rank_normalize_scores_are_in_unit_interval_with_ties():
    scores = torch.tensor([[2.0, 2.0, 5.0, 1.0]])
    ranks = rank_normalize_scores(scores)
    assert ranks.min().item() >= 0.0
    assert ranks.max().item() <= 1.0
    assert ranks[0, 0].item() == pytest.approx(ranks[0, 1].item())
    assert ranks[0, 2].item() == pytest.approx(1.0)


def test_teacher_student_gap_uses_ranked_scores_not_raw_dimensions():
    teacher = torch.tensor([[0.1, 0.2, 0.9]])
    student = torch.tensor([[10.0, 0.0, 5.0]])
    gap = compute_teacher_student_visual_gap(teacher, student)
    assert torch.all(gap.teacher_rank >= 0)
    assert torch.all(gap.teacher_rank <= 1)
    assert gap.gap_raw[0, 2].item() > 0
    assert gap.gap_pos[0, 0].item() == 0


def test_visual_prototype_rejects_missing_visual_tokens():
    with pytest.raises(ValueError, match="no visual"):
        compute_visual_prototype(torch.zeros((1, 3, 2)), [torch.tensor([], dtype=torch.long)])
