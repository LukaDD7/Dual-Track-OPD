from dual_track_opd.fc_opd.verifier import verify_geometry3k_response


def test_answer_letter_matches_choice_letter():
    result = verify_geometry3k_response(
        question="Find x.",
        choices=["30", "60"],
        response_text="<answer>B</answer>",
        answer_metadata="B",
    )
    assert result["answer_extracted"] == "B"
    assert result["correct"] is True
    assert result["reward"] == 1.0


def test_numeric_answer_matches_choice_value():
    result = verify_geometry3k_response(
        question="Find x.",
        choices=["30", "60"],
        response_text="<answer>60</answer>",
        answer_metadata="B",
    )
    assert result["answer_extracted"] == "60"
    assert result["correct"] is True


def test_malformed_no_answer_gets_zero_reward():
    result = verify_geometry3k_response(
        question="Find x.",
        choices=["30", "60"],
        response_text="I am not sure.",
        answer_metadata="B",
    )
    assert result["malformed"] is True
    assert result["format_valid"] is False
    assert result["reward"] == 0.0


def test_wrong_but_formatted_gets_partial_reward():
    result = verify_geometry3k_response(
        question="Find x.",
        choices=["30", "60"],
        response_text="<answer>A</answer>",
        answer_metadata="B",
    )
    assert result["correct"] is False
    assert result["format_valid"] is True
    assert result["reward"] == 0.25
