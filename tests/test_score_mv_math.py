import json

from dual_track_opd.eval.score_mv_math import (
    _parse_judge_verdict,
    extract_choice,
    score_choice_rows,
    summarize,
)


def test_extract_choice_prefers_final_answer_markers() -> None:
    response = "A is wrong. Reasoning... The final answer is: C\nD"
    assert extract_choice(response) == "D"
    assert extract_choice("The final answer is <answer>B</answer>") == "B"
    assert extract_choice("\\boxed{A}") == "A"
    assert extract_choice("b") == "B"
    assert extract_choice("no option") is None


def test_choice_full_denominator_and_truncation_stats() -> None:
    rows = [
        {"sample_id": "1", "prediction": "The answer is A", "ground_truth": "A"},
        {"sample_id": "2", "prediction": "", "ground_truth": "B", "finish_reason": "length"},
        {"sample_id": "3", "prediction": "C", "ground_truth": "C", "finish_reason": "length"},
    ]
    scored = score_choice_rows(rows)
    summary = summarize(scored)
    assert [row["correct"] for row in scored] == [True, False, True]
    assert summary == {
        "rows": 3,
        "correct": 2,
        "truncated": 2,
        "judge_unparsed_or_unextracted": 0,
        "accuracy_full_denominator": 2 / 3,
    }


def test_judge_verdicts_use_official_shapes() -> None:
    assert _parse_judge_verdict("True", "single-step") is True
    assert _parse_judge_verdict("false", "choice") is False
    assert _parse_judge_verdict(" 2/3 ", "multi-step") == (2, 3)
    assert _parse_judge_verdict("invalid", "multi-step") is None


def test_json_output_shape_matches_official_protocol(tmp_path) -> None:
    # This exercises metadata loading without requiring a judge endpoint.
    from dual_track_opd.eval.score_mv_math import load_metadata

    metadata = [
        {"problem_id": 1, "answer": "B", "answer_type": "choice"},
        {"problem_id": 2, "answer": "7", "answer_type": "single-step"},
        {"problem_id": 3, "answer": "(1) 1\n(2) 2", "answer_type": "multi-step"},
    ]
    path = tmp_path / "MV-MATH.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded = load_metadata(path)
    assert set(loaded) == {"1", "2", "3"}
    assert loaded["3"]["answer_type"] == "multi-step"
