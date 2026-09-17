from dual_track_opd.fc_opd.evidence_generation import classify_task_infer_evidence


def test_option_value_is_solve_like():
    result = classify_task_infer_evidence("B. 60", ["30", "60"])
    assert result["class"] == "solve_like"
    assert "option_only" in result["flags"]


def test_single_letter_is_not_clean_infer():
    result = classify_task_infer_evidence("A", ["A. 30", "B. 60"])
    assert result["class"] in {"solve_like", "unusable"}


def test_intermediate_geometry_fact_is_clean():
    result = classify_task_infer_evidence(
        "Using perpendicular bisector, X is midpoint of CD, so CX=12",
        ["24", "36"],
    )
    assert result["class"] == "clean_infer"


def test_full_solution_language_is_solve_like():
    result = classify_task_infer_evidence(
        "Therefore the answer is B because the final requested angle is 60.",
        ["30", "60"],
    )
    assert result["class"] == "solve_like"
    assert "solution_language" in result["flags"]
