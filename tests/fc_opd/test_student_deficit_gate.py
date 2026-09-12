from dual_track_opd.fc_opd.four_condition_offline_builder import (
    build_verifier_learning_value_gate,
    compute_student_deficit_capability_scores,
)


def _block(values):
    return {"actual_token_log_probs": values, "token_ids": [[0] for _ in values], "log_probs": [[0.0] for _ in values]}


def test_student_deficit_formula_and_chunk_gate():
    teacher = {
        "full": _block([1.0, 1.0, 0.0, 1.0]),
        "degraded": _block([0.0, 0.0, 1.0, 0.0]),
        "task_visible": _block([0.0, 0.0, 0.0, 0.0]),
        "free": _block([0.0, 0.0, 0.0, 0.0]),
        "task_infer": _block([0.0, 0.0, 0.0, 0.0]),
        "task_solve": _block([0.0, 0.0, 0.0, 0.0]),
    }
    student = {
        "full": _block([0.0, 1.0, 0.0, 0.0]),
        "degraded": _block([0.0, 0.0, 0.0, 0.0]),
        "task_visible": _block([0.0, 0.0, 0.0, 0.0]),
        "free": _block([0.0, 0.0, 0.0, 0.0]),
        "task_infer": _block([0.0, 0.0, 0.0, 0.0]),
        "task_solve": _block([0.0, 0.0, 0.0, 0.0]),
    }
    chunks = {
        "visible_evidence": [[0, 1]],
        "reasoning": [[1, 2]],
        "diagram_inference": [[2, 3]],
        "answer": [[3, 4]],
    }

    scores = compute_student_deficit_capability_scores(
        teacher_condition_scores=teacher,
        student_condition_scores=student,
        chunk_spans=chunks,
        verifier_learning_value_gate=build_verifier_learning_value_gate(None),
        max_capabilities_per_token=4,
    )

    visual = scores["visual_detail"]
    assert visual["student_deficit"][0] > 0
    assert visual["student_deficit"][1] == 0
    assert visual["teacher_attribution"][2] == 0
    assert visual["chunk_compatibility_prior"][3] == 0
    assert visual["chunk_compatibility"][3] > 0
    assert visual["final_token_weight"][3] > 0


def test_verifier_learning_value_gate_prioritizes_wrong_valid_rollouts():
    correct_gate = build_verifier_learning_value_gate(
        {"correct": True, "format_valid": True, "malformed": False, "reward": 1.0}
    )
    wrong_gate = build_verifier_learning_value_gate(
        {"correct": False, "format_valid": True, "malformed": False, "reward": 0.25}
    )
    malformed_gate = build_verifier_learning_value_gate(
        {"correct": None, "format_valid": False, "malformed": True, "reward": 0.0}
    )

    assert wrong_gate["chunk_gates"]["visible_evidence"] > correct_gate["chunk_gates"]["visible_evidence"]
    assert wrong_gate["chunk_gates"]["diagram_inference"] > correct_gate["chunk_gates"]["diagram_inference"]
    assert correct_gate["chunk_gates"]["answer"] > 0.0
    assert wrong_gate["chunk_gates"]["visible_evidence"] < malformed_gate["chunk_gates"]["visible_evidence"]
    assert wrong_gate["chunk_gates"]["diagram_inference"] < malformed_gate["chunk_gates"]["diagram_inference"]
    assert malformed_gate["chunk_gates"]["reasoning"] > correct_gate["chunk_gates"]["reasoning"]
    assert malformed_gate["chunk_gates"]["answer"] > correct_gate["chunk_gates"]["answer"]
