import math
from types import SimpleNamespace

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.online_batch import OnlineStudentScores
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK
from dual_track_opd.fc_opd.verl_post_rollout_hook import fc_opd_post_rollout_hook


def _condition_inputs():
    return ConditionInputs(
        full_image=ImageInput(path="/tmp/full.png"),
        degraded_image=ImageInput(path="/tmp/degraded.png", transform={"type": "lowres_nearest", "scale": 0.5}),
        free_caption="A geometry diagram.",
        task_evidence="A labeled angle is visible.",
        task_visible_evidence="A labeled angle is visible.",
        task_infer_evidence="The relation can be inferred.",
        task_solve_evidence="Solve using the relation.",
    )


def _response():
    return (
        "<visible_evidence>\nA labeled angle is visible.\n</visible_evidence>\n"
        "<diagram_inference>\nThe angles are complementary.\n</diagram_inference>\n"
        "<reasoning>\nThe answer is B.\n</reasoning>\n"
        "<answer>\nB\n</answer>"
    )


def _teacher_topk(token_ids, probability):
    ids = torch.tensor(token_ids, dtype=torch.long)
    alt = (ids + 1).remainder(256)
    topk_ids = torch.stack((ids, alt), dim=-1).unsqueeze(0)
    log_probs = torch.log(torch.tensor([probability, 1.0 - probability], dtype=torch.float32))
    return TeacherTopK(
        token_ids=topk_ids,
        log_probs=log_probs.reshape(1, 1, 2).expand(1, len(token_ids), 2).clone(),
        tail_log_prob=torch.full((1, len(token_ids)), math.log(1e-6)),
    )


class RecordingTeacher:
    def __init__(self):
        self.calls = []

    def __call__(self, sample, conditions):
        self.calls.append(tuple(sample.rollout_token_ids))
        return {Condition(condition): _teacher_topk(sample.rollout_token_ids, 0.8) for condition in conditions}


class RecordingStudent:
    def __init__(self):
        self.calls = []

    def __call__(self, sample, conditions):
        self.calls.append(tuple(sample.rollout_token_ids))
        length = len(sample.rollout_token_ids)
        logits = torch.zeros((1, length, 256), dtype=torch.float32)
        condition_log_probs = {Condition(condition): torch.full((length,), -0.5) for condition in conditions}
        return OnlineStudentScores(loss_logits=logits, condition_log_probs=condition_log_probs)


def _wrong_valid(_sample):
    return {"correct": False, "format_valid": True, "malformed": False, "reward": 0.0}


def test_post_rollout_hook_attaches_verl_tensors_and_masks_padding():
    tokenizer = ByteTokenizer()
    valid_ids = tuple(tokenizer.encode(_response()))
    pad = (0, 0, 0)
    responses = torch.tensor([valid_ids + pad], dtype=torch.long)
    response_mask = torch.tensor([[1] * len(valid_ids) + [0] * len(pad)], dtype=torch.bool)
    teacher = RecordingTeacher()
    student = RecordingStudent()
    batch = SimpleNamespace(
        batch={"responses": responses, "response_mask": response_mask},
        non_tensor_batch={
            "uid": ["sample-1"],
            "question": ["Find the angle."],
            "condition_inputs": [_condition_inputs().to_dict()],
            "choices": [["A", "B", "C"]],
            "answer": ["B"],
        },
    )

    updated, metrics = fc_opd_post_rollout_hook(
        batch=batch,
        tokenizer=tokenizer,
        processor=None,
        config={
            "algorithm": {
                "fc_opd": {
                    "teacher_scorer": teacher,
                    "student_scorer": student,
                    "verifier": _wrong_valid,
                }
            }
        },
        global_steps=7,
    )

    assert updated is batch
    assert teacher.calls == [valid_ids]
    assert student.calls == [valid_ids]
    assert batch.batch["fc_teacher_topk_indices"].shape == (1, 6, len(valid_ids) + len(pad), 2)
    assert batch.batch["fc_teacher_topk_log_probs"].shape == (1, 6, len(valid_ids) + len(pad), 2)
    assert batch.batch["fc_condition_weights"].shape == (1, 6, len(valid_ids) + len(pad))
    assert batch.batch["fc_condition_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert batch.batch["fc_condition_weights"][:, :, -len(pad) :].sum().item() == 0.0
    assert metrics["fc_opd/hook_num_samples"] == 1.0
    assert metrics["fc_opd/hook_active_weight"] > 0.0


def test_post_rollout_hook_requires_student_scorer():
    """The verl hook must fail fast instead of pretending CPU fallback is a real smoke."""
    tokenizer = ByteTokenizer()
    valid_ids = tuple(tokenizer.encode(_response()))
    batch = SimpleNamespace(
        batch={
            "responses": torch.tensor([valid_ids], dtype=torch.long),
            "response_mask": torch.ones((1, len(valid_ids)), dtype=torch.bool),
        },
        non_tensor_batch={
            "question": ["Find the angle."],
            "condition_inputs": [_condition_inputs().to_dict()],
        },
    )

    with pytest.raises(ValueError, match="student_scorer_fqn is required"):
        fc_opd_post_rollout_hook(
            batch=batch,
            tokenizer=tokenizer,
            processor=None,
            config={"algorithm": {"fc_opd": {"teacher_scorer": RecordingTeacher()}}},
            global_steps=1,
        )
