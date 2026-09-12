import math

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.online_batch import (
    OnlineFCOPDConfig,
    OnlineFCOPDSample,
    OnlineStudentScores,
    compute_online_fc_opd_batch,
)
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK


def _condition_inputs():
    return ConditionInputs(
        full_image=ImageInput(path="/tmp/full.png"),
        degraded_image=ImageInput(path="/tmp/degraded.png", transform={"type": "lowres_nearest", "scale": 0.1}),
        free_caption="A geometry diagram.",
        task_evidence="The diagram shows a labeled angle.",
        task_visible_evidence="A labeled angle is visible.",
        task_infer_evidence="The relevant angle relation can be inferred.",
        task_solve_evidence="Use the angle relation to solve.",
    )


def _response(answer="B"):
    return (
        "<visible_evidence>\nA labeled angle is visible.\n</visible_evidence>\n"
        "<diagram_inference>\nThe relation is complementary.\n</diagram_inference>\n"
        "<reasoning>\nThe missing angle is 60.\n</reasoning>\n"
        f"<answer>\n{answer}\n</answer>"
    )


def _sample(tokenizer, *, answer="B", metadata=None):
    text = _response(answer)
    return OnlineFCOPDSample(
        sample_uid="geometry3k:1",
        question="Find the angle.",
        condition_inputs=_condition_inputs(),
        rollout_token_ids=tuple(tokenizer.encode(text)),
        rollout_text=text,
        choices=("40", "60", "80", "100"),
        answer_metadata="A",
        metadata=metadata or {},
    )


def _teacher_topk(token_ids, probability):
    ids = torch.tensor(token_ids, dtype=torch.long)
    alt = (ids + 1).remainder(256)
    topk_ids = torch.stack((ids, alt), dim=-1).unsqueeze(0)
    probs = torch.tensor([probability, 1.0 - probability], dtype=torch.float32)
    log_probs = probs.log().reshape(1, 1, 2).expand(1, len(token_ids), 2).clone()
    return TeacherTopK(
        token_ids=topk_ids,
        log_probs=log_probs,
        tail_log_prob=torch.full((1, len(token_ids)), math.log(1e-6)),
    )


class RecordingTeacher:
    def __init__(self, probability_by_condition=None):
        self.calls = []
        self.probability_by_condition = probability_by_condition or {}

    def __call__(self, sample, conditions):
        self.calls.append(tuple(sample.rollout_token_ids))
        return {
            condition: _teacher_topk(
                sample.rollout_token_ids,
                self.probability_by_condition.get(Condition(condition), 0.7),
            )
            for condition in conditions
        }


class TrainableStudent:
    def __init__(self, seq_len, vocab_size=256):
        self.logits = torch.nn.Parameter(torch.zeros((1, seq_len, vocab_size), dtype=torch.float32))

    def __call__(self, sample, conditions):
        length = len(sample.rollout_token_ids)
        values = {}
        for condition in conditions:
            if Condition(condition) in {Condition.FULL, Condition.TASK_VISIBLE, Condition.TASK_INFER, Condition.TASK_SOLVE}:
                values[condition] = torch.full((length,), -0.6)
            else:
                values[condition] = torch.full((length,), -0.7)
        return OnlineStudentScores(loss_logits=self.logits, condition_log_probs=values)


class ScoresOnlyStudent(TrainableStudent):
    def __call__(self, sample, conditions):
        scores = super().__call__(sample, conditions)
        return OnlineStudentScores(loss_logits=None, condition_log_probs=scores.condition_log_probs)


def _wrong_valid(_sample):
    return {"correct": False, "format_valid": True, "malformed": False, "reward": 0.25}


def _correct(_sample):
    return {"correct": True, "format_valid": True, "malformed": False, "reward": 1.0}


def test_online_batch_uses_runtime_rollout_token_ids_and_no_offline_jsonl_required():
    tokenizer = ByteTokenizer()
    sample = _sample(tokenizer, answer="B")
    teacher = RecordingTeacher()
    student = TrainableStudent(len(sample.rollout_token_ids))

    output = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=student,
        verifier=_wrong_valid,
    )

    assert teacher.calls == [sample.rollout_token_ids]
    assert output.samples[0].response_token_ids.squeeze(0).tolist() == list(sample.rollout_token_ids)
    assert torch.isfinite(output.loss)
    assert output.samples[0].verifier_learning_value_gate["outcome_class"] == "wrong_but_format_valid"


def test_online_verifier_gate_favors_wrong_valid_over_correct_answer_chunk():
    tokenizer = ByteTokenizer()
    sample = _sample(tokenizer, answer="B")
    teacher = RecordingTeacher()

    wrong_student = TrainableStudent(len(sample.rollout_token_ids))
    wrong = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=wrong_student,
        verifier=_wrong_valid,
    ).samples[0]
    correct_student = TrainableStudent(len(sample.rollout_token_ids))
    correct = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=correct_student,
        verifier=_correct,
    ).samples[0]

    wrong_visible = wrong.verifier_learning_value_gate["chunk_gates"]["visible_evidence"]
    correct_visible = correct.verifier_learning_value_gate["chunk_gates"]["visible_evidence"]
    assert wrong_visible > correct_visible
    assert correct.verifier_learning_value_gate["chunk_gates"]["answer"] > 0.0
    answer_mask = correct.chunk_masks["answer"]
    wrong_answer_weight = sum(weight[answer_mask].sum() for weight in wrong.condition_weights.values())
    correct_answer_weight = sum(weight[answer_mask].sum() for weight in correct.condition_weights.values())
    assert correct_answer_weight.item() > 0.0
    assert correct_answer_weight.item() < wrong_answer_weight.item()


def test_online_grouped_loss_is_finite_and_backward_updates_current_student():
    tokenizer = ByteTokenizer()
    sample = _sample(tokenizer, answer="B")
    teacher = RecordingTeacher(
        {
            Condition.FULL: 0.85,
            Condition.DEGRADED: 0.20,
            Condition.FREE: 0.20,
            Condition.TASK_VISIBLE: 0.80,
            Condition.TASK_INFER: 0.82,
            Condition.TASK_SOLVE: 0.86,
        }
    )
    student = TrainableStudent(len(sample.rollout_token_ids))
    optimizer = torch.optim.SGD([student.logits], lr=0.1)

    before = student.logits.detach().clone()
    output = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=student,
        verifier=_wrong_valid,
    )
    assert torch.isfinite(output.loss)
    assert output.samples[0].grouped_loss_tensors
    assert all(torch.isfinite(value).all() for value in output.samples[0].grouped_loss_tensors.values())
    output.loss.backward()
    optimizer.step()
    assert not torch.allclose(before, student.logits.detach())


def test_online_loss_ignores_stale_offline_fields_when_allowed():
    tokenizer = ByteTokenizer()
    sample = _sample(
        tokenizer,
        answer="B",
        metadata={
            "teacher_condition_scores": {"full": "stale"},
            "offline_score_jsonl": "/tmp/stale.jsonl",
        },
    )
    teacher = RecordingTeacher({Condition.FULL: 0.9})
    student = TrainableStudent(len(sample.rollout_token_ids))

    output = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=student,
        verifier=_wrong_valid,
        config=OnlineFCOPDConfig(reject_stale_offline_fields=False),
    )

    assert teacher.calls == [sample.rollout_token_ids]
    assert torch.isfinite(output.loss)


def test_online_batch_can_skip_hook_loss_for_remote_student_scorer():
    tokenizer = ByteTokenizer()
    sample = _sample(tokenizer, answer="B")
    teacher = RecordingTeacher({Condition.FULL: 0.9})
    student = ScoresOnlyStudent(len(sample.rollout_token_ids))

    output = compute_online_fc_opd_batch(
        [sample],
        tokenizer=tokenizer,
        teacher_scorer=teacher,
        student_scorer=student,
        verifier=_wrong_valid,
        config=OnlineFCOPDConfig(compute_hook_loss=False),
    )

    assert output.loss.item() == pytest.approx(0.0)
    assert output.samples[0].condition_weights
    assert output.samples[0].grouped_loss_tensors


def test_online_batch_rejects_stale_offline_fields_by_default():
    tokenizer = ByteTokenizer()
    sample = _sample(tokenizer, metadata={"teacher_scores": {"full": "stale"}})
    teacher = RecordingTeacher()
    student = TrainableStudent(len(sample.rollout_token_ids))

    with pytest.raises(ValueError, match="stale offline fields"):
        compute_online_fc_opd_batch(
            [sample],
            tokenizer=tokenizer,
            teacher_scorer=teacher,
            student_scorer=student,
            verifier=_wrong_valid,
        )
