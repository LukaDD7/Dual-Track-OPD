"""Regression tests for multi-prompt and mixed-length online student scoring."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.online_batch import OnlineFCOPDSample
from dual_track_opd.fc_opd.student_scorer import StudentScorer
from dual_track_opd.fc_opd.teacher_prompts import RenderedTeacherPrompt


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 15


class _PromptProcessor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return str(messages[0]["content"][-1]["text"])

    def __call__(self, *, text, images, padding=False, return_tensors):
        del images, padding, return_tensors
        rows = []
        for prompt in text:
            rows.append([2, 3] if "prompt-A" in prompt else [7, 3])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
            "position_ids": torch.arange(2).repeat(len(rows), 1),
        }


class _PromptPositionModel:
    device = torch.device("cpu")

    def __call__(self, **kwargs):
        assert "position_ids" not in kwargs
        input_ids = kwargs["input_ids"]
        batch, width = input_ids.shape
        logits = torch.zeros((batch, width, 16), dtype=torch.float32)
        for row in range(batch):
            prompt_id = int(input_ids[row, 0])
            for position in range(width):
                logits[row, position, (prompt_id + position) % 16] = 6.0
        return SimpleNamespace(logits=logits)


def _sample(uid: str, question: str, response_ids: tuple[int, ...]) -> OnlineFCOPDSample:
    inputs = ConditionInputs(
        full_image=ImageInput("/unused.png"),
        degraded_image=ImageInput("/unused.png", {"type": "blur"}),
        free_caption="caption",
        task_evidence="evidence",
    )
    return OnlineFCOPDSample(
        sample_uid=uid,
        question=question,
        condition_inputs=inputs,
        rollout_token_ids=response_ids,
        rollout_text="display-only",
    )


def test_batched_scoring_groups_prompts_and_matches_serial(monkeypatch):
    import dual_track_opd.fc_opd.student_scorer as scorer_module

    def fake_render(condition, question, condition_inputs):
        del condition_inputs
        return RenderedTeacherPrompt(
            condition=Condition(condition),
            messages=({
                "role": "user",
                "content": [{"type": "text", "text": f"prompt-{question}"}],
            },),
            image_paths=(),
        )

    monkeypatch.setattr(scorer_module, "render_teacher_prompt", fake_render)
    scorer = object.__new__(StudentScorer)
    scorer._processor = _PromptProcessor()
    scorer._tokenizer = scorer._processor.tokenizer
    scorer._model = _PromptPositionModel()
    scorer._image_cls = None

    samples = [
        _sample("a-short", "A", (4,)),
        _sample("b-long", "B", (8, 9, 10)),
        _sample("a-medium", "A", (5, 6)),
    ]
    batched = scorer._score_batched(samples, [Condition.FULL])
    serial = [scorer._score_one(sample, [Condition.FULL]) for sample in samples]

    assert len(batched) == len(samples)
    for batch_result, serial_result, sample in zip(batched, serial, samples, strict=True):
        assert batch_result.loss_logits.shape[0] == len(sample.rollout_token_ids)
        assert torch.allclose(
            batch_result.loss_logits,
            serial_result.loss_logits.squeeze(0),
        )
        assert torch.allclose(
            batch_result.condition_log_probs[Condition.FULL],
            serial_result.condition_log_probs[Condition.FULL].squeeze(0),
        )

    # The two prompt groups must produce different position-sensitive scores.
    assert not torch.allclose(
        batched[0].condition_log_probs[Condition.FULL],
        batched[1].condition_log_probs[Condition.FULL][:1],
    )

    auxiliary_only = scorer._score_batched(samples[:2], [Condition.FREE])
    assert all(result.loss_logits is not None for result in auxiliary_only)
    assert all(set(result.condition_log_probs) == {Condition.FREE} for result in auxiliary_only)
