import math

import pytest

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.teacher_protocol import (
    TeacherMetadata,
    TeacherScoreResponse,
    ensure_exact_token_alignment,
    tokenizer_fingerprint,
    validate_score_response,
)
from dual_track_opd.fc_opd.teacher_transformers import (
    TransformersTeacherScorer,
    response_prediction_logits,
)


class TinyTokenizer:
    special_tokens_map = {"eos_token": "<eos>"}

    def __init__(self, reversed_vocab: bool = False):
        items = [("a", 0), ("b", 1), ("<eos>", 2)]
        self.vocab = dict(reversed(items) if reversed_vocab else items)

    def get_vocab(self):
        return self.vocab

    def get_added_vocab(self):
        return {"<eos>": 2}


def test_tokenizer_fingerprint_is_order_independent():
    assert tokenizer_fingerprint(TinyTokenizer()) == tokenizer_fingerprint(
        TinyTokenizer(reversed_vocab=True)
    )


def test_tokenizer_fingerprint_changes_with_vocab():
    tokenizer = TinyTokenizer()
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.vocab["c"] = 3
    assert tokenizer_fingerprint(tokenizer) != original


def test_validate_score_response_checks_shapes_and_values():
    metadata = TeacherMetadata(
        model_id="test",
        tokenizer_hash="hash",
        vocab_size=4,
        top_k=2,
        dtype="float32",
        git_revision="test",
    )
    response = TeacherScoreResponse(
        request_id="id",
        condition=Condition.FULL,
        token_ids=(1,),
        topk_token_ids=((1, 2),),
        topk_log_probs=((math.log(0.6), math.log(0.3)),),
        tail_log_prob=(math.log(0.1),),
        teacher_entropy=(1.0,),
        sampled_token_log_probs=(math.log(0.6),),
    )
    validate_score_response(response, metadata)


def test_validate_score_response_rejects_non_normalized_mass():
    metadata = TeacherMetadata(
        model_id="test",
        tokenizer_hash="hash",
        vocab_size=4,
        top_k=2,
        dtype="float32",
        git_revision="test",
    )
    response = TeacherScoreResponse(
        request_id="id",
        condition=Condition.FULL,
        token_ids=(1,),
        topk_token_ids=((1, 2),),
        topk_log_probs=((math.log(0.4), math.log(0.3)),),
        tail_log_prob=(math.log(0.1),),
        teacher_entropy=(1.0,),
        sampled_token_log_probs=(math.log(0.4),),
    )
    with pytest.raises(ValueError, match="sum to one"):
        validate_score_response(response, metadata)


def test_exact_token_alignment_is_a_hard_assertion():
    with pytest.raises(ValueError, match="token IDs differ"):
        ensure_exact_token_alignment([1, 2], [1, 3], context="test")


class EncodeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_transformers_backend_retokenization_gate_without_loading_a_model(caplog):
    scorer = object.__new__(TransformersTeacherScorer)
    scorer.tokenizer = EncodeTokenizer()
    from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
    from dual_track_opd.fc_opd.teacher_protocol import TeacherScoreRequest

    inputs = ConditionInputs(
        full_image=ImageInput("/tmp/full.png"),
        degraded_image=ImageInput(
            "/tmp/blur.png",
            {"type": "gaussian_blur", "sigma": 2.0},
        ),
        free_caption="caption",
        task_evidence="evidence",
    )
    request = TeacherScoreRequest(
        request_id="id",
        condition=Condition.FULL,
        question="question",
        condition_inputs=inputs,
        response_token_ids=(ord("A"),),
        tokenizer_hash="hash",
        response_text="B",
    )
    caplog.clear()
    with caplog.at_level("WARNING"):
        scorer._check_response_text(request)
    # The current student token IDs are the distillation target.  A teacher
    # tokenizer round-trip mismatch is logged, never silently substituted.
    assert tuple(request.response_token_ids) == (ord("A"),)
    assert "Teacher tokenizer round-trip mismatch" in caplog.text


def test_teacher_request_round_trips_exact_rollout_prompt():
    from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
    from dual_track_opd.fc_opd.teacher_protocol import TeacherScoreRequest

    request = TeacherScoreRequest(
        request_id="id",
        condition=Condition.FULL,
        question="question",
        condition_inputs=ConditionInputs(
            full_image=ImageInput("/tmp/full.png"),
            degraded_image=ImageInput("/tmp/blur.png", {"type": "gaussian_blur", "sigma": 2.0}),
            free_caption="caption",
            task_evidence="evidence",
        ),
        response_token_ids=(1, 2),
        tokenizer_hash="hash",
        prompt=({"role": "user", "content": "<image>\nExact question"},),
    )
    restored = TeacherScoreRequest.from_dict(request.to_dict())
    assert restored.prompt == request.prompt


def test_teacher_causal_slice_uses_positions_before_response_tokens():
    import torch

    full = torch.arange(1 * 8 * 3).reshape(1, 8, 3)
    # Three response tokens occupy input positions 5, 6, 7.  Their predictors
    # are logits positions 4, 5, 6; position 7 predicts a token after response.
    actual = response_prediction_logits(full, num_response_tokens=3)
    assert torch.equal(actual, full[:, 4:7, :])


def test_transformers_teacher_uses_exact_rollout_prompt_for_full_condition(tmp_path):
    import torch
    from PIL import Image
    from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
    from dual_track_opd.fc_opd.teacher_protocol import TeacherScoreRequest

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)

    class FakeProcessor:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered"

        def __call__(self, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    scorer = object.__new__(TransformersTeacherScorer)
    scorer.processor = FakeProcessor()
    scorer.device = torch.device("cpu")
    prompt = ({"role": "user", "content": "<image>\nExact rollout wording"},)
    request = TeacherScoreRequest(
        request_id="id",
        condition=Condition.FULL,
        question="a differently reconstructed question",
        condition_inputs=ConditionInputs(
            full_image=ImageInput(str(image_path)),
            degraded_image=ImageInput(str(image_path), {"type": "gaussian_blur", "sigma": 2.0}),
            free_caption="caption",
            task_evidence="evidence",
        ),
        response_token_ids=(1,),
        tokenizer_hash="hash",
        prompt=prompt,
    )

    scorer._prepare_prompt(request)
    assert scorer.processor.messages == list(prompt)
