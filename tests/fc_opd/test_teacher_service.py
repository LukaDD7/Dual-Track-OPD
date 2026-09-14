from dataclasses import replace

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.teacher_client import (
    TeacherClient,
    TeacherServiceError,
    score_teacher_conditions,
)
from dual_track_opd.fc_opd.teacher_protocol import TeacherScoreRequest
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server


def _inputs() -> ConditionInputs:
    return ConditionInputs(
        full_image=ImageInput("/tmp/full.png"),
        degraded_image=ImageInput(
            "/tmp/blur.png",
            {"type": "gaussian_blur", "sigma": 2.0},
        ),
        free_caption="A simple image.",
        task_evidence="One object is visible.",
    )


def _request(
    condition: Condition,
    tokenizer_hash: str,
    request_id: str = "request",
) -> TeacherScoreRequest:
    return TeacherScoreRequest(
        request_id=request_id,
        condition=condition,
        question="How many objects?",
        condition_inputs=_inputs(),
        response_token_ids=(3, 5, 7),
        tokenizer_hash=tokenizer_hash,
        response_text="<answer>1</answer>",
    )


def _url(server) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}"


def test_health_metadata_and_deterministic_scoring():
    scorer = SyntheticTeacherScorer(vocab_size=32, top_k=4, tokenizer_hash="same")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server), expected_tokenizer_hash="same")
        assert client.health()
        request = _request(Condition.FULL, "same")
        first = client.score([request])[0]
        second = client.score([request])[0]

    assert first == second
    assert first.token_ids == request.response_token_ids
    assert len(first.topk_token_ids) == len(request.response_token_ids)
    assert all(len(row) == 4 for row in first.topk_token_ids)


def test_teacher_diagnostic_endpoint_round_trips_backend_result():
    class DiagnosticSyntheticTeacher(SyntheticTeacherScorer):
        def diagnose_generation_alignment(self, request, *, max_new_tokens=4):
            return {
                "request_id": request.request_id,
                "max_new_tokens": max_new_tokens,
                "native_forced_topk_ids_match": True,
            }

    scorer = DiagnosticSyntheticTeacher(vocab_size=32, top_k=4, tokenizer_hash="same")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server), expected_tokenizer_hash="same")
        result = client.diagnose_generation_alignment(_request(Condition.FULL, "same"), max_new_tokens=3)

    assert result["request_id"] == "request"
    assert result["max_new_tokens"] == 3
    assert result["native_forced_topk_ids_match"] is True


def test_four_conditions_return_nonidentical_scores():
    scorer = SyntheticTeacherScorer(vocab_size=32, top_k=4, tokenizer_hash="same")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server), expected_tokenizer_hash="same")
        responses = client.score(
            [
                _request(condition, "same", request_id=condition.value)
                for condition in (
                    Condition.FULL,
                    Condition.BLUR,
                    Condition.FREE,
                    Condition.TASK,
                )
            ]
        )
    signatures = {response.topk_log_probs for response in responses}
    assert len(signatures) == 4


def test_client_rejects_tokenizer_mismatch_before_scoring():
    scorer = SyntheticTeacherScorer(tokenizer_hash="teacher")
    with running_teacher_server(scorer) as server:
        with pytest.raises(ValueError, match="hashes do not match"):
            TeacherClient(_url(server), expected_tokenizer_hash="student")


def test_server_rejects_request_tokenizer_mismatch():
    scorer = SyntheticTeacherScorer(tokenizer_hash="teacher")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server))
        request = _request(Condition.FULL, "wrong")
        with pytest.raises(ValueError, match="does not match"):
            client.score([request])


def test_score_teacher_conditions_returns_tensors_for_each_condition():
    scorer = SyntheticTeacherScorer(vocab_size=32, top_k=4, tokenizer_hash="same")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server), expected_tokenizer_hash="same")
        scores = score_teacher_conditions(
            (3, 5, 7),
            "How many objects?",
            _inputs(),
            (Condition.FULL, Condition.BLUR, Condition.FREE, Condition.TASK),
            client,
        )

    assert set(scores) == {
        Condition.FULL,
        Condition.BLUR,
        Condition.FREE,
        Condition.TASK,
    }
    for score in scores.values():
        score.validate()
        assert score.token_ids.shape == (1, 3, 4)
        assert score.log_probs.shape == (1, 3, 4)
        assert score.tail_log_prob.shape == (1, 3)
        assert score.entropy.shape == (1, 3)
        assert torch.isfinite(score.log_probs).all()


def test_service_error_is_explicit_and_never_falls_back():
    scorer = SyntheticTeacherScorer(vocab_size=8, top_k=4, tokenizer_hash="same")
    bad_request = replace(
        _request(Condition.TASK, "same"),
        response_token_ids=(999,),
    )
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server))
        with pytest.raises(TeacherServiceError, match="outside the synthetic vocabulary"):
            client.score([bad_request])


def test_client_rejects_teacher_response_with_changed_token_id(monkeypatch):
    scorer = SyntheticTeacherScorer(vocab_size=32, top_k=4, tokenizer_hash="same")
    with running_teacher_server(scorer) as server:
        client = TeacherClient(_url(server), expected_tokenizer_hash="same")
        original_request = client._request_json

        def tampered_request(method, path, payload=None):
            result = original_request(method, path, payload)
            if path == "/score":
                result["responses"][0]["token_ids"][1] = 6
            return result

        monkeypatch.setattr(client, "_request_json", tampered_request)
        with pytest.raises(TeacherServiceError, match="exact student request"):
            client.score([_request(Condition.FULL, "same")])
