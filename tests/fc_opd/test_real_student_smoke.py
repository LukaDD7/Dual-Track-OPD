from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.offline_loss import (
    FOUR_CONDITIONS,
    offline_record_to_tensors,
)
from dual_track_opd.fc_opd.offline_scoring import (
    ByteTokenizer,
    OfflineScoringConfig,
    iter_offline_scores,
    make_smoke_dataset,
)
from dual_track_opd.fc_opd.real_student_smoke import (
    FOUR_CLEAN_CONDITIONS,
    FOUR_CLEAN_CONDITION_ROUTER,
    SIX_C_CHUNK_GATED_ROUTER,
    SIX_C_SOLVE_CONDITIONS,
    StudentForwardOutput,
    TWO_CONDITIONS,
    TWO_CONDITION_ROUTER,
    detect_tied_parameter_groups,
    lm_head_embed_tied,
    response_logit_slice,
    run_real_student_min_train,
    run_real_student_record,
    run_real_student_smoke,
    teacher_scores_to_device,
)
import dual_track_opd.fc_opd.real_student_smoke as real_student_smoke
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server

TOP_K = 32
FAKE_PROMPT_LENGTH = 4


def _build_offline_payloads(num_samples: int = 2) -> list[dict]:
    tokenizer = ByteTokenizer()
    scorer = SyntheticTeacherScorer(
        vocab_size=320, top_k=TOP_K, tokenizer_hash=tokenizer_fingerprint(tokenizer)
    )
    with ExitStack() as stack:
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}", expected_tokenizer_hash=tokenizer_fingerprint(tokenizer)
        )
        config = OfflineScoringConfig(source_dataset="vstar", conditions=FOUR_CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(num_samples),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )
    return [record.payload for record in scored]


def _build_2c_offline_payloads(num_samples: int = 2) -> list[dict]:
    tokenizer = ByteTokenizer()
    scorer = SyntheticTeacherScorer(
        vocab_size=320, top_k=TOP_K, tokenizer_hash=tokenizer_fingerprint(tokenizer)
    )
    with ExitStack() as stack:
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}", expected_tokenizer_hash=tokenizer_fingerprint(tokenizer)
        )
        config = OfflineScoringConfig(source_dataset="vstar", conditions=TWO_CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(num_samples),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )
    return [record.payload for record in scored]


def _build_4c_clean_offline_payloads(num_samples: int = 2) -> list[dict]:
    tokenizer = ByteTokenizer()
    scorer = SyntheticTeacherScorer(
        vocab_size=320, top_k=TOP_K, tokenizer_hash=tokenizer_fingerprint(tokenizer)
    )
    with ExitStack() as stack:
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}", expected_tokenizer_hash=tokenizer_fingerprint(tokenizer)
        )
        config = OfflineScoringConfig(source_dataset="geometry3k", conditions=FOUR_CLEAN_CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(num_samples, dataset_name="geometry3k"),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )
    return [record.payload for record in scored]


def _manual_payload(
    conditions,
    *,
    format_valid: bool = True,
    legacy_visual: bool = False,
) -> dict:
    tokenizer = ByteTokenizer()
    response_text = "abcde"
    response_ids = tokenizer.encode(response_text)
    topk = [[token_id, (token_id + 1) % 256] for token_id in response_ids]
    log_probs = [[-0.4, -1.1] for _ in response_ids]
    block = {
        "token_ids": topk,
        "log_probs": log_probs,
        "tail_log_prob": [-4.0 for _ in response_ids],
        "entropy": [0.5 for _ in response_ids],
        "top_k": 2,
    }
    chunks = {
        "visible_evidence": [[0, 1]],
        "diagram_inference": [[1, 2]],
        "reasoning": [[2, 4]],
        "answer": [[4, 5]],
        "format_valid": format_valid,
        "errors": [] if format_valid else ["reasoning:close_tag_count=0", "answer:open_tag_count=0"],
    }
    if legacy_visual:
        chunks["visual_evidence"] = chunks.pop("visible_evidence")
    if not format_valid:
        chunks["visible_evidence"] = []
        chunks["diagram_inference"] = []
        chunks["reasoning"] = []
        chunks["answer"] = []
    return {
        "sample_uid": "manual-1",
        "question": "Question?",
        "response_text": response_text,
        "response_token_ids": response_ids,
        "tokenizer_hash": tokenizer_fingerprint(tokenizer),
        "condition_scores": {condition.value: dict(block) for condition in conditions},
        "chunk_spans": chunks,
    }


class FakeStudentModel(nn.Module):
    """Tiny non-causal LM whose logits depend on real trainable parameters."""

    def __init__(self, vocab_size: int, hidden: int = 8, *, tie_weights: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    def full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


class FakeStudentProvider:
    """CPU provider that mimics teacher-forced forward with a fake model."""

    def __init__(
        self,
        vocab_size: int,
        tokenizer: ByteTokenizer,
        *,
        override_hash: str | None = None,
        tie_weights: bool = False,
    ):
        self.model = FakeStudentModel(vocab_size, tie_weights=tie_weights)
        self.tokenizer = tokenizer
        self._hash = override_hash or tokenizer_fingerprint(tokenizer)
        self.vocab_size = vocab_size

    @property
    def tokenizer_hash(self) -> str:
        return self._hash

    def named_parameters(self):
        yield from self.model.named_parameters()

    def all_named_parameters(self):
        yield from self.model.named_parameters(remove_duplicate=False)

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def forward(self, record):
        response_ids = tuple(int(item) for item in record["response_token_ids"])
        response_text = str(record["response_text"])
        prompt_ids = torch.zeros((1, FAKE_PROMPT_LENGTH), dtype=torch.long)
        response_tensor = torch.tensor([response_ids], dtype=torch.long)
        full_ids = torch.cat([prompt_ids, response_tensor], dim=1)
        full_logits = self.model.full_logits(full_ids)
        response_logits = response_logit_slice(full_logits, FAKE_PROMPT_LENGTH, len(response_ids))
        decoded = self.tokenizer.decode(list(response_ids))
        reencoded = tuple(int(i) for i in self.tokenizer.encode(response_text))
        return StudentForwardOutput(
            response_logits=response_logits,
            prompt_length=FAKE_PROMPT_LENGTH,
            full_length=int(full_ids.shape[1]),
            used_token_ids=response_ids,
            reencoded_token_ids=reencoded,
            decoded_text=decoded,
            decoded_matches=(decoded == response_text),
        )


@pytest.fixture(scope="module")
def offline_payloads() -> list[dict]:
    return _build_offline_payloads(2)


def _provider_for(payloads, **kwargs) -> FakeStudentProvider:
    vocab = max(offline_record_to_tensors(p).vocab_floor for p in payloads)
    return FakeStudentProvider(vocab, ByteTokenizer(), **kwargs)


def test_response_logit_slice_picks_next_token_positions():
    seq_len, vocab = 10, 3
    full = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1).expand(1, seq_len, vocab)
    sliced = response_logit_slice(full, prompt_length=4, num_response_tokens=5)
    assert sliced.shape == (1, 5, vocab)
    # Positions 3..7 predict response tokens at absolute positions 4..8.
    assert sliced[0, :, 0].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_geometry3k_wrapper_passes_routing_mode_to_python():
    script = Path("scripts/hpc/run_fc_opd_geometry3k_4c_real_student_min_train_smoke.sh").read_text(
        encoding="utf-8"
    )

    assert '--routing-mode "${FC_OPD_ROUTING_MODE:-uniform_all_conditions}"' in script
    assert '--condition-set "${FC_OPD_CONDITION_SET:-4c-clean}"' in script


def test_response_logit_slice_rejects_out_of_range():
    full = torch.zeros((1, 5, 2))
    with pytest.raises(ValueError, match="exceeds sequence length"):
        response_logit_slice(full, prompt_length=3, num_response_tokens=5)


def test_real_student_record_backward_reaches_fake_parameter(offline_payloads):
    provider = _provider_for(offline_payloads)
    result = run_real_student_record(offline_payloads[0], provider)
    assert result.passed
    assert result.tokenizer_hash_matches
    assert result.seq_len == result.num_response_tokens
    assert result.logits_shape[0] == 1 and result.logits_shape[1] == result.seq_len
    assert result.loss_is_finite
    assert result.backward_succeeded
    assert result.grad_param_name is not None
    assert result.grad_param_norm > 0.0
    assert result.grad_is_finite
    assert result.consumed_conditions == set(FOUR_CONDITIONS)


def test_named_parameter_actually_receives_grad(offline_payloads):
    provider = _provider_for(offline_payloads)
    run_real_student_record(offline_payloads[0], provider)
    grads = {name: p.grad for name, p in provider.named_parameters()}
    assert grads["lm_head.weight"] is not None
    assert torch.isfinite(grads["lm_head.weight"]).all()
    assert grads["lm_head.weight"].float().norm() > 0.0


def test_tokenizer_hash_mismatch_raises(offline_payloads):
    provider = _provider_for(offline_payloads, override_hash="not-the-offline-hash")
    with pytest.raises(ValueError, match="tokenizer hash does not match"):
        run_real_student_record(offline_payloads[0], provider)


def test_response_length_mismatch_raises(offline_payloads):
    provider = _provider_for(offline_payloads)
    payload = dict(offline_payloads[0])
    payload["response_token_ids"] = list(payload["response_token_ids"]) + [0]
    with pytest.raises(ValueError):
        run_real_student_record(payload, provider)


def test_decoded_mismatch_is_rejected_but_can_be_relaxed(offline_payloads):
    provider = _provider_for(offline_payloads)
    payload = dict(offline_payloads[0])
    payload["response_text"] = payload["response_text"] + " (edited)"

    with pytest.raises(ValueError, match="decoded response text does not match"):
        run_real_student_record(payload, provider)

    relaxed = run_real_student_record(payload, provider, require_decoded_match=False)
    assert relaxed.loss_is_finite
    assert relaxed.backward_succeeded
    assert not relaxed.decoded_matches


def test_smoke_report_passes_over_all_records(offline_payloads):
    provider = _provider_for(offline_payloads)
    report = run_real_student_smoke(offline_payloads, provider)
    assert report.num_records == len(offline_payloads)
    assert report.passed
    assert all(r.four_conditions_consumed for r in report.results)


def _cpu_teacher_scores() -> dict:
    return {
        Condition.FULL: TeacherTopK(
            token_ids=torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64),
            log_probs=torch.log(torch.tensor([[[0.6, 0.4], [0.7, 0.3]]])),
            tail_log_prob=torch.log(torch.tensor([[0.01, 0.02]])),
            entropy=torch.tensor([[0.5, 0.4]]),
        ),
        Condition.BLUR: TeacherTopK(
            token_ids=torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int64),
            log_probs=torch.log(torch.tensor([[[0.5, 0.5], [0.6, 0.4]]])),
            tail_log_prob=None,
            entropy=None,
        ),
    }


def test_teacher_scores_to_device_preserves_type_and_moves_all_fields():
    scores = _cpu_teacher_scores()
    target = torch.device("cpu")
    moved = teacher_scores_to_device(scores, target)

    assert set(moved) == set(scores)
    for condition, topk in moved.items():
        assert isinstance(topk, TeacherTopK)  # dataclasses.replace preserves the type
        topk.validate()
        assert topk.token_ids.device == target
        assert topk.log_probs.device == target
        if topk.tail_log_prob is not None:
            assert topk.tail_log_prob.device == target
        if topk.entropy is not None:
            assert topk.entropy.device == target
    # The optional fields stay None when absent rather than being fabricated.
    assert moved[Condition.BLUR].tail_log_prob is None
    assert moved[Condition.BLUR].entropy is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_teacher_scores_to_device_moves_cpu_scores_to_cuda():
    scores = _cpu_teacher_scores()
    moved = teacher_scores_to_device(scores, "cuda")
    for topk in moved.values():
        assert topk.token_ids.is_cuda
        assert topk.log_probs.is_cuda
        if topk.tail_log_prob is not None:
            assert topk.tail_log_prob.is_cuda
        if topk.entropy is not None:
            assert topk.entropy.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_student_record_handles_cuda_logits_with_cpu_teacher_scores(offline_payloads):
    # Regression: CPU teacher tensors + CUDA student logits must not raise a
    # cross-device error inside compute_fc_opd_loss.
    payload = offline_payloads[0]
    vocab = offline_record_to_tensors(payload).vocab_floor
    provider = FakeStudentProvider(vocab, ByteTokenizer())

    class CudaProvider:
        # The fake model and its parameters stay on CPU; only the response
        # logits are moved to CUDA, reproducing the real cuda-logits /
        # cpu-teacher-scores device split that triggered the bug.
        tokenizer_hash = provider.tokenizer_hash

        def named_parameters(self):
            return provider.named_parameters()

        def zero_grad(self):
            provider.zero_grad()

        def forward(self, record):
            out = provider.forward(record)
            out.response_logits = out.response_logits.to("cuda")
            return out

    result = run_real_student_record(payload, CudaProvider())
    assert result.passed


def test_student_vocab_floor_is_checked(offline_payloads):
    # A provider whose model vocab is too small for the teacher token ids.
    tensors = offline_record_to_tensors(offline_payloads[0])
    provider = FakeStudentProvider(tensors.vocab_floor - 1, ByteTokenizer())
    with pytest.raises((ValueError, IndexError)):
        run_real_student_record(offline_payloads[0], provider)


# --- parameter / tied-weight reporting -------------------------------------


def test_result_reports_trainable_params_and_nonzero_grads(offline_payloads):
    provider = _provider_for(offline_payloads)
    result = run_real_student_record(offline_payloads[0], provider)
    assert result.num_trainable_params > 0
    assert result.num_trainable_param_tensors >= 1
    assert "lm_head.weight" in result.trainable_param_names
    assert result.nonzero_grad_param_names  # at least one param has a real grad


def test_tied_weights_are_detected_and_reported(offline_payloads):
    provider = _provider_for(offline_payloads, tie_weights=True)
    result = run_real_student_record(offline_payloads[0], provider)
    assert result.lm_head_embed_tied is True
    assert result.tied_parameter_names
    tied = result.tied_parameter_names[0]
    assert any("lm_head" in name for name in tied)
    assert any("embed" in name for name in tied)


def test_untied_weights_report_no_tie(offline_payloads):
    provider = _provider_for(offline_payloads, tie_weights=False)
    result = run_real_student_record(offline_payloads[0], provider)
    assert result.lm_head_embed_tied is False
    assert result.tied_parameter_names == []


def test_detect_tied_helpers_directly(offline_payloads):
    provider = _provider_for(offline_payloads, tie_weights=True)
    groups = detect_tied_parameter_groups(provider)
    assert groups
    assert lm_head_embed_tied(groups) is True
    assert lm_head_embed_tied([["a.weight", "b.weight"]]) is False


# --- real optimizer-step (min-train) smoke ---------------------------------


def test_min_train_changes_trainable_parameters(offline_payloads):
    provider = _provider_for(offline_payloads)
    report = run_real_student_min_train(offline_payloads, provider, num_steps=3, learning_rate=0.1)
    assert report.passed
    assert report.num_steps == 3
    assert report.tokenizer_hash_matches
    assert report.decoded_matches
    for step in report.steps:
        assert step.loss_is_finite
        assert step.grad_is_finite
        assert step.grad_norm > 0.0
        assert step.param_delta_norm > 0.0
        assert step.consumed_conditions == set(FOUR_CONDITIONS)
    assert report.every_step_updates_params
    assert report.four_conditions_consumed


def test_min_train_supports_explicit_2c_router():
    payloads = _build_2c_offline_payloads(2)
    provider = _provider_for(payloads)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=TWO_CONDITION_ROUTER,
        expected_conditions=TWO_CONDITIONS,
        num_steps=2,
        learning_rate=0.1,
    )

    assert report.passed
    assert report.expected_conditions_consumed
    assert report.consumed_conditions == set(TWO_CONDITIONS)
    assert not report.four_conditions_consumed


def test_min_train_supports_clean_4c_router():
    payloads = _build_4c_clean_offline_payloads(2)
    provider = _provider_for(payloads)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=FOUR_CLEAN_CONDITION_ROUTER,
        expected_conditions=FOUR_CLEAN_CONDITIONS,
        num_steps=2,
        learning_rate=0.1,
    )

    assert report.passed
    assert report.expected_conditions_consumed
    assert report.consumed_conditions == set(FOUR_CLEAN_CONDITIONS)
    assert not report.four_conditions_consumed


def test_uniform_all_conditions_does_not_call_router(monkeypatch):
    payloads = [_manual_payload(SIX_C_SOLVE_CONDITIONS)]
    provider = _provider_for(payloads)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("route_condition_weights should not be called")

    monkeypatch.setattr(real_student_smoke, "route_condition_weights", fail_if_called)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=SIX_C_CHUNK_GATED_ROUTER,
        expected_conditions=SIX_C_SOLVE_CONDITIONS,
        routing_mode="uniform_all_conditions",
        num_steps=1,
        learning_rate=0.1,
    )

    assert report.passed
    assert report.consumed_conditions == set(SIX_C_SOLVE_CONDITIONS)


def test_min_train_6c_chunk_gated_routes_visible_evidence():
    payloads = [_manual_payload(SIX_C_SOLVE_CONDITIONS)]
    provider = _provider_for(payloads)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=SIX_C_CHUNK_GATED_ROUTER,
        expected_conditions=(
            Condition.TASK_VISIBLE,
            Condition.FULL,
            Condition.TASK_INFER,
            Condition.TASK_SOLVE,
        ),
        routing_mode="chunk_gated",
        num_steps=1,
        learning_rate=0.1,
    )

    assert report.passed
    assert report.chunk_gated_valid_rows == 1
    assert report.fallback_bad_chunk_rows == 0
    assert Condition.TASK_VISIBLE in report.consumed_conditions
    assert Condition.FREE in report.unused_conditions
    assert Condition.DEGRADED in report.unused_conditions
    assert report.condition_token_weight_sums["task_visible"] > 0
    assert report.chunk_condition_weight_sums["visible_evidence"]["task_visible"] > 0


def test_min_train_6c_contrastive_consumes_free_and_degraded():
    payloads = [_manual_payload(SIX_C_SOLVE_CONDITIONS)]
    provider = _provider_for(payloads)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=SIX_C_CHUNK_GATED_ROUTER,
        expected_conditions=SIX_C_SOLVE_CONDITIONS,
        routing_mode="chunk_gated_contrastive",
        num_steps=1,
        learning_rate=0.1,
    )

    assert report.passed
    assert Condition.FREE in report.consumed_conditions
    assert Condition.DEGRADED in report.consumed_conditions
    assert Condition.FREE not in report.unused_conditions
    assert Condition.DEGRADED not in report.unused_conditions
    assert report.condition_token_weight_sums["free"] > 0
    assert report.condition_token_weight_sums["degraded"] > 0
    assert report.condition_token_weight_nonzero_counts["free"] > 0
    assert report.chunk_condition_weight_sums["visible_evidence"]["free"] > 0
    assert report.chunk_condition_weight_sums["visible_evidence"]["degraded"] > 0


def test_min_train_chunk_gated_falls_back_for_malformed_chunks():
    payloads = [_manual_payload(SIX_C_SOLVE_CONDITIONS, format_valid=False)]
    provider = _provider_for(payloads)
    report = run_real_student_min_train(
        payloads,
        provider,
        router_config=SIX_C_CHUNK_GATED_ROUTER,
        expected_conditions=SIX_C_SOLVE_CONDITIONS,
        routing_mode="chunk_gated",
        num_steps=1,
        learning_rate=0.1,
    )

    assert report.passed
    assert report.fallback_bad_chunk_rows == 1
    assert report.chunk_gated_valid_rows == 0
    assert report.consumed_conditions == set(SIX_C_SOLVE_CONDITIONS)


def test_min_train_actually_moves_a_parameter(offline_payloads):
    provider = _provider_for(offline_payloads)
    before = next(p for n, p in provider.named_parameters() if n == "lm_head.weight").detach().clone()
    run_real_student_min_train(offline_payloads, provider, num_steps=2, learning_rate=0.1)
    after = next(p for n, p in provider.named_parameters() if n == "lm_head.weight").detach()
    assert not torch.allclose(before, after)


def test_min_train_reports_tied_status(offline_payloads):
    provider = _provider_for(offline_payloads, tie_weights=True)
    report = run_real_student_min_train(offline_payloads, provider, num_steps=1, learning_rate=0.1)
    assert report.lm_head_embed_tied is True
    assert report.tied_parameter_names
    assert report.nonzero_grad_param_names


def test_min_train_rejects_tokenizer_mismatch(offline_payloads):
    provider = _provider_for(offline_payloads, override_hash="wrong-hash")
    with pytest.raises(ValueError, match="tokenizer hash does not match"):
        run_real_student_min_train(offline_payloads, provider, num_steps=1)


def test_min_train_requires_positive_steps(offline_payloads):
    provider = _provider_for(offline_payloads)
    with pytest.raises(ValueError, match="num_steps must be positive"):
        run_real_student_min_train(offline_payloads, provider, num_steps=0)


def test_min_train_decoded_mismatch_can_be_relaxed(offline_payloads):
    provider = _provider_for(offline_payloads)
    payload = dict(offline_payloads[0])
    payload["response_text"] = payload["response_text"] + " (edited)"
    with pytest.raises(ValueError, match="decoded response text does not match"):
        run_real_student_min_train([payload], provider, num_steps=1)
    relaxed = run_real_student_min_train(
        [payload], provider, num_steps=1, require_decoded_match=False
    )
    assert relaxed.steps[0].loss_is_finite
    assert not relaxed.decoded_matches
