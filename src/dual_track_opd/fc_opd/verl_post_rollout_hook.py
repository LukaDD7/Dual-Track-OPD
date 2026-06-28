"""Post-rollout FC-OPD hook for the thin verl trainer patch."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch

from .conditions import Condition, ConditionInputs, build_condition_inputs
from .online_batch import (
    OnlineFCOPDConfig,
    OnlineFCOPDSample,
    StudentForcedScorer,
    TeacherScorer,
    compute_online_fc_opd_batch,
)
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint
from .verl_integration import DEFAULT_VERL_CONDITION_ORDER, online_batch_output_to_verl_tensors


def fc_opd_post_rollout_hook(
    *,
    batch: Any,
    tokenizer: Any,
    processor: Any | None,
    config: Any,
    global_steps: int,
) -> tuple[Any, dict[str, float]]:
    """Attach online FC-OPD tensors to a verl rollout batch.

    The hook scores the current response tokens selected by ``response_mask``.
    It never reads offline score JSONL or stale precomputed teacher outputs.
    """

    del processor
    fc_config = _fc_opd_config(config)
    conditions = _conditions_from_config(fc_config)
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must have shape [B, T]")

    teacher_scorer = _build_teacher_scorer(fc_config, tokenizer)
    student_scorer = _build_student_scorer(fc_config)
    verifier = _optional_callable(fc_config, "verifier", "verifier_fqn")
    samples = [
        _sample_from_batch_row(
            batch=batch,
            tokenizer=tokenizer,
            row_index=index,
            response_token_ids=_valid_response_ids(responses[index], response_mask[index]),
            global_steps=global_steps,
        )
        for index in range(int(responses.shape[0]))
    ]
    output = compute_online_fc_opd_batch(
        samples,
        tokenizer=tokenizer,
        teacher_scorer=teacher_scorer,
        student_scorer=student_scorer,
        verifier=verifier,
        config=OnlineFCOPDConfig(conditions=conditions),
    )
    verl_tensors = online_batch_output_to_verl_tensors(
        output,
        condition_order=conditions,
        target_seq_len=int(responses.shape[1]),
        response_mask=response_mask,
    )
    for key, value in verl_tensors.as_batch_dict().items():
        batch.batch[key] = value.to(responses.device)

    metrics = {
        "fc_opd/hook_loss": float(output.loss.detach().cpu().item()),
        "fc_opd/hook_num_samples": float(len(samples)),
        "fc_opd/hook_active_weight": float(verl_tensors.condition_weights.detach().sum().cpu().item()),
    }
    for key, value in output.metrics.items():
        metrics[f"fc_opd/{key}"] = float(value.detach().cpu().item())
    return batch, metrics


def _fc_opd_config(config: Any) -> Mapping[str, Any]:
    algorithm = _config_get(config, "algorithm", {})
    return _config_get(algorithm, "fc_opd", {})


def _conditions_from_config(fc_config: Mapping[str, Any]) -> tuple[Condition, ...]:
    raw = _config_get(fc_config, "conditions", None) or DEFAULT_VERL_CONDITION_ORDER
    return tuple(Condition(condition) for condition in raw)


def _build_teacher_scorer(fc_config: Mapping[str, Any], tokenizer: Any) -> TeacherScorer:
    injected = _optional_callable(fc_config, "teacher_scorer", "teacher_scorer_fqn")
    if injected is not None:
        return injected
    teacher_url = _config_get(fc_config, "teacher_url", None)
    if not teacher_url:
        raise ValueError("algorithm.fc_opd.teacher_url or teacher_scorer_fqn is required")
    expected_hash = _config_get(fc_config, "expected_tokenizer_hash", None)
    if expected_hash is None:
        expected_hash = tokenizer_fingerprint(tokenizer)
    timeout = float(_config_get(fc_config, "teacher_timeout_seconds", 120.0))
    client = TeacherClient(str(teacher_url), expected_tokenizer_hash=str(expected_hash), timeout_seconds=timeout)

    def _score(sample: OnlineFCOPDSample, conditions: Sequence[Condition]):
        return score_teacher_conditions(
            response_token_ids=sample.rollout_token_ids,
            question=sample.question,
            condition_inputs=sample.condition_inputs,
            conditions=conditions,
            teacher_client=client,
            response_text=sample.rollout_text,
            request_prefix=f"{sample.sample_uid}:verl:{sample.metadata.get('global_steps', 0)}",
        )

    return _score


def _build_student_scorer(fc_config: Mapping[str, Any]) -> StudentForcedScorer:
    scorer = _optional_callable(fc_config, "student_scorer", "student_scorer_fqn")
    if scorer is None:
        raise ValueError(
            "algorithm.fc_opd.student_scorer_fqn is required; the trainer hook must not load or reuse offline scores"
        )
    return scorer


def _optional_callable(fc_config: Mapping[str, Any], direct_key: str, fqn_key: str) -> Callable[..., Any] | None:
    direct = _config_get(fc_config, direct_key, None)
    if direct is not None:
        if not callable(direct):
            raise TypeError(f"algorithm.fc_opd.{direct_key} must be callable")
        return direct
    fqn = _config_get(fc_config, fqn_key, None)
    if not fqn:
        return None
    loaded = _load_fqn(str(fqn))
    kwargs = _config_get(fc_config, f"{direct_key}_kwargs", {}) or {}
    if isinstance(kwargs, Mapping) and kwargs:
        loaded = loaded(**dict(kwargs))
    if not callable(loaded):
        raise TypeError(f"{fqn} did not resolve to a callable")
    return loaded


def _sample_from_batch_row(
    *,
    batch: Any,
    tokenizer: Any,
    row_index: int,
    response_token_ids: tuple[int, ...],
    global_steps: int,
) -> OnlineFCOPDSample:
    if not response_token_ids:
        raise ValueError(f"row {row_index}: response_mask selected no rollout tokens")
    question = _row_text(batch, row_index, ("question", "questions", "problem", "prompt_text"))
    if question is None and "prompts" in batch.batch:
        question = _decode_tokens(tokenizer, batch.batch["prompts"][row_index].detach().cpu().tolist())
    if question is None:
        raise ValueError("FC-OPD hook requires a question field or prompt tokens")

    condition_inputs = _condition_inputs_from_row(batch, row_index)
    rollout_text = _decode_tokens(tokenizer, response_token_ids)
    sample_uid = _row_text(batch, row_index, ("sample_uid", "uid", "id")) or f"verl:{global_steps}:{row_index}"
    return OnlineFCOPDSample(
        sample_uid=sample_uid,
        question=question,
        condition_inputs=condition_inputs,
        rollout_token_ids=response_token_ids,
        rollout_text=rollout_text,
        prompt=_row_value(batch, row_index, ("prompt", "raw_prompt", "messages")),
        images=_row_value(batch, row_index, ("images", "image_path", "multi_modal_inputs")),
        choices=tuple(str(item) for item in (_row_value(batch, row_index, ("choices", "options")) or ())),
        answer_metadata=_row_value(batch, row_index, ("answer_metadata", "answer", "gold_answer")),
        metadata={"global_steps": global_steps},
    )


def _condition_inputs_from_row(batch: Any, row_index: int) -> ConditionInputs:
    raw = _row_value(batch, row_index, ("fc_opd_condition_inputs", "condition_inputs"))
    if isinstance(raw, ConditionInputs):
        return raw
    if isinstance(raw, Mapping):
        record = dict(raw) if "condition_inputs" in raw else {"condition_inputs": raw}
        return build_condition_inputs(record)
    row_record: dict[str, Any] = {}
    for key in (
        "condition_inputs",
        "full_image",
        "degraded_image",
        "free_caption",
        "task_evidence",
        "task_visible_evidence",
        "task_infer_evidence",
        "task_solve_evidence",
    ):
        value = _row_value(batch, row_index, (key,))
        if value is not None:
            row_record[key] = value
    if "condition_inputs" not in row_record and {"full_image", "degraded_image"}.issubset(row_record):
        row_record = {"condition_inputs": row_record}
    return build_condition_inputs(row_record)


def _valid_response_ids(response_row: torch.Tensor, mask_row: torch.Tensor) -> tuple[int, ...]:
    selected = response_row.detach().cpu()[mask_row.detach().cpu().bool()].tolist()
    return tuple(int(item) for item in selected)


def _row_text(batch: Any, row_index: int, keys: Sequence[str]) -> str | None:
    value = _row_value(batch, row_index, keys)
    return None if value is None else str(value)


def _row_value(batch: Any, row_index: int, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in getattr(batch, "non_tensor_batch", {}):
            return _select_row(batch.non_tensor_batch[key], row_index)
        if key in getattr(batch, "batch", {}):
            return _select_row(batch.batch[key], row_index)
    return None


def _select_row(value: Any, row_index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[row_index]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[row_index]
    try:
        return value[row_index]
    except Exception:
        return value


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _load_fqn(fqn: str) -> Callable[..., Any]:
    module_name, _, attr = fqn.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"expected a fully qualified name, got: {fqn}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)
