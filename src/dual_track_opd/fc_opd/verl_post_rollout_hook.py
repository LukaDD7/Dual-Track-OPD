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
from .teacher_client import (
    TeacherClient,
    score_teacher_conditions,
    score_teacher_conditions_multi_sample,
)
from .teacher_protocol import tokenizer_fingerprint
from .va_opd_loss import compute_rollout_va_weights
from .verl_integration import DEFAULT_VERL_CONDITION_ORDER, online_batch_output_to_verl_tensors


# No-op student scorer for VA-OPD mode (student scores not used, avoids GPU allocation).
_NOOP_SCORER: StudentForcedScorer = lambda sample, conditions: None


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
    teacher_scorer = _build_teacher_scorer(fc_config, tokenizer, samples, conditions)
    # VA-OPD mode: full+degraded only → skip chunk parsing, student scorer, routing
    _is_va_opd = set(conditions) == {Condition.FULL, Condition.DEGRADED} and len(conditions) == 2
    pre_scored_students = None if _is_va_opd else _pre_score_students(fc_config, samples, conditions)
    student_scorer = _NOOP_SCORER if _is_va_opd else _build_student_scorer(fc_config)
    output = compute_online_fc_opd_batch(
        samples,
        tokenizer=tokenizer,
        teacher_scorer=teacher_scorer,
        student_scorer=student_scorer,
        verifier=verifier,
        config=OnlineFCOPDConfig(
            conditions=conditions,
            compute_hook_loss=bool(_config_get(fc_config, "compute_hook_loss", True)),
            skip_routing=_is_va_opd,
        ),
        pre_scored_students=pre_scored_students,
    )
    verl_tensors = online_batch_output_to_verl_tensors(
        output,
        condition_order=conditions,
        target_seq_len=int(responses.shape[1]),
        response_mask=response_mask,
    )
    B = int(responses.shape[0])
    for key, value in verl_tensors.as_batch_dict().items():
        if key == "fc_condition_ids":
            # condition_ids is per-condition ([C]); broadcast to [B, C] so
            # verl's batch.reorder works (all non_tensor entries need leading B).
            import numpy as np
            arr = np.array(value.cpu().tolist(), dtype=np.int64)  # [C]
            batch.non_tensor_batch[key] = np.tile(arr, (B, 1))    # [B, C]
        else:
            batch.batch[key] = value.to(responses.device)
    # Pass loss coefficient as a per-sample tensor [B] (must match batch_size).
    loss_coef = float(_config_get(fc_config, "loss_coef", 0.0))
    batch.batch["fc_opd_coef"] = torch.full((B,), loss_coef, device=responses.device)
    # Pass loss_mode — store as a list of strings [B] to survive batch.reorder.
    loss_mode = str(_config_get(fc_config, "loss_mode", "forward"))
    import numpy as np
    batch.non_tensor_batch["fc_opd_loss_mode"] = np.array([loss_mode] * B, dtype=object)
    batch.non_tensor_batch["fc_prompt_ids"] = np.array(
        [sample.sample_uid for sample in samples],
        dtype=object,
    )
    if _is_va_opd:
        if verl_tensors.teacher_sampled_log_probs is None:
            raise ValueError("VA-OPD requires exact teacher sampled token log-probs")
        full_idx = conditions.index(Condition.FULL)
        degraded_idx = conditions.index(Condition.DEGRADED)
        sampled_log_probs = verl_tensors.teacher_sampled_log_probs.to(responses.device)
        va_pos = (sampled_log_probs[:, full_idx] - sampled_log_probs[:, degraded_idx]).clamp_min(0.0)
        teacher_valid = (
            torch.ones_like(response_mask, dtype=torch.bool, device=responses.device)
            if verl_tensors.teacher_valid_mask is None
            else (
                verl_tensors.teacher_valid_mask[:, full_idx].to(responses.device, dtype=torch.bool)
                & verl_tensors.teacher_valid_mask[:, degraded_idx].to(responses.device, dtype=torch.bool)
            )
        )
        batch.batch["fc_rollout_weights"] = compute_rollout_va_weights(
            va_pos,
            response_mask=response_mask & teacher_valid,
            prompt_ids=[sample.sample_uid for sample in samples],
        ).to(responses.device)

    # ── Formal pipeline verification log (VA-OPD reproducibility) ───
    _log_pipeline_verification(
        conditions=conditions,
        loss_mode=loss_mode,
        B=B,
        T=int(responses.shape[1]),
        sampled_lp_shape=getattr(verl_tensors.teacher_sampled_log_probs, "shape", None),
        top_k=int(verl_tensors.teacher_topk_indices.shape[-1]) if verl_tensors.teacher_topk_indices is not None else None,
    )

    metrics = {
        "fc_opd/hook_loss": float(output.loss.detach().cpu().item()),
        "fc_opd/hook_num_samples": float(len(samples)),
        "fc_opd/hook_active_weight": float(verl_tensors.condition_weights.detach().sum().cpu().item()),
    }
    if verl_tensors.teacher_valid_mask is not None:
        valid = verl_tensors.teacher_valid_mask.to(device=response_mask.device, dtype=torch.bool)
        metrics["fc_opd/teacher_valid_ratio"] = float(
            (valid & response_mask.unsqueeze(1)).float().sum().cpu().item()
            / response_mask.unsqueeze(1).expand_as(valid).float().sum().clamp_min(1.0).cpu().item()
        )
    for key, value in output.metrics.items():
        metrics[f"fc_opd/{key}"] = float(value.detach().cpu().item())
    return batch, metrics


def _fc_opd_config(config: Any) -> Mapping[str, Any]:
    algorithm = _config_get(config, "algorithm", {})
    return _config_get(algorithm, "fc_opd", {})


def _conditions_from_config(fc_config: Mapping[str, Any]) -> tuple[Condition, ...]:
    raw = _config_get(fc_config, "conditions", None) or DEFAULT_VERL_CONDITION_ORDER
    return tuple(Condition(condition) for condition in raw)


def _build_teacher_scorer(
    fc_config: Mapping[str, Any],
    tokenizer: Any,
    samples: list[OnlineFCOPDSample],
    conditions: tuple[Condition, ...],
) -> TeacherScorer:
    injected = _optional_callable(fc_config, "teacher_scorer", "teacher_scorer_fqn")
    if injected is not None:
        return injected
    # Support both legacy teacher_url (single) and teacher_urls (comma-separated list).
    teacher_urls_str = _config_get(fc_config, "teacher_urls", None)
    if teacher_urls_str is None:
        teacher_url = _config_get(fc_config, "teacher_url", None)
        if not teacher_url:
            raise ValueError("algorithm.fc_opd.teacher_urls or teacher_url or teacher_scorer_fqn is required")
        teacher_urls = [str(teacher_url)]
    else:
        teacher_urls = [u.strip() for u in str(teacher_urls_str).split(",") if u.strip()]

    expected_hash = _config_get(fc_config, "expected_tokenizer_hash", None)
    if expected_hash is None:
        expected_hash = tokenizer_fingerprint(tokenizer)
    timeout = float(_config_get(fc_config, "teacher_timeout_seconds", 120.0))

    # Split samples evenly across teachers.  Each teacher handles a contiguous
    # chunk so response_text indexing stays trivially aligned.
    num_teachers = len(teacher_urls)
    sample_tuples = [
        (sample.rollout_token_ids, sample.question, sample.condition_inputs)
        for sample in samples
    ]
    response_texts = [sample.rollout_text for sample in samples]

    if num_teachers == 1:
        client = TeacherClient(teacher_urls[0], expected_tokenizer_hash=str(expected_hash), timeout_seconds=timeout)
        pre_scored = score_teacher_conditions_multi_sample(
            sample_tuples, conditions, client,
            response_texts=response_texts, request_prefix="verl_batch",
        )
        lookup: list[dict[Condition, TeacherTopK]] = pre_scored
    else:
        # Fan out to multiple teachers concurrently (HTTP I/O — threads are fine).
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_size = (len(samples) + num_teachers - 1) // num_teachers
        futures: dict[Any, int] = {}  # future → teacher_idx
        partial_results: dict[int, list[dict[Condition, TeacherTopK]]] = {}

        def _score_chunk(teacher_idx: int, start: int, end: int):
            client = TeacherClient(teacher_urls[teacher_idx], expected_tokenizer_hash=str(expected_hash), timeout_seconds=timeout)
            return score_teacher_conditions_multi_sample(
                sample_tuples[start:end], conditions, client,
                response_texts=response_texts[start:end],
                request_prefix=f"verl_batch_t{teacher_idx}",
            )

        with ThreadPoolExecutor(max_workers=num_teachers) as executor:
            for t_idx in range(num_teachers):
                start = t_idx * chunk_size
                end = min(start + chunk_size, len(samples))
                if start >= end:
                    break
                futures[executor.submit(_score_chunk, t_idx, start, end)] = t_idx

            for future in as_completed(futures):
                t_idx = futures[future]
                partial_results[t_idx] = future.result()

        # Reassemble in original sample order.
        lookup = [{} for _ in samples]
        for t_idx in sorted(partial_results):
            start = t_idx * chunk_size
            chunk = partial_results[t_idx]
            for offset, result in enumerate(chunk):
                lookup[start + offset] = result

    def _score(sample: OnlineFCOPDSample, conds: Sequence[Condition]):
        for idx, s in enumerate(samples):
            if s.sample_uid == sample.sample_uid:
                return {Condition(c): lookup[idx][Condition(c)] for c in conds}
        raise ValueError(f"sample {sample.sample_uid} not found in pre-scored batch")

    return _score


def _build_student_scorer(fc_config: Mapping[str, Any]) -> StudentForcedScorer:
    # 1) Directly-injected callable (tests, custom injectors) — no instantiation.
    direct = _config_get(fc_config, "student_scorer", None)
    if direct is not None:
        if not callable(direct):
            raise TypeError("algorithm.fc_opd.student_scorer must be callable")
        return direct

    # 2) Must instantiate via FQN.
    fqn = _config_get(fc_config, "student_scorer_fqn", None)
    if not fqn:
        raise ValueError(
            "algorithm.fc_opd.student_scorer_fqn is required for verl FC-OPD smoke; "
            "a directly-injected student_scorer callable works for tests"
        )
    kwargs = _config_get(fc_config, "student_scorer_kwargs", {}) or {}

    # 3) Try direct CUDA instantiation.  Catches "No CUDA GPUs are available"
    #    from CPU-only Ray actors (verl TaskRunner) and falls through to Ray proxy.
    if torch.cuda.is_available():
        try:
            loaded = _load_fqn(str(fqn))
            if isinstance(kwargs, Mapping) and kwargs:
                loaded = loaded(**dict(kwargs))
            if not callable(loaded):
                raise TypeError(f"{fqn} did not resolve to a callable")
            return loaded
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "cuda" not in msg and "gpu" not in msg:
                raise

    # 4) Ray proxy — spawns a detached Ray GPU actor with StudentScorer.
    from .ray_student_scorer import build_ray_student_scorer_proxy
    return build_ray_student_scorer_proxy(
        student_scorer_fqn=str(fqn),
        student_scorer_kwargs=dict(kwargs),
    )


def _pre_score_students(
    fc_config: Mapping[str, Any],
    samples: list[OnlineFCOPDSample],
    conditions: tuple[Condition, ...],
) -> list[OnlineStudentScores] | None:
    """Pre-score all student samples in one HTTP batch, matching the teacher pattern."""
    fqn = str(_config_get(fc_config, "student_scorer_fqn", ""))
    if not fqn:
        return None
    kwargs = dict(_config_get(fc_config, "student_scorer_kwargs", {}) or {})
    try:
        loaded = _load_fqn(fqn)
        if kwargs:
            loaded = loaded(**kwargs)
        result = loaded(samples, conditions)
        return list(result) if isinstance(result, list) else [result]
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "student pre-scoring failed, falling back to per-sample scoring",
            exc_info=True,
        )
        return None


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
    # verl drops most non-standard columns; look in extra_info first
    question = None
    extra = _row_value(batch, row_index, ("extra_info",))
    if isinstance(extra, Mapping):
        question = str(extra.get("question", "")) or None
    if question is None:
        question = _row_text(batch, row_index, ("question", "questions", "problem", "prompt_text"))
    if question is None and "prompts" in batch.batch:
        # Fallback: decode prompt tokens (use skip_special_tokens to avoid
        # leaking vision/padding tokens into the teacher prompt).
        question = _decode_tokens(tokenizer, batch.batch["prompts"][row_index].detach().cpu().tolist())
    if question is None:
        raise ValueError("FC-OPD hook requires a question field or prompt tokens")

    condition_inputs = _condition_inputs_from_row(batch, row_index)
    rollout_text = _decode_tokens(tokenizer, response_token_ids, skip_special_tokens=False)
    sample_uid = _row_text(batch, row_index, ("sample_uid", "uid", "id")) or f"verl:{global_steps}:{row_index}"
    # verl drops non-standard columns; read verl-hidden fields from extra_info
    extra = _row_value(batch, row_index, ("extra_info",)) or {}
    choices = tuple(str(item) for item in (
        extra.get("choices") or _row_value(batch, row_index, ("choices", "options")) or ()
    ))
    answer_metadata = (
        extra.get("answer") or extra.get("answer_metadata")
        or _row_value(batch, row_index, ("answer_metadata", "answer", "gold_answer"))
    )
    return OnlineFCOPDSample(
        sample_uid=sample_uid,
        question=question,
        condition_inputs=condition_inputs,
        rollout_token_ids=response_token_ids,
        rollout_text=rollout_text,
        prompt=_row_value(batch, row_index, ("prompt", "raw_prompt", "messages")),
        images=_row_value(batch, row_index, ("images", "image_path", "multi_modal_inputs")),
        choices=choices,
        answer_metadata=answer_metadata,
        metadata={"global_steps": global_steps},
    )


def _condition_inputs_from_row(batch: Any, row_index: int) -> ConditionInputs:
    raw = _row_value(batch, row_index, ("fc_opd_condition_inputs", "condition_inputs"))
    # verl AgentLoop drops non-standard non-tensor columns; fall back to extra_info
    if raw is None:
        extra = _row_value(batch, row_index, ("extra_info",))
        if isinstance(extra, Mapping):
            raw = extra.get("condition_inputs") or extra.get("fc_opd_condition_inputs")
            # Parquet struct round-trips as list-of-tuples via pyarrow
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                try:
                    raw = dict(raw)
                except (TypeError, ValueError):
                    pass
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


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        )
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


def _log_pipeline_verification(
    *,
    conditions: tuple[Condition, ...],
    loss_mode: str,
    B: int,
    T: int,
    sampled_lp_shape: tuple[int, ...] | None,
    top_k: int | None,
) -> None:
    """Print formal pipeline verification header for VA-OPD reproducibility."""
    import logging as _logging
    import sys as _sys
    _log = _logging.getLogger(__name__)
    lines = [
        "=" * 72,
        "  VA-OPD Pipeline Verification (arXiv 2605.21924 §3.2-3.3)",
        "=" * 72,
        f"  C          = {len(conditions)}  conditions: {[c.value for c in conditions]}",
        f"  loss_mode  = {loss_mode}",
        f"  batch      = [B={B}, T={T}]",
        f"  top_k      = {top_k}",
        f"  exact_lp   = {sampled_lp_shape}  (None=tail-fallback)",
        f"  Student    = raw image + canonical question  (choices allowed, no XML)",
        f"  Teacher    = raw image + same canonical question  (no format bias)",
        f"  KL         = {'JSD(P_T, P_S)' if loss_mode == 'va_opd_jsd' else 'reverse KL(P_S || P_T)' if loss_mode == 'va_opd' else 'forward'}",
        f"  Formula §3.2: w^(k) = softmax(z_score(ā^(k)) / τ), sums to 1 per prompt",
        f"  Formula §3.3: L_group = 0.5·mean(L_HighVA) + 0.5·mean(L_LowVA)",
        f"  Total (formula 7): L = Σ_k w^(k)·L_group^(k)   (pure distillation, no GRPO)",
        "=" * 72,
    ]
    for line in lines:
        _log.warning(line)
    print("\n".join(lines), file=_sys.stderr, flush=True)
