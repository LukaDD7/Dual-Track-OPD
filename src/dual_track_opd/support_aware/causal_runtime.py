"""HuggingFace runtime adapters for causal state probes.

The runtime is intentionally separate from selection, statistics, and report
code.  It loads no model at import time, uses exact response token IDs for all
fixed-trajectory probes, and releases vocabulary logits after every chunk.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .causal_image import ImageConditions
from .causal_schema import ContinuationEstimate, continuation_estimate
from .diagnostic import generation_record_from_token_ids
from .prefix_intervention import generate_continuation, prefix_leakage_reason
from .verifier import verify_answer
from .visual_js import full_vocab_js_statistics, token_signal_rows


DIRECT_ANSWER_INSTRUCTION = (
    "Use the problem and partial reasoning above. Return only the final answer "
    "as exactly one line: \\boxed{<answer>}. Do not add reasoning."
)


@dataclass(frozen=True)
class RuntimeModels:
    student_model: Any
    student_processor: Any
    teacher_model: Any | None
    teacher_processor: Any | None
    tokenizer_hash: str


@dataclass(frozen=True)
class FixedTrajectoryResult:
    token_signals: tuple[Mapping[str, Any], ...]
    decoded_tokens: tuple[str, ...]
    prompt_token_hash: str
    layout_metadata: Mapping[str, Any]


def model_identity(model_path: str) -> dict[str, Any]:
    path = Path(os.path.expandvars(model_path)).expanduser()
    identity: dict[str, Any] = {"path": str(path), "is_local": path.exists()}
    if path.is_dir():
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            candidate = path / name
            if candidate.is_file():
                identity[f"{name}_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return identity


def load_runtime_models(
    *,
    student_model_path: str,
    student_device: str,
    dtype: str,
    teacher_model_path: str | None = None,
    teacher_device: str | None = None,
) -> RuntimeModels:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype) if dtype != "float32" else torch.float32

    def load(path_value: str, device: str):
        resolved = os.path.expandvars(path_value)
        processor = AutoProcessor.from_pretrained(
            resolved,
            local_files_only=Path(resolved).exists(),
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            resolved,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=Path(resolved).exists(),
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        return model, processor

    student_model, student_processor = load(student_model_path, student_device)
    student_hash = tokenizer_fingerprint(student_processor.tokenizer)
    teacher_model = teacher_processor = None
    if teacher_model_path is not None:
        teacher_model, teacher_processor = load(
            teacher_model_path,
            teacher_device or student_device,
        )
        teacher_hash = tokenizer_fingerprint(teacher_processor.tokenizer)
        if teacher_hash != student_hash:
            raise ValueError(
                "teacher/student tokenizer mismatch; relay, transport, and token-level "
                "teacher-path probes require identical token-ID semantics"
            )
    return RuntimeModels(
        student_model=student_model,
        student_processor=student_processor,
        teacher_model=teacher_model,
        teacher_processor=teacher_processor,
        tokenizer_hash=student_hash,
    )


def _prompt_inputs(processor, image: Image.Image, prompt_text: str) -> dict[str, torch.Tensor]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return dict(processor(text=[rendered], images=[image], return_tensors="pt"))


def _layout_signature(inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "input_ids": inputs["input_ids"].detach().cpu().tolist(),
        "attention_shape": list(inputs["attention_mask"].shape),
    }
    for key in ("image_grid_thw", "video_grid_thw", "token_type_ids", "mm_token_type_ids"):
        if key in inputs:
            value = inputs[key]
            signature[key] = {
                "shape": list(value.shape),
                "values": value.detach().cpu().tolist() if value.numel() <= 128 else None,
            }
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            signature[f"{key}_shape"] = list(inputs[key].shape)
    return signature


def _validate_counterfactual_layout(inputs_by_condition: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    names = list(inputs_by_condition)
    if not names:
        raise ValueError("at least one image condition is required")
    reference = inputs_by_condition[names[0]]
    reference_ids = reference["input_ids"].detach().cpu()
    for name in names[1:]:
        value = inputs_by_condition[name]
        if not torch.equal(reference_ids, value["input_ids"].detach().cpu()):
            raise RuntimeError(f"{name}: image intervention changed prompt token IDs")
        for key in ("image_grid_thw", "video_grid_thw"):
            left = reference.get(key)
            right = value.get(key)
            if (left is None) != (right is None):
                raise RuntimeError(f"{name}: image intervention changed {key} presence")
            if left is not None and not torch.equal(left.detach().cpu(), right.detach().cpu()):
                raise RuntimeError(f"{name}: image intervention changed {key}")
        for key in ("pixel_values", "pixel_values_videos"):
            if key in reference and key in value and reference[key].shape != value[key].shape:
                raise RuntimeError(f"{name}: image intervention changed {key} shape")


def _append_response_prefix(
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = {key: value.to(device) for key, value in prompt_inputs.items()}
    if int(encoded["input_ids"].shape[0]) != 1:
        raise ValueError("causal fixed-trajectory scoring requires batch size one")
    response = torch.tensor(
        [list(int(value) for value in response_ids)],
        dtype=encoded["input_ids"].dtype,
        device=device,
    )
    encoded["input_ids"] = torch.cat((encoded["input_ids"], response), dim=1)
    extension = torch.ones(
        (1, len(response_ids)),
        dtype=encoded["attention_mask"].dtype,
        device=device,
    )
    encoded["attention_mask"] = torch.cat((encoded["attention_mask"], extension), dim=1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in encoded:
            token_extension = torch.zeros(
                (1, len(response_ids)),
                dtype=encoded[key].dtype,
                device=device,
            )
            encoded[key] = torch.cat((encoded[key], token_extension), dim=-1)
    encoded.pop("position_ids", None)
    return encoded


def _supports_logits_to_keep(model: Any) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "logits_to_keep" in parameters


@torch.inference_mode()
def response_chunk_logits(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: Sequence[int],
    *,
    start: int,
    end: int,
    device: str,
    prefer_logits_to_keep: bool = True,
) -> tuple[torch.Tensor, bool]:
    """Return logits predicting response tokens ``[start:end]``.

    The model sees the exact fixed response prefix through ``end``.  When the
    model supports ``logits_to_keep`` only ``chunk_size + 1`` sequence logits
    are returned; the final next-token position is dropped.
    """

    total = len(response_ids)
    if not 0 <= start < end <= total:
        raise ValueError("response chunk must satisfy 0 <= start < end <= length")
    prefix = tuple(int(value) for value in response_ids[:end])
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    model_inputs = _append_response_prefix(prompt_inputs, prefix, device=device)
    used_keep = bool(prefer_logits_to_keep and _supports_logits_to_keep(model))
    kwargs: dict[str, Any] = {"use_cache": False}
    if used_keep:
        kwargs["logits_to_keep"] = end - start + 1
    outputs = model(**model_inputs, **kwargs)
    logits = outputs.logits
    if used_keep and int(logits.shape[1]) == end - start + 1:
        selected = logits[0, :-1, :]
    else:
        selected = logits[0, prompt_length - 1 + start : prompt_length - 1 + end, :]
        used_keep = False
    if int(selected.shape[0]) != end - start:
        raise RuntimeError("fixed-trajectory logits do not align with response chunk")
    result = selected.detach()
    del outputs, logits, model_inputs
    return result, used_keep


def fixed_trajectory_visual_statistics(
    model: Any,
    processor: Any,
    *,
    prompt_text: str,
    images: ImageConditions,
    response_ids: Sequence[int],
    device: str,
    chunk_size: int = 64,
    prefer_logits_to_keep: bool = True,
) -> FixedTrajectoryResult:
    if not response_ids:
        raise ValueError("fixed trajectory must contain response token IDs")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    image_map = {"full": images.full, "degraded": images.degraded, "null": images.null}
    prompt_inputs = {
        name: _prompt_inputs(processor, image, prompt_text)
        for name, image in image_map.items()
    }
    _validate_counterfactual_layout(prompt_inputs)
    prompt_ids = tuple(int(value) for value in prompt_inputs["full"]["input_ids"][0].tolist())

    js_degraded: list[float] = []
    js_null: list[float] = []
    entropy_full: list[float] = []
    entropy_degraded: list[float] = []
    entropy_null: list[float] = []
    keep_used_everywhere = True
    for start in range(0, len(response_ids), chunk_size):
        end = min(len(response_ids), start + chunk_size)
        full_logits, full_keep = response_chunk_logits(
            model, prompt_inputs["full"], response_ids,
            start=start, end=end, device=device, prefer_logits_to_keep=prefer_logits_to_keep,
        )
        degraded_logits, degraded_keep = response_chunk_logits(
            model, prompt_inputs["degraded"], response_ids,
            start=start, end=end, device=device, prefer_logits_to_keep=prefer_logits_to_keep,
        )
        degraded_stats = full_vocab_js_statistics(full_logits, degraded_logits)
        null_logits, null_keep = response_chunk_logits(
            model, prompt_inputs["null"], response_ids,
            start=start, end=end, device=device, prefer_logits_to_keep=prefer_logits_to_keep,
        )
        null_stats = full_vocab_js_statistics(full_logits, null_logits)
        keep_used_everywhere &= full_keep and degraded_keep and null_keep
        js_degraded.extend(float(value) for value in degraded_stats.js.cpu().tolist())
        js_null.extend(float(value) for value in null_stats.js.cpu().tolist())
        entropy_full.extend(float(value) for value in degraded_stats.entropy_left.cpu().tolist())
        entropy_degraded.extend(float(value) for value in degraded_stats.entropy_right.cpu().tolist())
        entropy_null.extend(float(value) for value in null_stats.entropy_right.cpu().tolist())
        del full_logits, degraded_logits, null_logits, degraded_stats, null_stats
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    rows = token_signal_rows(
        token_ids=response_ids,
        js_full_degraded=js_degraded,
        js_full_null=js_null,
        entropy_full=entropy_full,
        entropy_degraded=entropy_degraded,
        entropy_null=entropy_null,
    )
    tokenizer = processor.tokenizer
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        decoded_tokens = tuple(str(value) for value in tokenizer.convert_ids_to_tokens(list(response_ids)))
    else:
        decoded_tokens = tuple(
            tokenizer.decode([int(value)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for value in response_ids
        )
    return FixedTrajectoryResult(
        token_signals=tuple(rows),
        decoded_tokens=decoded_tokens,
        prompt_token_hash=hash_token_ids(prompt_ids),
        layout_metadata={
            "conditions": {name: _layout_signature(value) for name, value in prompt_inputs.items()},
            "response_chunk_size": chunk_size,
            "logits_to_keep_used_everywhere": keep_used_everywhere,
            "image_intervention": dict(images.metadata),
        },
    )


def condition_continuations(
    model: Any,
    processor: Any,
    *,
    images: ImageConditions,
    prompt_text: str,
    prefix_ids: Sequence[int],
    gold_answer: Any,
    k: int,
    max_continuation_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> tuple[ContinuationEstimate, ...]:
    if k <= 0:
        raise ValueError("continuation K must be positive")
    estimates = []
    for condition, image in (("full", images.full), ("degraded", images.degraded), ("null", images.null)):
        correctness: list[bool | None] = []
        seeds: list[int] = []
        hashes: list[str] = []
        for rollout_index in range(k):
            rollout_seed = seed + rollout_index
            generated = generate_continuation(
                model,
                processor,
                image=image,
                prompt_text=prompt_text,
                prefix_ids=prefix_ids,
                max_continuation_tokens=max_continuation_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=rollout_seed,
                device=device,
            )["generation"]
            verdict = verify_answer(generated.response_text_display, gold_answer)
            correctness.append(verdict.get("correct"))
            seeds.append(rollout_seed)
            hashes.append(generated.response_token_hash)
        estimates.append(continuation_estimate(
            condition,
            correctness,
            seeds=seeds,
            response_hashes=hashes,
        ))
    return tuple(estimates)


def teacher_relay_estimate(
    *,
    models: RuntimeModels,
    image: Image.Image,
    prompt_text: str,
    student_prefix_ids: Sequence[int],
    gold_answer: Any,
    relay_length: int,
    k: int,
    max_student_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    student_device: str,
    teacher_device: str,
) -> ContinuationEstimate:
    if models.teacher_model is None or models.teacher_processor is None:
        raise ValueError("teacher relay requires a loaded teacher")
    correctness: list[bool | None] = []
    seeds: list[int] = []
    hashes: list[str] = []
    teacher_eos = models.teacher_processor.tokenizer.eos_token_id
    for rollout_index in range(k):
        relay_seed = seed + 2 * rollout_index
        relay = generate_continuation(
            models.teacher_model,
            models.teacher_processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=student_prefix_ids,
            max_continuation_tokens=relay_length,
            temperature=temperature,
            top_p=top_p,
            seed=relay_seed,
            device=teacher_device,
        )
        relay_ids = tuple(int(value) for value in relay["continuation_token_ids"])
        if relay_ids and teacher_eos is not None and relay_ids[-1] == int(teacher_eos):
            relay_ids = relay_ids[:-1]
        hybrid_prefix = tuple(int(value) for value in student_prefix_ids) + relay_ids
        student_seed = relay_seed + 1
        hybrid_text = models.student_processor.tokenizer.decode(
            list(hybrid_prefix),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if prefix_leakage_reason(hybrid_text, gold_answer) is not None:
            # Relay gain must measure a useful intermediate state, not a short
            # teacher segment that already states an extractable final answer.
            correctness.append(None)
            seeds.append(student_seed)
            hashes.append(hash_token_ids(hybrid_prefix))
            continue
        generated = generate_continuation(
            models.student_model,
            models.student_processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=hybrid_prefix,
            max_continuation_tokens=max_student_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=student_seed,
            device=student_device,
        )["generation"]
        verdict = verify_answer(generated.response_text_display, gold_answer)
        correctness.append(verdict.get("correct"))
        seeds.append(student_seed)
        hashes.append(generated.response_token_hash)
    return continuation_estimate(
        f"relay_l{relay_length}", correctness, seeds=seeds, response_hashes=hashes
    )


def transport_estimate(
    *,
    models: RuntimeModels,
    image: Image.Image,
    prompt_text: str,
    teacher_prefix_ids: Sequence[int],
    gold_answer: Any,
    k: int,
    max_continuation_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    student_device: str,
    condition: str = "teacher_transport",
) -> ContinuationEstimate:
    correctness: list[bool | None] = []
    seeds: list[int] = []
    hashes: list[str] = []
    for rollout_index in range(k):
        rollout_seed = seed + rollout_index
        generated = generate_continuation(
            models.student_model,
            models.student_processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=teacher_prefix_ids,
            max_continuation_tokens=max_continuation_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=rollout_seed,
            device=student_device,
        )["generation"]
        verdict = verify_answer(generated.response_text_display, gold_answer)
        correctness.append(verdict.get("correct"))
        seeds.append(rollout_seed)
        hashes.append(generated.response_token_hash)
    return continuation_estimate(
        condition, correctness, seeds=seeds, response_hashes=hashes
    )


def direct_answer_estimate(
    model: Any,
    processor: Any,
    *,
    image: Image.Image,
    prompt_text: str,
    prefix_ids: Sequence[int],
    gold_answer: Any,
    k: int,
    max_answer_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> ContinuationEstimate:
    """Answer-only decoding from a text-rendered partial assistant turn.

    This probe intentionally creates a new user turn and therefore records a
    text-level answer-leakage estimand, not an exact-token continuation claim.
    """

    prefix_text = processor.tokenizer.decode(
        list(int(value) for value in prefix_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    leakage_reason = prefix_leakage_reason(prefix_text, gold_answer)
    if leakage_reason is not None:
        raise ValueError(f"teacher prefix is not answer-free: {leakage_reason}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": prefix_text}]},
        {"role": "user", "content": [{"type": "text", "text": DIRECT_ANSWER_INSTRUCTION}]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = int(inputs["input_ids"].shape[1])
    correctness: list[bool | None] = []
    seeds: list[int] = []
    hashes: list[str] = []
    for rollout_index in range(k):
        rollout_seed = seed + rollout_index
        random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        model.generation_config.do_sample = temperature > 0
        model.generation_config.max_new_tokens = max_answer_tokens
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
        model.generation_config.temperature = temperature if temperature > 0 else None
        model.generation_config.top_p = top_p if temperature > 0 else None
        with torch.inference_mode():
            outputs = model.generate(**inputs)
        response_ids = tuple(int(value) for value in outputs[0, input_width:].tolist())
        generation = generation_record_from_token_ids(
            model=model,
            tokenizer=processor.tokenizer,
            response_token_ids=response_ids,
            prompt_token_ids=tuple(int(value) for value in inputs["input_ids"][0].tolist()),
            max_new_tokens=max_answer_tokens,
        )
        verdict = verify_answer(generation.response_text_display, gold_answer)
        correctness.append(verdict.get("correct"))
        seeds.append(rollout_seed)
        hashes.append(generation.response_token_hash)
    return continuation_estimate(
        "answer_leakage", correctness, seeds=seeds, response_hashes=hashes
    )


def teacher_path_support_statistics(
    *,
    models: RuntimeModels,
    image: Image.Image,
    prompt_text: str,
    teacher_token_ids: Sequence[int],
    student_device: str,
    teacher_device: str,
    chunk_size: int = 64,
    top_k: int = 32,
) -> list[dict[str, float | int]]:
    """Student NLL, full-vocab S/T JS, and top-k overlap on teacher states."""

    if models.teacher_model is None or models.teacher_processor is None:
        raise ValueError("teacher-path support requires a loaded teacher")
    student_inputs = _prompt_inputs(models.student_processor, image, prompt_text)
    teacher_inputs = _prompt_inputs(models.teacher_processor, image, prompt_text)
    if not torch.equal(student_inputs["input_ids"], teacher_inputs["input_ids"]):
        raise RuntimeError("teacher/student rendered prompt IDs differ")
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(teacher_token_ids), chunk_size):
        end = min(len(teacher_token_ids), start + chunk_size)
        student_logits, _ = response_chunk_logits(
            models.student_model, student_inputs, teacher_token_ids,
            start=start, end=end, device=student_device,
        )
        teacher_logits, _ = response_chunk_logits(
            models.teacher_model, teacher_inputs, teacher_token_ids,
            start=start, end=end, device=teacher_device,
        )
        stats = full_vocab_js_statistics(student_logits, teacher_logits)
        student_logp = torch.log_softmax(student_logits.float(), dim=-1)
        ids = torch.tensor(teacher_token_ids[start:end], dtype=torch.long, device=student_logp.device)
        sampled = student_logp[torch.arange(end - start, device=student_logp.device), ids]
        k = min(top_k, int(student_logits.shape[-1]))
        student_top = torch.topk(student_logits, k=k, dim=-1).indices.cpu()
        teacher_top = torch.topk(teacher_logits, k=k, dim=-1).indices.cpu()
        for offset in range(end - start):
            overlap = len(set(student_top[offset].tolist()).intersection(teacher_top[offset].tolist())) / k
            rows.append({
                "position": start + offset,
                "token_id": int(teacher_token_ids[start + offset]),
                "student_nll": float(-sampled[offset].cpu()),
                "student_teacher_js": float(stats.js[offset].cpu()),
                "topk_overlap": float(overlap),
            })
        del student_logits, teacher_logits, stats, student_logp, sampled
    return rows
