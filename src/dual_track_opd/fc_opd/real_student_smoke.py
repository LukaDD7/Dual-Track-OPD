"""Real student-logits adapter smoke for the offline FC-OPD loss.

This is the bridge from *synthetic* student logits (``offline_loss``) to *real*
student-model logits. It loads a real Qwen3-VL / Qwen3.5-VL student, runs a
teacher-forced multimodal forward on records from an offline-score JSONL,
extracts the logits at the exact response-token positions, feeds them into the
existing FC-OPD loss/router path, and verifies that ``backward`` reaches real
model parameters.

It is **not** verl integration. No teacher service is required, and
``third_party/verl`` is never imported. The model-specific (transformers) code is
imported lazily so the alignment and loss logic stay importable and CPU-testable
with a fake provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch

from .conditions import Condition, ConditionInputs, ImageInput
from .loss import FCOPDLossConfig, compute_fc_opd_loss
from .offline_loss import (
    FOUR_CONDITIONS,
    FOUR_CONDITION_ROUTER,
    load_offline_score_records,
    offline_record_to_tensors,
)
from .router import RouterConfig, route_condition_weights
from .teacher_prompts import render_teacher_prompt
from .teacher_protocol import tokenizer_fingerprint

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

TWO_CONDITIONS: tuple[Condition, ...] = (Condition.FULL, Condition.BLUR)
FOUR_CLEAN_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK,
)
SIX_C_SOLVE_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK_VISIBLE,
    Condition.TASK_INFER,
    Condition.TASK_SOLVE,
)
TWO_CONDITION_ROUTER = RouterConfig(
    mode="chunk",
    chunk_condition={
        "visual_evidence": Condition.FULL,
        "reasoning": Condition.FULL,
        "answer": Condition.FULL,
    },
    invalid_format_condition=Condition.BLUR,
)
FOUR_CLEAN_CONDITION_ROUTER = RouterConfig(
    mode="chunk",
    chunk_condition={
        "visual_evidence": Condition.TASK,
        "reasoning": Condition.FREE,
        "answer": Condition.FULL,
    },
    invalid_format_condition=Condition.DEGRADED,
)
SIX_C_CHUNK_GATED_ROUTER = RouterConfig(
    mode="chunk_gated",
    invalid_format_condition=Condition.FULL,
)
ROUTING_MODES = (
    "chunk",
    "uniform_all_conditions",
    "chunk_gated",
    "chunk_gated_primary",
    "chunk_gated_contrastive",
    "student_deficit_chunk_gated",
)
STUDENT_DEFICIT_ROUTER = RouterConfig(mode="student_deficit_chunk_gated", invalid_format_condition=Condition.FULL)


# ---------------------------------------------------------------------------
# Pure, CPU-testable alignment helpers
# ---------------------------------------------------------------------------


def response_logit_slice(
    full_logits: torch.Tensor,
    prompt_length: int,
    num_response_tokens: int,
) -> torch.Tensor:
    """Slice the logits that predict the teacher-forced response tokens.

    For a causal LM over ``[prompt(P), response(T)]`` the logits at position
    ``i`` predict token ``i + 1``. The response tokens occupy absolute positions
    ``P .. P + T - 1``, so the logits that predict them are positions
    ``P - 1 .. P + T - 2`` — i.e. ``full_logits[:, P - 1 : P - 1 + T, :]``.
    """

    if full_logits.ndim != 3:
        raise ValueError("full_logits must have shape [batch, seq, vocab]")
    if prompt_length < 1:
        raise ValueError("prompt_length must be at least 1 for next-token alignment")
    if num_response_tokens < 1:
        raise ValueError("num_response_tokens must be positive")
    start = prompt_length - 1
    end = start + num_response_tokens
    if end > full_logits.shape[1]:
        raise ValueError(
            f"response slice [{start}:{end}] exceeds sequence length {full_logits.shape[1]}"
        )
    return full_logits[:, start:end, :]


def condition_inputs_from_record(record: Mapping[str, Any]) -> ConditionInputs:
    """Rebuild the validated ConditionInputs stored in an offline-score record."""

    raw = record.get("condition_inputs")
    if not isinstance(raw, Mapping):
        raise ValueError("offline record is missing condition_inputs")
    full = raw["full_image"]
    degraded = raw["degraded_image"]
    inputs = ConditionInputs(
        full_image=ImageInput(path=str(full["path"])),
        degraded_image=ImageInput(
            path=str(degraded["path"]),
            transform=dict(degraded.get("transform", {})),
        ),
        free_caption=str(raw["free_caption"]),
        task_evidence=str(raw["task_evidence"]),
        task_visible_evidence=(
            None if raw.get("task_visible_evidence") is None else str(raw["task_visible_evidence"])
        ),
        task_infer_evidence=(
            None if raw.get("task_infer_evidence") is None else str(raw["task_infer_evidence"])
        ),
        task_solve_evidence=(
            None if raw.get("task_solve_evidence") is None else str(raw["task_solve_evidence"])
        ),
        verified_facts=None if raw.get("verified_facts") is None else str(raw["verified_facts"]),
        verified_facts_source=(
            None if raw.get("verified_facts_source") is None else str(raw["verified_facts_source"])
        ),
    )
    inputs.validate()
    return inputs


def teacher_scores_to_device(
    teacher_scores: Mapping[Condition, Any],
    device: torch.device | str,
) -> dict[Condition, Any]:
    """Move every ``TeacherTopK`` tensor field onto ``device``.

    Uses ``dataclasses.replace`` so the ``TeacherTopK`` type is preserved. The
    optional ``tail_log_prob`` / ``entropy`` fields are moved only when present.
    """

    device = torch.device(device)
    moved: dict[Condition, Any] = {}
    for condition, scores in teacher_scores.items():
        moved[condition] = replace(
            scores,
            token_ids=scores.token_ids.to(device),
            log_probs=scores.log_probs.to(device),
            tail_log_prob=(
                None if scores.tail_log_prob is None else scores.tail_log_prob.to(device)
            ),
            entropy=None if scores.entropy is None else scores.entropy.to(device),
        )
    return moved


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@dataclass
class StudentForwardOutput:
    """Result of a teacher-forced student forward for one record."""

    response_logits: torch.Tensor  # [1, T, V], connected to model parameters
    prompt_length: int
    full_length: int
    used_token_ids: tuple[int, ...]
    reencoded_token_ids: tuple[int, ...]
    decoded_text: str
    decoded_matches: bool


class StudentLogitsProvider(Protocol):
    @property
    def tokenizer_hash(self) -> str: ...

    def forward(self, record: Mapping[str, Any]) -> StudentForwardOutput: ...

    def named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]: ...

    def zero_grad(self) -> None: ...


def _all_named_parameters(provider: StudentLogitsProvider) -> Iterable[tuple[str, torch.Tensor]]:
    """All named parameters including tied duplicates, with a safe fallback."""

    getter = getattr(provider, "all_named_parameters", None)
    if getter is not None:
        yield from getter()
    else:
        yield from provider.named_parameters()


def detect_tied_parameter_groups(provider: StudentLogitsProvider) -> list[list[str]]:
    """Group parameter names that share storage (tied weights)."""

    by_storage: dict[int, list[str]] = {}
    for name, param in _all_named_parameters(provider):
        by_storage.setdefault(param.data_ptr(), []).append(name)
    return [sorted(names) for names in by_storage.values() if len(names) > 1]


def lm_head_embed_tied(tied_groups: Sequence[Sequence[str]]) -> bool:
    """True when an lm-head-like name and an embedding-like name share storage."""

    for names in tied_groups:
        has_head = any(("lm_head" in name) or ("output_embed" in name) for name in names)
        has_embed = any("embed" in name for name in names)
        if has_head and has_embed:
            return True
    return False


def _trainable_summary(provider: StudentLogitsProvider, *, max_names: int = 8) -> tuple[int, int, list[str]]:
    num_tensors = 0
    num_elements = 0
    names: list[str] = []
    for name, param in provider.named_parameters():
        num_tensors += 1
        num_elements += int(param.numel())
        if len(names) < max_names:
            names.append(name)
    return num_tensors, num_elements, names


def _finite_nonzero_grads(
    provider: StudentLogitsProvider, *, max_names: int = 8
) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for name, param in provider.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad = grad.detach().float()
        if bool(torch.isfinite(grad).all().item()) and float(grad.norm().item()) > 0.0:
            found.append((name, float(grad.norm().item())))
            if len(found) >= max_names:
                break
    return found


# ---------------------------------------------------------------------------
# Real Hugging Face student provider (transformers imported lazily)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealStudentConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    freeze_all_but_lm_head: bool = False
    max_prompt_length: int | None = None
    max_response_tokens: int | None = None
    condition: Condition = Condition.FULL


class HFStudentProvider:
    """Teacher-forced multimodal forward over a real Qwen-VL student model."""

    def __init__(self, model: Any, processor: Any, config: RealStudentConfig):
        self._model = model
        self._processor = processor
        self._tokenizer = getattr(processor, "tokenizer", processor)
        self._config = config
        self._torch_dtype = _DTYPES[config.dtype]

    @classmethod
    def load(cls, config: RealStudentConfig) -> "HFStudentProvider":
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except ImportError:  # pragma: no cover - older transformers
            from transformers import AutoModelForCausalLM as _AutoModel

        processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
        model = _AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=_DTYPES[config.dtype],
            trust_remote_code=True,
        )
        model.to(config.device)
        model.eval()

        if config.freeze_all_but_lm_head:
            for param in model.parameters():
                param.requires_grad_(False)
            head = model.get_output_embeddings()
            if head is None:
                raise ValueError("model has no output embeddings to keep trainable")
            for param in head.parameters():
                param.requires_grad_(True)
        else:
            for param in model.parameters():
                param.requires_grad_(True)
        return cls(model, processor, config)

    @property
    def tokenizer_hash(self) -> str:
        return tokenizer_fingerprint(self._tokenizer)

    def named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                yield name, param

    def all_named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        # remove_duplicate=False so tied weights (e.g. lm_head / embed_tokens)
        # surface under all of their names for tied-parameter detection.
        yield from self._model.named_parameters(remove_duplicate=False)

    def zero_grad(self) -> None:
        self._model.zero_grad(set_to_none=True)

    def forward(self, record: Mapping[str, Any]) -> StudentForwardOutput:
        from PIL import Image

        question = str(record["question"])
        response_text = str(record["response_text"])
        response_ids = tuple(int(item) for item in record["response_token_ids"])
        if self._config.max_response_tokens and len(response_ids) > self._config.max_response_tokens:
            raise ValueError(
                f"response has {len(response_ids)} tokens > max {self._config.max_response_tokens}"
            )

        condition_inputs = condition_inputs_from_record(record)
        rendered = render_teacher_prompt(self._config.condition, question, condition_inputs)
        image_path = condition_inputs.full_image.path
        image = Image.open(image_path).convert("RGB")

        text = self._processor.apply_chat_template(
            list(rendered.messages), tokenize=False, add_generation_prompt=True
        )
        proc = self._processor(text=[text], images=[image], return_tensors="pt")
        prompt_ids = proc["input_ids"]
        prompt_length = int(prompt_ids.shape[1])
        if self._config.max_prompt_length and prompt_length > self._config.max_prompt_length:
            raise ValueError(
                f"prompt has {prompt_length} tokens > max {self._config.max_prompt_length}"
            )

        response_tensor = torch.tensor([response_ids], dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, response_tensor], dim=1).to(self._config.device)
        attention_mask = torch.ones_like(full_ids)

        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in proc.items():
            if key in ("input_ids", "attention_mask"):
                continue
            if torch.is_floating_point(value):
                model_inputs[key] = value.to(self._config.device, self._torch_dtype)
            else:
                model_inputs[key] = value.to(self._config.device)
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = attention_mask

        outputs = self._model(**model_inputs)
        full_logits = outputs.logits
        response_logits = response_logit_slice(full_logits, prompt_length, len(response_ids))

        reencoded = tuple(
            int(item) for item in self._tokenizer.encode(response_text, add_special_tokens=False)
        )
        decoded = self._tokenizer.decode(response_ids)
        return StudentForwardOutput(
            response_logits=response_logits,
            prompt_length=prompt_length,
            full_length=int(full_ids.shape[1]),
            used_token_ids=response_ids,
            reencoded_token_ids=reencoded,
            decoded_text=decoded,
            decoded_matches=(decoded == response_text),
        )


# ---------------------------------------------------------------------------
# Per-record run + verification
# ---------------------------------------------------------------------------


@dataclass
class RealStudentResult:
    sample_uid: str
    tokenizer_hash_matches: bool
    seq_len: int
    num_response_tokens: int
    prompt_length: int
    full_length: int
    student_vocab_size: int
    logits_shape: tuple[int, ...]
    decoded_matches: bool
    reencoded_matches: bool
    loss_value: float
    loss_is_finite: bool
    backward_succeeded: bool
    grad_param_name: str | None
    grad_param_norm: float
    grad_is_finite: bool
    consumed_conditions: set[Condition]
    num_trainable_params: int = 0
    num_trainable_param_tensors: int = 0
    trainable_param_names: list[str] = field(default_factory=list)
    nonzero_grad_param_names: list[str] = field(default_factory=list)
    lm_head_embed_tied: bool = False
    tied_parameter_names: list[list[str]] = field(default_factory=list)
    expected_conditions: tuple[Condition, ...] = FOUR_CONDITIONS
    routing_mode: str = "chunk"
    chunk_format_valid: bool = True
    fallback_bad_chunk_rows: int = 0
    available_conditions: tuple[Condition, ...] = ()
    condition_token_weight_sums: dict[str, float] = field(default_factory=dict)
    condition_token_weight_nonzero_counts: dict[str, int] = field(default_factory=dict)
    chunk_condition_weight_sums: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    capability_weight_sums: dict[str, float] = field(default_factory=dict)
    consumed_capabilities: set[str] = field(default_factory=set)
    unused_capabilities: set[str] = field(default_factory=set)

    @property
    def unused_conditions(self) -> set[Condition]:
        return set(self.available_conditions) - set(self.consumed_conditions)

    @property
    def four_conditions_consumed(self) -> bool:
        return set(FOUR_CONDITIONS).issubset(self.consumed_conditions)

    @property
    def expected_conditions_consumed(self) -> bool:
        return set(self.expected_conditions).issubset(self.consumed_conditions)

    @property
    def passed(self) -> bool:
        return (
            self.tokenizer_hash_matches
            and self.seq_len == self.num_response_tokens
            and len(self.logits_shape) == 3
            and self.logits_shape[0] == 1
            and self.logits_shape[1] == self.seq_len
            and self.loss_is_finite
            and self.backward_succeeded
            and self.grad_param_name is not None
            and self.grad_is_finite
            and self.grad_param_norm > 0.0
            and self.expected_conditions_consumed
            and (
                self.routing_mode != "student_deficit_chunk_gated"
                or bool(self.consumed_capabilities)
            )
        )


def _first_finite_nonzero_grad(
    provider: StudentLogitsProvider,
) -> tuple[str | None, float, bool]:
    found = _finite_nonzero_grads(provider, max_names=1)
    if not found:
        return None, 0.0, False
    name, norm = found[0]
    return name, norm, True


def _uniform_condition_weights(
    teacher_scores: Mapping[Condition, Any],
    response_mask: torch.Tensor,
    *,
    expected_conditions: Sequence[Condition] | None = None,
) -> dict[Condition, torch.Tensor]:
    available = tuple(condition for condition in (expected_conditions or tuple(teacher_scores)) if condition in teacher_scores)
    if not available:
        available = tuple(teacher_scores)
    if not available:
        raise ValueError("uniform routing requires at least one available condition")
    value = 1.0 / len(available)
    weights = {
        condition: torch.zeros_like(response_mask, dtype=torch.float32)
        for condition in teacher_scores
    }
    for condition in available:
        weights[condition][response_mask.bool()] = value
    return weights


def _routing_diagnostics(
    *,
    weights: Mapping[Condition, torch.Tensor],
    chunk_masks: Mapping[str, torch.Tensor],
    teacher_scores: Mapping[Condition, Any],
) -> dict[str, Any]:
    condition_sums = {
        condition.value: float(weight.detach().float().sum().item())
        for condition, weight in weights.items()
    }
    condition_counts = {
        condition.value: int((weight.detach().float() > 0).sum().item())
        for condition, weight in weights.items()
    }
    chunk_sums: dict[str, dict[str, float]] = {}
    for chunk_name, mask in chunk_masks.items():
        chunk_mask = mask.detach().float()
        chunk_sums[chunk_name] = {
            condition.value: float((weight.detach().float() * chunk_mask).sum().item())
            for condition, weight in weights.items()
        }
    return {
        "available_conditions": tuple(teacher_scores),
        "condition_token_weight_sums": condition_sums,
        "condition_token_weight_nonzero_counts": condition_counts,
        "chunk_condition_weight_sums": chunk_sums,
    }


def _capability_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    scores = record.get("capability_scores", {})
    if not isinstance(scores, Mapping):
        return {"capability_weight_sums": {}, "consumed_capabilities": set(), "unused_capabilities": set()}
    sums: dict[str, float] = {}
    consumed: set[str] = set()
    unused: set[str] = set()
    for capability, block in scores.items():
        if not isinstance(block, Mapping):
            continue
        weights = block.get("final_token_weight", [])
        total = sum(float(value) for value in weights) if isinstance(weights, Sequence) else 0.0
        sums[str(capability)] = total
        if total > 0:
            consumed.add(str(capability))
        else:
            unused.add(str(capability))
    return {
        "capability_weight_sums": sums,
        "consumed_capabilities": consumed,
        "unused_capabilities": unused,
    }


def _condition_weights_for_record(
    *,
    routing_mode: str,
    teacher_scores: Mapping[Condition, Any],
    chunk_masks: Mapping[str, torch.Tensor],
    router_config: RouterConfig,
    response_mask: torch.Tensor,
    format_valid: torch.Tensor,
    expected_conditions: Sequence[Condition],
    capability_scores: Mapping[str, Any] | None = None,
) -> tuple[dict[Condition, torch.Tensor], bool]:
    if routing_mode == "uniform_all_conditions":
        return _uniform_condition_weights(
            teacher_scores,
            response_mask,
            expected_conditions=expected_conditions,
        ), False
    if routing_mode in {"chunk_gated", "chunk_gated_primary", "chunk_gated_contrastive"} and not bool(format_valid.bool().all().item()):
        return _uniform_condition_weights(
            teacher_scores,
            response_mask,
            expected_conditions=expected_conditions,
        ), True
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"routing_mode must be one of {ROUTING_MODES}")
    effective_router = router_config
    if routing_mode == "chunk_gated" and router_config.mode != "chunk_gated":
        effective_router = SIX_C_CHUNK_GATED_ROUTER
    elif routing_mode == "chunk_gated_primary":
        effective_router = RouterConfig(
            mode="chunk_gated_primary",
            invalid_format_condition=router_config.invalid_format_condition,
            chunk_condition_routing=router_config.chunk_condition_routing,
            delta_threshold=router_config.delta_threshold,
            entropy_max=router_config.entropy_max,
            topk_mass_min=router_config.topk_mass_min,
            max_conditions_per_token=router_config.max_conditions_per_token,
            normalize_weights_per_token=router_config.normalize_weights_per_token,
        )
    elif routing_mode == "chunk_gated_contrastive":
        effective_router = RouterConfig(
            mode="chunk_gated_contrastive",
            invalid_format_condition=router_config.invalid_format_condition,
            delta_threshold=router_config.delta_threshold,
            entropy_max=router_config.entropy_max,
            topk_mass_min=router_config.topk_mass_min,
            max_conditions_per_token=max(router_config.max_conditions_per_token, 4),
            normalize_weights_per_token=router_config.normalize_weights_per_token,
        )
    elif routing_mode == "student_deficit_chunk_gated":
        effective_router = STUDENT_DEFICIT_ROUTER
    return route_condition_weights(
        signals={"capability_scores": capability_scores or {}},
        chunk_masks=chunk_masks,
        router_config=effective_router,
        response_mask=response_mask,
        available_conditions=tuple(teacher_scores),
        format_valid=format_valid,
    ), False


def run_real_student_record(
    record: Mapping[str, Any],
    provider: StudentLogitsProvider,
    *,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    expected_conditions: tuple[Condition, ...] = FOUR_CONDITIONS,
    routing_mode: str = "chunk",
    require_decoded_match: bool = True,
    device: torch.device | str = "cpu",
) -> RealStudentResult:
    """Run a teacher-forced real-student forward and the FC-OPD loss for one record."""

    offline_hash = str(record.get("tokenizer_hash", ""))
    tokenizer_hash_matches = provider.tokenizer_hash == offline_hash
    if not tokenizer_hash_matches:
        raise ValueError(
            "student tokenizer hash does not match the offline tokenizer_hash: "
            f"{provider.tokenizer_hash} != {offline_hash}"
        )

    tensors = offline_record_to_tensors(record, device=device)
    provider.zero_grad()
    output = provider.forward(record)
    logits = output.response_logits

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"student logits must have shape [1, T, V], got {tuple(logits.shape)}")
    if logits.shape[1] != tensors.seq_len:
        raise ValueError(
            f"response logits T={logits.shape[1]} != condition-score T={tensors.seq_len}"
        )
    if logits.shape[-1] <= tensors.vocab_floor - 1:
        raise ValueError("student vocab is smaller than the largest teacher token id")
    if require_decoded_match and not output.decoded_matches:
        raise ValueError("decoded response text does not match the offline record")

    # Real student logits can live on a different device (e.g. cuda) than the
    # CPU-built offline teacher tensors. Move every loss-side tensor onto the
    # student logits device so compute_fc_opd_loss stays single-device.
    target_device = logits.device
    teacher_scores = teacher_scores_to_device(tensors.teacher_scores, target_device)
    chunk_masks = {name: mask.to(target_device) for name, mask in tensors.chunk_masks.items()}
    response_mask = tensors.response_mask.to(target_device)
    format_valid = tensors.format_valid.to(target_device)

    weights, fallback_bad_chunk = _condition_weights_for_record(
        routing_mode=routing_mode,
        teacher_scores=teacher_scores,
        chunk_masks=chunk_masks,
        router_config=router_config,
        response_mask=response_mask,
        format_valid=format_valid,
        expected_conditions=expected_conditions,
        capability_scores=record.get("capability_scores") if isinstance(record.get("capability_scores"), Mapping) else None,
    )
    loss, metrics = compute_fc_opd_loss(
        logits,
        teacher_scores,
        chunk_masks,
        weights,
        response_mask,
        loss_config or FCOPDLossConfig(),
    )

    loss_is_finite = bool(torch.isfinite(loss).all().item())
    backward_succeeded = False
    try:
        loss.backward()
        backward_succeeded = True
    except RuntimeError:
        backward_succeeded = False

    nonzero_grads = _finite_nonzero_grads(provider)
    grad_name = nonzero_grads[0][0] if nonzero_grads else None
    grad_norm = nonzero_grads[0][1] if nonzero_grads else 0.0
    grad_finite = bool(nonzero_grads)
    consumed = {
        condition
        for condition in tensors.teacher_scores
        if metrics.get(f"selection/{condition.value}", torch.zeros(())).item() > 0.0
    }
    diagnostics = _routing_diagnostics(
        weights=weights,
        chunk_masks=chunk_masks,
        teacher_scores=teacher_scores,
    )
    capability_diagnostics = _capability_diagnostics(record)

    num_tensors, num_elements, trainable_names = _trainable_summary(provider)
    tied_groups = detect_tied_parameter_groups(provider)

    return RealStudentResult(
        sample_uid=str(record.get("sample_uid", "unknown")),
        tokenizer_hash_matches=tokenizer_hash_matches,
        seq_len=tensors.seq_len,
        num_response_tokens=tensors.num_response_tokens,
        prompt_length=output.prompt_length,
        full_length=output.full_length,
        student_vocab_size=int(logits.shape[-1]),
        logits_shape=tuple(int(dim) for dim in logits.shape),
        decoded_matches=output.decoded_matches,
        reencoded_matches=output.reencoded_token_ids == output.used_token_ids,
        loss_value=float(loss.detach().float().item()),
        loss_is_finite=loss_is_finite,
        backward_succeeded=backward_succeeded,
        grad_param_name=grad_name,
        grad_param_norm=grad_norm,
        grad_is_finite=grad_finite,
        consumed_conditions=consumed,
        num_trainable_params=num_elements,
        num_trainable_param_tensors=num_tensors,
        trainable_param_names=trainable_names,
        nonzero_grad_param_names=[name for name, _ in nonzero_grads],
        lm_head_embed_tied=lm_head_embed_tied(tied_groups),
        tied_parameter_names=tied_groups,
        expected_conditions=expected_conditions,
        routing_mode=routing_mode,
        chunk_format_valid=bool(format_valid.bool().all().item()),
        fallback_bad_chunk_rows=1 if fallback_bad_chunk else 0,
        available_conditions=diagnostics["available_conditions"],
        condition_token_weight_sums=diagnostics["condition_token_weight_sums"],
        condition_token_weight_nonzero_counts=diagnostics["condition_token_weight_nonzero_counts"],
        chunk_condition_weight_sums=diagnostics["chunk_condition_weight_sums"],
        metrics={key: float(value.item()) for key, value in metrics.items()},
        capability_weight_sums=capability_diagnostics["capability_weight_sums"],
        consumed_capabilities=capability_diagnostics["consumed_capabilities"],
        unused_capabilities=capability_diagnostics["unused_capabilities"],
    )


@dataclass
class RealStudentSmokeReport:
    results: list[RealStudentResult] = field(default_factory=list)

    @property
    def num_records(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> bool:
        return self.num_records > 0 and all(result.passed for result in self.results)


def run_real_student_smoke(
    records: Iterable[Mapping[str, Any]],
    provider: StudentLogitsProvider,
    *,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    expected_conditions: tuple[Condition, ...] = FOUR_CONDITIONS,
    routing_mode: str = "chunk",
    require_decoded_match: bool = True,
    device: torch.device | str = "cpu",
    expect_capability_scores: bool = False,
) -> RealStudentSmokeReport:
    report = RealStudentSmokeReport()
    for record in records:
        if expect_capability_scores and not isinstance(record.get("capability_scores"), Mapping):
            raise ValueError(f"{record.get('sample_uid', 'unknown')}: missing capability_scores")
        report.results.append(
            run_real_student_record(
                record,
                provider,
                router_config=router_config,
                loss_config=loss_config,
                expected_conditions=expected_conditions,
                routing_mode=routing_mode,
                require_decoded_match=require_decoded_match,
                device=device,
            )
        )
    return report


# ---------------------------------------------------------------------------
# Real optimizer-step smoke
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealMinTrainStep:
    step: int
    loss: float
    loss_is_finite: bool
    grad_norm: float
    grad_is_finite: bool
    param_delta_norm: float
    consumed_conditions: set[Condition]
    consumed_capabilities: set[str] = field(default_factory=set)


@dataclass
class RealMinTrainReport:
    num_records: int
    num_steps: int
    learning_rate: float
    tokenizer_hash_matches: bool
    decoded_matches: bool
    reencoded_matches: bool
    require_decoded_match: bool
    num_trainable_params: int
    num_trainable_param_tensors: int
    trainable_param_names: list[str] = field(default_factory=list)
    nonzero_grad_param_names: list[str] = field(default_factory=list)
    lm_head_embed_tied: bool = False
    tied_parameter_names: list[list[str]] = field(default_factory=list)
    expected_conditions: tuple[Condition, ...] = FOUR_CONDITIONS
    routing_mode: str = "chunk"
    skipped_bad_chunk_rows: int = 0
    fallback_bad_chunk_rows: int = 0
    chunk_gated_valid_rows: int = 0
    available_conditions: tuple[Condition, ...] = ()
    condition_token_weight_sums: dict[str, float] = field(default_factory=dict)
    condition_token_weight_nonzero_counts: dict[str, int] = field(default_factory=dict)
    chunk_condition_weight_sums: dict[str, dict[str, float]] = field(default_factory=dict)
    capability_weight_sums: dict[str, float] = field(default_factory=dict)
    consumed_capabilities: set[str] = field(default_factory=set)
    unused_capabilities: set[str] = field(default_factory=set)
    steps: list[RealMinTrainStep] = field(default_factory=list)

    @property
    def all_loss_finite(self) -> bool:
        return all(step.loss_is_finite for step in self.steps)

    @property
    def all_grads_finite(self) -> bool:
        return all(step.grad_is_finite for step in self.steps)

    @property
    def every_step_updates_params(self) -> bool:
        return all(step.param_delta_norm > 0.0 for step in self.steps)

    @property
    def consumed_conditions(self) -> set[Condition]:
        consumed: set[Condition] = set()
        for step in self.steps:
            consumed |= step.consumed_conditions
        return consumed

    @property
    def four_conditions_consumed(self) -> bool:
        return set(FOUR_CONDITIONS).issubset(self.consumed_conditions)

    @property
    def expected_conditions_consumed(self) -> bool:
        return set(self.expected_conditions).issubset(self.consumed_conditions)

    @property
    def unused_conditions(self) -> set[Condition]:
        return set(self.available_conditions) - set(self.consumed_conditions)

    @property
    def loss_decreased(self) -> bool:
        return bool(self.steps) and self.steps[-1].loss < self.steps[0].loss

    @property
    def passed(self) -> bool:
        return (
            self.num_records > 0
            and self.num_steps > 0
            and self.tokenizer_hash_matches
            and (self.decoded_matches or not self.require_decoded_match)
            and self.all_loss_finite
            and self.all_grads_finite
            and self.every_step_updates_params
            and self.expected_conditions_consumed
            and (
                self.routing_mode != "student_deficit_chunk_gated"
                or bool(self.consumed_capabilities)
            )
        )


@dataclass
class _PreparedRecord:
    record: Mapping[str, Any]
    seq_len: int
    vocab_floor: int
    teacher_scores: dict[Condition, Any] | None = None
    chunk_masks: dict[str, torch.Tensor] | None = None
    response_mask: torch.Tensor | None = None
    format_valid: torch.Tensor | None = None
    device: torch.device | None = None


def _params_grad_norm(params: Sequence[torch.Tensor]) -> tuple[float, bool]:
    norms = []
    finite = True
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        norms.append(grad.norm())
        finite &= bool(torch.isfinite(grad).all().item())
    if not norms:
        return 0.0, finite
    return float(torch.stack(norms).norm().item()), finite


def run_real_student_min_train(
    records: Sequence[Mapping[str, Any]],
    provider: StudentLogitsProvider,
    *,
    num_steps: int = 3,
    learning_rate: float = 1e-4,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    expected_conditions: tuple[Condition, ...] = FOUR_CONDITIONS,
    routing_mode: str = "chunk",
    require_decoded_match: bool = True,
    device: torch.device | str = "cpu",
    expect_capability_scores: bool = False,
) -> RealMinTrainReport:
    """Run a few real optimizer steps and verify trainable parameters change.

    Re-runs the teacher-forced forward each step (logits depend on the current
    parameters), computes the FC-OPD loss against the recorded teacher scores,
    backpropagates, and steps Adam over the trainable parameters. The recorded
    offline tensors are device-moved once and reused.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not records:
        raise ValueError("no offline-score records were provided")
    if expect_capability_scores:
        missing = [str(record.get("sample_uid", "unknown")) for record in records if not isinstance(record.get("capability_scores"), Mapping)]
        if missing:
            raise ValueError(f"missing capability_scores in records: {missing[:5]}")
    loss_config = loss_config or FCOPDLossConfig()

    tokenizer_hash_matches = True
    for record in records:
        if provider.tokenizer_hash != str(record.get("tokenizer_hash", "")):
            tokenizer_hash_matches = False
    if not tokenizer_hash_matches:
        raise ValueError("student tokenizer hash does not match an offline tokenizer_hash")

    prepared = []
    for record in records:
        cpu_tensors = offline_record_to_tensors(record, device="cpu")
        prepared.append(
            _PreparedRecord(
                record=record,
                seq_len=cpu_tensors.seq_len,
                vocab_floor=cpu_tensors.vocab_floor,
            )
        )

    params = [param for _, param in provider.named_parameters()]
    if not params:
        raise ValueError("provider exposes no trainable parameters")
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    num_tensors, num_elements, trainable_names = _trainable_summary(provider)
    tied_groups = detect_tied_parameter_groups(provider)
    report = RealMinTrainReport(
        num_records=len(records),
        num_steps=num_steps,
        learning_rate=learning_rate,
        tokenizer_hash_matches=tokenizer_hash_matches,
        decoded_matches=True,
        reencoded_matches=True,
        require_decoded_match=require_decoded_match,
        num_trainable_params=num_elements,
        num_trainable_param_tensors=num_tensors,
        trainable_param_names=trainable_names,
        lm_head_embed_tied=lm_head_embed_tied(tied_groups),
        tied_parameter_names=tied_groups,
        expected_conditions=expected_conditions,
        routing_mode=routing_mode,
    )

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float32)
        consumed: set[Condition] = set()
        consumed_capabilities: set[str] = set()

        for prep in prepared:
            output = provider.forward(prep.record)
            logits = output.response_logits
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != prep.seq_len:
                raise ValueError(
                    f"student logits must be [1, {prep.seq_len}, V], got {tuple(logits.shape)}"
                )
            if step == 0:
                report.decoded_matches &= output.decoded_matches
                report.reencoded_matches &= output.reencoded_token_ids == output.used_token_ids
                if require_decoded_match and not output.decoded_matches:
                    raise ValueError("decoded response text does not match the offline record")

            if prep.teacher_scores is None or prep.device != logits.device:
                cpu_tensors = offline_record_to_tensors(prep.record, device="cpu")
                prep.device = logits.device
                prep.teacher_scores = teacher_scores_to_device(cpu_tensors.teacher_scores, logits.device)
                prep.chunk_masks = {
                    name: mask.to(logits.device) for name, mask in cpu_tensors.chunk_masks.items()
                }
                prep.response_mask = cpu_tensors.response_mask.to(logits.device)
                prep.format_valid = cpu_tensors.format_valid.to(logits.device)

            weights, fallback_bad_chunk = _condition_weights_for_record(
                routing_mode=routing_mode,
                teacher_scores=prep.teacher_scores,
                chunk_masks=prep.chunk_masks,
                router_config=router_config,
                response_mask=prep.response_mask,
                format_valid=prep.format_valid,
                expected_conditions=expected_conditions,
                capability_scores=prep.record.get("capability_scores") if isinstance(prep.record.get("capability_scores"), Mapping) else None,
            )
            if step == 0 and routing_mode == "chunk_gated":
                if fallback_bad_chunk:
                    report.fallback_bad_chunk_rows += 1
                else:
                    report.chunk_gated_valid_rows += 1
            elif step == 0 and routing_mode in {"chunk_gated_primary", "chunk_gated_contrastive"}:
                if fallback_bad_chunk:
                    report.fallback_bad_chunk_rows += 1
                else:
                    report.chunk_gated_valid_rows += 1
            loss, metrics = compute_fc_opd_loss(
                logits,
                prep.teacher_scores,
                prep.chunk_masks,
                weights,
                prep.response_mask,
                loss_config,
            )
            total_loss = total_loss.to(loss.device) + loss
            consumed |= {
                condition
                for condition in prep.teacher_scores
                if metrics.get(f"selection/{condition.value}", torch.zeros(())).item() > 0.0
            }
            if step == 0:
                diagnostics = _routing_diagnostics(
                    weights=weights,
                    chunk_masks=prep.chunk_masks,
                    teacher_scores=prep.teacher_scores,
                )
                report.available_conditions = tuple(
                    dict.fromkeys((*report.available_conditions, *diagnostics["available_conditions"]))
                )
                for condition, value in diagnostics["condition_token_weight_sums"].items():
                    report.condition_token_weight_sums[condition] = (
                        report.condition_token_weight_sums.get(condition, 0.0) + float(value)
                    )
                for condition, value in diagnostics["condition_token_weight_nonzero_counts"].items():
                    report.condition_token_weight_nonzero_counts[condition] = (
                        report.condition_token_weight_nonzero_counts.get(condition, 0) + int(value)
                    )
                for chunk_name, condition_values in diagnostics["chunk_condition_weight_sums"].items():
                    chunk_report = report.chunk_condition_weight_sums.setdefault(chunk_name, {})
                    for condition, value in condition_values.items():
                        chunk_report[condition] = chunk_report.get(condition, 0.0) + float(value)
                capability_diagnostics = _capability_diagnostics(prep.record)
                for capability, value in capability_diagnostics["capability_weight_sums"].items():
                    report.capability_weight_sums[capability] = (
                        report.capability_weight_sums.get(capability, 0.0) + float(value)
                    )
                report.consumed_capabilities |= capability_diagnostics["consumed_capabilities"]
                report.unused_capabilities |= capability_diagnostics["unused_capabilities"]
            capability_diagnostics = _capability_diagnostics(prep.record)
            consumed_capabilities |= capability_diagnostics["consumed_capabilities"]

        loss_is_finite = bool(torch.isfinite(total_loss).all().item())
        total_loss.backward()
        grad_norm, grad_is_finite = _params_grad_norm(params)

        snapshot = [param.detach().clone() for param in params]
        optimizer.step()
        delta = torch.stack(
            [(param.detach() - old).float().norm() for param, old in zip(params, snapshot, strict=True)]
        ).norm()

        if step == 0:
            report.nonzero_grad_param_names = [name for name, _ in _finite_nonzero_grads(provider)]

        report.steps.append(
            RealMinTrainStep(
                step=step,
                loss=float(total_loss.detach().float().item()),
                loss_is_finite=loss_is_finite,
                grad_norm=grad_norm,
                grad_is_finite=grad_is_finite,
                param_delta_norm=float(delta.item()),
                consumed_conditions=consumed,
                consumed_capabilities=consumed_capabilities,
            )
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _result_to_json(result: RealStudentResult) -> dict[str, Any]:
    return {
        "sample_uid": result.sample_uid,
        "tokenizer_hash_matches": result.tokenizer_hash_matches,
        "T": result.seq_len,
        "prompt_length": result.prompt_length,
        "full_length": result.full_length,
        "logits_shape": list(result.logits_shape),
        "student_vocab_size": result.student_vocab_size,
        "decoded_matches": result.decoded_matches,
        "reencoded_matches": result.reencoded_matches,
        "loss": result.loss_value,
        "loss_is_finite": result.loss_is_finite,
        "backward_succeeded": result.backward_succeeded,
        "grad_param_name": result.grad_param_name,
        "grad_param_norm": result.grad_param_norm,
        "grad_is_finite": result.grad_is_finite,
        "num_trainable_params": result.num_trainable_params,
        "num_trainable_param_tensors": result.num_trainable_param_tensors,
        "trainable_param_names": result.trainable_param_names,
        "nonzero_grad_param_names": result.nonzero_grad_param_names,
        "lm_head_embed_tied": result.lm_head_embed_tied,
        "tied_parameter_names": result.tied_parameter_names,
        "expected_conditions": [condition.value for condition in result.expected_conditions],
        "consumed_conditions": sorted(c.value for c in result.consumed_conditions),
        "available_conditions": sorted(c.value for c in result.available_conditions),
        "unused_conditions": sorted(c.value for c in result.unused_conditions),
        "routing_mode": result.routing_mode,
        "chunk_format_valid": result.chunk_format_valid,
        "fallback_bad_chunk_rows": result.fallback_bad_chunk_rows,
        "condition_token_weight_sums": result.condition_token_weight_sums,
        "condition_token_weight_nonzero_counts": result.condition_token_weight_nonzero_counts,
        "chunk_condition_weight_sums": result.chunk_condition_weight_sums,
        "capability_weight_sums": result.capability_weight_sums,
        "consumed_capabilities": sorted(result.consumed_capabilities),
        "unused_capabilities": sorted(result.unused_capabilities),
        "four_conditions_consumed": result.four_conditions_consumed,
        "expected_conditions_consumed": result.expected_conditions_consumed,
        "passed": result.passed,
    }


def _condition_set(value: str) -> tuple[RouterConfig, tuple[Condition, ...]]:
    if value == "4c":
        return FOUR_CONDITION_ROUTER, FOUR_CONDITIONS
    if value == "4c-clean":
        return FOUR_CLEAN_CONDITION_ROUTER, FOUR_CLEAN_CONDITIONS
    if value == "6c-solve":
        return SIX_C_CHUNK_GATED_ROUTER, SIX_C_SOLVE_CONDITIONS
    if value == "2c":
        return TWO_CONDITION_ROUTER, TWO_CONDITIONS
    raise argparse.ArgumentTypeError("condition set must be '4c', '4c-clean', '6c-solve', or '2c'")


def _expected_conditions_for_routing(
    score_conditions: tuple[Condition, ...],
    *,
    routing_mode: str,
    router_config: RouterConfig,
) -> tuple[Condition, ...]:
    if routing_mode == "student_deficit_chunk_gated":
        return tuple()
    if routing_mode not in {"chunk_gated", "chunk_gated_primary", "chunk_gated_contrastive"}:
        return score_conditions
    available = set(score_conditions)
    expected: list[Condition] = []
    routing = (
        {
            "visible_evidence": (Condition.FULL, Condition.DEGRADED, Condition.TASK_VISIBLE, Condition.FREE),
            "diagram_inference": (Condition.TASK_INFER, Condition.TASK_VISIBLE, Condition.FULL, Condition.DEGRADED),
            "reasoning": (Condition.TASK_SOLVE, Condition.TASK_INFER),
            "answer": (Condition.TASK_SOLVE, Condition.TASK_INFER),
        }
        if routing_mode == "chunk_gated_contrastive"
        else router_config.chunk_condition_routing
    )
    max_conditions = max(router_config.max_conditions_per_token, 4) if routing_mode == "chunk_gated_contrastive" else router_config.max_conditions_per_token
    for candidates in routing.values():
        for condition_like in list(candidates)[:max_conditions]:
            condition = Condition(condition_like)
            if condition in available and condition not in expected:
                expected.append(condition)
            elif condition is Condition.TASK and Condition.TASK_VISIBLE in available and Condition.TASK_VISIBLE not in expected:
                expected.append(Condition.TASK_VISIBLE)
            elif condition is Condition.TASK_VISIBLE and Condition.TASK in available and Condition.TASK not in expected:
                expected.append(Condition.TASK)
    if not expected:
        return score_conditions
    return tuple(expected)


def main(argv: Sequence[str] | None = None) -> int:
    default_model = os.path.join(
        os.environ.get("DTOPD_MODEL_ROOT", ""), "Qwen3-VL-4B-Instruct"
    )
    parser = argparse.ArgumentParser(
        description="Real student-logits adapter smoke for the offline FC-OPD loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True, help="offline-score JSONL path")
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=tuple(_DTYPES))
    parser.add_argument("--condition-set", choices=("4c", "4c-clean", "6c-solve", "2c"), default="4c")
    parser.add_argument("--routing-mode", choices=ROUTING_MODES, default="chunk")
    parser.add_argument("--expect-capability-scores", action="store_true")
    parser.add_argument("--freeze-all-but-lm-head", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument(
        "--allow-retokenize-mismatch",
        action="store_true",
        help="treat a decoded-text mismatch as a warning instead of a failure",
    )
    args = parser.parse_args(argv)

    records = load_offline_score_records(args.scores)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"FAIL: no records found in {args.scores}", file=sys.stderr)
        return 1

    config = RealStudentConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        freeze_all_but_lm_head=args.freeze_all_but_lm_head,
        max_prompt_length=args.max_prompt_length,
        max_response_tokens=args.max_response_tokens,
    )
    provider = HFStudentProvider.load(config)
    router_config, score_conditions = _condition_set(args.condition_set)
    expected_conditions = _expected_conditions_for_routing(
        score_conditions,
        routing_mode=args.routing_mode,
        router_config=router_config,
    )

    report = run_real_student_smoke(
        records,
        provider,
        router_config=router_config,
        expected_conditions=expected_conditions,
        routing_mode=args.routing_mode,
        require_decoded_match=not args.allow_retokenize_mismatch,
        device="cpu",
        expect_capability_scores=args.expect_capability_scores,
    )

    for result in report.results:
        print(json.dumps(_result_to_json(result)))
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "condition_set": args.condition_set,
                "routing_mode": args.routing_mode,
                "num_records": report.num_records,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    if not report.passed:
        print("FAIL: real student-logits smoke did not pass", file=sys.stderr)
        return 1
    print("PASS: real student-logits smoke is valid")
    return 0


def min_train_main(argv: Sequence[str] | None = None) -> int:
    default_model = os.path.join(
        os.environ.get("DTOPD_MODEL_ROOT", ""), "Qwen3-VL-4B-Instruct"
    )
    parser = argparse.ArgumentParser(
        description="Real student optimizer-step smoke for the offline FC-OPD loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True, help="offline-score JSONL path")
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=tuple(_DTYPES))
    parser.add_argument("--condition-set", choices=("4c", "4c-clean", "6c-solve", "2c"), default="4c")
    parser.add_argument("--routing-mode", choices=ROUTING_MODES, default="chunk")
    parser.add_argument("--expect-capability-scores", action="store_true")
    parser.add_argument(
        "--freeze-all-but-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only the LM head (and any tied embedding) trainable",
    )
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument(
        "--allow-retokenize-mismatch",
        action="store_true",
        help="treat a decoded-text mismatch as a warning instead of a failure",
    )
    args = parser.parse_args(argv)

    records = load_offline_score_records(args.scores)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"FAIL: no records found in {args.scores}", file=sys.stderr)
        return 1

    config = RealStudentConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        freeze_all_but_lm_head=args.freeze_all_but_lm_head,
        max_prompt_length=args.max_prompt_length,
        max_response_tokens=args.max_response_tokens,
    )
    provider = HFStudentProvider.load(config)
    router_config, score_conditions = _condition_set(args.condition_set)
    expected_conditions = _expected_conditions_for_routing(
        score_conditions,
        routing_mode=args.routing_mode,
        router_config=router_config,
    )

    report = run_real_student_min_train(
        records,
        provider,
        num_steps=args.steps,
        learning_rate=args.lr,
        router_config=router_config,
        expected_conditions=expected_conditions,
        routing_mode=args.routing_mode,
        require_decoded_match=not args.allow_retokenize_mismatch,
        expect_capability_scores=args.expect_capability_scores,
    )

    for step in report.steps:
        print(
            json.dumps(
                {
                    "step": step.step,
                    "loss": step.loss,
                    "grad_norm": step.grad_norm,
                    "param_delta_norm": step.param_delta_norm,
                    "consumed_conditions": sorted(c.value for c in step.consumed_conditions),
                    "consumed_capabilities": sorted(step.consumed_capabilities),
                }
            )
        )

    summary = {
        "model_path": args.model_path,
        "condition_set": args.condition_set,
        "routing_mode": args.routing_mode,
        "num_records": report.num_records,
        "num_steps": report.num_steps,
        "learning_rate": report.learning_rate,
        "tokenizer_hash_matches": report.tokenizer_hash_matches,
        "decoded_matches": report.decoded_matches,
        "reencoded_matches": report.reencoded_matches,
        "num_trainable_params": report.num_trainable_params,
        "num_trainable_param_tensors": report.num_trainable_param_tensors,
        "trainable_param_names": report.trainable_param_names,
        "nonzero_grad_param_names": report.nonzero_grad_param_names,
        "lm_head_embed_tied": report.lm_head_embed_tied,
        "tied_parameter_names": report.tied_parameter_names,
        "all_loss_finite": report.all_loss_finite,
        "all_grads_finite": report.all_grads_finite,
        "every_step_updates_params": report.every_step_updates_params,
        "consumed_conditions": sorted(c.value for c in report.consumed_conditions),
        "available_conditions": sorted(c.value for c in report.available_conditions),
        "unused_conditions": sorted(c.value for c in report.unused_conditions),
        "four_conditions_consumed": report.four_conditions_consumed,
        "expected_conditions": [condition.value for condition in report.expected_conditions],
        "expected_conditions_consumed": report.expected_conditions_consumed,
        "condition_token_weight_sums": report.condition_token_weight_sums,
        "condition_token_weight_nonzero_counts": report.condition_token_weight_nonzero_counts,
        "chunk_condition_weight_sums": report.chunk_condition_weight_sums,
        "capability_weight_sums": report.capability_weight_sums,
        "consumed_capabilities": sorted(report.consumed_capabilities),
        "unused_capabilities": sorted(report.unused_capabilities),
        "grouped_loss_values": {},
        "skipped_bad_chunk_rows": report.skipped_bad_chunk_rows,
        "fallback_bad_chunk_rows": report.fallback_bad_chunk_rows,
        "chunk_gated_valid_rows": report.chunk_gated_valid_rows,
        "initial_loss": report.steps[0].loss,
        "final_loss": report.steps[-1].loss,
        "loss_decreased": report.loss_decreased,
        "passed": report.passed,
    }
    print(json.dumps(summary, indent=2))
    if not report.passed:
        print("FAIL: real student optimizer-step smoke did not pass", file=sys.stderr)
        return 1
    print("PASS: real student optimizer-step smoke is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
