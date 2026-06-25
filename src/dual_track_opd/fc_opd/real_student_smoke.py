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
from dataclasses import dataclass, field
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
        verified_facts=None if raw.get("verified_facts") is None else str(raw["verified_facts"]),
        verified_facts_source=(
            None if raw.get("verified_facts_source") is None else str(raw["verified_facts_source"])
        ),
    )
    inputs.validate()
    return inputs


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
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def four_conditions_consumed(self) -> bool:
        return set(FOUR_CONDITIONS).issubset(self.consumed_conditions)

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
            and self.four_conditions_consumed
        )


def _first_finite_nonzero_grad(
    provider: StudentLogitsProvider,
) -> tuple[str | None, float, bool]:
    for name, param in provider.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad = grad.detach().float()
        finite = bool(torch.isfinite(grad).all().item())
        norm = float(grad.norm().item())
        if finite and norm > 0.0:
            return name, norm, True
    return None, 0.0, False


def run_real_student_record(
    record: Mapping[str, Any],
    provider: StudentLogitsProvider,
    *,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
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

    weights = route_condition_weights(
        signals={},
        chunk_masks=tensors.chunk_masks,
        router_config=router_config,
        response_mask=tensors.response_mask,
        available_conditions=tuple(tensors.teacher_scores),
        format_valid=tensors.format_valid,
    )
    loss, metrics = compute_fc_opd_loss(
        logits,
        tensors.teacher_scores,
        tensors.chunk_masks,
        weights,
        tensors.response_mask,
        loss_config or FCOPDLossConfig(),
    )

    loss_is_finite = bool(torch.isfinite(loss).all().item())
    backward_succeeded = False
    try:
        loss.backward()
        backward_succeeded = True
    except RuntimeError:
        backward_succeeded = False

    grad_name, grad_norm, grad_finite = _first_finite_nonzero_grad(provider)
    consumed = {
        condition
        for condition in tensors.teacher_scores
        if metrics.get(f"selection/{condition.value}", torch.zeros(())).item() > 0.0
    }

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
        metrics={key: float(value.item()) for key, value in metrics.items()},
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
    require_decoded_match: bool = True,
    device: torch.device | str = "cpu",
) -> RealStudentSmokeReport:
    report = RealStudentSmokeReport()
    for record in records:
        report.results.append(
            run_real_student_record(
                record,
                provider,
                router_config=router_config,
                loss_config=loss_config,
                require_decoded_match=require_decoded_match,
                device=device,
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
        "consumed_conditions": sorted(c.value for c in result.consumed_conditions),
        "four_conditions_consumed": result.four_conditions_consumed,
        "passed": result.passed,
    }


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

    report = run_real_student_smoke(
        records,
        provider,
        require_decoded_match=not args.allow_retokenize_mismatch,
        device="cpu",
    )

    for result in report.results:
        print(json.dumps(_result_to_json(result)))
    print(
        json.dumps(
            {
                "model_path": args.model_path,
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


if __name__ == "__main__":
    raise SystemExit(main())
