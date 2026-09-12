"""Student-rollout FC-OPD signal audit.

This module generates fixed student responses first, then teacher-force scores
those exact response token IDs under the requested conditions. It is the
OPD-compatible audit path; the fixed-response dataset audit remains a
protocol/path check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .conditions import Condition, ConditionInputs
from .dataset_adapters import load_normalized_records
from .dataset_signal_audit import (
    DatasetAuditResult,
    build_audit_condition_inputs,
    compute_pairwise_kd_gradient_cosines,
    hash_text,
    hash_token_ids,
    materialize_gaussian_blur,
    summarize_audit_samples,
)
from .offline_scoring import DEFAULT_TEACHER_URL, derive_degraded_path
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint
from .signal_decomposer import TeacherTopK, compute_condition_signals
from .teacher_prompts import render_teacher_prompt


class StudentRolloutTokenizer(Protocol):
    def encode(self, text: str, **kwargs: object) -> list[int]: ...

    def get_vocab(self) -> Mapping[str, int]: ...


class StudentRolloutGenerator(Protocol):
    tokenizer: StudentRolloutTokenizer

    def generate(
        self,
        *,
        question: str,
        image_path: str,
        prompt_text: str,
        seed: int,
    ) -> str: ...


ROLLOUT_RESPONSE_FORMATS = ("answer_only", "fc_opd_structured", "fc_opd_structured_v2")
STRUCTURED_ROLLOUT_INSTRUCTION = """Respond using exactly this XML structure:
<visual_evidence>
Question-relevant visual observations from the image. Do not use the gold answer.
</visual_evidence>
<reasoning>
Briefly reason from the visual evidence and answer choices.
</reasoning>
<answer>
Final option letter and short answer.
</answer>"""
STRUCTURED_ROLLOUT_V2_INSTRUCTION = """Respond using exactly this XML structure:
<visible_evidence>
Directly visible visual facts from the image. Keep this block concise.
</visible_evidence>
<diagram_inference>
Intermediate visual/geometric facts derived from diagram marks. Keep this block concise. Do not write the final answer here.
</diagram_inference>
<reasoning>
Briefly reason from the evidence and answer choices. Keep this block concise.
</reasoning>
<answer>
Final option letter and short answer. This block must be present.
</answer>
Always emit all four XML blocks exactly once. Do not continue after </answer>."""


@dataclass(frozen=True)
class RolloutPrompt:
    text: str
    format_mode: str


@dataclass(frozen=True)
class RolloutDiversityDiagnostics:
    unique_response_per_prompt_mean: float | None
    duplicate_rollout_rate: float | None
    all_rollouts_identical_per_prompt_count: int
    short_response_rate: float | None


class LegacyBatchRolloutGenerator(Protocol):
    tokenizer: StudentRolloutTokenizer

    def generate(
        self,
        *,
        question: str,
        image_path: str,
        num_return_sequences: int,
        seed: int,
    ) -> list[str]: ...


@dataclass(frozen=True)
class StudentRolloutAuditConfig:
    dataset: Path
    dataset_type: str = "auto"
    source_dataset: str = "vision-opd-6k"
    limit: int = 4
    conditions: tuple[Condition, ...] = (Condition.FULL, Condition.BLUR)
    teacher_url: str = DEFAULT_TEACHER_URL
    student_model_path: str = "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"
    rollouts_per_prompt: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    blur_sigma: float = 2.0
    degraded_dir: str | None = None
    materialize_degraded_images: bool = False
    output_dir: Path | None = None
    rollout_response_format: str = "fc_opd_structured"
    min_response_tokens_for_warning: int = 16

    def __post_init__(self) -> None:
        if self.rollout_response_format not in ROLLOUT_RESPONSE_FORMATS:
            raise ValueError(f"rollout_response_format must be one of {ROLLOUT_RESPONSE_FORMATS}")
        if self.rollouts_per_prompt <= 0:
            raise ValueError("rollouts_per_prompt must be positive")
        if self.min_response_tokens_for_warning <= 0:
            raise ValueError("min_response_tokens_for_warning must be positive")


@dataclass
class StudentRolloutAuditResult(DatasetAuditResult):
    expected_rows: int = 0
    actual_rows: int = 0


class FixedFakeRolloutGenerator:
    """Deterministic test/smoke rollout generator."""

    def __init__(self, tokenizer: StudentRolloutTokenizer):
        self.tokenizer = tokenizer

    def generate(
        self,
        *,
        question: str,
        image_path: str,
        prompt_text: str,
        seed: int,
    ) -> str:
        del image_path
        if "<visible_evidence>" in prompt_text:
            return (
                "<visible_evidence>\n"
                f"Fake visible evidence seed {seed} for: {question[:48]}\n"
                "</visible_evidence>\n"
                "<diagram_inference>\n"
                "This is a deterministic intermediate diagram inference.\n"
                "</diagram_inference>\n"
                "<reasoning>\n"
                "This is a deterministic test rollout with routed reasoning.\n"
                "</reasoning>\n"
                "<answer>\n"
                f"{chr(ord('A') + seed % 4)}. fake answer\n"
                "</answer>"
            )
        if "<visual_evidence>" not in prompt_text:
            return f"{chr(ord('A') + seed % 4)}. fake"
        return (
            "<visual_evidence>\n"
            f"Fake rollout seed {seed} for: {question[:48]}\n"
            "</visual_evidence>\n"
            "<reasoning>\n"
            "This is a deterministic test rollout with visual evidence and reasoning.\n"
            "</reasoning>\n"
            "<answer>\n"
            f"{chr(ord('A') + seed % 4)}. fake answer\n"
            "</answer>"
        )


class HFQwenStudentRolloutGenerator:
    """Lazy Hugging Face Qwen-VL rollout generator for HPC smoke audits."""

    def __init__(self, config: StudentRolloutAuditConfig):
        from PIL import Image
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._image_cls = Image
        model_path = os.path.expandvars(config.student_model_path.removeprefix("hf:"))
        self.processor = AutoProcessor.from_pretrained(model_path)
        torch_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=config.device if config.device == "cuda" else None,
        )
        if config.device != "cuda":
            self.model.to(config.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.config = config

    def generate(
        self,
        *,
        question: str,
        image_path: str,
        prompt_text: str,
        seed: int,
    ) -> str:
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        image = self._image_cls.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        generation_kwargs: dict[str, object] = {
            "do_sample": self.config.temperature > 0,
            "max_new_tokens": self.config.max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        outputs = self.model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[:, input_len:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def score_conditions(
        self,
        *,
        response_token_ids: Sequence[int],
        question: str,
        condition_inputs: ConditionInputs,
        conditions: Sequence[Condition],
        response_text: str | None = None,
        top_k: int = 32,
    ) -> dict[Condition, TeacherTopK]:
        """Teacher-force score the same response under student-side conditions."""

        del response_text
        import torch

        outputs: dict[Condition, TeacherTopK] = {}
        response_ids = tuple(int(item) for item in response_token_ids)
        response_tensor_cpu = torch.tensor([response_ids], dtype=torch.long)
        for condition in conditions:
            rendered = render_teacher_prompt(condition, question, condition_inputs)
            images = None
            if rendered.image_paths:
                images = [self._image_cls.open(rendered.image_paths[0]).convert("RGB")]
            prompt = self.processor.apply_chat_template(
                list(rendered.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            proc = self.processor(text=[prompt], images=images, return_tensors="pt")
            prompt_ids = proc["input_ids"]
            prompt_len = int(prompt_ids.shape[1])
            response_tensor = response_tensor_cpu.to(dtype=prompt_ids.dtype)
            full_ids = torch.cat([prompt_ids, response_tensor], dim=1).to(self.model.device)
            model_inputs: dict[str, torch.Tensor] = {}
            for key, value in proc.items():
                if key in {"input_ids", "attention_mask"}:
                    continue
                model_inputs[key] = value.to(self.model.device)
            model_inputs["input_ids"] = full_ids
            model_inputs["attention_mask"] = torch.ones_like(full_ids)
            with torch.no_grad():
                logits = self.model(**model_inputs).logits[:, prompt_len - 1 : prompt_len - 1 + len(response_ids), :]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                values, indices = torch.topk(log_probs, k=min(int(top_k), log_probs.shape[-1] - 1), dim=-1)
                tail_mass = (1.0 - values.exp().sum(dim=-1)).clamp_min(torch.finfo(torch.float32).tiny)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            outputs[Condition(condition)] = TeacherTopK(
                token_ids=indices.cpu(),
                log_probs=values.cpu(),
                tail_log_prob=tail_mass.log().cpu(),
                entropy=entropy.cpu(),
            )
        return outputs


def run_student_rollout_signal_audit(
    config: StudentRolloutAuditConfig,
    *,
    rollout_generator: StudentRolloutGenerator | None = None,
    teacher_client: TeacherClient | None = None,
) -> StudentRolloutAuditResult:
    records = load_normalized_records(
        config.dataset,
        config.dataset_type,
        source_dataset=config.source_dataset,
    )[: config.limit]
    rollout_generator = rollout_generator or HFQwenStudentRolloutGenerator(config)
    tokenizer = rollout_generator.tokenizer
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    teacher_client = teacher_client or TeacherClient(
        config.teacher_url,
        expected_tokenizer_hash=tokenizer_hash,
    )

    samples: list[dict[str, Any]] = []
    for prompt_index, record in enumerate(records):
        samples.extend(
            _score_rollouts_for_record(
                record,
                prompt_index=prompt_index,
                config=config,
                rollout_generator=rollout_generator,
                tokenizer_hash=tokenizer_hash,
                teacher_client=teacher_client,
            )
        )

    summary = summarize_audit_samples(samples, config=_summary_config(config), num_loaded=len(records))
    summary["expected_rows"] = len(records) * config.rollouts_per_prompt
    summary["actual_rows"] = len(samples)
    summary["student_rollout_config"] = {
        "student_model_path": config.student_model_path,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "rollout_response_format": config.rollout_response_format,
        "min_response_tokens_for_warning": config.min_response_tokens_for_warning,
    }
    summary.update(_rollout_diversity_summary(samples, config))
    result = StudentRolloutAuditResult(
        samples=samples,
        summary=summary,
        expected_rows=summary["expected_rows"],
        actual_rows=summary["actual_rows"],
    )
    if config.output_dir:
        _write_outputs(result, config.output_dir, config.source_dataset)
    return result


def _score_rollouts_for_record(
    record: Mapping[str, Any],
    *,
    prompt_index: int,
    config: StudentRolloutAuditConfig,
    rollout_generator: StudentRolloutGenerator,
    tokenizer_hash: str,
    teacher_client: TeacherClient,
) -> list[dict[str, Any]]:
    question = str(record.get("question") or "")
    image_path = str(record.get("image_path") or "")
    degraded_image_path = derive_degraded_path(image_path, config.blur_sigma, config.degraded_dir)
    if config.materialize_degraded_images and image_path:
        materialize_gaussian_blur(image_path, degraded_image_path, config.blur_sigma)

    rows: list[dict[str, Any]] = []
    prompt = build_rollout_prompt(question, response_format=config.rollout_response_format)
    for rollout_id in range(config.rollouts_per_prompt):
        generation_seed = rollout_seed(
            base_seed=config.seed,
            source_index=int(record.get("source_index", prompt_index)),
            rollout_id=rollout_id,
        )
        response_text = rollout_generator.generate(
            question=question,
            image_path=image_path,
            prompt_text=prompt.text,
            seed=generation_seed,
        )
        token_ids = tuple(int(item) for item in rollout_generator.tokenizer.encode(response_text))
        condition_inputs = build_audit_condition_inputs(
            record,
            question=question,
            answer=None,
            full_image_path=image_path,
            degraded_image_path=degraded_image_path,
            blur_sigma=config.blur_sigma,
            task_evidence_mode="none",
        )
        teacher_scores = score_teacher_conditions(
            token_ids,
            question,
            condition_inputs,
            config.conditions,
            teacher_client,
            response_text=response_text,
            request_prefix=f"{record.get('sample_uid')}:rollout-{rollout_id}",
        )
        signals = compute_condition_signals(teacher_scores)
        gradient_cosines = {}
        errors: list[str] = []
        try:
            gradient_cosines = compute_pairwise_kd_gradient_cosines(teacher_scores)
            if not gradient_cosines:
                errors.append("gradient_cosine_unavailable: requested condition pair not present")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gradient_cosine_unavailable: {exc}")
        image_exists = Path(image_path).expanduser().is_file() if image_path else False
        degraded_image_exists = (
            Path(degraded_image_path).expanduser().is_file() if degraded_image_path else False
        )
        rows.append(
            {
                "sample_uid": str(record.get("sample_uid") or ""),
                "rollout_id": rollout_id,
                "rollout_uid": f"{record.get('sample_uid')}:rollout-{rollout_id}",
                "source_dataset": config.source_dataset,
                "source_index": int(record.get("source_index", prompt_index)),
                "question": question,
                "image_path": image_path,
                "degraded_image_path": degraded_image_path,
                "image_exists": image_exists,
                "degraded_image_exists": degraded_image_exists,
                "bbox_image_path": str(record.get("bbox_image_path") or ""),
                "bbox_image_paths": list(record.get("bbox_image_paths") or []),
                "bbox_image_exists": bool(record.get("bbox_image_exists", False)),
                "crop_bbox_policy": "metadata_only_default_no_crop_condition",
                "bbox_metadata_unused_by_default": True,
                "response_source": "student_rollout",
                "response_text": response_text,
                "response_token_ids": list(token_ids),
                "response_token_count": len(token_ids),
                "response_length": len(token_ids),
                "response_text_hash": hash_text(response_text),
                "response_token_hash": hash_token_ids(token_ids),
                "response_provenance_note": "student rollout sampled before teacher forced-scoring",
                "generation_seed": generation_seed,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
                "rollout_response_format": config.rollout_response_format,
                "rollout_prompt": prompt.text,
                "rollout_prompt_hash": hash_text(prompt.text),
                "student_model_path": config.student_model_path,
                "prompt_hash": hash_text(question),
                "tokenizer_hash": tokenizer_hash,
                "teacher_model_id": teacher_client.metadata.model_id,
                "condition_score_available": {
                    condition.value: condition in teacher_scores for condition in config.conditions
                },
                "condition_entropy_mean": {
                    condition.value: (
                        None
                        if scores.entropy is None
                        else float(scores.entropy.float().mean().item())
                    )
                    for condition, scores in teacher_scores.items()
                },
                "condition_signals": {
                    name: [float(item) for item in signal[0].tolist()]
                    for name, signal in signals.items()
                },
                "gradient_cosines": gradient_cosines,
                "leakage_warnings": [],
                "errors": errors,
                "metadata": {
                    "crop_bbox_policy": "metadata_only_default_no_crop_condition",
                    "response_source": "student_rollout",
                    "rollout_response_format": config.rollout_response_format,
                    "rollout_response_format_note": response_format_note(config.rollout_response_format),
                },
            }
        )
    return rows


def build_rollout_prompt(question: str, *, response_format: str) -> RolloutPrompt:
    if response_format == "answer_only":
        return RolloutPrompt(text=question, format_mode=response_format)
    if response_format == "fc_opd_structured":
        return RolloutPrompt(
            text=f"{question}\n\n{STRUCTURED_ROLLOUT_INSTRUCTION}",
            format_mode=response_format,
        )
    if response_format == "fc_opd_structured_v2":
        return RolloutPrompt(
            text=f"{question}\n\n{STRUCTURED_ROLLOUT_V2_INSTRUCTION}",
            format_mode=response_format,
        )
    raise ValueError(f"unsupported rollout_response_format: {response_format}")


def response_format_note(response_format: str) -> str:
    if response_format == "answer_only":
        return "mechanical smoke only; too short for token-level FC-OPD training"
    if response_format == "fc_opd_structured_v2":
        return "OPD-compatible structured response with visible evidence, diagram inference, reasoning, and answer spans"
    return "OPD-compatible structured response with visual evidence, reasoning, and answer spans"


def rollout_seed(*, base_seed: int, source_index: int, rollout_id: int) -> int:
    return int(base_seed) + int(source_index) * 1000 + int(rollout_id)


def _rollout_diversity_summary(
    samples: Sequence[Mapping[str, Any]],
    config: StudentRolloutAuditConfig,
) -> dict[str, Any]:
    by_prompt: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        by_prompt.setdefault(str(sample.get("sample_uid", "")), []).append(sample)
    unique_counts = [
        len({str(sample.get("response_text_hash", "")) for sample in prompt_samples})
        for prompt_samples in by_prompt.values()
    ]
    duplicate_rollouts = 0
    total_rollouts = 0
    identical_prompt_count = 0
    for prompt_samples in by_prompt.values():
        hashes = [str(sample.get("response_text_hash", "")) for sample in prompt_samples]
        counts = {hash_value: hashes.count(hash_value) for hash_value in set(hashes)}
        duplicate_rollouts += sum(max(0, count - 1) for count in counts.values())
        total_rollouts += len(hashes)
        if len(set(hashes)) == 1 and len(hashes) > 1:
            identical_prompt_count += 1
    response_lengths = [int(sample.get("response_length", 0)) for sample in samples]
    short_count = sum(length < config.min_response_tokens_for_warning for length in response_lengths)
    duplicate_rate = None if total_rollouts == 0 else duplicate_rollouts / total_rollouts
    short_rate = None if not response_lengths else short_count / len(response_lengths)
    warnings: list[str] = []
    if duplicate_rate is not None and duplicate_rate > 0.25:
        warnings.append("duplicate_rollout_rate_high")
    if short_rate is not None and short_rate > 0:
        warnings.append("short_response_rate_nonzero")
    if config.rollout_response_format == "answer_only":
        warnings.append("answer_only_is_mechanical_smoke_only")
    return {
        "unique_response_per_prompt_mean": (
            None if not unique_counts else sum(unique_counts) / len(unique_counts)
        ),
        "duplicate_rollout_rate": duplicate_rate,
        "all_rollouts_identical_per_prompt_count": identical_prompt_count,
        "short_response_rate": short_rate,
        "min_response_tokens_for_warning": config.min_response_tokens_for_warning,
        "rollout_diversity_warnings": warnings,
    }


def _summary_config(config: StudentRolloutAuditConfig) -> Any:
    @dataclass(frozen=True)
    class SummaryConfig:
        dataset: Path = config.dataset
        dataset_type: str = config.dataset_type
        source_dataset: str = config.source_dataset
        conditions: tuple[Condition, ...] = config.conditions
        dry_run: bool = False
        high_signal_threshold: float = 0.01
        high_cosine_threshold: float = 0.98

    return SummaryConfig()


def _write_outputs(result: StudentRolloutAuditResult, output_dir: Path, source_dataset: str) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source_dataset}_student_rollout_signal_audit"
    result.jsonl_path = output_dir / f"{stem}.jsonl"
    result.summary_json_path = output_dir / f"{stem}_summary.json"
    with result.jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in result.samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    result.summary_json_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_conditions(value: str) -> tuple[Condition, ...]:
    conditions = tuple(Condition(item.strip()) for item in value.split(",") if item.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-type", default="auto")
    parser.add_argument("--source-dataset", default="vision-opd-6k")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--conditions", type=_parse_conditions, default=(Condition.FULL, Condition.BLUR))
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--student-model-path", default="hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct")
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--rollout-response-format",
        choices=ROLLOUT_RESPONSE_FORMATS,
        default="fc_opd_structured",
    )
    parser.add_argument("--min-response-tokens-for-warning", type=int, default=16)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--degraded-dir")
    parser.add_argument("--materialize-degraded-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_student_rollout_signal_audit(
        StudentRolloutAuditConfig(
            dataset=args.dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            limit=args.limit,
            conditions=args.conditions,
            teacher_url=args.teacher_url,
            student_model_path=args.student_model_path,
            rollouts_per_prompt=args.rollouts_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            rollout_response_format=args.rollout_response_format,
            min_response_tokens_for_warning=args.min_response_tokens_for_warning,
            blur_sigma=args.blur_sigma,
            degraded_dir=args.degraded_dir,
            materialize_degraded_images=args.materialize_degraded_images,
            output_dir=args.output_dir,
        )
    )
    print(f"wrote {result.actual_rows}/{result.expected_rows} student-rollout audit rows")
    print(f"jsonl: {result.jsonl_path}")
    print(f"summary: {result.summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
