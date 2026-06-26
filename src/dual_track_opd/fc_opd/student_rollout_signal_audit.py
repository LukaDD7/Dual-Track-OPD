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

from .conditions import Condition
from .dataset_adapters import load_normalized_records
from .dataset_signal_audit import (
    DatasetAuditResult,
    build_audit_condition_inputs,
    hash_text,
    hash_token_ids,
    materialize_gaussian_blur,
    summarize_audit_samples,
)
from .offline_scoring import DEFAULT_TEACHER_URL, derive_degraded_path
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint
from .signal_decomposer import compute_condition_signals


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
        num_return_sequences: int,
        seed: int,
    ) -> list[str]:
        del image_path
        return [
            (
                "<visual_evidence>\n"
                f"Fake rollout {index} for: {question[:48]}\n"
                "</visual_evidence>\n"
                "<reasoning>\n"
                "This is a deterministic test rollout.\n"
                "</reasoning>\n"
                "<answer>\n"
                f"{chr(ord('A') + (seed + index) % 4)}\n"
                "</answer>"
            )
            for index in range(num_return_sequences)
        ]


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
        num_return_sequences: int,
        seed: int,
    ) -> list[str]:
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        image = self._image_cls.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[:, input_len:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


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
    }
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

    rollouts = rollout_generator.generate(
        question=question,
        image_path=image_path,
        num_return_sequences=config.rollouts_per_prompt,
        seed=config.seed + prompt_index,
    )
    rows: list[dict[str, Any]] = []
    for rollout_id, response_text in enumerate(rollouts):
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
                "bbox_image_path": str(record.get("bbox_image_path") or ""),
                "bbox_image_paths": list(record.get("bbox_image_paths") or []),
                "bbox_metadata_unused_by_default": True,
                "response_source": "student_rollout",
                "response_text": response_text,
                "response_token_ids": list(token_ids),
                "response_token_count": len(token_ids),
                "response_length": len(token_ids),
                "response_text_hash": hash_text(response_text),
                "response_token_hash": hash_token_ids(token_ids),
                "response_provenance_note": "student rollout sampled before teacher forced-scoring",
                "generation_seed": config.seed + prompt_index,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
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
                "gradient_cosines": {},
                "leakage_warnings": [],
                "errors": [],
                "metadata": {
                    "crop_bbox_policy": "metadata_only_default_no_crop_condition",
                    "response_source": "student_rollout",
                },
            }
        )
    return rows


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
