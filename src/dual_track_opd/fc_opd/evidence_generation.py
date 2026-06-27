"""Auditable free/task evidence cache generation for clean-data 4C FC-OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .dataset_adapters import load_normalized_records
from .geometry3k_adapter import (
    inspect_geometry3k_dataset,
    is_generated_degraded_image_path,
    load_geometry3k_records,
)
from .virl39k_adapter import load_virl39k_records

FREE_CAPTION_PROMPTS = {
    "natural": (
        "Describe the image objectively. Focus on visible objects, attributes, spatial relations, "
        "text, colors, shapes, symbols, and fine-grained details. Do not answer any question. "
        "Do not mention any option letter. Do not infer hidden intent. Return only visual "
        "observations grounded in the image."
    ),
    "geometry": (
        "Describe the geometry diagram objectively. Mention all visible points, lines, angles, "
        "shapes, labels, length marks, angle marks, parallel or perpendicular markers, circles, "
        "triangles, and spatial relations. Do not solve the problem. Do not answer any question. "
        "Return only diagram observations."
    ),
}
TASK_EVIDENCE_PROMPTS = {
    "natural": (
        "You are given an image and a multiple-choice visual reasoning question. Do not answer "
        "the question. Do not output an option letter. Do not say 'the answer is ...'. Do not use "
        "any gold answer or label. Write concise question-relevant visual evidence needed to solve "
        "the question. Mention only observable evidence from the image. Return evidence only, not "
        "the final answer."
    ),
    "geometry": (
        "You are given a geometry diagram and a question. Do not solve the problem. Do not output "
        "the final answer. Do not output an option letter. Do not use any gold answer. Extract the "
        "question-relevant diagram evidence: visible labels, points, lines, angles, equal-length "
        "marks, parallel or perpendicular relations, circles, triangles, and any given numeric "
        "values. Return only evidence grounded in the diagram."
    ),
}
TASK_VISIBLE_PROMPTS = TASK_EVIDENCE_PROMPTS
TASK_INFER_PROMPTS = {
    "natural": (
        "You are given an image and a multiple-choice visual reasoning question. Do not answer "
        "the question. Do not output an option letter. Infer only intermediate visual relations "
        "or constraints needed for solving. Do not compute the final requested quantity."
    ),
    "geometry": (
        "You are given a geometry diagram and a question. May infer intermediate geometric facts "
        "from diagram marks and structure, and may write usable equations. Must not compute the "
        "final requested quantity, choose an option, or write a complete solution. If a derived "
        "numeric value is exactly the asked final answer, omit it."
    ),
}
TASK_SOLVE_PROMPTS = {
    "natural": (
        "You are given an image and a multiple-choice visual reasoning question. Solve the problem "
        "fully using only the image, question, and choices provided in this prompt. Do not use any "
        "dataset gold answer or hidden label."
    ),
    "geometry": (
        "You are given a geometry diagram and a question. Solve the problem fully using only the "
        "image, question, and choices provided in this prompt. Do not use any dataset gold answer "
        "or hidden label."
    ),
}

CONDITION_SETS: dict[str, tuple[str, ...]] = {
    "4c-clean": ("full", "degraded", "free", "task_visible"),
    "5c-infer": ("full", "degraded", "free", "task_visible", "task_infer"),
    "6c-solve": ("full", "degraded", "free", "task_visible", "task_infer", "task_solve"),
}


class EvidenceGenerator(Protocol):
    model_id: str

    def generate_free_caption(self, *, image_path: str, prompt: str, seed: int) -> str: ...

    def generate_task_evidence(
        self,
        *,
        image_path: str,
        question: str,
        choices: Sequence[str],
        prompt: str,
        seed: int,
    ) -> str: ...


class TemplateEvidenceGenerator:
    """Deterministic lightweight generator for tests and dry pipeline checks."""

    model_id = "template-evidence-generator"

    def generate_free_caption(self, *, image_path: str, prompt: str, seed: int) -> str:
        del prompt, seed
        name = Path(image_path).name or "diagram"
        return f"Observable diagram/image evidence is present in {name}; labels, shapes, and spatial relations should be inspected."

    def generate_task_evidence(
        self,
        *,
        image_path: str,
        question: str,
        choices: Sequence[str],
        prompt: str,
        seed: int,
    ) -> str:
        del seed
        basis = question.splitlines()[0][:80]
        if "Solve the problem fully" in prompt or "Solve the problem" in prompt:
            return f"Teacher-inferred solution should be generated from {Path(image_path).name}: {basis}"
        if "usable equations" in prompt or "intermediate" in prompt:
            return f"Intermediate diagram relations and equations should be inferred from {Path(image_path).name}: {basis}"
        return f"Question-relevant visual evidence should be extracted from {Path(image_path).name}: {basis}"


@dataclass(frozen=True)
class EvidenceGenerationConfig:
    dataset: Path
    output_jsonl: Path
    summary_json: Path
    dataset_type: str = "geometry3k"
    source_dataset: str = "geometry3k"
    generator_model_path: str | None = None
    generator_url: str | None = None
    limit: int | None = None
    start_index: int = 0
    end_index: int | None = None
    seed: int = 42
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 256
    resume: bool = False
    skip_existing: bool = False
    strict_leakage: bool = False
    strict_condition_validation: bool = False
    condition_set: str = "4c-clean"
    include_prompts_in_output: bool = False
    task_evidence_mode: str = "visible"
    allow_degraded_source_images: bool = False

    def __post_init__(self) -> None:
        if self.condition_set not in CONDITION_SETS:
            raise ValueError(f"condition_set must be one of {sorted(CONDITION_SETS)}")
        if self.task_evidence_mode not in {"visible", "infer", "solve"}:
            raise ValueError("task_evidence_mode must be visible, infer, or solve")


@dataclass
class EvidenceGenerationResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def run_evidence_generation(
    config: EvidenceGenerationConfig,
    *,
    generator: EvidenceGenerator | None = None,
) -> EvidenceGenerationResult:
    if config.skip_existing and config.output_jsonl.is_file() and config.summary_json.is_file():
        return EvidenceGenerationResult(
            rows=_read_jsonl(config.output_jsonl),
            summary=json.loads(config.summary_json.read_text(encoding="utf-8")),
        )
    records = _load_records(config)
    selected = _select(records, config.start_index, config.end_index, config.limit)
    generator = generator or _build_generator(config)
    existing = _read_jsonl(config.output_jsonl) if config.resume else []
    existing_ids = {str(row.get("sample_uid", "")) for row in existing}
    mode = "a" if config.resume and config.output_jsonl.is_file() else "w"
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict[str, Any]] = []
    with config.output_jsonl.open(mode, encoding="utf-8") as handle:
        for record in selected:
            if str(record["sample_uid"]) in existing_ids:
                continue
            row = build_evidence_row(record, config=config, generator=generator)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            new_rows.append(row)
    rows = [*existing, *new_rows]
    summary = summarize_evidence(rows, config=config, generator=generator)
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return EvidenceGenerationResult(rows=rows, summary=summary)


def build_evidence_row(
    record: Mapping[str, Any],
    *,
    config: EvidenceGenerationConfig,
    generator: EvidenceGenerator,
) -> dict[str, Any]:
    domain = "geometry" if config.dataset_type == "geometry3k" else "natural"
    free_prompt = FREE_CAPTION_PROMPTS[domain]
    task_visible_prompt = TASK_VISIBLE_PROMPTS[domain]
    task_infer_prompt = TASK_INFER_PROMPTS[domain]
    task_solve_prompt = TASK_SOLVE_PROMPTS[domain]
    image_path = str(record.get("image_path", ""))
    choices = list(record.get("choices", []))
    source_index = int(record.get("source_index", 0))
    seed = int(config.seed) + source_index
    free_caption = generator.generate_free_caption(image_path=image_path, prompt=free_prompt, seed=seed)
    task_visible_evidence = generator.generate_task_evidence(
        image_path=image_path,
        question=str(record["question"]),
        choices=choices,
        prompt=task_visible_prompt,
        seed=seed + 17,
    )
    task_infer_evidence = None
    task_solve_evidence = None
    if "task_infer" in CONDITION_SETS[config.condition_set] or config.task_evidence_mode in {"infer", "solve"}:
        task_infer_evidence = generator.generate_task_evidence(
            image_path=image_path,
            question=str(record["question"]),
            choices=choices,
            prompt=task_infer_prompt,
            seed=seed + 29,
        )
    if "task_solve" in CONDITION_SETS[config.condition_set] or config.task_evidence_mode == "solve":
        task_solve_evidence = generator.generate_task_evidence(
            image_path=image_path,
            question=str(record["question"]),
            choices=choices,
            prompt=task_solve_prompt,
            seed=seed + 43,
        )
    task_evidence_by_mode = {
        "visible": task_visible_evidence,
        "infer": task_infer_evidence or task_visible_evidence,
        "solve": task_solve_evidence or task_infer_evidence or task_visible_evidence,
    }
    task_evidence = task_evidence_by_mode[config.task_evidence_mode]
    leakage = detect_evidence_leakage(
        free_caption=free_caption,
        task_evidence=task_evidence,
        task_visible_evidence=task_visible_evidence,
        task_infer_evidence=task_infer_evidence,
        task_solve_evidence=task_solve_evidence,
        answer=record.get("answer") or record.get("gold") or record.get("answer_metadata"),
    )
    errors = []
    if config.strict_leakage and any(item.startswith("hard_") for item in leakage):
        errors.append("hard_leakage_detected")
    degraded_source_image = is_generated_degraded_image_path(image_path)
    if degraded_source_image and not config.allow_degraded_source_images:
        errors.append("degraded_source_image_path")
    prompt_hashes = {
        "free_caption_prompt": text_hash(free_prompt),
        "task_visible_prompt": text_hash(task_visible_prompt),
        "task_infer_prompt": text_hash(task_infer_prompt),
        "task_solve_prompt": text_hash(task_solve_prompt),
    }
    contamination = contamination_flags(
        task_visible_evidence=task_visible_evidence,
        task_infer_evidence=task_infer_evidence,
        task_solve_evidence=task_solve_evidence,
        answer=record.get("answer") or record.get("gold") or record.get("answer_metadata"),
    )
    task_infer_classification = classify_task_infer_evidence(task_infer_evidence or "", choices)
    condition_validation_warnings = []
    if task_infer_classification["class"] == "solve_like":
        condition_validation_warnings.append("task_infer_solve_like")
        if config.strict_condition_validation:
            errors.append("task_infer_solve_like")
    elif task_infer_classification["class"] == "unusable":
        condition_validation_warnings.append("task_infer_unusable")
        if config.strict_condition_validation:
            errors.append("task_infer_unusable")
    row = {
        "sample_uid": str(record["sample_uid"]),
        "source_dataset": config.source_dataset,
        "source_index": source_index,
        "question": str(record["question"]),
        "choices": choices,
        "image_path": image_path,
        "image_hash": file_sha256(image_path),
        "degraded_source_image": degraded_source_image,
        "degraded_source_image_allowed": bool(config.allow_degraded_source_images),
        "condition_evidence": {
            "free": free_caption,
            "task_visible": task_visible_evidence,
            "task_infer": task_infer_evidence,
            "task_solve": task_solve_evidence,
        },
        "free_caption": free_caption,
        "task_evidence": task_evidence,
        "task_visible_evidence": task_visible_evidence,
        "task_infer_evidence": task_infer_evidence,
        "task_solve_evidence": task_solve_evidence,
        "task_infer_class": task_infer_classification["class"],
        "task_infer_solve_like": task_infer_classification["class"] == "solve_like",
        "task_infer_classification_flags": task_infer_classification["flags"],
        "condition_validation_warnings": condition_validation_warnings,
        "strict_condition_validation": bool(config.strict_condition_validation),
        "free_caption_prompt_hash": prompt_hashes["free_caption_prompt"],
        "task_evidence_prompt_hash": prompt_hashes["task_visible_prompt"],
        "task_visible_prompt_hash": prompt_hashes["task_visible_prompt"],
        "task_infer_prompt_hash": prompt_hashes["task_infer_prompt"],
        "task_solve_prompt_hash": prompt_hashes["task_solve_prompt"],
        "prompt_hashes": prompt_hashes,
        "generator_model_id": generator.model_id,
        "generator_metadata": {
            "generator_model_path": config.generator_model_path,
            "generator_url": config.generator_url,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "seed": seed,
            "condition_set": config.condition_set,
            "task_evidence_mode": config.task_evidence_mode,
        },
        "condition_set_name": config.condition_set,
        "conditions": list(CONDITION_SETS[config.condition_set]),
        "no_gold_field_used": True,
        "gold_answer_seen_by_prompt": False,
        "answer_letter_forbidden": True,
        "final_answer_forbidden": True,
        **contamination,
        "leakage_warnings": leakage,
        "errors": errors,
    }
    if config.include_prompts_in_output:
        row.update(
            {
                "free_caption_prompt": free_prompt,
                "task_evidence_prompt": task_visible_prompt,
                "task_visible_prompt": task_visible_prompt,
                "task_infer_prompt": task_infer_prompt,
                "task_solve_prompt": task_solve_prompt,
            }
        )
    return row


def detect_evidence_leakage(
    *,
    free_caption: str,
    task_evidence: str,
    answer: Any,
    task_visible_evidence: str | None = None,
    task_infer_evidence: str | None = None,
    task_solve_evidence: str | None = None,
) -> list[str]:
    text = "\n".join(
        item
        for item in (
            free_caption,
            task_evidence,
            task_visible_evidence,
            task_infer_evidence,
            task_solve_evidence,
        )
        if item
    )
    lower = text.lower()
    warnings: list[str] = []
    if "answer is" in lower or "the answer" in lower:
        warnings.append("answer_phrase_leakage")
    for letter in ("A", "B", "C", "D", "E"):
        if f"option {letter.lower()}" in lower or f"answer {letter.lower()}" in lower:
            warnings.append(f"answer_letter_leakage:{letter}")
    if answer is not None:
        normalized_answer = _normalize(str(answer))
        if len(normalized_answer) >= 3 and normalized_answer in _normalize(text):
            warnings.append("soft_answer_text_overlap")
    return sorted(set(warnings))


def contamination_flags(
    *,
    task_visible_evidence: str,
    task_infer_evidence: str | None,
    task_solve_evidence: str | None,
    answer: Any,
) -> dict[str, bool]:
    visible_lower = task_visible_evidence.lower()
    infer_lower = (task_infer_evidence or "").lower()
    solve_lower = (task_solve_evidence or "").lower()
    answer_text = "" if answer is None else _normalize(str(answer))
    generated_text = _normalize("\n".join(item for item in (task_visible_evidence, task_infer_evidence, task_solve_evidence) if item))
    exact_answer = bool(answer_text and len(answer_text) >= 3 and answer_text in generated_text)
    answer_letter = any(
        phrase in f"{visible_lower}\n{infer_lower}\n{solve_lower}"
        for letter in ("a", "b", "c", "d", "e")
        for phrase in (f"answer {letter}", f"option {letter}")
    )
    visible_final = "answer is" in visible_lower or "the answer" in visible_lower
    infer_final = "answer is" in infer_lower or "the answer" in infer_lower
    return {
        "answer_letter_in_generated_evidence": answer_letter,
        "exact_answer_in_generated_evidence": exact_answer,
        "final_solution_detected_in_task_visible": visible_final,
        "final_solution_detected_in_task_infer": infer_final,
        "solution_contaminated": bool(task_solve_evidence and ("solution" in solve_lower or "answer" in solve_lower)),
    }


def classify_task_infer_evidence(text: str | None, choices: Sequence[str] = ()) -> dict[str, Any]:
    """Classify whether task_infer evidence stayed intermediate or solved."""

    raw = (text or "").strip()
    lower = raw.lower()
    flags: list[str] = []
    if not raw:
        return {"class": "unusable", "flags": ["empty"]}
    if len(raw) < 20:
        flags.append("too_short")
    if re.fullmatch(r"\s*[A-Da-d]\s*(?:[\.\):;-]\s*)?(?:[-+]?\d+(?:\.\d+)?|\S{1,20})?\s*", raw):
        flags.append("option_only")
    if re.match(r"\s*[A-Da-d]\s*[\.\):;-]", raw):
        flags.append("starts_with_option")
    solution_phrases = (
        "answer is",
        "therefore the answer",
        "so the answer",
        "choose",
    )
    if any(phrase in lower for phrase in solution_phrases):
        flags.append("solution_language")
    normalized_raw = _normalize(raw)
    for choice in choices:
        choice_text = str(choice).strip()
        if not choice_text:
            continue
        normalized_choice = _normalize(choice_text)
        if len(normalized_choice) >= 2 and normalized_choice in normalized_raw:
            flags.append("choice_phrase_emitted")
            break
    final_markers = (
        "therefore",
        "hence",
        "final answer",
        "the correct option",
        "we get",
        "is equal to",
    )
    if any(marker in lower for marker in final_markers) and (
        "solution_language" in flags or "choice_phrase_emitted" in flags
    ):
        flags.append("final_solution_like")
    if "option_only" in flags or "starts_with_option" in flags or "solution_language" in flags or "final_solution_like" in flags:
        return {"class": "solve_like", "flags": sorted(set(flags))}
    if "too_short" in flags:
        return {"class": "unusable", "flags": sorted(set(flags))}
    return {"class": "clean_infer", "flags": sorted(set(flags))}


def validate_evidence_row(row: Mapping[str, Any]) -> list[str]:
    errors = []
    if row.get("no_gold_field_used") is not True:
        errors.append("no_gold_field_used_not_true")
    free_prompt = str(row.get("free_caption_prompt", ""))
    if str(row.get("question", "")) and str(row.get("question", "")) in free_prompt:
        errors.append("free_caption_prompt_contains_question")
    task_prompt = str(row.get("task_evidence_prompt") or row.get("task_visible_prompt") or "").lower()
    if task_prompt and ("do not" not in task_prompt or "answer" not in task_prompt):
        errors.append("task_evidence_prompt_missing_answer_forbid_instruction")
    if row.get("gold_answer_seen_by_prompt") is not False:
        errors.append("gold_answer_seen_by_prompt_not_false")
    if is_generated_degraded_image_path(str(row.get("image_path", ""))) and row.get("degraded_source_image_allowed") is not True:
        errors.append("degraded_source_image_path")
    if row.get("final_solution_detected_in_task_visible"):
        errors.append("final_solution_detected_in_task_visible")
    if row.get("final_solution_detected_in_task_infer") and row.get("strict_condition_validation"):
        errors.append("final_solution_detected_in_task_infer")
    if row.get("task_infer_class") in {"solve_like", "unusable"} and row.get("strict_condition_validation"):
        errors.append(f"task_infer_{row.get('task_infer_class')}")
    if row.get("errors"):
        errors.extend(str(item) for item in row["errors"])
    return errors


def summarize_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: EvidenceGenerationConfig,
    generator: EvidenceGenerator,
) -> dict[str, Any]:
    validation_errors = [error for row in rows for error in validate_evidence_row(row)]
    leakage_rows = [row for row in rows if row.get("leakage_warnings")]
    degraded_source_image_count = sum(
        1 for row in rows if is_generated_degraded_image_path(str(row.get("image_path", "")))
    )
    task_infer_class_counts = Counter(str(row.get("task_infer_class", "missing")) for row in rows)
    return {
        "source_dataset": config.source_dataset,
        "dataset_type": config.dataset_type,
        "num_rows": len(rows),
        "generator_model_id": generator.model_id,
        "condition_set_name": config.condition_set,
        "conditions": list(CONDITION_SETS[config.condition_set]),
        "leakage_warning_rate": None if not rows else len(leakage_rows) / len(rows),
        "degraded_source_image_count": degraded_source_image_count,
        "task_infer_clean_count": task_infer_class_counts.get("clean_infer", 0),
        "task_infer_solve_like_count": task_infer_class_counts.get("solve_like", 0),
        "task_infer_unusable_count": task_infer_class_counts.get("unusable", 0),
        "task_infer_class_counts": dict(task_infer_class_counts),
        "validation_error_count": len(validation_errors),
        "validation_errors_top10": validation_errors[:10],
        "output_jsonl": str(config.output_jsonl),
        "summary_json": str(config.summary_json),
        "no_gold_field_used": True,
    }


def _build_generator(config: EvidenceGenerationConfig) -> EvidenceGenerator:
    if config.generator_url:
        raise NotImplementedError("generator_url endpoint is not implemented; teacher scorer service is unchanged")
    if config.generator_model_path:
        return HFQwenEvidenceGenerator(config)
    return TemplateEvidenceGenerator()


class HFQwenEvidenceGenerator:
    """Local Transformers VLM generator; loaded only when explicitly requested."""

    def __init__(self, config: EvidenceGenerationConfig):
        from PIL import Image
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._image_cls = Image
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(os.path.expandvars(config.generator_model_path or ""))
        self.model = AutoModelForImageTextToText.from_pretrained(
            os.path.expandvars(config.generator_model_path or ""),
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.config = config
        self.model_id = os.path.expandvars(config.generator_model_path or "hf-generator")

    def _generate(self, image_path: str, text: str, seed: int) -> str:
        torch = self._torch
        random.seed(seed)
        torch.manual_seed(seed)
        image = self._image_cls.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated = outputs[:, inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_free_caption(self, *, image_path: str, prompt: str, seed: int) -> str:
        return self._generate(image_path, prompt, seed)

    def generate_task_evidence(
        self,
        *,
        image_path: str,
        question: str,
        choices: Sequence[str],
        prompt: str,
        seed: int,
    ) -> str:
        text = f"{prompt}\n\nQuestion:\n{question}"
        if choices:
            text += "\n\nChoices:\n" + "\n".join(choices)
        return self._generate(image_path, text, seed)


def _load_records(config: EvidenceGenerationConfig) -> list[dict[str, Any]]:
    if config.dataset_type == "geometry3k":
        return load_geometry3k_records(config.dataset, source_dataset=config.source_dataset)
    if config.dataset_type == "virl39k":
        return load_virl39k_records(config.dataset, source_dataset=config.source_dataset)
    if config.dataset_type == "generic_vqa":
        return load_normalized_records(config.dataset, "auto", source_dataset=config.source_dataset)
    raise ValueError("dataset_type must be geometry3k, virl39k, or generic_vqa")


def _select(records: Sequence[dict[str, Any]], start: int, end: int | None, limit: int | None) -> list[dict[str, Any]]:
    output = list(records[start : end if end is not None else len(records)])
    return output[:limit] if limit is not None else output


def file_sha256(path: str | Path) -> str | None:
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-type", choices=("geometry3k", "virl39k", "generic_vqa"), default="geometry3k")
    parser.add_argument("--source-dataset", default="geometry3k")
    parser.add_argument("--generator-model-path")
    parser.add_argument("--generator-url")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--condition-set", choices=tuple(CONDITION_SETS), default="4c-clean")
    parser.add_argument("--include-prompts-in-output", action="store_true")
    parser.add_argument("--task-evidence-mode", choices=("visible", "infer", "solve"), default="visible")
    parser.add_argument("--allow-degraded-source-images", action="store_true")
    parser.add_argument("--strict-leakage", action="store_true")
    parser.add_argument("--strict-condition-validation", action="store_true")
    parser.add_argument(
        "--dry-run-inspect",
        action="store_true",
        help="Load and summarize dataset records without generating evidence or requiring output paths.",
    )
    parser.add_argument("--inspect-examples", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run_inspect:
        if args.dataset_type != "geometry3k":
            raise ValueError("--dry-run-inspect is currently implemented for dataset-type geometry3k")
        print(
            json.dumps(
                inspect_geometry3k_dataset(
                    args.dataset,
                    source_dataset=args.source_dataset,
                    examples=args.inspect_examples,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.output_jsonl is None:
        raise SystemExit("--output-jsonl is required unless --dry-run-inspect is set")
    if args.summary_json is None:
        raise SystemExit("--summary-json is required unless --dry-run-inspect is set")
    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=args.dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            generator_model_path=args.generator_model_path,
            generator_url=args.generator_url,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            resume=args.resume,
            skip_existing=args.skip_existing,
            condition_set=args.condition_set,
            include_prompts_in_output=args.include_prompts_in_output,
            task_evidence_mode=args.task_evidence_mode,
            allow_degraded_source_images=args.allow_degraded_source_images,
            strict_leakage=args.strict_leakage,
            strict_condition_validation=args.strict_condition_validation,
        )
    )
    print(json.dumps(result.summary, indent=2))
    return 0 if result.summary["validation_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
