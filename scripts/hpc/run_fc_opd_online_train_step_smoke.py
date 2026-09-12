"""Online FC-OPD one-step training smoke.

This script is intentionally not an offline replay smoke. It reads condition
evidence for prompt construction, samples a fresh rollout from the current
student, immediately scores that same rollout with the teacher service and the
current student, computes online FC-OPD loss, and takes one optimizer step on a
small selected student parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.online_batch import (
    OnlineFCOPDConfig,
    OnlineFCOPDSample,
    OnlineStudentScores,
    compute_online_fc_opd_batch,
)
from dual_track_opd.fc_opd.student_rollout_signal_audit import (
    HFQwenStudentRolloutGenerator,
    StudentRolloutAuditConfig,
    build_rollout_prompt,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient, score_teacher_conditions
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_prompts import render_teacher_prompt


def main() -> None:
    args = _parse_args()
    evidence_row = _read_jsonl_row(args.evidence_jsonl, args.row_index)
    condition_inputs = _condition_inputs_from_evidence_row(evidence_row, args.degraded_image_path)
    rollout_config = StudentRolloutAuditConfig(
        dataset=args.evidence_jsonl,
        dataset_type="generic_jsonl",
        source_dataset=str(evidence_row.get("source_dataset", "online_smoke")),
        student_model_path=args.student_model,
        teacher_url=args.teacher_url,
        limit=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        dtype=args.dtype,
        rollout_response_format="fc_opd_structured",
    )
    student = HFQwenStudentRolloutGenerator(rollout_config)
    _select_trainable_parameter(student.model, args.trainable_name_contains)
    optimizer = torch.optim.SGD((param for param in student.model.parameters() if param.requires_grad), lr=args.lr)

    question = str(evidence_row["question"])
    prompt = build_rollout_prompt(question, response_format="fc_opd_structured")
    rollout_text = student.generate(
        question=question,
        image_path=str(evidence_row["image_path"]),
        prompt_text=prompt.text,
        seed=args.seed,
    )
    rollout_token_ids = tuple(int(item) for item in student.tokenizer.encode(rollout_text))
    if not rollout_token_ids:
        raise RuntimeError("student produced an empty rollout")

    teacher_client = TeacherClient(args.teacher_url, expected_tokenizer_hash=tokenizer_fingerprint(student.tokenizer))
    teacher_scorer = _TeacherScorer(teacher_client)
    student_scorer = _HFStudentScorer(student, top_k=args.student_top_k)
    sample = OnlineFCOPDSample(
        sample_uid=str(evidence_row.get("sample_uid", f"online-smoke:{args.row_index}")),
        question=question,
        condition_inputs=condition_inputs,
        rollout_token_ids=rollout_token_ids,
        rollout_text=rollout_text,
        prompt=prompt.text,
        images=(str(evidence_row["image_path"]),),
        choices=tuple(str(item) for item in evidence_row.get("choices", [])),
        answer_metadata=evidence_row.get("answer_metadata") or evidence_row.get("answer"),
    )

    before = _selected_parameter_vector(student.model).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    output = compute_online_fc_opd_batch(
        [sample],
        tokenizer=student.tokenizer,
        teacher_scorer=teacher_scorer,
        student_scorer=student_scorer,
        config=OnlineFCOPDConfig(),
    )
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"online FC-OPD loss is not finite: {output.loss}")
    output.loss.backward()
    optimizer.step()
    after = _selected_parameter_vector(student.model).detach()
    delta_norm = torch.linalg.vector_norm(after - before).item()
    if delta_norm <= 0.0 or not math.isfinite(delta_norm):
        raise RuntimeError("optimizer step did not change the selected student parameter")

    report = {
        "sample_uid": sample.sample_uid,
        "rollout_num_tokens": len(rollout_token_ids),
        "loss": float(output.loss.detach().cpu().item()),
        "selected_parameter_delta_norm": delta_norm,
        "verifier_outcome": output.samples[0].verifier_learning_value_gate["outcome_class"],
        "teacher_conditions": [condition.value for condition in output.samples[0].teacher_scores],
        "metrics": {key: float(value.detach().cpu().item()) for key, value in output.metrics.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _TeacherScorer:
    def __init__(self, client: TeacherClient):
        self.client = client

    def __call__(self, sample: OnlineFCOPDSample, conditions: Sequence[Condition]):
        return score_teacher_conditions(
            response_token_ids=sample.rollout_token_ids,
            question=sample.question,
            condition_inputs=sample.condition_inputs,
            conditions=conditions,
            teacher_client=self.client,
            response_text=sample.rollout_text,
            request_prefix=f"{sample.sample_uid}:online",
        )


class _HFStudentScorer:
    def __init__(self, student: HFQwenStudentRolloutGenerator, *, top_k: int):
        self.student = student
        self.top_k = top_k

    def __call__(self, sample: OnlineFCOPDSample, conditions: Sequence[Condition]) -> OnlineStudentScores:
        loss_logits = self._condition_logits(sample, Condition.FULL, grad=True)
        condition_log_probs = {}
        with torch.no_grad():
            response_ids = torch.tensor(sample.rollout_token_ids, dtype=torch.long, device=loss_logits.device).reshape(1, -1)
            for condition in conditions:
                logits = self._condition_logits(sample, Condition(condition), grad=False)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                condition_log_probs[Condition(condition)] = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).cpu()
        return OnlineStudentScores(loss_logits=loss_logits, condition_log_probs=condition_log_probs)

    def _condition_logits(self, sample: OnlineFCOPDSample, condition: Condition, *, grad: bool) -> torch.Tensor:
        rendered = render_teacher_prompt(condition, sample.question, sample.condition_inputs)
        images = None
        if rendered.image_paths:
            images = [self.student._image_cls.open(rendered.image_paths[0]).convert("RGB")]
        prompt = self.student.processor.apply_chat_template(
            list(rendered.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        proc = self.student.processor(text=[prompt], images=images, return_tensors="pt")
        prompt_ids = proc["input_ids"]
        prompt_len = int(prompt_ids.shape[1])
        response_tensor = torch.tensor([sample.rollout_token_ids], dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, response_tensor], dim=1).to(self.student.model.device)
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in proc.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            model_inputs[key] = value.to(self.student.model.device)
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = torch.ones_like(full_ids)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = self.student.model(**model_inputs).logits[
                :, prompt_len - 1 : prompt_len - 1 + len(sample.rollout_token_ids), :
            ]
        return logits


def _read_jsonl_row(path: str | Path, index: int) -> dict[str, Any]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if index < 0 or index >= len(rows):
        raise IndexError(f"row_index {index} outside evidence row count {len(rows)}")
    row = rows[index]
    if not isinstance(row, Mapping):
        raise ValueError("evidence row must be a JSON object")
    return dict(row)


def _condition_inputs_from_evidence_row(row: Mapping[str, Any], degraded_image_path: str | None) -> ConditionInputs:
    image_path = str(row["image_path"])
    degraded_path = degraded_image_path or str(row.get("degraded_image_path") or image_path)
    evidence = row.get("condition_evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    return ConditionInputs(
        full_image=ImageInput(path=image_path),
        degraded_image=ImageInput(
            path=degraded_path,
            transform=dict(row.get("degraded_transform") or {"type": "lowres_nearest", "scale": 0.1}),
        ),
        free_caption=str(row.get("free_caption") or evidence.get("free") or "Image evidence is available."),
        task_evidence=str(row.get("task_evidence") or evidence.get("task_visible") or "Task evidence is available."),
        task_visible_evidence=str(row.get("task_visible_evidence") or evidence.get("task_visible") or "Visible evidence."),
        task_infer_evidence=str(row.get("task_infer_evidence") or evidence.get("task_infer") or "Inference evidence."),
        task_solve_evidence=str(row.get("task_solve_evidence") or evidence.get("task_solve") or "Solve evidence."),
    )


def _select_trainable_parameter(model: torch.nn.Module, name_contains: str) -> None:
    selected = None
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if selected is None and name_contains in name:
            selected = (name, param)
    if selected is None:
        for name, param in model.named_parameters():
            selected = (name, param)
            break
    if selected is None:
        raise RuntimeError("student model has no parameters")
    name, param = selected
    param.requires_grad_(True)
    print(f"selected trainable parameter: {name} shape={tuple(param.shape)}")


def _selected_parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    values = [param.detach().reshape(-1) for param in model.parameters() if param.requires_grad]
    if not values:
        raise RuntimeError("no trainable parameter selected")
    return torch.cat(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", required=True, help="Evidence cache JSONL, not offline score JSONL.")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--student-model", default=os.environ.get("FC_OPD_STUDENT_MODEL", "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--teacher-url", default=os.environ.get("FC_OPD_TEACHER_URL", "http://127.0.0.1:18080"))
    parser.add_argument("--degraded-image-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--student-top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--trainable-name-contains", default="lm_head")
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
