"""Render condition-specific Qwen-style teacher messages."""

from __future__ import annotations

from dataclasses import dataclass

from .conditions import Condition, ConditionInputs


@dataclass(frozen=True)
class RenderedTeacherPrompt:
    condition: Condition
    messages: tuple[dict[str, object], ...]
    image_paths: tuple[str, ...]


def _text_prompt(question: str) -> str:
    return f"Question:\n{question}"


def render_teacher_prompt(
    condition: Condition | str,
    question: str,
    inputs: ConditionInputs,
) -> RenderedTeacherPrompt:
    """Render one teacher condition — clean VA-OPD: image + question, no format bias."""

    condition = Condition(condition)
    if not question.strip():
        raise ValueError("question must be non-empty")

    if condition is Condition.FULL:
        image_path = inputs.full_image.path
        text = _text_prompt(question)
    elif condition in {Condition.BLUR, Condition.DEGRADED}:
        image_path = inputs.degraded_image.path
        text = _text_prompt(question)
    elif condition is Condition.FREE:
        image_path = None
        text = f"Image description:\n{inputs.free_caption}\n\nQuestion:\n{question}"
    elif condition in {Condition.TASK, Condition.TASK_VISIBLE}:
        image_path = None
        text = f"Evidence:\n{inputs.task_visible_text}\n\nQuestion:\n{question}"
    elif condition is Condition.TASK_INFER:
        if inputs.task_infer_evidence is None:
            raise ValueError("task_infer condition requires task_infer_evidence")
        image_path = None
        text = f"Context:\n{inputs.task_infer_evidence}\n\nQuestion:\n{question}"
    elif condition is Condition.TASK_SOLVE:
        if inputs.task_solve_evidence is None:
            raise ValueError("task_solve condition requires task_solve_evidence")
        image_path = None
        text = f"Context:\n{inputs.task_solve_evidence}\n\nQuestion:\n{question}"
    else:
        if inputs.verified_facts is None or inputs.verified_facts_source is None:
            raise ValueError("fact condition requires verified facts and their source")
        image_path = None
        text = f"Facts (source: {inputs.verified_facts_source}):\n{inputs.verified_facts}\n\nQuestion:\n{question}"

    content: list[dict[str, object]] = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": text})
    return RenderedTeacherPrompt(
        condition=condition,
        messages=({"role": "user", "content": content},),
        image_paths=() if image_path is None else (image_path,),
    )
