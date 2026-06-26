"""Render condition-specific Qwen-style teacher messages."""

from __future__ import annotations

from dataclasses import dataclass

from .conditions import Condition, ConditionInputs


STRUCTURED_RESPONSE_INSTRUCTION = """Continue using exactly this structure:
<visual_evidence>
Only image-grounded facts relevant to the question.
</visual_evidence>
<reasoning>
Reason from the available evidence and the question.
</reasoning>
<answer>
Final answer.
</answer>"""


@dataclass(frozen=True)
class RenderedTeacherPrompt:
    condition: Condition
    messages: tuple[dict[str, object], ...]
    image_paths: tuple[str, ...]


def _text_prompt(context: str, question: str) -> str:
    return f"{context}\n\nQuestion:\n{question}\n\n{STRUCTURED_RESPONSE_INSTRUCTION}"


def render_teacher_prompt(
    condition: Condition | str,
    question: str,
    inputs: ConditionInputs,
) -> RenderedTeacherPrompt:
    """Render one teacher condition without generating or altering evidence."""

    condition = Condition(condition)
    if not question.strip():
        raise ValueError("question must be non-empty")

    if condition is Condition.FULL:
        image_path = inputs.full_image.path
        text = _text_prompt("Use the full image to answer the question.", question)
    elif condition in {Condition.BLUR, Condition.DEGRADED}:
        image_path = inputs.degraded_image.path
        text = _text_prompt("Use the degraded image to answer the question.", question)
    elif condition is Condition.FREE:
        image_path = None
        text = _text_prompt(
            f"Image description:\n{inputs.free_caption}",
            question,
        )
    elif condition is Condition.TASK:
        image_path = None
        text = _text_prompt(
            f"Question-conditioned visual evidence:\n{inputs.task_evidence}",
            question,
        )
    else:
        if inputs.verified_facts is None or inputs.verified_facts_source is None:
            raise ValueError("fact condition requires verified facts and their source")
        image_path = None
        text = _text_prompt(
            "Externally verified visual facts "
            f"(source: {inputs.verified_facts_source}):\n{inputs.verified_facts}",
            question,
        )

    content: list[dict[str, object]] = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": text})
    return RenderedTeacherPrompt(
        condition=condition,
        messages=({"role": "user", "content": content},),
        image_paths=() if image_path is None else (image_path,),
    )
