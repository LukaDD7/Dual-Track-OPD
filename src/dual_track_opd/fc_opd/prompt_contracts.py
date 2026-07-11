"""Canonical prompts shared by rollout datasets and teacher diagnostics."""

from __future__ import annotations


def geometry3k_training_prompt(question: str) -> list[dict[str, str]]:
    """Return the Qwen3-VL-Instruct image-question contract.

    Qwen3-VL exposes reasoning behavior through the model variant and chat
    template.  Injecting a legacy instruction that demands literal ``<think>``
    tags makes the 32B Instruct teacher terminate immediately on Geometry3K.
    Keep the user turn minimal and let the model's own template control its
    reasoning.  A short final-answer constraint makes evaluation reliable
    without imposing legacy chain-of-thought markup.
    """
    question = str(question).strip()
    if not question:
        raise ValueError("Geometry3K question must be non-empty")
    return [{
        "role": "user",
        "content": f"<image>\n{question}\n\nPut the final answer in \\boxed{{}}.",
    }]
