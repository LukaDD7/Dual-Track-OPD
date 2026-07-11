"""Canonical prompts shared by rollout datasets and teacher diagnostics."""

from __future__ import annotations


def geometry3k_training_prompt(question: str) -> list[dict[str, str]]:
    question = str(question).strip()
    if not question:
        raise ValueError("Geometry3K question must be non-empty")
    return [{
        "role": "user",
        "content": (
            f"<image>\n{question}\n\n"
            "You FIRST think about the reasoning process as an internal monologue "
            "and then provide the final answer. The reasoning process MUST BE "
            "enclosed within <think> </think> tags. The final answer MUST BE put "
            "in \\boxed{}."
        ),
    }]
