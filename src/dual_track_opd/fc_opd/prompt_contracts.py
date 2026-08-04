"""Canonical prompts shared by rollout datasets and teacher diagnostics."""

from __future__ import annotations


def geometry3k_training_prompt(question: str) -> list[dict[str, str]]:
    """Return the canonical Qwen3-VL-Instruct image-question contract (v1).

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


def geometry3k_training_prompt_boxed_only(question: str) -> list[dict[str, str]]:
    """Concise-closing variant (v2, ``boxed_only``) for the truncation ablation.

    Same image-question contract as v1 with one changed variable: the closing
    instruction now asks for a brief reasoning followed by a single final
    ``\\boxed{}`` line and an explicit stop.  This targets the observed 96%
    clip-rate behavior where the student keeps writing reasoning until the
    response budget runs out.  It does not touch the reward scorer and does not
    inject literal ``<think>`` markup (reasoning mode stays model-controlled).
    """
    question = str(question).strip()
    if not question:
        raise ValueError("Geometry3K question must be non-empty")
    return [{
        "role": "user",
        "content": (
            f"<image>\n{question}\n\n"
            "Briefly reason. Then output your final answer as exactly one line: "
            "\\boxed{<answer>}. Stop immediately after that line; write nothing else."
        ),
    }]


# Version → prompt builder for FCOPDDataset.  ``v1`` is the canonical default;
# ``boxed_only`` is the separately versioned truncation ablation (2048 tokens).
GEOMETRY3K_PROMPT_BUILDERS = {
    "v1": geometry3k_training_prompt,
    "boxed_only": geometry3k_training_prompt_boxed_only,
}


def get_geometry3k_prompt_builder(prompt_version: str):
    """Return the prompt builder for a versioned geometry3k prompt, failing fast on unknown versions."""
    try:
        return GEOMETRY3K_PROMPT_BUILDERS[prompt_version]
    except KeyError:
        raise ValueError(
            f"unknown geometry3k prompt_version: {prompt_version!r}; "
            f"expected one of {sorted(GEOMETRY3K_PROMPT_BUILDERS)}"
        ) from None
