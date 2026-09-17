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


def geometry3k_training_prompt_answer_only(question: str) -> list[dict[str, str]]:
    """No-reasoning variant (v3, ``answer_only``) for the closing-behavior gate.

    This is a validation-only prompt ablation, not an approved training
    contract.  It tests whether explicitly suppressing visible reasoning makes
    Qwen3.5 terminate reliably inside the 2048-token response budget.  Accuracy
    must be compared with ``boxed_only`` before this prompt can be considered
    for training.
    """
    question = str(question).strip()
    if not question:
        raise ValueError("Geometry3K question must be non-empty")
    return [{
        "role": "user",
        "content": (
            f"<image>\n{question}\n\n"
            "Return exactly one line and nothing else: "
            "\\boxed{<final answer>}. Do not show reasoning."
        ),
    }]


# Version → prompt builder for FCOPDDataset.  ``v1`` is the canonical default;
# ``boxed_only`` and ``answer_only`` are separately versioned 2048-token
# closing-behavior ablations.
GEOMETRY3K_PROMPT_BUILDERS = {
    "v1": geometry3k_training_prompt,
    "boxed_only": geometry3k_training_prompt_boxed_only,
    "answer_only": geometry3k_training_prompt_answer_only,
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


def clean_geometry3k_question_rows(rows, prompt_version: str = "v1"):
    """Normalize question rows to the versioned prompt contract.

    Dataset-agnostic helper used by ``FCOPDDataset._clean_prompts`` so the
    mapping logic is unit-testable without importing verl/transformers.
    Rows are copied; non-empty questions get a ``prompt`` field built by the
    selected versioned builder.
    """
    builder = get_geometry3k_prompt_builder(prompt_version)
    out = []
    for row in rows:
        row = dict(row)
        question = str(row.get("question", "")).strip()
        if question:
            row["prompt"] = builder(question)
        out.append(row)
    return out
