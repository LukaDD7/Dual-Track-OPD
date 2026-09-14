"""Fresh no-prefix evaluation statistics for the STP-OPD mechanics pilot.

Pure statistics (no model I/O): Jeffreys posterior means, prompt-macro
comparisons, and the handoff §7 P1 mechanics gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def jeffreys_posterior_mean(pass_count: int, k: int) -> float:
    """Beta(1/2, 1/2) posterior mean: (pass + 1/2) / (k + 1)."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= pass_count <= k:
        raise ValueError("pass_count must lie in [0, k]")
    return (float(pass_count) + 0.5) / (float(k) + 1.0)


def per_prompt_posterior_means(
    passes_by_prompt: Mapping[str, int],
    k: int,
) -> dict[str, float]:
    """Jeffreys posterior mean per prompt from pass counts."""

    return {
        str(prompt_id): jeffreys_posterior_mean(int(pass_count), k)
        for prompt_id, pass_count in passes_by_prompt.items()
    }


def prompt_macro_lift(
    treatment: Mapping[str, float],
    baseline: Mapping[str, float],
) -> float:
    """Mean of per-prompt treatment-minus-baseline differences."""

    common = sorted(set(treatment) & set(baseline))
    if not common:
        raise ValueError("treatment and baseline share no prompts")
    diffs = [treatment[prompt_id] - baseline[prompt_id] for prompt_id in common]
    return sum(diffs) / len(diffs)


def p1_mechanics_gate(
    a3_posterior: Mapping[str, float],
    a0_posterior: Mapping[str, float],
    *,
    improved_prompts_required: int = 4,
    lift_required: float = 0.10,
) -> dict[str, object]:
    """Handoff §7 gate: >= 4/7 prompts improve; A3 minus A0 lift >= 0.10."""

    common = sorted(set(a3_posterior) & set(a0_posterior))
    improved = [
        prompt_id
        for prompt_id in common
        if a3_posterior[prompt_id] > a0_posterior[prompt_id]
    ]
    lift = prompt_macro_lift(a3_posterior, a0_posterior)
    passed = (
        len(common) >= improved_prompts_required
        and len(improved) >= improved_prompts_required
        and lift >= lift_required
    )
    return {
        "n_prompts": len(common),
        "n_improved": len(improved),
        "improved_prompts": improved,
        "prompt_macro_lift": round(lift, 4),
        "improved_prompts_required": improved_prompts_required,
        "lift_required": lift_required,
        "passed": bool(passed),
    }


def support_transition_buckets(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    stable_threshold: float = 0.5,
) -> dict[str, int]:
    """Count prompts by transition bucket: rare->stable, rare->no-correct."""

    rare_to_stable = 0
    rare_to_no_correct = 0
    for prompt_id in set(before) & set(after):
        pre = before[prompt_id]
        post = after[prompt_id]
        if pre < stable_threshold and post >= stable_threshold:
            rare_to_stable += 1
        elif pre < stable_threshold and post <= pre:
            rare_to_no_correct += 1
    return {
        "rare_to_stable": rare_to_stable,
        "rare_to_no_correct": rare_to_no_correct,
    }
