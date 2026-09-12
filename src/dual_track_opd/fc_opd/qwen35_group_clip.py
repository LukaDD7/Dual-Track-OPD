"""Pure aggregation helpers for training-rollout group-level clip analysis.

Kept import-light (no transformers/verl) so they are unit-testable in any env.
The CLI wrapper lives in ``scripts/qwen35_group_clip_analysis.py``.
"""

from __future__ import annotations

import statistics


def group_key(uid: str) -> str:
    """Prompt-group key: uid = {sample_uid}_{rollout}_{output}."""
    parts = str(uid).rsplit("_", 2)
    return parts[0] if len(parts) == 3 else str(uid)


def clean_response(text: str) -> str:
    return text[len("assistant\n"):] if text.startswith("assistant\n") else text


def aggregate_group_metrics(records) -> dict:
    """Aggregate sequence/group metrics from per-row records.

    Each record is a dict with keys: ``group``, ``length``, ``is_clip``,
    ``is_correct``, ``has_boxed``, ``cap``.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    lengths = [r["length"] for r in records]
    seq_clip = sum(1 for r in records if r["is_clip"])
    seq_eos = n - seq_clip
    seq_boxed = sum(1 for r in records if r["has_boxed"])
    seq_correct = sum(1 for r in records if r["is_correct"])

    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)

    g_any_clip = sum(1 for g in groups.values() if any(x["is_clip"] for x in g))
    g_all_clip = sum(1 for g in groups.values() if all(x["is_clip"] for x in g))
    g_all_eos = sum(1 for g in groups.values() if all(not x["is_clip"] for x in g))
    g_any_correct = sum(1 for g in groups.values() if any(x["is_correct"] for x in g))
    g_clipped_counts = [sum(1 for x in g if x["is_clip"]) for g in groups.values()]

    def _pct(p):
        s = sorted(lengths)
        return s[min(n - 1, int(n * p))]

    return {
        "n": n,
        "cap": records[0]["cap"],
        "length": {
            "mean": round(statistics.mean(lengths), 1),
            "p50": _pct(0.50),
            "p90": _pct(0.90),
            "p95": _pct(0.95),
            "max": max(lengths),
        },
        "sequence": {
            "clip_rate": round(seq_clip / n, 4),
            "n_clip": seq_clip,
            "eos_rate": round(seq_eos / n, 4),
            "n_eos": seq_eos,
            "boxed_rate": round(seq_boxed / n, 4),
            "n_boxed": seq_boxed,
            "accuracy_rate": round(seq_correct / n, 4),
            "n_correct": seq_correct,
        },
        "group": {
            "n_groups": len(groups),
            "any_clip_rate": round(g_any_clip / len(groups), 4),
            "n_groups_any_clip": g_any_clip,
            "all_clip_rate": round(g_all_clip / len(groups), 4),
            "n_groups_all_clip": g_all_clip,
            "all_eos_rate": round(g_all_eos / len(groups), 4),
            "n_groups_all_eos": g_all_eos,
            "any_correct_rate": round(g_any_correct / len(groups), 4),
            "n_groups_any_correct": g_any_correct,
            "mean_clipped_per_group": round(statistics.mean(g_clipped_counts), 3),
            "rollouts_per_group": round(n / len(groups), 2),
        },
    }
