"""Prompt-paired summary + bootstrap CIs for Track B fixed offline eval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prompt_stats(rows: list[dict]) -> dict[str, float]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["sample_uid"])].append(row)
    per_prompt = {}
    for uid, samples in by_uid.items():
        correct = sum(1 for sample in samples if sample["correct"])
        n = len(samples)
        per_prompt[uid] = {
            "n": n,
            "acc": correct / n,
            "pass_at_n": 1.0 if correct > 0 else 0.0,
            "boxed_rate": sum(1 for sample in samples if sample["boxed"]) / n,
            "clip_rate": sum(1 for sample in samples if sample["clipped"]) / n,
            "eos_rate": sum(1 for sample in samples if sample["eos"]) / n,
            "mean_len": float(np.mean([sample["length"] for sample in samples])),
        }
    return per_prompt


def _bootstrap_delta_ci(
    base: dict[str, float],
    treated: dict[str, float],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict:
    uids = sorted(set(base).intersection(treated))
    if len(uids) < 2:
        return {
            "paired_prompt_count": len(uids),
            "base_mean": None,
            "treated_mean": None,
            "mean_delta": None,
            "ci95": [None, None],
        }
    deltas = np.asarray([treated[uid][metric] - base[uid][metric] for uid in uids])
    rng = np.random.default_rng(seed)
    samples = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(axis=1)
    return {
        "paired_prompt_count": len(uids),
        "base_mean": float(np.mean([base[uid][metric] for uid in uids])),
        "treated_mean": float(np.mean([treated[uid][metric] for uid in uids])),
        "mean_delta": float(deltas.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


METRICS = ("acc", "pass_at_n", "boxed_rate", "clip_rate", "eos_rate", "mean_len")


def _cross_seed_summary(
    conditions: list[str],
    stats_by_condition: dict[str, dict[str, dict]],
    paired: dict[str, dict],
) -> dict[str, dict]:
    """Mean delta acc / pass@8 across the seeds of each training step (descriptive)."""

    result: dict[str, dict] = {}
    step_groups: dict[str, list[str]] = {}
    for condition in conditions:
        for step in ("step60", "step120"):
            if condition.endswith(f"_{step}"):
                step_groups.setdefault(step, []).append(condition)
    for step, members in step_groups.items():
        deltas = [paired[member]["acc"]["mean_delta"] for member in members if paired.get(member, {}).get("acc", {}).get("mean_delta") is not None]
        pass_deltas = [
            paired[member]["pass_at_n"]["mean_delta"]
            for member in members
            if paired.get(member, {}).get("pass_at_n", {}).get("mean_delta") is not None
        ]
        if not deltas:
            continue
        result[step] = {
            "seed_conditions": members,
            "acc_mean_delta_across_seeds": float(np.mean(deltas)),
            "acc_delta_range": [float(min(deltas)), float(max(deltas))],
            "pass8_mean_delta_across_seeds": (
                float(np.mean(pass_deltas)) if pass_deltas else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--conditions", required=True,
                        help="comma-separated condition tags, first is base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    stats_by_condition: dict[str, dict[str, dict]] = {}
    for condition in conditions:
        path = eval_dir / f"{condition}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing eval dump: {path}")
        stats_by_condition[condition] = _prompt_stats(_read_rows(path))

    base_condition = conditions[0]
    comparisons = {}
    for condition in conditions[1:]:
        comparisons[condition] = {
            metric: _bootstrap_delta_ci(
                stats_by_condition[base_condition],
                stats_by_condition[condition],
                metric=metric,
                seed=args.seed,
                resamples=args.resamples,
            )
            for metric in METRICS
        }

    result = {
        "base_condition": base_condition,
        "conditions": conditions,
        "prompt_counts": {
            condition: len(stats) for condition, stats in stats_by_condition.items()
        },
        "paired_comparisons": comparisons,
        "cross_seed_summary": _cross_seed_summary(
            conditions, stats_by_condition, comparisons
        ),
        "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.resamples,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
