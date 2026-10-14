#!/usr/bin/env python3
"""Prepare the ViRL-medium + Cauldron-broad Project15 RL pilot pool.

This pool deliberately excludes MMF-hard. The prior MMF-only RL arm improved
long-chain diagnostics but regressed broad MCQ/VQA benchmarks. The next pilot
keeps ViRL39K as the reasoning/math signal and Cauldron as broad MCQ/short-answer
coverage. All selected answers are rule-verifiable by ``mmf_reward.py``.

The pool is not difficulty-calibrated by step2175 pass rates yet. It is the
fast, CPU-side v1 pilot pool; after the 50-step run, replace it with a
pass-rate-filtered v2 pool.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from prep_mixed_rl_pools import (
    IMAGES_INLINE_SCHEMA,
    gt_rule_kind,
    inline_cauldron_bytes,
    write_shards,
)


def load_parquet_rows(pattern: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet shards matching {pattern}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def virl_source(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    source = str(extra.get("source") or "")
    if source and source != "ViRL39K":
        return source
    uid = str(row.get("sample_uid") or "")
    if uid.startswith("ViRL39K:") and len(uid.split(":")) > 2:
        return uid.split(":", 2)[1].split("-", 1)[0]
    return "unknown"


def select_by_source(
    rows: list[dict[str, Any]], total: int, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[virl_source(row)].append(row)

    eligible = sum(len(group) for group in by_source.values())
    target = min(total, eligible)
    quota = {
        source: int(len(group) * target / eligible)
        for source, group in by_source.items()
    }
    remainder = target - sum(quota.values())
    order = sorted(
        by_source,
        key=lambda source: len(by_source[source]) * target / eligible - quota[source],
        reverse=True,
    )
    for source in order[:remainder]:
        quota[source] += 1

    picked: dict[str, set[int]] = {}
    residual = 0
    for source, group in by_source.items():
        count = min(quota[source], len(group))
        indexes = set(rng.sample(range(len(group)), count) if count < len(group) else range(len(group)))
        picked[source] = indexes
        residual += quota[source] - count

    if residual:
        for source in sorted(
            by_source,
            key=lambda source: len(by_source[source]) - len(picked[source]),
            reverse=True,
        ):
            group = by_source[source]
            available = [i for i in range(len(group)) if i not in picked[source]]
            if not available:
                continue
            take = min(residual, len(available))
            picked[source].update(rng.sample(available, take))
            residual -= take
            if residual == 0:
                break

    selected: list[dict[str, Any]] = []
    for source in sorted(by_source):
        group = by_source[source]
        selected.extend(group[i] for i in sorted(picked[source]))
    return selected, {source: len(picked[source]) for source in sorted(picked)}


def select_cauldron(
    rows: list[dict[str, Any]], per_subset: int, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subset[str(row.get("data_source", ""))].append(row)

    selected: list[dict[str, Any]] = []
    taken: dict[str, int] = {}
    for subset in sorted(by_subset):
        group = by_subset[subset]
        if len(group) > per_subset:
            indexes = sorted(rng.sample(range(len(group)), per_subset))
            group = [group[i] for i in indexes]
        selected.extend(group)
        taken[subset] = len(group)
    return selected, taken


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.setdefault("hint", "")
        row.setdefault("hint_reason", "")
        row.setdefault("prompt_with_hint", [])
        row.setdefault("ability", "reasoning")

        extra = row.setdefault("extra_info", {})
        extra["question"] = str(row.get("question", ""))
        extra["source"] = str(extra.get("source") or row.get("data_source", ""))
        extra.setdefault("gt_source", "")
        extra.setdefault("original_answer", "")

        reward_model = row.setdefault("reward_model", {})
        reward_model.setdefault("style", "rule")
        reward_model.setdefault("ground_truth", "")

        row.setdefault("data_source", str(extra["source"]))
        row.setdefault("prompt", [{"role": "user", "content": str(row.get("question", ""))}])
        row.setdefault("sample_uid", f"{row.get('data_source', 'unknown')}:{id(row)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy")
    base = root / "fc-opd-storage/outputs/fc_opd/sft_rl"
    parser.add_argument("--virl-train", default=str(root / "fc-opd-storage/outputs/fc_opd/virl39k_va_opd/train.parquet"))
    parser.add_argument("--cauldron-dir", default=str(base / "hint_pools/cauldron_pool_100k"))
    parser.add_argument("--out-dir", type=Path, default=base / "virl_cauldron_rl_v1")
    parser.add_argument("--virl-rows", type=int, default=15000)
    parser.add_argument("--cauldron-per-subset", type=int, default=1000)
    parser.add_argument("--keep-virl-latex", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--prefix", default="virl_cauldron_train")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    allowed = {"letter", "yesno", "number"}
    if args.keep_virl_latex:
        allowed.add("latex")

    virl_all = load_parquet_rows(args.virl_train)
    virl_eligible = [row for row in virl_all if gt_rule_kind((row.get("reward_model") or {}).get("ground_truth")) in allowed]
    virl_selected, virl_taken = select_by_source(virl_eligible, args.virl_rows, rng)

    cauldron_all = load_parquet_rows(f"{args.cauldron_dir}/cauldron_hint_pool__part_*.parquet")
    cauldron_eligible = [
        row
        for row in cauldron_all
        if gt_rule_kind((row.get("reward_model") or {}).get("ground_truth")) in {"letter", "yesno", "number"}
    ]
    cauldron_selected_raw, cauldron_taken = select_cauldron(
        cauldron_eligible, args.cauldron_per_subset, rng
    )
    cauldron_selected = []
    dropped_missing_image = 0
    for row in cauldron_selected_raw:
        if inline_cauldron_bytes(row):
            cauldron_selected.append(row)
        else:
            dropped_missing_image += 1

    rows = normalize_rows(virl_selected + cauldron_selected)
    paths = write_shards(rows, args.out_dir, args.prefix, args.shard_rows)

    stats = {
        "pool": "virl_cauldron_rl_v1",
        "virl_train": str(args.virl_train),
        "cauldron_dir": str(args.cauldron_dir),
        "seed": args.seed,
        "virl_total": len(virl_all),
        "virl_rule_ok": len(virl_eligible),
        "virl_selected": len(virl_selected),
        "virl_per_source": virl_taken,
        "virl_gt_kinds": dict(Counter(gt_rule_kind((r.get("reward_model") or {}).get("ground_truth")) for r in virl_selected)),
        "cauldron_total": len(cauldron_all),
        "cauldron_rule_ok": len(cauldron_eligible),
        "cauldron_selected": len(cauldron_selected),
        "cauldron_per_subset": cauldron_taken,
        "cauldron_gt_kinds": dict(Counter(gt_rule_kind((r.get("reward_model") or {}).get("ground_truth")) for r in cauldron_selected)),
        "cauldron_dropped_missing_image": dropped_missing_image,
        "total_rows": len(rows),
        "virl_share": round(len(virl_selected) / len(rows), 4) if rows else 0.0,
        "cauldron_share": round(len(cauldron_selected) / len(rows), 4) if rows else 0.0,
        "reward": "file:///inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/sft_rl/mmf_reward.py",
        "difficulty_calibration": "none; v1 fast pilot before step2175 pass-rate filtering",
        "shard_rows": args.shard_rows,
        "shards": paths,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "virl_cauldron_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in stats.items() if key != "shards"}, indent=2))
    print(f"wrote {len(paths)} shards to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
