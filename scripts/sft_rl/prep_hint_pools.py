#!/usr/bin/env python3
"""Prepare full-scale PTD-PO hint-generation pools (MMF + Cauldron).

Two pools, aligned with the SFT-then-RL track's RL data design (runbook §7.9):

  mmf       MMFineReason answer-non-null pool (117,688 rows) filtered to
            rule-verifiable GT forms (letter / yes-no / numeric / LaTeX) ->
            ~95K rows. This is the core PTD pool: pass_rate=0 hardest samples,
            maximal PTD activation. SFT/RL overlap is intentionally kept
            (arXiv 2604.23747 §3 practice; runbook 2026-08-22 decision).

  cauldron  the_cauldron 14 closed-form subsets extracted from sft_full train
            shards. sft_full keeps each Cauldron row as a multi-turn
            conversation sharing one image, so this script expands every
            (user, assistant) turn pair into an independent QA sample and
            re-normalizes <image> placeholders per sample. Images stay as
            external image_url references (no byte copy). --max-per-subset
            gives deterministic stratified control of the pool size.

Output schema matches what build_mmf_hints.py consumes (question / images /
reward_model.ground_truth / extra_info), sharded to <=1500 rows per file.

Usage:
  python3 scripts/sft_rl/prep_hint_pools.py --pool mmf \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/mmf_pool
  python3 scripts/sft_rl/prep_hint_pools.py --pool cauldron \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/cauldron_pool \
    --max-per-subset 0
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CLOSED_FORM_SUBSETS = (
    "vqav2", "visual7w", "aokvqa", "tallyqa", "vsr", "ai2d", "scienceqa",
    "tqa", "chartqa", "textvqa", "docvqa", "infographic_vqa", "iconqa", "raven",
)


def gt_type(gt: str) -> str:
    a = gt.strip()
    if re.fullmatch(r"\(?[A-Ea-e]\)?\.?", a):
        return "letter"
    if re.fullmatch(r"(yes|no|true|false)", a.lower()):
        return "yesno"
    if re.fullmatch(r"-?\d+(\.\d+)?", a):
        return "pure_number"
    if re.search(r"\d", a):
        return "has_number"
    return "other"


def normalize_placeholders(question: str, n_images: int) -> str:
    """Re-normalize <image> count so an expanded turn is a standalone sample.

    In sft_full only the first turn of a Cauldron conversation carries the
    <image> placeholder; later turns share the image but have none. After
    expansion each sample must carry exactly len(images) placeholders.
    """
    body = question.replace("<image>", "").strip()
    if n_images <= 0:
        return body
    return ("\n".join(["<image>"] * n_images) + "\n" + body).strip()


def clean_cauldron_answer(answer: str) -> str:
    """Strip assistant-turn boilerplate like 'Answer: B' -> 'B'.

    Keeps the underlying GT directly comparable for rule rewards and hint QC.
    """
    a = answer.strip()
    a = re.sub(r"^\s*(?:the\s+)?answer\s*(?:is)?\s*[:\-]\s*", "", a, flags=re.IGNORECASE)
    return a.strip()


def extract_turn_pairs(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(messages):
        if (
            messages[i].get("role") == "user"
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "assistant"
        ):
            pairs.append((str(messages[i].get("content", "")), str(messages[i + 1].get("content", ""))))
            i += 2
        else:
            i += 1
    return pairs


MMF_SCHEMA = pa.schema([
    pa.field("data_source", pa.string()),
    pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("ability", pa.string()),
    pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    pa.field("extra_info", pa.struct([
        ("question", pa.string()), ("source", pa.string()),
        ("gt_source", pa.string()), ("original_answer", pa.string()),
    ])),
    pa.field("question", pa.string()),
    pa.field("sample_uid", pa.string()),
])

CAULDRON_SCHEMA = pa.schema([
    pa.field("data_source", pa.string()),
    pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("image_url", pa.string())]))),
    pa.field("ability", pa.string()),
    pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    pa.field("extra_info", pa.struct([
        ("question", pa.string()), ("source", pa.string()),
        ("gt_source", pa.string()), ("original_answer", pa.string()),
    ])),
    pa.field("question", pa.string()),
    pa.field("sample_uid", pa.string()),
])


def write_shards(rows: list[dict[str, Any]], out_dir: Path, prefix: str,
                 schema: pa.Schema, shard_rows: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i in range(0, len(rows), shard_rows):
        p = out_dir / f"{prefix}__part_{i // shard_rows:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[i : i + shard_rows], schema=schema), p, compression="zstd")
        paths.append(str(p))
    return paths


def prep_mmf(args: argparse.Namespace) -> dict[str, Any]:
    pf = pq.ParquetFile(args.input)
    total = pf.metadata.num_rows
    kept: list[dict[str, Any]] = []
    gt_stats: Counter[str] = Counter()
    # iter_batches avoids the "nested chunked array outputs" failure that hits
    # to_pylist on this single-row-group, 5.9GB nested file
    for batch in pf.iter_batches(batch_size=1024):
        t = pa.Table.from_batches([batch], schema=pf.schema_arrow)
        for r in t.to_pylist():
            gt = str((r.get("reward_model") or {}).get("ground_truth", ""))
            k = gt_type(gt)
            gt_stats[k] += 1
            if k == "other" and not args.keep_other:
                continue
            kept.append(r)
    paths = write_shards(kept, Path(args.out_dir), "mmf_hint_pool", MMF_SCHEMA, args.shard_rows)
    return {
        "pool": "mmf",
        "input": str(args.input),
        "total_rows": total,
        "kept_rows": len(kept),
        "gt_stats": dict(gt_stats),
        "dropped_other": gt_stats["other"] if not args.keep_other else 0,
        "shards": paths,
    }


def prep_cauldron(args: argparse.Namespace) -> dict[str, Any]:
    files = sorted(glob.glob(f"{args.input_dir}/sft_full_train__part_*.parquet"))
    if not files:
        raise SystemExit(f"no sft_full_train__part_*.parquet under {args.input_dir}")
    by_src: dict[str, list[dict[str, Any]]] = {}
    conv_rows = 0
    for f in files:
        t = pq.read_table(f, columns=["source", "messages", "images", "image_hash"])
        for r in t.to_pylist():
            src = str(r.get("source", ""))
            if src not in CLOSED_FORM_SUBSETS:
                continue
            conv_rows += 1
            imgs = r.get("images") or []
            image_urls = [{"image_url": str(i.get("image_url", ""))} for i in imgs]
            n_img = len(image_urls)
            ih = str(r.get("image_hash", ""))
            for turn_idx, (user, assistant) in enumerate(extract_turn_pairs(r.get("messages") or [])):
                q = normalize_placeholders(user, n_img)
                ans = clean_cauldron_answer(assistant)
                if not q or not ans:
                    continue
                uid = f"{src}:{ih}:{turn_idx}"
                by_src.setdefault(src, []).append({
                    "data_source": src,
                    "prompt": [{"role": "user", "content": q}],
                    "images": image_urls,
                    "ability": "reasoning",
                    "reward_model": {"style": "rule", "ground_truth": ans},
                    "extra_info": {
                        "question": q,
                        "source": src,
                        "gt_source": "assistant_turn",
                        "original_answer": "",
                    },
                    "question": q,
                    "sample_uid": uid,
                })

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    per_subset: dict[str, int] = {}
    for src in sorted(by_src):
        rows = by_src[src]
        if args.max_per_subset > 0 and len(rows) > args.max_per_subset:
            idx = sorted(rng.sample(range(len(rows)), args.max_per_subset))
            rows = [rows[i] for i in idx]
        per_subset[src] = len(rows)
        selected.extend(rows)

    paths = write_shards(selected, Path(args.out_dir), "cauldron_hint_pool",
                         CAULDRON_SCHEMA, args.shard_rows)
    return {
        "pool": "cauldron",
        "input_dir": str(args.input_dir),
        "subsets": list(CLOSED_FORM_SUBSETS),
        "conversation_rows": conv_rows,
        "qa_pairs_total": sum(len(v) for v in by_src.values()),
        "kept_rows": len(selected),
        "per_subset": per_subset,
        "max_per_subset": args.max_per_subset,
        "seed": args.seed,
        "shards": paths,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=("mmf", "cauldron"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path,
                    default=Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
                                 "fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/"
                                 "mmfinereason_rl.parquet"),
                    help="mmf: single RL parquet")
    ap.add_argument("--input-dir", type=Path,
                    default=Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
                                 "fc-opd-storage/outputs/fc_opd/sft_rl/sft_full/train"),
                    help="cauldron: sft_full train shard dir")
    ap.add_argument("--keep-other", action="store_true",
                    help="mmf: keep free-text GT rows (default: drop non-verifiable)")
    ap.add_argument("--max-per-subset", type=int, default=0,
                    help="cauldron: deterministic per-subset cap (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-rows", type=int, default=1500)
    args = ap.parse_args()

    if args.pool == "mmf":
        stats = prep_mmf(args)
    else:
        stats = prep_cauldron(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "hint_pool_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "shards"},
                     ensure_ascii=False, indent=2))
    print(f"wrote {len(stats['shards'])} shards under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
