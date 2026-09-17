#!/usr/bin/env python3
"""Convert VisualWebInstruct to verl SFT + RL/GRPO parquet (repo schema).

Input : dataset/VisualWebInstruct/data/train_batch_*.parquet
        columns: question, answer, question_images, solution_images, url, idx
Output: fc-opd-storage/outputs/fc_opd/sft_rl/visualwebinstruct/
          vwinstruct_sft__part_*.parquet    (SFT: Q -> A response target)
          vwinstruct_rl_train__part_*.parquet (RL: rule-verifiable GT only)
          vwinstruct_stats.json

Policies
  SFT (main product, all rows):
    - keep rows with non-empty question, answer, and >=1 image across
      question_images + solution_images
    - inject "<image>" placeholders at the START of the question when the
      question text has fewer placeholders than images (source data often
      references figures verbally without placeholders)
    - response = answer verbatim (no CoT in source)
  RL (rule-verifiable subset):
    - repo 4-class GT filter (letter/yesno/pure_number/has_number)
    - drop answer in {"", "Not found"} and "other" free-text
    - placeholder check same as SFT
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SFT_SCHEMA = pa.schema([
    pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("source", pa.string()),
    pa.field("pass_rate", pa.float64()),
    pa.field("image_hash", pa.string()),
])

RL_SCHEMA = pa.schema([
    pa.field("data_source", pa.string()),
    pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("ability", pa.string()),
    pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    pa.field("extra_info", pa.struct([
        ("question", pa.string()), ("source", pa.string()),
        ("gt_source", pa.string()), ("gt_type", pa.string()),
        ("original_answer", pa.string()),
    ])),
    pa.field("question", pa.string()),
    pa.field("sample_uid", pa.string()),
])


def gt_type(gt: str) -> str:
    """Identical to sample_mmf_rl.py (repo GT policy)."""
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


def fix_placeholders(question: str, n_imgs: int) -> str | None:
    """Make count("<image>") == n_imgs; inject at front. None = unfixable."""
    n_ph = question.count("<image>")
    if n_ph == n_imgs:
        return question
    if n_ph < n_imgs:
        return "<image>" * (n_imgs - n_ph) + question
    return None  # more placeholders than images


def norm_images(imgs) -> list[dict]:
    if isinstance(imgs, dict):
        imgs = [imgs]
    return [{"bytes": i.get("bytes"), "path": i.get("path") or ""} for i in (imgs or [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--skip-rl", action="store_true", help="only write SFT shards")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/train_batch_*.parquet"))
    if not files:
        raise SystemExit(f"no train_batch_*.parquet under {args.input_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows: list[dict] = []
    rl_rows: list[dict] = []
    stats = {"total": 0, "sft_kept": 0, "rl_kept": 0, "no_answer": 0,
             "no_images": 0, "placeholder_unfixable": 0}
    gt_stats: Counter = Counter()

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                stats["total"] += 1
                question = str(r.get("question") or "")
                answer = str(r.get("answer") or "")
                q_imgs = norm_images(r.get("question_images"))
                s_imgs = norm_images(r.get("solution_images"))
                imgs = q_imgs + s_imgs
                if not question or answer.strip().lower() in {"", "not found"}:
                    stats["no_answer"] += 1
                    continue
                if not imgs:
                    stats["no_images"] += 1
                    continue
                q2 = fix_placeholders(question, len(imgs))
                if q2 is None:
                    stats["placeholder_unfixable"] += 1
                    continue
                # SFT row
                h = ",".join(hashlib_sha16(i["bytes"]) for i in imgs)
                sft_rows.append({
                    "messages": [
                        {"role": "user", "content": q2},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": imgs,
                    "source": "VisualWebInstruct",
                    "pass_rate": -1.0,
                    "image_hash": h,
                })
                stats["sft_kept"] += 1
                # RL row (rule-verifiable only)
                t = gt_type(answer)
                gt_stats[t] += 1
                if t == "other":
                    continue
                url = str(r.get("url") or "")[:200]
                idx = str(r.get("idx") or stats["total"])
                rl_rows.append({
                    "data_source": "VisualWebInstruct",
                    "prompt": [{"role": "user", "content": q2}],
                    "images": imgs,
                    "ability": "reasoning",
                    "reward_model": {"style": "rule", "ground_truth": answer.strip()},
                    "extra_info": {
                        "question": q2,
                        "source": "VisualWebInstruct",
                        "gt_source": "answer",
                        "gt_type": t,
                        "original_answer": answer,
                    },
                    "question": q2,
                    "sample_uid": f"VWI:{idx}",
                })
                stats["rl_kept"] += 1
        print(f"[scan] {f}: sft={stats['sft_kept']} rl={stats['rl_kept']}", flush=True)

    # write SFT shards
    n_sft = 0
    sft_paths = []
    for i in range(0, len(sft_rows), args.shard_rows):
        p = out_dir / f"vwinstruct_sft__part_{n_sft:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(sft_rows[i : i + args.shard_rows], schema=SFT_SCHEMA), p)
        sft_paths.append(str(p))
        n_sft += 1

    # write RL shards
    n_rl = 0
    rl_paths = []
    if not args.skip_rl:
        for i in range(0, len(rl_rows), args.shard_rows):
            p = out_dir / f"vwinstruct_rl_train__part_{n_rl:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(rl_rows[i : i + args.shard_rows], schema=RL_SCHEMA), p)
            rl_paths.append(str(p))
            n_rl += 1

    stats_out = {
        "total_rows": stats["total"],
        "sft_kept": stats["sft_kept"],
        "rl_kept": stats["rl_kept"],
        "no_answer": stats["no_answer"],
        "no_images": stats["no_images"],
        "placeholder_unfixable": stats["placeholder_unfixable"],
        "gt_stats": dict(gt_stats),
        "sft_shards": sft_paths,
        "rl_shards": rl_paths,
    }
    (out_dir / "vwinstruct_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"SFT: {stats['sft_kept']} rows -> {n_sft} shards; RL: {stats['rl_kept']} rows -> {n_rl} shards")
    print(f"GT stats: {dict(gt_stats)}")


def hashlib_sha16(b: bytes | None) -> str:
    import hashlib
    return hashlib.sha256(b or b"").hexdigest()[:16]


if __name__ == "__main__":
    main()
