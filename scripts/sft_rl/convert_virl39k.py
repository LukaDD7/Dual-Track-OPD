#!/usr/bin/env python3
"""Convert ViRL39K to verl SFT + RL/GRPO parquet (repo schema).

Input : dataset/ViRL39K/39Krelease.parquet  (+ images/ tree on disk)
Output: fc-opd-storage/outputs/fc_opd/sft_rl/virl39k/
          virl39k_sft__part_*.parquet     (SFT needs target text -> RL-only
          virl39k_rl_train__part_*.parquet (<=1500 rows, verl GRPO schema)
          virl39k_rl_stats.json

RL path (main product):
  - answer unboxing: GT is "\\boxed{A}"-style; strip \\boxed{...} before the
    repo 4-class GT filter (letter/yesno/pure_number/has_number, drop other).
  - <image> placeholder injection: 851 rows have 0 placeholders but >=1
    image (images referenced mid-text like "figure (a)"). Insert <image> at
    the start of the question so verl's image-count assertion passes.
  - images loaded from the extracted images/ tree (relative paths in the
    'image' column).

SFT: skipped — ViRL39K ships no long-CoT teacher responses, so there is no
SFT target. For instruction-response SFT style, sample_mmf_rl.py has
teacher-response requirements that this set cannot satisfy.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def gt_type(gt: str) -> str:
    """Identical to sample_mmf_rl.py / prep_mmfinereason_586k_rl.py."""
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


def unbox(answer: str) -> str:
    r"""Strip a single outer \boxed{...} wrapper, if present."""
    a = answer.strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", a)
    return m.group(1).strip() if m else a


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--keep-other", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    rows: list[dict] = []
    stats = {"total": 0, "placeholder_injected": 0, "missing_image": 0,
             "gt_other_dropped": 0}
    gt_stats: Counter = Counter()
    src_stats: Counter = Counter()

    pf = pq.ParquetFile(args.input_parquet)
    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            stats["total"] += 1
            answer = (r.get("answer") or "").strip()
            if not answer:
                continue
            gt = unbox(answer)
            t = gt_type(gt)
            gt_stats[t] += 1
            if t == "other" and not args.keep_other:
                stats["gt_other_dropped"] += 1
                continue
            question = str(r.get("question") or "")
            rel_paths = r.get("image") or []
            # load image bytes
            imgs = []
            ok = True
            for p in rel_paths:
                fp = image_root / p
                if not fp.exists():
                    stats["missing_image"] += 1
                    ok = False
                    break
                imgs.append({"bytes": fp.read_bytes(), "path": p})
            if not ok or not imgs:
                continue
            # placeholder fix: verl asserts count("<image>") == len(images)
            if question.count("<image>") != len(imgs):
                if question.count("<image>") < len(imgs):
                    # inject at the front (images are referenced mid-text)
                    question = "<image>" * (len(imgs) - question.count("<image>")) + question
                    stats["placeholder_injected"] += 1
                else:
                    # more placeholders than images: drop (cannot fix safely)
                    continue
            source = str(r.get("source") or "ViRL39K")
            qid = str(r.get("qid") or stats["total"])
            src_stats[source] += 1
            rows.append({
                "data_source": "ViRL39K",
                "prompt": [{"role": "user", "content": question}],
                "images": imgs,
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "question": question,
                    "source": source,
                    "gt_source": "answer_unboxed",
                    "gt_type": t,
                    "original_answer": answer,
                },
                "question": question,
                "sample_uid": f"ViRL39K:{qid}",
            })
        print(f"[scan] row-group {rg + 1}/{pf.metadata.num_row_groups}: pool={len(rows)}", flush=True)

    n_shards = 0
    paths = []
    for i in range(0, len(rows), args.shard_rows):
        p = out_dir / f"virl39k_rl_train__part_{n_shards:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[i : i + args.shard_rows], schema=RL_SCHEMA), p)
        paths.append(str(p))
        n_shards += 1

    stats_out = {
        "total_rows": stats["total"],
        "gt_stats": dict(gt_stats),
        "gt_other_dropped": stats["gt_other_dropped"],
        "placeholder_injected": stats["placeholder_injected"],
        "missing_image": stats["missing_image"],
        "kept_rows": len(rows),
        "shards": paths,
        "by_source": dict(src_stats.most_common()),
    }
    (out_dir / "virl39k_rl_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"kept {len(rows)} rows -> {n_shards} shards under {out_dir}")
    print(f"GT stats: {dict(gt_stats)}")
    print(f"stats: { {k: v for k, v in stats.items() if k != 'total'} }")


if __name__ == "__main__":
    main()
