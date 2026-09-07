#!/usr/bin/env python3
"""Convert PAPOGalaxy/PAPO_MMK12_test -> verl RL val parquet (repo schema).

Why: ViRL39K is train-only (38,348 rows, no val shard). The verl GRPO launch
scripts FATAL without a val file. The PTD-PO paper evaluates on MMK12, whose
held-out test split is published as ``PAPOGalaxy/PAPO_MMK12_test``. Using that
external test set as the RL val keeps train (full 38,348) and val strictly
disjoint, matches the paper, and needs zero hold-out from train — unlike
``split_virl39k_val.py`` (local hold-out), which would drop 500 rows from
train and change the training distribution.

Paper-MMK12 vs ViRL-MMK12 overlap: ViRL39K bundles ``MMK12`` (12,661 rows) and
``MMK12Converted`` (209 rows) from ``FanqingM/MMK12``. ``PAPO_MMK12_test`` is a
repackaged *test* split (2,000 rows) of the same MMK12 benchmark. It ships as a
single self-contained parquet with images inlined as {bytes, path} structs, so
no images/ tree or ViRL image-root is needed.

Input : dataset/MMK12_test/train-00000-of-00001.parquet  (~167 MB, 2,000 rows)
        columns: image (List<{bytes,path}>), problem (str), answer (str)
Output: fc-opd-storage/outputs/fc_opd/sft_rl/virl39k_mmk12_test/
          virl39k_rl_val__part_0000.parquet
          virl39k_mmk12_test_stats.json

Notes:
  - split name is ``train`` (not ``test``) — verified via HF datasets-server.
  - GT uses the same unbox + 4-class filter as convert_virl39k (letter /
    yesno / pure_number / has_number, drop other), so mmf_reward.py grades it
    identically.
  - ``problem`` already leads with ``<image>``; placeholder count == image
    count is re-asserted (verl hard constraint), mirroring convert_virl39k.

Usage:
  python3 scripts/sft_rl/convert_mmk12_test.py \
    --input-parquet dataset/MMK12_test/train-00000-of-00001.parquet \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/virl39k_mmk12_test
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the exact RL schema + GT classification so this val shard is drop-in
# interchangeable with convert_virl39k / split_virl39k_val output.
from convert_virl39k import RL_SCHEMA, gt_type, unbox  # noqa: E402  (same-dir import)


def _img_bytes(img: dict) -> bytes | None:
    b = img.get("bytes")
    if b is not None:
        return b
    p = img.get("path")
    if p:
        try:
            return Path(p).read_bytes()
        except OSError:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    DTOPD = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy")
    base = DTOPD / "fc-opd-storage/outputs/fc_opd/sft_rl"
    ap.add_argument("--input-parquet", type=Path,
                    default=DTOPD / "dataset/MMK12_test/train-00000-of-00001.parquet")
    ap.add_argument("--out-dir", type=Path, default=base / "virl39k_mmk12_test")
    ap.add_argument("--shard-rows", type=int, default=1500)
    ap.add_argument("--prefix", default="virl39k_rl_val")
    ap.add_argument("--keep-other", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.input_parquet)
    pf = pq.ParquetFile(str(src))
    gt_stats: Counter = Counter()
    stats = {"total": 0, "missing_image": 0, "placeholder_mismatch_dropped": 0,
             "gt_other_dropped": 0}

    rows: list[dict] = []
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
            question = str(r.get("problem") or "")
            raw_imgs = r.get("images") or []
            imgs = []
            ok = True
            for img in raw_imgs:
                b = _img_bytes(img) if isinstance(img, dict) else None
                if b is None:
                    stats["missing_image"] += 1
                    ok = False
                    break
                imgs.append({"bytes": b, "path": (img or {}).get("path")})
            if not ok or not imgs:
                continue
            # verl asserts count("<image>") == len(images)
            if question.count("<image>") != len(imgs):
                stats["placeholder_mismatch_dropped"] += 1
                continue
            sample_uid = f"MMK12_test:{stats['total']}"
            rows.append({
                "data_source": "MMK12_test",
                "prompt": [{"role": "user", "content": question}],
                "images": imgs,
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "question": question,
                    "source": "MMK12_test",
                    "gt_source": "answer_unboxed",
                    "gt_type": t,
                    "original_answer": answer,
                },
                "question": question,
                "sample_uid": sample_uid,
            })
        print(f"[scan] row-group {rg + 1}/{pf.metadata.num_row_groups}: val={len(rows)}", flush=True)

    n_shards = 0
    paths = []
    for i in range(0, len(rows), args.shard_rows):
        p = out_dir / f"{args.prefix}__part_{n_shards:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[i:i + args.shard_rows], schema=RL_SCHEMA), p)
        paths.append(str(p))
        n_shards += 1

    stats_out = {
        "input": str(src),
        "total_rows": stats["total"],
        "gt_stats": dict(gt_stats),
        "gt_other_dropped": stats["gt_other_dropped"],
        "missing_image": stats["missing_image"],
        "placeholder_mismatch_dropped": stats["placeholder_mismatch_dropped"],
        "val_rows": len(rows),
        "shards": paths,
        "note": "PAPO_MMK12_test split is named 'train' in the HF repo; used as external RL val "
                "(train/val disjoint by construction, matches PTD-PO paper eval on MMK12).",
    }
    (out_dir / "virl39k_mmk12_test_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"val {len(rows)} rows -> {n_shards} shards under {out_dir}")
    print(f"GT stats: {dict(gt_stats)}")
    print(f"dropped: other={stats['gt_other_dropped']} missing_image={stats['missing_image']} "
          f"ph_mismatch={stats['placeholder_mismatch_dropped']}")


if __name__ == "__main__":
    main()