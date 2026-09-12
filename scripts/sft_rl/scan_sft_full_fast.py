#!/usr/bin/env python3
"""Fast mismatch scan for sft_full shards (no image decode).

The token/feature counts that verl's Qwen3-VL forward checks depend only on
image *dimensions* (smart_resize + image_grid_thw), not on pixel content. We
monkeypatch verl's process_image to return a same-size blank RGB image, then
reuse scan_oversized_samples._worker verbatim. The resulting n_tok / n_feat /
seqlen are identical to the real pipeline at a fraction of the cost (no 47GB
image-store decode).

Usage:
  python scan_sft_full_fast.py --sft-train-dir .../sft_full/train \
      --sft-val-dir .../sft_full/val --model $MODEL --out bad.json \
      --max-length 12288 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path

from PIL import Image


def blank_process_image(image, image_patch_size: int = 14):
    path = image.get("image_url") or image.get("path")
    if path is None:
        raise ValueError(f"image without path/image_url: {image}")
    with Image.open(path) as im:
        w, h = im.size
    return Image.new("RGB", (w, h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-train-dir", required=True)
    ap.add_argument("--sft-val-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # patch before forking workers so children inherit it
    import verl.utils.dataset.multiturn_sft_dataset as mtd

    mtd.process_image = blank_process_image

    from scan_oversized_samples import _worker

    shards = (
        sorted(glob.glob(f"{args.sft_train_dir}/sft_full_train__part_*.parquet"))
        + sorted(glob.glob(f"{args.sft_val_dir}/sft_full_val__part_*.parquet"))
    )
    if not shards:
        raise SystemExit("no sft_full shards found")

    groups = [shards[w :: args.workers] for w in range(args.workers)]
    groups = [g for g in groups if g]

    mp.set_start_method("fork", force=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(
            _worker,
            [(g, args.model, args.max_length, True) for g in groups],
        )

    bad = [b for r in results for b in r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bad, ensure_ascii=False, indent=1))
    print(f"total bad: {len(bad)}")
    from collections import Counter

    print("by shard:", dict(Counter(b["shard"] for b in bad)))
    print("by source:", dict(Counter(b.get("source", "?") for b in bad)))


if __name__ == "__main__":
    main()
