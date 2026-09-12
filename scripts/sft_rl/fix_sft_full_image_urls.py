#!/usr/bin/env python3
"""Repair sft_full shards written by an early prep_sft_full.py run.

The 08:37 (2026-08-22) run wrote images as [{"path": <rel>}] with paths relative
to $LZY_ROOT. verl's VLM SFT pipeline (MultiTurnSFTDataset -> process_image ->
qwen_vl_utils.fetch_image) requires the "image_url" key and resolves local paths
against the training process CWD (the backend example dir), so both the key and
the relative prefix break loading. This rewrites every shard in place to
images=[{"image_url": <absolute path>}], preserving rows/order/split exactly.

Usage:
  python fix_sft_full_image_urls.py [--sft-dir .../sft_full] [--root $LZY_ROOT]
"""

from __future__ import annotations

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        pa.field(
            "messages",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        pa.field("images", pa.list_(pa.struct([("image_url", pa.string())]))),
        pa.field("source", pa.string()),
        pa.field("image_hash", pa.string()),
    ]
)


def fix_shard(path: str, root: str) -> int:
    t = pq.read_table(path)
    rows = t.to_pylist()
    out = []
    for r in rows:
        imgs = []
        for im in r.get("images") or []:
            p = im.get("path")
            if p is None:
                p = im.get("image_url", "")
            if p and not p.startswith("/") and not p.startswith("file://"):
                p = os.path.join(root, p)
            imgs.append({"image_url": p})
        out.append(
            {
                "messages": r["messages"],
                "images": imgs,
                "source": r["source"],
                "image_hash": r["image_hash"],
            }
        )
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), tmp)
    os.replace(tmp, path)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", default="fc-opd-storage/outputs/fc_opd/sft_rl/sft_full")
    ap.add_argument("--root", default=os.environ.get("LZY_ROOT", "/inspire/hdd/global_user/mengweicheng-240108120092/lzy"))
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.sft_dir, "train", "sft_full_train__part_*.parquet"))
        + glob.glob(os.path.join(args.sft_dir, "val", "sft_full_val__part_*.parquet"))
    )
    if not files:
        raise SystemExit(f"no sft_full shards under {args.sft_dir}")
    total = 0
    for f in files:
        n = fix_shard(f, args.root)
        total += n
        print(f"fixed {f}: {n} rows")
    print(f"TOTAL rows: {total} across {len(files)} shards")


if __name__ == "__main__":
    main()
