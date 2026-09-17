#!/usr/bin/env python3
"""Ensure <image> placeholders match the images list in warmup shards.

Issue found 2026-08-19: the_cauldron `raven` rows carry 2 images (context +
candidate) but the converter emitted a single <image>; verl's processor asserts
`image_offset == len(images)`. Fix in place: pad missing placeholders into the
first user message, then rewrite the shard.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def fix_rows(rows: list[dict]) -> tuple[int, int]:
    fixed = 0
    for r in rows:
        imgs = r.get("images") or []
        n_img = len(imgs)
        user_msgs = [m for m in r.get("messages", []) if m.get("role") == "user"]
        n_ph = sum(m.get("content", "").count("<image>") for m in user_msgs)
        if n_ph < n_img and user_msgs:
            user_msgs[0]["content"] = "<image>" * (n_img - n_ph) + user_msgs[0]["content"]
            fixed += 1
        elif n_ph > n_img:
            print(f"WARN over-placeholder: images={n_img} ph={n_ph} src={r.get('source')}")
    return fixed, 0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fix_image_placeholders.py <warmup-dir>")
    base = Path(sys.argv[1])
    files = sorted(glob.glob(f"{base}/sft_warmup_*__part_*.parquet"))
    if not files:
        raise SystemExit(f"no shards under {base}")
    total_fixed = 0
    for f in files:
        t = pq.read_table(f)
        rows = t.to_pylist()
        fixed, _ = fix_rows(rows)
        if fixed:
            schema = t.schema
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), f)
            total_fixed += fixed
            print(f"[fixed] {Path(f).name}: {fixed} rows")
    print(f"done: {total_fixed} rows fixed across {len(files)} shards")


if __name__ == "__main__":
    main()
