#!/usr/bin/env python3
"""Split SFT and RL pools, enforcing disjointness by image-hash + prompt text.

Design intent (DeReason-style): SFT gets the broad-coverage pool, RL gets the
hard verifiable pool; any row whose image+prompt appears in both pools is
reported and kept only in SFT. Every split is also divided into train/val by
sample hash so val never enters training.

Usage:
  split_sft_rl.py --sft a.parquet b.parquet --rl c.parquet \
      --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/split
"""

from __future__ import annotations

import argparse
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq


def row_key(messages, image_hash) -> str:
    text = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
    return hashlib.sha256(f"{text}::{image_hash}".encode()).hexdigest()


def load(path: str):
    t = pq.read_table(path)
    rows = t.to_pylist()
    keys = [row_key(r["messages"], r.get("image_hash", "")) for r in rows]
    return t, rows, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", nargs="+", required=True)
    parser.add_argument("--rl", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import pathlib

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sft_rows = []
    for p in args.sft:
        _, rows, _ = load(p)
        sft_rows.extend(rows)
    rl_kept = []
    sft_keys = {row_key(r["messages"], r.get("image_hash", "")) for r in sft_rows}
    rl_total = 0
    for p in args.rl:
        _, rows, keys = load(p)
        rl_total += len(rows)
        for r, k in zip(rows, keys):
            if k in sft_keys:
                print(f"drop from RL (in SFT): {k}")
                continue
            rl_kept.append(r)

    rng = __import__("random").Random(args.seed)

    def split(rows, name):
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        val, train = rows[:n_val], rows[n_val:]
        for subset, out in (("train", train), ("val", val)):
            path = f"{args.out_dir}/{name}_{subset}.parquet"
            pq.write_table(pa.Table.from_pylist(out), path)
            print(f"wrote {path}: {len(out)} rows")

    split(sft_rows, "sft")
    split(rl_kept, "rl")
    print(f"SFT pool={len(sft_rows)} RL pool={len(rl_kept)} "
          f"(dropped {rl_total - len(rl_kept)} overlapping rows)")


if __name__ == "__main__":
    main()
