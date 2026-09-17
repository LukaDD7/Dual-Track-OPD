#!/usr/bin/env python3
"""Per-dataset SFT pools for the one-epoch single-source attribution runs.

The warmup mixed MMFineReason + the_cauldron in one pool (3 epochs, sft_6939).
This script rebuilds each source SEPARATELY at full scale for 1-epoch SFT
(dataset-level attribution: what does each source alone contribute):

  --dataset mmf      : ALL mmfinereason_sft__part_*.parquet rows (122,603)
  --dataset cauldron : ALL rows of the same 16 subsets the warmup sampled from
                       (DEFAULT_CAULDRON_SUBSETS, no per-subset cap -> 382,932)

Pipeline (same guards as prep_sft_warmup.py):
  - read_nested_safe per part (pyarrow nested-chunk workaround)
  - filter_overlength: drop rows whose image-token-expanded length would
    exceed --max-seq-len (12288 = run_sft_warmup.sh max_length; rows are
    DROPPED, never truncated)
  - deterministic seed shuffle, val split, shards <= --shard-rows rows

Shards keep the warmup filename prefix (sft_warmup_train__part_%04d /
sft_warmup_val__part_%04d) so scripts/sft_rl/run_sft_warmup.sh's directory
glob picks them up unchanged. Output schema matches warmup: messages /
images(bytes inline) / source / image_hash (MMF's pass_rate column is dropped).

Launch (after prep):
  SFT_RL_NAME=qwen3vl_sft_mmf122k_1ep SFT_RL_DATA=<out-dir> \
  SFT_RL_GPUS=0,1,2,3,4,5,6,7 SFT_RL_EPOCHS=1 \
  bash scripts/sft_rl/run_sft_warmup.sh
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_sft_warmup import (  # noqa: E402
    DEFAULT_CAULDRON_SUBSETS,
    filter_overlength,
    read_nested_safe,
    subset_of_part,
)

SCHEMA = pa.schema([
    pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("source", pa.string()),
    pa.field("image_hash", pa.string()),
])


def load_mmf(mmf_dir: str) -> list[dict]:
    files = sorted(glob.glob(f"{mmf_dir}/mmfinereason_sft__part_*.parquet"))
    if not files:
        raise SystemExit(f"no MMF converted parts under {mmf_dir}")
    rows: list[dict] = []
    for f in files:
        rows.extend(read_nested_safe(f))
    return rows


def load_cauldron(cauldron_dir: str, subsets: list[str]) -> list[dict]:
    keep = set(subsets)
    files = [
        f for f in sorted(glob.glob(f"{cauldron_dir}/cauldron_sft__part_*.parquet"))
        if subset_of_part(Path(f)) in keep
    ]
    if not files:
        raise SystemExit(f"no cauldron parts for subsets {sorted(keep)} under {cauldron_dir}")
    rows: list[dict] = []
    for f in files:
        rows.extend(read_nested_safe(f))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mmf", "cauldron"), required=True)
    parser.add_argument("--mmf-dir", required=True)
    parser.add_argument("--cauldron-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subsets", nargs="*", default=DEFAULT_CAULDRON_SUBSETS)
    parser.add_argument("--val-frac", type=float, default=-1.0,
                        help="default: 0.02 for mmf (~2.4K rows), 0.008 for cauldron "
                             "(~3K rows, aligned to warmup val 3,032)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--model", default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit-rows", type=int, default=-1,
                        help="smoke cap: read at most N raw rows before filtering")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mmf":
        rows = load_mmf(args.mmf_dir)
        val_frac = args.val_frac if args.val_frac >= 0 else 0.02
        label = "mmf"
    else:
        rows = load_cauldron(args.cauldron_dir, args.subsets)
        val_frac = args.val_frac if args.val_frac >= 0 else 0.008
        label = "cauldron"
    print(f"[{label}] raw rows: {len(rows)}")

    if 0 < args.limit_rows < len(rows):
        rows = rows[: args.limit_rows]
        print(f"[{label}] limit-rows cap applied: {len(rows)} raw rows")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows, dropped = filter_overlength(rows, tok, args.max_seq_len)
    print(f"[{label}] length guard (max_seq_len={args.max_seq_len}): kept {len(rows)}, "
          f"dropped {sum(dropped.values())}")
    for k in sorted(dropped, key=lambda k: -dropped[k])[:10]:
        print(f"  dropped {dropped[k]:6d}  {k}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"[{label}] split val_frac={val_frac}: train {len(train)} / val {len(val)}")

    for name, part_rows in (("sft_warmup_train", train), ("sft_warmup_val", val)):
        paths = []
        for i in range(0, len(part_rows), args.shard_rows):
            chunk = part_rows[i : i + args.shard_rows]
            path = out_dir / f"{name}__part_{i // args.shard_rows:04d}.parquet"
            table = pa.Table.from_pylist(chunk, schema=SCHEMA)
            pq.write_table(table, path)
            paths.append(str(path))
        print(f"[{label}] wrote {len(paths)} shards for {name}: {len(part_rows)} rows")

    print(
        f"\nNext: SFT_RL_NAME=qwen3vl_sft_{label}_1ep SFT_RL_DATA={out_dir} "
        f"SFT_RL_GPUS=<gpus> SFT_RL_EPOCHS=1 bash scripts/sft_rl/run_sft_warmup.sh"
    )


if __name__ == "__main__":
    main()
