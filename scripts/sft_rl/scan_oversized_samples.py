#!/usr/bin/env python3
"""Find warmup rows whose processed sequence exceeds max_length after image
expansion (verl truncates input_ids but not multi_modal_inputs -> model forward
crashes with "Image features and image tokens do not match").

Usage: scan_oversized_samples.py --warmup-dir DIR --model PATH --out JSON
       [--max-length 8192] [--workers 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf


def _worker(args) -> list[dict]:
    shard_paths, model, max_length, mismatch_only = args
    from transformers import AutoProcessor
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    cfg = OmegaConf.create({
        "max_length": max_length,
        "truncation": "right",
        "pad_mode": "no_padding",
        "image_key": "images",
        "use_dynamic_bsz": True,
    })
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    import json as _json
    cid = int(_json.load(open(f"{model}/config.json"))["image_token_id"])
    import torch
    bad = []
    for sp in shard_paths:
        ds = MultiTurnSFTDataset([sp], model, cfg, proc, max_samples=-1)
        rows = pq.read_table(sp).to_pylist()
        for i in range(len(ds)):
            it = ds[i]
            n_tok = int((it["input_ids"] == cid).sum())
            g = it["multi_modal_inputs"].get("image_grid_thw")
            n_feat = int(sum(int(t) * (int(h) // 2) * (int(w) // 2)
                             for t, h, w in g.tolist())) if g is not None else 0
            if n_tok != n_feat or (not mismatch_only and len(it["input_ids"]) >= max_length):
                bad.append({
                    "shard": Path(sp).name,
                    "idx": i,
                    "n_tok": n_tok,
                    "n_feat": n_feat,
                    "seqlen": len(it["input_ids"]),
                    "grids": g.tolist() if g is not None else None,
                    "source": rows[i].get("source", "?") if i < len(rows) else "?",
                })
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name-prefix", default="sft_warmup",
                        help="shard name prefix, e.g. sft_warmup or sft_full")
    parser.add_argument("--mismatch-only", action="store_true",
                        help="report only rows where image tokens != features "
                             "(the actual crash set)")
    args = parser.parse_args()

    shards = (sorted(glob.glob(f"{args.warmup_dir}/{args.name_prefix}_train__part_*.parquet")) +
              sorted(glob.glob(f"{args.warmup_dir}/{args.name_prefix}_val__part_*.parquet")))
    if not shards:
        raise SystemExit("no shards")
    groups = [shards[w::args.workers] for w in range(args.workers)]
    groups = [g for g in groups if g]

    with mp.Pool(args.workers) as pool:
        results = pool.map(
            _worker, [(g, args.model, args.max_length, args.mismatch_only) for g in groups]
        )

    bad = [b for r in results for b in r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bad, ensure_ascii=False, indent=1))
    print(f"total bad: {len(bad)}")
    from collections import Counter
    by_shard = Counter(b["shard"] for b in bad)
    print("by shard:", dict(by_shard))


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
