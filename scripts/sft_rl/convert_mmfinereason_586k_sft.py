#!/usr/bin/env python3
"""Convert MMFineReason-SFT-586K to verl SFT parquet (repo warmup/full schema).

Input : dataset/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data/train-*.parquet
        (same schema as the 123K release: question contains <image>,
         qwen3vl_235b_thinking_response is the long CoT, image is a struct)
Output: fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason_586k/
          mmfinereason586k_sft__part_*.parquet (<=1500 rows, single row-group)
        columns: messages, images, source, pass_rate, image_hash

Notes
  - 586K is a superset of the 123K release: 585,744 rows, of which 125,602
    (21.4%) are identical (image bytes + question) to rows already in the 123K
    set. --drop-123k-overlap (default ON, needs the 123K dir) removes them so
    SFT pools do not double-train the same rows; disable to keep everything.
  - 17,901 rows have answer=null (FineVision-visualwebinstruct) — answer is
    not needed for SFT (teacher CoT is the target), so they are kept unless
    empty qwen3vl_235b_thinking_response.
  - pass_rate retained for downstream difficulty-based sampling.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        pa.field(
            "messages",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        pa.field("source", pa.string()),
        pa.field("pass_rate", pa.float64()),
        pa.field("image_hash", pa.string()),
    ]
)


def row_key(question: str, imgs: list[dict]) -> str:
    h = ",".join(hashlib.sha256(i.get("bytes", b"")).hexdigest()[:16] for i in imgs)
    return f"{h}::{question}"


def load_123k_keys(input_dir_123k: str) -> set[str]:
    keys: set[str] = set()
    for f in sorted(glob.glob(f"{input_dir_123k}/train-*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=["question", "image"]).to_pylist():
                imgs = r.get("image") or []
                if isinstance(imgs, dict):
                    imgs = [imgs]
                keys.add(row_key(str(r.get("question", "")), imgs))
    print(f"123K overlap keys loaded: {len(keys)}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="dataset/MMFineReason-SFT-586K-.../data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-dir-123k", default=None,
                        help="dataset/MMFineReason-SFT-123K-.../data for overlap drop")
    parser.add_argument("--drop-123k-overlap", action="store_true",
                        help="drop rows already present in the 123K release (default when --input-dir-123k given)")
    parser.add_argument("--keep-123k-overlap", action="store_true",
                        help="keep all 586K rows even if 123K dir given")
    parser.add_argument("--shard-rows", type=int, default=1500)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet under {args.input_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_overlap = bool(args.input_dir_123k) and not args.keep_123k_overlap
    overlap_keys = load_123k_keys(args.input_dir_123k) if drop_overlap else set()

    rows: list[dict] = []
    stats = {"total": 0, "kept": 0, "no_response": 0, "overlap_dropped": 0}
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                stats["total"] += 1
                question = str(r.get("question", "") or "")
                response = str(r.get("qwen3vl_235b_thinking_response", "") or "")
                if not question or not response:
                    stats["no_response"] += 1
                    continue
                imgs = r.get("image") or []
                if isinstance(imgs, dict):
                    imgs = [imgs]
                imgs = [{"bytes": i.get("bytes"), "path": i.get("path") or ""} for i in imgs]
                if drop_overlap and row_key(question, imgs) in overlap_keys:
                    stats["overlap_dropped"] += 1
                    continue
                hashes = ",".join(hashlib.sha256(i.get("bytes", b"")).hexdigest()[:16] for i in imgs)
                rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response},
                        ],
                        "images": imgs,
                        "source": str(r.get("source", "")),
                        "pass_rate": float(r.get("pass_rate", -1.0)),
                        "image_hash": hashes,
                    }
                )
                stats["kept"] += 1
        print(f"[scan] {f}: running kept={stats['kept']}", flush=True)

    n_shards = 0
    for i in range(0, len(rows), args.shard_rows):
        tbl = pa.Table.from_pylist(rows[i : i + args.shard_rows], schema=SCHEMA)
        pq.write_table(tbl, out_dir / f"mmfinereason586k_sft__part_{n_shards:04d}.parquet")
        n_shards += 1
    print(f"kept rows: {stats['kept']} -> {n_shards} shards under {out_dir}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
