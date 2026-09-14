#!/usr/bin/env python3
"""Convert MMFineReason-SFT-123K to verl SFT parquet (messages + images columns).

Input : dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data/train-*.parquet
        (question contains <image>; qwen3vl_235b_thinking_response is the long CoT)
Output: fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/mmfinereason_sft.parquet
        columns: messages, images, source, pass_rate, image_hash
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--combine", action="store_true",
                        help="merge per-shard outputs under --output-dir into --output")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found under {args.input_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.combine:
        parts = sorted(output.parent.glob(f"{output.stem}__part_*.parquet"))
        tables = [pq.read_table(p) for p in parts]
        pq.write_table(pa.concat_tables(tables), output)
        print(f"combined {len(parts)} parts -> {output}: {sum(t.num_rows for t in tables)} rows")
        return

    for f in files:
        part_path = output.parent / f"{output.stem}__part_{f.stem}.parquet"
        if part_path.exists():
            print(f"[skip] {f.name} already converted", flush=True)
            continue
        messages_all, images_all, sources_all, passes_all, hashes_all = [], [], [], [], []
        t = pq.read_table(f)
        for row in t.to_pylist():
            question = row.get("question", "")
            response = row.get("qwen3vl_235b_thinking_response", "")
            if not question or not response:
                continue
            messages_all.append(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
            )
            imgs = row.get("image") or []
            if isinstance(imgs, dict):
                imgs = [imgs]
            images_all.append(imgs)
            sources_all.append(str(row.get("source", "")))
            passes_all.append(float(row.get("pass_rate", -1.0)))
            hashes_all.append(
                ",".join(hashlib.sha256(i.get("bytes", b"")).hexdigest()[:16] for i in imgs)
            )

        out = pa.table(
            {
                "messages": pa.array(
                    messages_all,
                    type=pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
                ),
                "images": pa.array(
                    images_all,
                    type=pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ),
                "source": pa.array(sources_all, type=pa.string()),
                "pass_rate": pa.array(passes_all, type=pa.float64()),
                "image_hash": pa.array(hashes_all, type=pa.string()),
            }
        )
        pq.write_table(out, part_path)
        print(f"[write] {part_path}: {out.num_rows} rows", flush=True)
    print("done; run with --combine to merge per-shard parts into one parquet")


if __name__ == "__main__":
    main()
