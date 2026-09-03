#!/usr/bin/env python3
"""Convert the_cauldron subsets to verl SFT parquet (messages + images columns).

Input : dataset/the_cauldron/<subset>/train-*.parquet  (HF parquet: images, texts)
Output: fc-opd-storage/outputs/fc_opd/sft_rl/cauldron/cauldron_sft.parquet
        columns: messages(list<struct<role,content>>), images(list<struct<bytes,path>>),
                 source(string), image_hash(string)
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def image_hash(img: dict) -> str:
    data = img.get("bytes") or b""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = str(img.get("path", "")).encode()
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


def texts_to_messages(texts) -> list[dict]:
    """Cauldron texts is a list of turns; each turn dict has user+assistant."""
    messages: list[dict] = []
    for turn in texts or []:
        if isinstance(turn, dict):
            if turn.get("user"):
                messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def _subset_row_counts(files) -> dict[str, int]:
    """Total rows per subset from parquet metadata (fast, no data read)."""
    counts: dict[str, int] = {}
    for f in files:
        subset = f.parent.name
        try:
            counts[subset] = counts.get(subset, 0) + pq.ParquetFile(f).metadata.num_rows
        except Exception:
            counts[subset] = counts.get(subset, 0) + pq.read_table(f, columns=[]).num_rows
    return counts


def _per_file_allowances(files, counts: dict[str, int], max_per_subset: int,
                         seed: int) -> dict[Path, int | None]:
    """Stratified per-file row allowance so each subset totals <= max_per_subset.

    Returns {path: allowance} where allowance is None when the cap does not
    apply to that subset (total rows already within cap).
    """
    allowances: dict[Path, int | None] = {}
    for f in files:
        subset = f.parent.name
        total = counts[subset]
        if max_per_subset <= 0 or total <= max_per_subset:
            allowances[f] = None
            continue
        # Largest remainder rounding keeps the subset total <= cap.
        file_rows = pq.ParquetFile(f).metadata.num_rows
        share = max_per_subset * file_rows / total
        base = int(share)
        remainder = share - base
        rng = random.Random(seed ^ hash(f.name))
        extra = 1 if rng.random() < remainder else 0
        allowances[f] = base + extra
    return allowances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="dataset/the_cauldron")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows-per-file", type=int, default=-1,
                        help="cap rows per subset for quick partial conversion")
    parser.add_argument("--combine", action="store_true",
                        help="merge per-subset outputs under --output-dir into --output")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="only convert these subset names (dir names)")
    parser.add_argument("--max-per-subset", type=int, default=0,
                        help="cap total rows per subset (stratified seeded sample); "
                             "0 = keep everything")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for --max-per-subset sampling")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("*/train-*.parquet"))
    if args.subsets:
        files = [f for f in files if f.parent.name in args.subsets]
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

    counts = _subset_row_counts(files)
    allowances = _per_file_allowances(files, counts, args.max_per_subset, args.seed)
    capped = {s for s, n in counts.items()
              if args.max_per_subset > 0 and n > args.max_per_subset}
    if capped:
        print(f"[sampling] per-subset cap={args.max_per_subset} seed={args.seed}; "
              f"capped subsets={sorted(capped)}", flush=True)

    # Per-file incremental writes: safe against interruption / ongoing download.
    for f in files:
        subset = f.parent.name
        part_path = output.parent / f"{output.stem}__part_{subset}_{f.stem}.parquet"
        if part_path.exists():
            print(f"[skip] {subset}/{f.name} already converted", flush=True)
            continue
        print(f"[convert] {f} ...", flush=True)
        t = pq.read_table(f)
        rows = t.to_pylist()
        if args.max_rows_per_file > 0:
            rows = rows[: args.max_rows_per_file]
        allowance = allowances[f]
        if allowance is not None and allowance < len(rows):
            rng = random.Random(args.seed ^ hash(f.name))
            idx = sorted(rng.sample(range(len(rows)), allowance))
            rows = [rows[i] for i in idx]
        bucket = {"messages": [], "images": [], "sources": [], "hashes": []}
        for row in rows:
            msgs = texts_to_messages(row.get("texts"))
            imgs = row.get("images") or []
            if not msgs:
                continue
            # verl matches <image> placeholders to image entries; ensure the
            # placeholder count equals the images count (raven has 2 images).
            if imgs:
                n_ph = sum(
                    m.get("content", "").count("<image>")
                    for m in msgs if m.get("role") == "user"
                )
                if n_ph < len(imgs):
                    msgs[0]["content"] = "<image>" * (len(imgs) - n_ph) + msgs[0]["content"]
            bucket["messages"].append(msgs)
            bucket["images"].append(imgs)
            bucket["sources"].append(subset)
            bucket["hashes"].append(",".join(image_hash(i) for i in imgs))
        print(f"[convert] {subset}/{f.name}: {len(rows)} rows", flush=True)
        if bucket["messages"]:
            part = pa.table(
                {
                    "messages": pa.array(
                        bucket["messages"],
                        type=pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
                    ),
                    "images": pa.array(
                        bucket["images"],
                        type=pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    ),
                    "source": pa.array(bucket["sources"], type=pa.string()),
                    "image_hash": pa.array(bucket["hashes"], type=pa.string()),
                }
            )
            pq.write_table(part, part_path)
            print(f"[write] {part_path}", flush=True)

    print("done; run with --combine to merge per-subset parts into one parquet")


if __name__ == "__main__":
    main()
