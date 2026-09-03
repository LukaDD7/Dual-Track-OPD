#!/usr/bin/env python3
"""Build the SFT warmup train/val parquets for the overnight window.

Design (T2 long-CoT-dominant warmup, aligned to arXiv:2604.23747):
  - MMFineReason: long-CoT teacher responses, stratified sample by source
    (--mmf-n rows total) -> format anchoring for RL. Default 100K so long CoT
    dominates the mix (previous 10K made long CoT only ~13% of rows).
  - the_cauldron: broad-coverage GT-answer tasks, stratified per-subset sample
    (--cauldron-per-subset rows per subset) from the 16 chosen subsets. Short-GT
    downsampled so it does not swamp the long CoT.

Outputs (verl SFT columns messages/images/source/image_hash):
  <out-dir>/sft_warmup_train__part_*.parquet  (sharded <=1500 rows/file:
    verl SFT reads via pandas, which trips pyarrow's nested-chunked limitation
    on large files; small shards avoid it)
  <out-dir>/sft_warmup_val__part_*.parquet    (--val-frac, deterministic)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import math
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# ---- length guard (same failure mode that crashed the 20260825_0705 run) ----
# Rows whose image-token-expanded template length >= MAX_SEQ_LEN crash verl's
# forward ("Image features and image tokens do not match") or train on a
# truncated long-CoT tail. Estimate that length exactly like the real
# pipeline: qwen_vl_utils smart_resize (factor=32, min=4*factor^2,
# max=16384*factor^2) -> processor smart_resize (65536/16777216) ->
# pads = grid.prod()//merge^2 with patch=16, merge=2.
# See scripts/sft_rl/scan_warmup_t2_exact.py (validated 1:1 against the real
# MultiTurnSFTDataset on all 156,196 warmup_t2 rows, 2026-08-25).
MAX_SEQ_LEN = 12288           # verl SFT max_length (run_sft_warmup.sh)
TEMPLATE_OVERHEAD = 160       # system + roles + vision markers + spacing
TEXT_MARGIN = 64              # est-vs-exact template drift safety pad
MERGE = 2
PATCH = 16


def smart_resize(height, width, factor, min_pixels, max_pixels):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_pads_from_size(w: int, h: int) -> int:
    """Exact image-token count for the verl/Qwen3-VL SFT pipeline."""
    factor = MERGE * PATCH
    h1, w1 = smart_resize(h, w, factor, 4 * factor * factor, 16384 * factor * factor)
    h2, w2 = smart_resize(h1, w1, factor, 65536, 16777216)
    return ((h2 // PATCH) // MERGE) * ((w2 // PATCH) // MERGE)


def row_expanded_len(row: dict, tok) -> tuple[int, int]:
    """(estimated template length, n_images). Returns (-1, n) on bad image."""
    text_tokens = 0
    n_images = 0
    for m in row["messages"]:
        content = m.get("content", "")
        if "<image>" in content:
            n_images += content.count("<image>")
            content = content.replace("<image>", "")
        try:
            text_tokens += len(tok.encode(content, add_special_tokens=False))
        except Exception:
            return -1, n_images
    img_tokens = 0
    for im in row.get("images") or []:
        b = im.get("bytes") if isinstance(im, dict) else None
        if not b:
            return -1, n_images
        try:
            with Image.open(io.BytesIO(b)) as pic:
                w, h = pic.size
            img_tokens += image_pads_from_size(w, h)
        except Exception:
            return -1, n_images
    return text_tokens + img_tokens + TEMPLATE_OVERHEAD + TEXT_MARGIN, n_images


def filter_overlength(rows: list[dict], tok, max_seq_len: int) -> tuple[list[dict], "dict[str, int]"]:
    """Drop rows that would exceed max_seq_len after image expansion."""
    from collections import Counter

    dropped: Counter = Counter()
    out: list[dict] = []
    for r in rows:
        est, n_img = row_expanded_len(r, tok)
        if est < 0:
            dropped[f"{r.get('source', '?')}::est_error"] += 1
        elif est > max_seq_len:
            dropped[f"{r.get('source', '?')}::seq_too_long"] += 1
        else:
            out.append(r)
    return out, dict(dropped)

DEFAULT_CAULDRON_SUBSETS = [
    "vqav2", "visual7w", "aokvqa", "tallyqa", "vsr",
    "ai2d", "scienceqa", "tqa", "chartqa", "textvqa", "docvqa",
    "infographic_vqa", "iconqa", "raven", "screen2words", "textcaps",
]


def row_hash(row) -> str:
    text = "".join(m.get("content", "") for m in row["messages"] if m.get("role") == "user")
    return hashlib.sha256(f"{row.get('image_hash','')}::{text}".encode()).hexdigest()


def read_nested_safe(path) -> list[dict]:
    """Read a parquet with nested struct columns, row-group by row-group.

    pyarrow (this env) fails with ArrowNotImplementedError on nested structs
    when a file has multiple row groups (chunked arrays). Per-row-group reads
    produce single chunks and convert fine.
    """
    try:
        return pq.read_table(path).to_pylist()
    except Exception:
        pf = pq.ParquetFile(path)
        rows: list[dict] = []
        for i in range(pf.metadata.num_row_groups):
            rows.extend(pf.read_row_group(i).to_pylist())
        return rows


def _file_row_counts(files) -> dict[str, int]:
    return {str(f): pq.ParquetFile(f).metadata.num_rows for f in files}


def subset_of_part(path: Path) -> str:
    """Converted part filename: cauldron_sft__part_<subset>_<orig>.parquet."""
    return path.stem.split("__part_", 1)[1].rsplit("_train-", 1)[0]


def _allowances(files, per_file_counts, per_subset_target: int, seed: int) -> dict[str, int | None]:
    by_subset: dict[str, list] = {}
    for f in files:
        by_subset.setdefault(subset_of_part(f), []).append(f)
    out: dict[str, int | None] = {}
    for subset, fs in by_subset.items():
        total = sum(per_file_counts[str(f)] for f in fs)
        if per_subset_target <= 0 or total <= per_subset_target:
            for f in fs:
                out[str(f)] = None
            continue
        rng = random.Random(seed ^ hash(subset))
        for f in fs:
            n = per_file_counts[str(f)]
            share = per_subset_target * n / total
            base = int(share)
            extra = 1 if rng.random() < (share - base) else 0
            out[str(f)] = base + extra
    return out


def sample_cauldron(cauldron_dir: str, subsets: list[str], per_subset: int,
                    seed: int) -> list[dict]:
    files = [Path(f) for f in sorted(glob.glob(f"{cauldron_dir}/cauldron_sft__part_*.parquet"))
             if subset_of_part(Path(f)) in subsets]
    counts = _file_row_counts(files)
    allowances = _allowances(files, counts, per_subset, seed)
    rows_out: list[dict] = []
    for f in files:
        allow = allowances[str(f)]
        rows = read_nested_safe(f)
        if allow is not None and allow < len(rows):
            rng = random.Random(seed ^ hash(str(f)))
            idx = sorted(rng.sample(range(len(rows)), allow))
            rows = [rows[i] for i in idx]
        rows_out.extend(rows)
    return rows_out


def sample_mmfinereason(mmf_dir: str, n: int, seed: int) -> list[dict]:
    files = sorted(glob.glob(f"{mmf_dir}/mmfinereason_sft__part_*.parquet"))
    if not files:
        raise SystemExit(f"no MMF converted parts under {mmf_dir}")
    rows: list[dict] = []
    for f in files:
        rows.extend(read_nested_safe(f))
    if n <= 0 or n >= len(rows):
        return rows
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    total = len(rows)
    rng = random.Random(seed)
    out: list[dict] = []
    targets: dict[str, int] = {}
    for src, rs in sorted(by_src.items()):
        share = n * len(rs) / total
        targets[src] = int(share)
    # largest-remainder distribution of the leftover
    leftover = n - sum(targets.values())
    rem = sorted(by_src, key=lambda s: (n * len(by_src[s]) / total) % 1.0, reverse=True)
    for i, src in enumerate(rem):
        if i >= leftover:
            break
        targets[src] += 1
    for src, k in targets.items():
        rs = by_src[src]
        if k >= len(rs):
            out.extend(rs)
        else:
            idx = sorted(rng.sample(range(len(rs)), k))
            out.extend(rs[i] for i in idx)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmf-dir", required=True,
                        help="dir containing mmfinereason_sft__part_*.parquet")
    parser.add_argument("--cauldron-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mmf-n", type=int, default=100000)
    parser.add_argument("--cauldron-per-subset", type=int, default=4000)
    parser.add_argument("--subsets", nargs="*", default=DEFAULT_CAULDRON_SUBSETS)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help="drop rows whose image-expanded template length "
                             "exceeds this (must match the training "
                             "max_length); pass 0 to disable the guard")
    parser.add_argument("--model", default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct",
                        help="tokenizer for the length guard")
    args = parser.parse_args()

    tok = None
    if args.max_seq_len > 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mmf_rows = sample_mmfinereason(args.mmf_dir, args.mmf_n, args.seed)
    print(f"MMF sampled: {len(mmf_rows)} rows")
    by_src = {}
    for r in mmf_rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("  by source:", dict(sorted(by_src.items(), key=lambda x: -x[1])[:8]))

    cauldron_rows = sample_cauldron(args.cauldron_dir, args.subsets,
                                    args.cauldron_per_subset, args.seed)
    print(f"Cauldron sampled: {len(cauldron_rows)} rows")
    by_sub = {}
    for r in cauldron_rows:
        by_sub[r["source"]] = by_sub.get(r["source"], 0) + 1
    print("  by subset:", dict(sorted(by_sub.items())))

    all_rows = mmf_rows + cauldron_rows

    if tok is not None:
        all_rows, dropped = filter_overlength(all_rows, tok, args.max_seq_len)
        total_dropped = sum(dropped.values())
        print(f"length guard (max_seq_len={args.max_seq_len}): kept {len(all_rows)}, "
              f"dropped {total_dropped}")
        for k in sorted(dropped, key=lambda k: -dropped[k])[:10]:
            print(f"  dropped {dropped[k]:6d}  {k}")

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_frac))
    val, train = all_rows[:n_val], all_rows[n_val:]

    schema = pa.schema([
        pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        pa.field("source", pa.string()),
        pa.field("image_hash", pa.string()),
    ])
    for name, rows in (("sft_warmup_train", train), ("sft_warmup_val", val)):
        paths = []
        for i in range(0, len(rows), 1500):
            part = rows[i : i + 1500]
            path = out_dir / f"{name}__part_{i // 1500:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(part, schema=schema), path)
            paths.append(str(path))
        print(f"wrote {len(paths)} shards for {name}: {len(rows)} rows total")
        print("  " + "\n  ".join(paths))


if __name__ == "__main__":
    main()
