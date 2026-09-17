#!/usr/bin/env python3
"""Build the full-pool SFT train/val parquets for the SFT-then-RL retrain.

Data
  - MMFineReason: ALL converted rows (mmfinereason_sft__part_*.parquet, 122,603)
  - the_cauldron: ALL rows of the same 16 subsets used by the group doc
    (vqav2/visual7w/aokvqa/tallyqa/vsr/ai2d/scienceqa/tqa/chartqa/textvqa/
     docvqa/infographic_vqa/iconqa/raven/screen2words/textcaps)

Length policy (no silent truncation; keeps SFT <-> RL consistent):
  - teacher CoT (assistant content) tokens <= --max-cot-tokens
    (default 8192 = RL max_response_length; paper-style length filter)
  - estimated full sequence (prompt text + image tokens + CoT + template
    overhead) <= --max-seq-tokens (default 12288)
  Rows failing either check are DROPPED (never truncated) and reported by source.

Images
  - --image-mode path (default): extract image bytes once into <out>/images,
    keyed by sha256 of the bytes (dedup), and write shards with
    images=[{"image_url": <abs path>}]; this qwen_vl_utils version's
    fetch_image resolves "image_url" as a local path, so the parquet shards
    stay small (no embedded bytes).
  - --image-mode embed: keep bytes inline (warmup format), shards are large.

Split
  - deterministic train/val by row hash (image_hash + user text), default val
    0.4% (~2K rows of the filtered pool).
  - shards <= --shard-rows rows (default 1500; single row-group per file avoids
    the pyarrow nested-chunk limitation in verl's pandas reader).

Output schema: messages / images / source / image_hash (same as warmup shards).

Usage:
  python prep_sft_full.py \
    --mmf-dir .../sft_rl/mmfinereason \
    --cauldron-dir .../sft_rl/cauldron \
    --out-dir .../sft_rl/sft_full \
    --model .../models/Qwen3-VL-8B-Instruct \
    --workers 16
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import multiprocessing as mp
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DEFAULT_SUBSETS = [
    "vqav2", "visual7w", "aokvqa", "tallyqa", "vsr",
    "ai2d", "scienceqa", "tqa", "chartqa", "textvqa", "docvqa",
    "infographic_vqa", "iconqa", "raven", "screen2words", "textcaps",
]
VQA_SUBSETS = {"vqav2", "visual7w", "aokvqa", "tallyqa", "vsr"}

PATCH = 28
MAX_PIXELS = 1280 * 28 * 28  # Qwen2.5/3-VL processor default
TEMPLATE_OVERHEAD = 128      # system + roles + spacing
FALLBACK_IMG_TOKENS = 512

SCHEMA_PATH = pa.schema(
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
SCHEMA_EMBED = pa.schema(
    [
        pa.field(
            "messages",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        pa.field(
            "images",
            pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ),
        pa.field("source", pa.string()),
        pa.field("image_hash", pa.string()),
    ]
)


def read_nested_safe(path) -> list[dict]:
    """Read a parquet with nested struct columns, row-group by row-group.

    pyarrow fails with ArrowNotImplementedError on nested structs when a file
    has multiple row groups (chunked arrays). Per-row-group reads produce
    single chunks and convert fine.
    """
    try:
        return pq.read_table(path).to_pylist()
    except Exception:
        pf = pq.ParquetFile(path)
        rows: list[dict] = []
        for i in range(pf.metadata.num_row_groups):
            rows.extend(pf.read_row_group(i).to_pylist())
        return rows


def row_key(row: dict) -> str:
    text = "".join(m.get("content", "") for m in row["messages"] if m.get("role") == "user")
    return hashlib.sha256(f"{row.get('image_hash', '')}::{text}".encode()).hexdigest()


def img_tokens_from_size(w: int, h: int) -> int:
    """Qwen3-VL style estimate: 28x28 patches, 2x2 merge, cap at MAX_PIXELS."""
    if w <= 0 or h <= 0:
        return FALLBACK_IMG_TOKENS
    if w * h > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (w * h))
        w, h = int(w * scale), int(h * scale)
    hp = (w + PATCH - 1) // PATCH
    vp = (h + PATCH - 1) // PATCH
    return max(1, (hp // 2) * (vp // 2))


def subset_of_part(path: Path) -> str:
    return path.stem.split("__part_", 1)[1].rsplit("_train-", 1)[0]


def _file_worker(args) -> tuple[list[dict], Counter, Counter, int, int]:
    files, model, max_cot, max_seq, image_store, embed, max_rows = args
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    out: list[dict] = []
    kept: Counter = Counter()
    dropped: Counter = Counter()
    img_bytes_written = 0
    n_processed = 0
    Path(image_store).mkdir(parents=True, exist_ok=True)

    for f in files:
        for row in read_nested_safe(f):
            if max_rows > 0 and n_processed >= max_rows:
                return out, kept, dropped, img_bytes_written, n_processed
            n_processed += 1
            source = row.get("source") or os.path.basename(f)
            messages = row.get("messages") or []
            user_text = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
            assistant_text = "".join(m.get("content", "") for m in messages if m.get("role") == "assistant")
            images = row.get("images") or []

            # safety: <image> placeholders must match image count (verl asserts this)
            n_ph = user_text.count("<image>")
            if n_ph != len(images):
                dropped[f"{source}::placeholder_mismatch"] += 1
                continue

            try:
                cot_tokens = len(tok.encode(assistant_text, add_special_tokens=False))
                user_tokens = len(tok.encode(user_text, add_special_tokens=False))
            except Exception:
                dropped[f"{source}::tokenize_error"] += 1
                continue

            img_tokens = 0
            bad_img = False
            for im in images:
                b = im.get("bytes") if isinstance(im, dict) else None
                if b:
                    try:
                        with Image.open(BytesIO(b)) as imobj:
                            w, h = imobj.size
                        img_tokens += img_tokens_from_size(w, h)
                    except Exception:
                        bad_img = True
                        break
                else:
                    img_tokens += FALLBACK_IMG_TOKENS
            if bad_img:
                dropped[f"{source}::image_decode_error"] += 1
                continue

            seq_est = user_tokens + cot_tokens + img_tokens + TEMPLATE_OVERHEAD
            if cot_tokens > max_cot:
                dropped[f"{source}::cot_too_long"] += 1
                continue
            if seq_est > max_seq:
                dropped[f"{source}::seq_too_long"] += 1
                continue

            if embed:
                new_images = [{"bytes": im.get("bytes", b""), "path": im.get("path")} for im in images]
            else:
                new_images = []
                for im in images:
                    b = im.get("bytes") if isinstance(im, dict) else None
                    if b:
                        digest = hashlib.sha256(b).hexdigest()
                        target = os.path.join(image_store, f"{digest}.img")
                        if not os.path.exists(target):
                            with open(target, "wb") as fh:
                                fh.write(b)
                            img_bytes_written += len(b)
                        new_images.append({"image_url": target})
                    elif im.get("path"):
                        new_images.append({"image_url": im["path"]})
                    else:
                        dropped[f"{source}::image_missing"] += 1
                        break
                else:
                    out.append(
                        {
                            "messages": messages,
                            "images": new_images,
                            "source": source,
                            "image_hash": row.get("image_hash", ""),
                        }
                    )
                    kept[source] += 1
                    continue
                continue

    return out, kept, dropped, img_bytes_written, n_processed


def write_shards(rows: list[dict], out_prefix: str, schema, shard_rows: int) -> int:
    n_shards = 0
    for i in range(0, len(rows), shard_rows):
        chunk = rows[i : i + shard_rows]
        tbl = pa.Table.from_pylist(chunk, schema=schema)
        pq.write_table(tbl, f"{out_prefix}__part_{n_shards:04d}.parquet")
        n_shards += 1
    return n_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmf-dir", required=True)
    parser.add_argument("--cauldron-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True, help="Qwen3-VL model dir for tokenizer")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--max-cot-tokens", type=int, default=8192)
    parser.add_argument("--max-seq-tokens", type=int, default=12288)
    parser.add_argument("--val-frac", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--image-mode", choices=["path", "embed"], default="path")
    parser.add_argument("--max-rows", type=int, default=-1, help="dry-run cap")
    args = parser.parse_args()

    # resolve to absolute so image_url / image-store paths are launch-independent
    # (verl workers run from the backend example dir, not from the repo root)
    out_dir = Path(args.out_dir).resolve()
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    image_store = str(out_dir / "images")
    for d in (train_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    mmf_files = sorted(glob.glob(f"{args.mmf_dir}/mmfinereason_sft__part_*.parquet"))
    cauldron_files = [
        f
        for f in sorted(glob.glob(f"{args.cauldron_dir}/cauldron_sft__part_*.parquet"))
        if subset_of_part(Path(f)) in set(args.subsets)
    ]
    if not mmf_files:
        raise SystemExit(f"no MMF parts under {args.mmf_dir}")
    if not cauldron_files:
        raise SystemExit(f"no Cauldron parts for subsets under {args.cauldron_dir}")

    all_files = [(f, "reasoning") for f in mmf_files] + [
        (f, "vqa" if subset_of_part(Path(f)) in VQA_SUBSETS else "general")
        for f in cauldron_files
    ]
    print(f"files: {len(mmf_files)} MMF + {len(cauldron_files)} cauldron = {len(all_files)}")

    # deterministic grouping: shards of the same source go to the same worker
    # (tokenizer is per-worker), and within a worker files stay ordered.
    groups = [all_files[w::args.workers] for w in range(args.workers)]
    groups = [g for g in groups if g]

    mp.set_start_method("fork", force=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(
            _file_worker,
            [
                (
                    [f for f, _ in g],
                    args.model,
                    args.max_cot_tokens,
                    args.max_seq_tokens,
                    image_store,
                    args.image_mode == "embed",
                    args.max_rows,
                )
                for g in groups
            ],
        )

    all_rows: list[dict] = []
    kept_total: Counter = Counter()
    dropped_total: Counter = Counter()
    img_bytes_written = 0
    processed = 0
    for rows, kept, dropped, ibw, n in results:
        all_rows.extend(rows)
        kept_total.update(kept)
        dropped_total.update(dropped)
        img_bytes_written += ibw
        processed += n

    print(f"processed rows: {processed}")
    print(f"kept rows: {len(all_rows)}  (by source: {dict(kept_total)})")
    print(f"dropped: {dict(dropped_total)}")
    print(
        f"image store bytes written: {img_bytes_written / 1e9:.2f} GB "
        f"(files: {len(os.listdir(image_store)) if os.path.isdir(image_store) else 0})"
    )

    # deterministic train/val by row hash
    val_threshold = int(args.val_frac * (2**32))
    rng_seed = hashlib.sha256(f"split::{args.seed}".encode()).hexdigest()
    train_rows, val_rows = [], []
    for row in all_rows:
        key = row_key(row)
        h = hashlib.sha256(f"{rng_seed}::{key}".encode()).hexdigest()
        (val_rows if int(h[:8], 16) < val_threshold else train_rows).append(row)

    schema = SCHEMA_PATH if args.image_mode == "path" else SCHEMA_EMBED
    n_train = write_shards(train_rows, str(train_dir / "sft_full_train"), schema, args.shard_rows)
    n_val = write_shards(val_rows, str(val_dir / "sft_full_val"), schema, args.shard_rows)
    print(f"train rows: {len(train_rows)} ({n_train} shards) -> {train_dir}")
    print(f"val rows: {len(val_rows)} ({n_val} shards) -> {val_dir}")
    print(
        "\nNext: RL stage reuses the same prompt set (paper-canonical SFT-then-RL, "
        "arXiv 2604.23747: full dataset for both SFT and RL); no disjoint split required. "
        "Launch SFT with scripts/sft_rl/run_sft_full.sh"
    )


if __name__ == "__main__":
    main()
