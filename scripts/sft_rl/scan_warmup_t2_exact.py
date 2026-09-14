#!/usr/bin/env python3
"""Exact, image-decode-free oversized-row scan for warmup_t2 shards (bytes images).

Same math as scan_sft_full_exact.py (verified against the real pipeline there),
adapted to warmup_t2's storage format: images live in parquet `bytes`, not
file paths, so we read dimensions from the in-row image header only.

Two classes of bad rows are reported:
  - mismatch rows: n_image_tokens_in_first_max_length != n_features.
    These crash verl's forward with "Image features and image tokens do not
    match" (right-truncation cuts input_ids but not multi_modal_inputs).
  - oversized rows: full template length >= max_length. These lose their tail
    (the long-CoT response) to right-truncation, so they train on a corrupted
    target even when they don't crash.

Usage:
  python scan_warmup_t2_exact.py --warmup-dir DIR --model PATH \
      --out bad.json --max-length 12288 --workers 48
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import multiprocessing as mp
from pathlib import Path

from PIL import Image


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
    """Exact pads for the real pipeline:
    1) qwen_vl_utils fetch_image smart-resizes with qwen_vl_utils defaults
       (min=4*factor^2, max=16384*factor^2, factor=patch*merge=32);
    2) the Qwen2VLImageProcessor preprocess then smart-resizes again with the
       model's own min/max (65536 / 16777216);
    3) image_grid_thw = resized//patch_size, pads = grid.prod()//merge^2.
    Verified equal to the real MultiTurnSFTDataset output on sft_full images.
    """
    factor = 32
    h1, w1 = smart_resize(h, w, factor, 4 * factor * factor, 16384 * factor * factor)
    h2, w2 = smart_resize(h1, w1, factor, 65536, 16777216)
    return ((h2 // 16) // 2) * ((w2 // 16) // 2)


def _worker(args) -> list[dict]:
    shard_paths, model, max_length = args
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")

    bad = []
    for sp in shard_paths:
        import pyarrow.parquet as pq

        rows = pq.read_table(sp).to_pylist()
        for i, r in enumerate(rows):
            messages = r["messages"]
            imgs = r.get("images") or []
            # warmup_t2 stores images as in-row bytes; read only the header
            pads = []
            for im in imgs:
                with Image.open(io.BytesIO(im["bytes"])) as pic:
                    w, h = pic.size
                pads.append(image_pads_from_size(w, h))
            n_feat = sum(pads)

            # build the exact template content: replace <image> with literal pads
            # (vision_start + image_pad*pads + vision_end), then tokenize
            new_messages = []
            for m in messages:
                content = m["content"]
                if isinstance(content, str) and "<image>" in content:
                    pi = 0
                    parts = []
                    for seg in content.split("<image>"):
                        if parts:
                            parts.append(
                                "<|vision_start|>"
                                + "<|image_pad|>" * pads[pi]
                                + "<|vision_end|>"
                            )
                            pi += 1
                        parts.append(seg)
                    content = "".join(parts)
                new_messages.append({"role": m["role"], "content": content})

            enc = tok.apply_chat_template(
                new_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
            )
            ids = enc["input_ids"]
            if isinstance(ids[0], list):
                ids = ids[0]
            elif hasattr(ids, "tolist"):
                ids = ids.tolist()
            n_tok = sum(1 for t in ids[:max_length] if t == pad_id)
            rec = {
                "shard": Path(sp).name,
                "idx": i,
                "n_tok": n_tok,
                "n_feat": n_feat,
                "seqlen": len(ids),
                "grids": None,
                "source": r.get("source", "?"),
                "n_images": len(imgs),
                "pads": pads,
            }
            if n_tok != n_feat:
                rec["kind"] = "mismatch"
                bad.append(rec)
            elif len(ids) >= max_length:
                rec["kind"] = "oversize_truncated"
                bad.append(rec)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--name-prefix", default="sft_warmup")
    args = ap.parse_args()

    shards = (
        sorted(glob.glob(f"{args.warmup_dir}/{args.name_prefix}_train__part_*.parquet"))
        + sorted(glob.glob(f"{args.warmup_dir}/{args.name_prefix}_val__part_*.parquet"))
    )
    if not shards:
        raise SystemExit("no shards found")

    groups = [shards[w :: args.workers] for w in range(args.workers)]
    groups = [g for g in groups if g]

    mp.set_start_method("fork", force=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(
            _worker,
            [(g, args.model, args.max_length) for g in groups],
        )

    bad = [b for r in results for b in r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bad, ensure_ascii=False, indent=1))
    print(f"total bad: {len(bad)}")
    from collections import Counter

    print("by kind:", dict(Counter(b.get("kind") for b in bad)))
    print("by source:", dict(Counter(b["source"] for b in bad)))
    print("by shard:", dict(Counter(b["shard"] for b in bad)))


if __name__ == "__main__":
    main()
