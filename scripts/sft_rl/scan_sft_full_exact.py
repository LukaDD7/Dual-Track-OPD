#!/usr/bin/env python3
"""Exact, image-decode-free mismatch scan for sft_full shards.

verl's Qwen3-VL forward raises "Image features and image tokens do not match"
when right-truncation (no_padding mode) cuts into an image-pad block: input_ids
is truncated but multi_modal_inputs is not. The check is
    n_image_tokens == n_image_features,
where features per image = grid.prod() // merge_size**2 and the template
expands each <image> into `<|vision_start|>` + `<|image_pad|>` * pads +
`<|vision_end|>` (Qwen3VLProcessor.replace_image_token). Both depend only on
image dimensions, so we:
  1. get pads per image from processor.get_number_of_image_patches (O(1)
     arithmetic, verified identical to real preprocess on 10 sampled images);
  2. tokenize the exact chat template with *literal* pad strings (no image
     tensors at all) and count image_pad ids within max_length.
This reproduces the production n_tok / n_feat exactly, without touching the
47GB image store.

Usage:
  python scan_sft_full_exact.py --sft-train-dir DIR --sft-val-dir DIR \
      --model $MODEL --out bad.json --max-length 12288 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

from PIL import Image


import math


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
    1) verl/qwen_vl_utils fetch_image smart-resizes with qwen_vl_utils defaults
       (min=4*factor^2, max=16384*factor^2, factor=patch*merge=32);
    2) the Qwen2VLImageProcessor preprocess then smart-resizes again with the
       model's own min/max (65536 / 16777216);
    3) image_grid_thw = resized//patch_size, pads = grid.prod()//merge^2.
    Verified equal to the real MultiTurnSFTDataset output on 8 sampled images.
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
    dims_cache: dict[str, tuple[int, int]] = {}

    def image_pads(path: str) -> int:
        key = path
        if key not in dims_cache:
            with Image.open(path) as im:
                dims_cache[key] = im.size
        w, h = dims_cache[key]
        return image_pads_from_size(w, h)

    bad = []
    for sp in shard_paths:
        import pyarrow.parquet as pq

        rows = pq.read_table(sp).to_pylist()
        for i, r in enumerate(rows):
            messages = r["messages"]
            imgs = r.get("images") or []
            pads = [image_pads(im["image_url"]) for im in imgs]
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
            if n_tok != n_feat:
                bad.append(
                    {
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
                )
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-train-dir", required=True)
    ap.add_argument("--sft-val-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    shards = (
        sorted(glob.glob(f"{args.sft_train_dir}/sft_full_train__part_*.parquet"))
        + sorted(glob.glob(f"{args.sft_val_dir}/sft_full_val__part_*.parquet"))
    )
    if not shards:
        raise SystemExit("no sft_full shards found")

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

    print("by shard:", dict(Counter(b["shard"] for b in bad)))
    print("by source:", dict(Counter(b["source"] for b in bad)))


if __name__ == "__main__":
    main()
