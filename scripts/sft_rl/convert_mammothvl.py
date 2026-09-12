#!/usr/bin/env python3
"""Convert MAmmoTH-VL (ov_2M / si_10M) to verl SFT parquet (repo schema).

Input : dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M.json (or si_10M)
        rows: {id, image: str | list[str] | null, conversations:
               [{from: human|gpt, value}], source}
        image paths reference the tar.gz shard trees
        (single_image_data/shard_*.tar.gz, multi_image_data/shard_*.tar.gz),
        which unpack to: LLaVA-OneVision-Data/..., M4-Instruct-Data/...,
        Cambrian-10M/..., tinychart_train/..., etc.
Output: fc-opd-storage/outputs/fc_opd/sft_rl/mammothvl_ov2m/
          mammothvl_ov2m_sft__part_*.parquet (<=1500 rows)
          mammothvl_ov2m_stats.json
        columns: messages, images, source, pass_rate, image_hash

Policies
  - images resolved from --image-root (the extracted shard tree); rows with
    missing image files are dropped (stats reported).
  - <image> count must equal len(images): source rows use LLaVA-style
    "<image>" tokens; normalize by injecting/stripping as needed (inject at
    first human turn front; multi-image rows whose count can't be fixed are
    dropped).
  - multi-turn conversations preserved (human/gpt -> user/assistant).
  - SFT only: MAmmoTH-VL is instruction-following data, GT answers are free
    text (not rule-verifiable for RL).

Usage:
  python3 scripts/sft_rl/convert_mammothvl.py \
    --input-json dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M.json \
    --image-root <extracted-shards-root> \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/mammothvl_ov2m
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson
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

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=0, help="debug: max rows to read")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    # Streaming buffer: flush a parquet shard every `shard_rows` instead of
    # accumulating the whole 2M-row corpus (with embedded image bytes) in RAM.
    buf: list[dict] = []
    n_shards = 0

    def write_shard(chunk: list[dict], idx: int) -> None:
        pq.write_table(
            pa.Table.from_pylist(chunk, schema=SCHEMA),
            out_dir / f"mammothvl_ov2m_sft__part_{idx:04d}.parquet",
        )

    stats = {"total": 0, "kept": 0, "no_image_field": 0, "missing_image": 0,
             "placeholder_fixed": 0, "placeholder_unfixable": 0, "bad_conversation": 0}
    src_stats: Counter = Counter()

    with open(args.input_json, "rb") as f:
        for r in ijson.items(f, "item"):
            stats["total"] += 1
            if args.limit and stats["total"] > args.limit:
                break
            img = r.get("image")
            if isinstance(img, str):
                img_paths = [img] if img else []
            elif isinstance(img, list):
                img_paths = [str(p) for p in img if p]
            else:
                img_paths = []
            if not img_paths:
                stats["no_image_field"] += 1
                continue
            # resolve files
            imgs = []
            ok = True
            for p in img_paths:
                fp = image_root / p
                if not fp.exists():
                    ok = False
                    break
                imgs.append({"bytes": fp.read_bytes(), "path": p})
            if not ok:
                stats["missing_image"] += 1
                continue
            convs = r.get("conversations") or []
            if not convs or convs[0].get("from") != "human":
                stats["bad_conversation"] += 1
                continue
            messages = []
            n_ph = 0
            bad = False
            for c in convs:
                role = ROLE_MAP.get(c.get("from"))
                if role is None:
                    bad = True
                    break
                content = str(c.get("value") or "")
                messages.append({"role": role, "content": content})
                n_ph += content.count("<image>")
            if bad:
                stats["bad_conversation"] += 1
                continue
            # normalize placeholder count to len(imgs)
            if n_ph != len(imgs):
                first = messages[0]["content"]
                stripped = first.replace("<image>", "").lstrip()
                if n_ph < len(imgs) or n_ph == 0:
                    messages[0]["content"] = "<image>" * len(imgs) + stripped
                    stats["placeholder_fixed"] += 1
                else:
                    # n_ph > len(imgs) and >0: would need to strip mid-conversation
                    messages[0]["content"] = "<image>" * len(imgs) + stripped
                    stats["placeholder_fixed"] += 1
                    # multi-turn placeholders beyond the first turn remain; verl
                    # counts across all messages, so strip them too
                    for m in messages[1:]:
                        m["content"] = m["content"].replace("<image>", "")
            h = ",".join(hashlib.sha256(i["bytes"]).hexdigest()[:16] for i in imgs)
            source = str(r.get("source") or "") or "mammoth"
            src_stats[source] += 1
            row = {
                "messages": messages,
                "images": imgs,
                "source": f"MAmmoTH-VL/{source}",
                "pass_rate": -1.0,
                "image_hash": h,
            }
            del imgs  # embedded bytes now owned by buf; free the per-row list
            buf.append(row)
            stats["kept"] += 1
            if len(buf) >= args.shard_rows:
                write_shard(buf, n_shards)
                n_shards += 1
                buf = []
            if stats["total"] % 200000 == 0:
                print(f"[scan] rows read={stats['total']} kept={stats['kept']}", flush=True)

    if buf:
        write_shard(buf, n_shards)
        n_shards += 1
        buf = []
    stats_out = {
        **stats,
        "kept_rows": stats["kept"],
        "shards": n_shards,
        "by_source": dict(src_stats.most_common()),
    }
    (out_dir / "mammothvl_ov2m_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"kept {stats['kept']} rows -> {n_shards} shards under {out_dir}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
