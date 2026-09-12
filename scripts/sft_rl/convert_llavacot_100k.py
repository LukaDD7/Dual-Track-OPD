#!/usr/bin/env python3
"""Convert LLaVA-CoT-100K to verl SFT parquet (repo schema).

Input : dataset/LLaVA-CoT-100k/train.jsonl
        image paths like "sqa/train/20839/image.png" (one per row), zip parts
        already concatenated + extracted to dataset/LLaVA-CoT-100k/images_extracted/
        (the archive unpacks without a top-level folder: sqa/, chartqa/, ...)
        conversations: [{from: human|gpt, value: str}] — multi-turn supported,
        first human turn usually references the image but WITHOUT <image>
        placeholders (LLaVA convention: single image attached per row).
Output: fc-opd-storage/outputs/fc_opd/sft_rl/llavacot100k/
          llavacot100k_sft__part_*.parquet (<=1500 rows)
          columns: messages, images, source, pass_rate, image_hash

Policies
  - one image per row (LLaVA-CoT format); full conversation is preserved as
    multi-turn messages (human/gpt -> user/assistant).
  - <image> placeholder injection: prepend exactly one "<image>" to the FIRST
    human turn if absent (verl asserts count == len(images); LLaVA-CoT rows
    reference the image implicitly).
  - rows whose image file is missing from images_extracted/ are dropped.
  - SFT-only (teacher CoT is the target; GT not rule-verifiable for RL).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
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

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-root", required=True,
                        help="directory containing sqa/ chartqa/ ... trees")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    rows: list[dict] = []
    stats = {"total": 0, "kept": 0, "missing_image": 0, "placeholder_injected": 0,
             "bad_conversation": 0}
    sub_source_stats: Counter = Counter()

    with open(args.input_jsonl) as f:
        for line in f:
            stats["total"] += 1
            r = json.loads(line)
            image_rel = str(r.get("image") or "")
            fp = image_root / image_rel if image_rel else None
            if not fp or not fp.exists():
                stats["missing_image"] += 1
                continue
            convs = r.get("conversations") or []
            if not convs or convs[0]["from"] != "human":
                stats["bad_conversation"] += 1
                continue
            messages = []
            n_ph_total = 0
            for c in convs:
                role = ROLE_MAP.get(c.get("from"))
                if role is None:
                    stats["bad_conversation"] += 1
                    break
                messages.append({"role": role, "content": str(c.get("value") or "")})
                n_ph_total += messages[-1]["content"].count("<image>")
            else:
                # inject exactly one <image> into the first user turn if none
                # anywhere in the conversation (LLaVA convention: implicit image)
                if n_ph_total == 0:
                    messages[0]["content"] = "<image>" + messages[0]["content"]
                    stats["placeholder_injected"] += 1
                elif n_ph_total != 1:
                    # multiple placeholders for a single image: normalize to 1
                    first = messages[0]["content"].replace("<image>", "")
                    messages[0]["content"] = "<image>" + first.strip()
                    stats["placeholder_injected"] += 1
                bytes_ = fp.read_bytes()
                h = hashlib.sha256(bytes_).hexdigest()[:16]
                sub = image_rel.split("/", 1)[0] if "/" in image_rel else "unknown"
                sub_source_stats[sub] += 1
                rows.append({
                    "messages": messages,
                    "images": [{"bytes": bytes_, "path": image_rel}],
                    "source": f"LLaVA-CoT-100K/{sub}",
                    "pass_rate": -1.0,
                    "image_hash": h,
                })
                stats["kept"] += 1
                continue
            # (break path: bad conversation, already counted)

    n_shards = 0
    for i in range(0, len(rows), args.shard_rows):
        tbl = pa.Table.from_pylist(rows[i : i + args.shard_rows], schema=SCHEMA)
        pq.write_table(tbl, out_dir / f"llavacot100k_sft__part_{n_shards:04d}.parquet")
        n_shards += 1
    stats_out = {
        **stats,
        "kept_rows": stats["kept"],
        "shards": n_shards,
        "by_subsource": dict(sub_source_stats.most_common()),
    }
    (out_dir / "llavacot100k_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"kept {stats['kept']} rows -> {n_shards} shards under {out_dir}")
    print(f"stats: {stats}")
    print(f"by subsource: {dict(sub_source_stats.most_common())}")


if __name__ == "__main__":
    main()
